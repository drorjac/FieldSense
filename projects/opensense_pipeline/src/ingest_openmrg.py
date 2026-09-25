"""
OpenMRG (Gothenburg, Sweden) raw archive -> OpenSense-1.0 xarray datasets.

The published archive is raw instrument data, not rainfall:

* ``cml/cml.nc`` - 794,887 timesteps x 728 sublinks of TSL/RSL in dBm at 10 s.
* ``radar/radar.nc`` - 26,496 timesteps of pseudo-dBZ on a 48x37 grid at 5 min.
* ``gauges/`` - municipal and SMHI gauge CSVs at 1 min / 15 min.

This module cuts an event window out of each, runs the CML retrieval chain, and
returns datasets carrying the coordinate names ``mergeplg`` expects.

Retrieval chain, per sublink:

1. total loss ``A = TSL - RSL``
2. wet/dry classification by rolling standard deviation (Schleiss & Berne 2010)
3. baseline = last dry level, held through the wet period
4. ``A_rain = A - baseline``, floored at zero
5. wet-antenna attenuation subtracted (Schleiss et al. 2013)
6. ``R = (A_rain / (k L))^(1/alpha)`` with ITU-R P.838-3 coefficients
7. the two directions of a link are averaged into one ``cml_id``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import poligrain as plg
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[3]

from core.itu_p838 import get_k_alpha

RAW = REPO_ROOT / "dataset/open_datasets/OpenMRG_Sweden/raw/extracted"
PROCESSED = REPO_ROOT / "dataset/open_datasets/OpenMRG_Sweden/processed"

# UTM 32N covers Gothenburg; mergeplg needs a projected CRS in metres.
CRS = "EPSG:32632"

# Radar is stored as pseudo-dBZ with scale_factor 0.4 and add_offset -30, both
# applied automatically on read. Z = a R^b inverts to R = (Z/a)^(1/b); these
# are fallbacks only, the file's own zr_a/zr_b attributes win.
ZR_A, ZR_B = 200.0, 1.5


@dataclass(frozen=True)
class Event:
    """A named rainfall event window."""

    key: str
    label: str
    start: str
    end: str
    note: str = ""


# Chosen from the municipal gauge record: the three wettest regimes in the
# summer, spanning a documented convective cell, a mixed event already used
# elsewhere in this repository, and a widespread frontal day.
EVENTS = {
    "torslanda": Event(
        "torslanda", "Torslanda convective cell", "2015-07-28T14:00",
        "2015-07-28T20:00",
        "the event the dataset ships an animation of; localized and intense"),
    "aug25": Event(
        "aug25", "25 August mixed event", "2015-08-25T04:00",
        "2015-08-25T11:00",
        "same event projects/spatial_interpolation evaluates on"),
    "jun17": Event(
        "jun17", "17 June widespread rain", "2015-06-17T11:00",
        "2015-06-17T18:00",
        "wettest day of the record, 26.5 mm domain mean"),
}


# --------------------------------------------------------------------------
# CML
# --------------------------------------------------------------------------
def load_cml_metadata() -> pd.DataFrame:
    """Sublink metadata: geometry, frequency, polarization, length."""
    df = pd.read_csv(RAW / "cml/cml_metadata.csv")
    df["Sublink"] = df["Sublink"].astype(int)
    return df.set_index("Sublink")


def _wet_dry_rolling_std(attenuation: np.ndarray, window: int,
                         threshold_db: float) -> np.ndarray:
    """Wet flag per sample from the rolling standard deviation of total loss.

    Schleiss & Berne (2010): rain makes the signal fluctuate, so a window whose
    spread exceeds a threshold is classified wet. Operates along axis 0 (time).
    """
    s = pd.DataFrame(attenuation)
    roll = s.rolling(window=window, center=True, min_periods=max(2, window // 4))
    return (roll.std().to_numpy() > threshold_db)


def _baseline_from_dry(attenuation: np.ndarray, wet: np.ndarray,
                       window: int) -> np.ndarray:
    """Dry-weather reference level per sublink.

    The dry level is the attenuation with no rain on the path: free-space loss
    plus hardware offsets, drifting slowly with temperature. It is estimated as
    a long rolling median over samples classified dry, then carried across wet
    spells.

    A rolling median of the dry samples beats holding the single last dry
    value, which inherits that sample's noise into the whole wet spell.
    """
    dry_only = pd.DataFrame(np.where(wet, np.nan, attenuation))
    base = (dry_only
            .rolling(window=window, center=True, min_periods=1)
            .median()
            .ffill()
            .bfill())

    # A link wet for the entire window has no dry reference; its own lower
    # decile is the best available proxy.
    all_wet = base.isna().all(axis=0)
    if all_wet.any():
        fallback = pd.DataFrame(attenuation).quantile(0.1)
        base.loc[:, all_wet] = fallback[all_wet].to_numpy()
    return base.to_numpy()


def retrieve_cml_rain(ds_raw: xr.Dataset, meta: pd.DataFrame,
                      wet_window: int = 30, wet_threshold_db: float = 0.8,
                      baseline_window: int = 1080,
                      min_attenuation_db: float = 0.1,
                      waa_max_db: float = 0.5,
                      waa_rate_per_mm_h: float = 0.28,
                      min_length_km: float = 0.5) -> xr.Dataset:
    """Raw TSL/RSL for one window -> per-sublink path rain rate (mm/h).

    Window lengths are in samples; at the native 10 s spacing ``wet_window``
    30 is 5 minutes and ``baseline_window`` 1080 is three hours.

    Two defaults are calibration choices, not physics, and both were checked
    against the municipal gauges on the 25 August event (see the project
    README for the table):

    ``baseline_window``
        One hour is too short: a centred median over a multi-hour rain event
        absorbs the rain into the dry reference and suppresses the retrieval
        to ~0.68 of gauge totals. Three hours gives 0.94.
    ``waa_max_db``
        The dominant magnitude knob. The Schleiss et al. (2013) value of
        2.3 dB drives this network to 0.35 of gauge totals, because a 2 km
        link at 23 GHz only develops ~2.7 dB of rain attenuation at 10 mm/h -
        subtracting 2.3 dB of it removes most of the signal. 0.5 dB gives
        0.79. Correlation against gauges is flat (0.75-0.80) across the whole
        range, so this shifts magnitude only, never skill.
    """
    sublinks = ds_raw.sublink.to_numpy().astype(int)
    known = [s for s in sublinks if s in meta.index]
    ds_raw = ds_raw.sel(sublink=known)
    m = meta.loc[known]

    tsl = ds_raw.tsl.to_numpy()
    rsl = ds_raw.rsl.to_numpy()
    total_loss = tsl - rsl                      # dB

    # TSL is often a constant nominal value; a NaN in either channel is a gap.
    bad = ~np.isfinite(total_loss)
    total_loss = pd.DataFrame(total_loss).ffill().bfill().to_numpy()

    wet = _wet_dry_rolling_std(total_loss, wet_window, wet_threshold_db)
    baseline = _baseline_from_dry(total_loss, wet, baseline_window)

    # Rain attenuation is the excess over the dry reference, everywhere.
    #
    # It is tempting to also force a_rain to zero wherever the classifier says
    # dry, but that silently deletes steady rain: the rolling-standard-
    # deviation test keys on fluctuation, and widespread stratiform rain
    # attenuates steadily, so hours of real rain get labelled dry. Subtracting
    # the baseline already drives genuinely dry periods to ~0.
    a_rain = np.clip(total_loss - baseline, 0.0, None)

    # Below the quantization floor the retrieval only amplifies noise.
    a_rain[a_rain < min_attenuation_db] = 0.0

    # Wet-antenna attenuation is a saturating function of the rain rate we are
    # solving for, so it is removed by fixed-point iteration.
    lengths = m["Length_km"].to_numpy()[None, :]
    freqs = m["Frequency_GHz"].to_numpy()
    pols = m["Polarization"].str.lower().to_numpy()
    ka = np.array([get_k_alpha(float(f), str(p))
                   for f, p in zip(freqs, pols)])
    k, alpha = ka[:, 0][None, :], ka[:, 1][None, :]

    rain = np.zeros_like(a_rain)
    for _ in range(8):
        waa = waa_max_db * (1.0 - np.exp(-waa_rate_per_mm_h * rain))
        effective = np.clip(a_rain - waa, 0.0, None)
        rain = (effective / (k * lengths)) ** (1.0 / alpha)
    rain[~np.isfinite(rain)] = 0.0
    rain[bad] = np.nan

    # Short links cannot resolve rain: their path attenuation is below the
    # quantization floor, so the retrieval is pure noise amplification.
    too_short = lengths.ravel() < min_length_km
    rain[:, too_short] = np.nan

    out = xr.Dataset(
        {"R": (("time", "sublink"), rain),
         "A_rain": (("time", "sublink"), a_rain),
         "wet": (("time", "sublink"), wet)},
        coords={"time": ds_raw.time.to_numpy(),
                "sublink": np.asarray(known, dtype=int)},
    )
    for col, name in [("Link", "link_id"), ("Frequency_GHz", "frequency"),
                      ("Length_km", "length"), ("Polarization", "polarization"),
                      ("NearLatitude_DecDeg", "site_0_lat"),
                      ("NearLongitude_DecDeg", "site_0_lon"),
                      ("FarLatitude_DecDeg", "site_1_lat"),
                      ("FarLongitude_DecDeg", "site_1_lon")]:
        out.coords[name] = ("sublink", m[col].to_numpy())
    return out


def sublinks_to_cmls(ds_sub: xr.Dataset) -> xr.Dataset:
    """Average the two directions of each link into one ``cml_id``.

    Both directions traverse the same physical path, so averaging halves the
    retrieval noise. Links where only one direction survived are kept as-is.
    The two directions report Near/Far swapped; taking the first occurrence
    fixes one orientation per link, which is all the line geometry needs.

    Done with numpy rather than ``xarray.groupby`` because the sublink
    dimension carries several non-index coordinates and groupby tries to align
    them against each other.
    """
    link_ids = np.asarray(ds_sub.link_id)
    uniq, inverse = np.unique(link_ids, return_inverse=True)

    rain = np.asarray(ds_sub.R)                      # (time, sublink)
    n_time, n_link = rain.shape[0], uniq.size
    averaged = np.full((n_time, n_link), np.nan, dtype=float)
    for j in range(n_link):
        members = rain[:, inverse == j]
        with np.errstate(invalid="ignore"):
            averaged[:, j] = np.nanmean(members, axis=1)

    out = xr.Dataset(
        {"R": (("time", "cml_id"), averaged)},
        coords={"time": np.asarray(ds_sub.time),
                "cml_id": uniq},
    )

    first_idx = np.array([np.flatnonzero(inverse == j)[0] for j in range(n_link)])
    for name in ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon",
                 "frequency", "length"):
        out.coords[name] = ("cml_id", np.asarray(ds_sub[name])[first_idx])

    x0, y0 = plg.spatial.project_point_coordinates(
        out.site_0_lon, out.site_0_lat, CRS)
    x1, y1 = plg.spatial.project_point_coordinates(
        out.site_1_lon, out.site_1_lat, CRS)
    out.coords["site_0_x"] = ("cml_id", np.asarray(x0))
    out.coords["site_0_y"] = ("cml_id", np.asarray(y0))
    out.coords["site_1_x"] = ("cml_id", np.asarray(x1))
    out.coords["site_1_y"] = ("cml_id", np.asarray(y1))
    out.coords["x"] = ("cml_id", (np.asarray(x0) + np.asarray(x1)) / 2)
    out.coords["y"] = ("cml_id", (np.asarray(y0) + np.asarray(y1)) / 2)
    out.R.attrs["units"] = "mm h-1"
    return out


# --------------------------------------------------------------------------
# radar
# --------------------------------------------------------------------------
def load_radar(start: str, end: str) -> xr.Dataset:
    """Radar window as rain rate (mm/h) on a projected grid."""
    ds = xr.open_dataset(RAW / "radar/radar.nc").sel(time=slice(start, end))

    # xarray applies scale_factor/add_offset on read, so ds.data is already
    # dBZ - applying the 0.4/-30 transform again would be double-scaling. The
    # _FillValue of 255 has likewise already become NaN.
    dbz = ds.data.astype("float32")

    # The file carries its own Z-R coefficients; prefer them over a textbook
    # Marshall-Palmer pair. OpenMRG ships zr_b = 1.5, not the usual 1.6.
    zr_a = float(ds.data.attrs.get("zr_a", ZR_A))
    zr_b = float(ds.data.attrs.get("zr_b", ZR_B))

    z = 10.0 ** (dbz / 10.0)
    rain = (z / zr_a) ** (1.0 / zr_b)

    out = xr.Dataset({"R": rain.fillna(0.0)})
    out.coords["latitudes"] = ds.lat
    out.coords["longitudes"] = ds.lon

    xs, ys = plg.spatial.project_point_coordinates(ds.lon, ds.lat, CRS)
    out.coords["x_grid"], out.coords["y_grid"] = xs, ys

    # mergeplg also reads 1-D x/y for grid bookkeeping. A lat/lon grid is not
    # exactly rectilinear once projected, so these are the centre row and
    # centre column rather than exact axes - the same approximation
    # mergeplg.io makes for this dataset. Over Gothenburg the departure is
    # well under one 1 km cell; x_grid/y_grid stay exact and are what the
    # geometry actually uses.
    xs_v, ys_v = np.asarray(xs), np.asarray(ys)
    mid_y, mid_x = xs_v.shape[0] // 2, xs_v.shape[1] // 2
    out.coords["x"] = ("x", xs_v[mid_y, :])
    out.coords["y"] = ("y", ys_v[:, mid_x])
    out.R.attrs.update(units="mm h-1",
                       note=f"Z = {zr_a} R^{zr_b} (from file attributes)")
    return out


# --------------------------------------------------------------------------
# gauges
# --------------------------------------------------------------------------
def load_city_gauges(start: str, end: str) -> xr.Dataset:
    """Municipal tipping-bucket gauges as rain rate (mm/h)."""
    values = pd.read_csv(RAW / "gauges/city/CityGauges-2015JJA.csv",
                         parse_dates=["Time_UTC"]).set_index("Time_UTC")
    values = values.loc[start:end]
    meta = pd.read_csv(RAW / "gauges/city/CityGauges-metadata.csv")

    name_col = next(c for c in meta.columns if "ame" in c)
    lat_col = next(c for c in meta.columns if "atitude" in c)
    lon_col = next(c for c in meta.columns if "ongitude" in c)
    meta = meta.set_index(name_col)

    stations = [c for c in values.columns if c in meta.index]
    values = values[stations]

    # The CSV carries 1-minute accumulations in mm; x60 gives mm/h.
    da = xr.DataArray(
        values.to_numpy() * 60.0,
        dims=("time", "id"),
        coords={"time": values.index.tz_localize(None).to_numpy(),
                "id": stations},
    )
    ds = xr.Dataset({"R": da})
    ds.coords["lat"] = ("id", meta.loc[stations, lat_col].to_numpy())
    ds.coords["lon"] = ("id", meta.loc[stations, lon_col].to_numpy())
    x, y = plg.spatial.project_point_coordinates(ds.lon, ds.lat, CRS)
    ds.coords["x"], ds.coords["y"] = x, y
    ds.R.attrs["units"] = "mm h-1"
    return ds


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def build_event(event: Event, resample: str = "5min",
                cache: bool = True, baseline_margin_h: float = 4.0) -> tuple:
    """Return (radar, cml, gauges) for one event, all on a common time axis.

    Everything is resampled to the radar's native 5-minute step, which is the
    coarsest of the three and the grid the merge runs on.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    paths = {n: PROCESSED / f"{event.key}_{n}.nc"
             for n in ("radar", "cml", "gauge")}
    if cache and all(p.exists() for p in paths.values()):
        print(f"  [cache] {event.key}")
        return tuple(xr.open_dataset(paths[n]) for n in ("radar", "cml", "gauge"))

    print(f"  radar   {event.start} .. {event.end}")
    rad = load_radar(event.start, event.end)

    # The baseline estimator needs dry weather on both sides of the event to
    # anchor on, so the raw window is padded and trimmed back afterwards.
    pad = pd.Timedelta(hours=baseline_margin_h)
    print(f"  cml     loading raw TSL/RSL (+/-{baseline_margin_h} h for baseline)")
    raw = xr.open_dataset(RAW / "cml/cml.nc").sel(
        time=slice(str(pd.Timestamp(event.start) - pad),
                   str(pd.Timestamp(event.end) + pad))).load()
    meta = load_cml_metadata()
    print(f"          {raw.sizes['time']} samples x {raw.sizes['sublink']} sublinks")
    sub = retrieve_cml_rain(raw, meta)
    sub = sub.sel(time=slice(event.start, event.end))
    cml = sublinks_to_cmls(sub)

    print("  gauges  loading")
    gauge = load_city_gauges(event.start, event.end)

    # Common 5-minute axis. Rain rate averages; that is the right operator for
    # an intensity, unlike an accumulation which would sum.
    cml = cml.resample(time=resample).mean()
    gauge = gauge.resample(time=resample).mean()
    rad = rad.resample(time=resample).mean()

    times = np.intersect1d(np.intersect1d(rad.time, cml.time), gauge.time)
    rad, cml, gauge = (d.sel(time=times) for d in (rad, cml, gauge))

    for name, ds in (("radar", rad), ("cml", cml), ("gauge", gauge)):
        ds.attrs.update(event=event.key, label=event.label,
                        source="OpenMRG", doi="10.5281/zenodo.7107689",
                        license="CC-BY-SA-4.0")
        if cache:
            ds.to_netcdf(paths[name])
    return rad, cml, gauge


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", choices=sorted(EVENTS), default="aug25")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    ev = EVENTS[args.event]
    print(f"{ev.label}  ({ev.note})")
    rad, cml, gauge = build_event(ev, cache=not args.no_cache)
    print(f"\nradar  {dict(rad.sizes)}  max {float(rad.R.max()):.1f} mm/h")
    print(f"cml    {dict(cml.sizes)}  max {float(cml.R.max()):.1f} mm/h")
    print(f"gauge  {dict(gauge.sizes)}  max {float(gauge.R.max()):.1f} mm/h")
