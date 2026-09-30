"""
OpenMRG (Gothenburg, Sweden) raw archive -> OpenSense-1.0 xarray datasets.

The published archive is raw instrument data, not rainfall:

* ``cml/cml.nc`` - 794,887 timesteps x 728 sublinks of TSL/RSL in dBm at 10 s.
* ``radar/radar.nc`` - 26,496 timesteps of pseudo-dBZ on a 48x37 grid at 5 min.
* ``gauges/`` - municipal and SMHI gauge CSVs at 1 min / 15 min.

This module cuts an event window out of each, runs the CML retrieval chain, and
returns datasets carrying the coordinate names ``mergeplg`` expects.

The retrieval chain is ``core.opensense.retrieval`` - the same code every
other source in this project runs through - followed by averaging the two
directions of each link into one ``cml_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import poligrain as plg
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[3]

from core.opensense import conventions as cv  # noqa: E402
from core.opensense.retrieval import (RetrievalConfig, combine_sublinks,  # noqa: E402
                                      retrieve, retrieve_improved,
                                      total_loss_from)

from core import data_paths as dp  # noqa: E402

# inputs, from ~/data/cml (or data/interim): see core/data_paths.py
CML_NC = dp.data_path(dp.OPENMRG_CML)
CML_META = dp.data_path(dp.OPENMRG_CML_META)
RADAR_NC = dp.data_path(dp.OPENMRG_RADAR)
GAUGES = dp.data_path(dp.OPENMRG_GAUGES)
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
    df = pd.read_csv(CML_META)
    df["Sublink"] = df["Sublink"].astype(int)
    return df.set_index("Sublink")


def retrieve_cml_rain(ds_raw: xr.Dataset, meta: pd.DataFrame,
                      cfg: RetrievalConfig | None = None) -> xr.Dataset:
    """Raw TSL/RSL for one window -> per-sublink path rain rate (mm/h).

    The chain itself is ``core.opensense.retrieval``; this function only
    joins the archive's bespoke sublink metadata table onto the signals.
    ``cfg`` defaults to the standard windows at OpenMRG's native 10 s - a
    5-minute wet/dry window and a 3-hour baseline.

    Two defaults are calibration choices, not physics, and both were checked
    against the municipal gauges on the 25 August event (see the project
    README for the table):

    ``baseline_window``
        One hour is too short: a centred median over a multi-hour rain event
        absorbs the rain into the dry reference and suppresses the retrieval
        to ~0.68 of gauge totals. Three hours gives 0.94.
    ``waa_max_db``
        The dominant magnitude knob. The Schleiss et al. (2013) value of
        2.3 dB drives this network to 0.35 of gauge totals on this event;
        0.5 dB gives 0.79. ``retrieval_benchmark.py`` shows why no static
        value transfers between events, and what does better.
    """
    cfg = cfg or RetrievalConfig.for_interval(10.0)
    sublinks = ds_raw.sublink.to_numpy().astype(int)
    known = [s for s in sublinks if s in meta.index]
    ds_raw = ds_raw.sel(sublink=known)
    m = meta.loc[known]

    loss = total_loss_from(ds_raw.rsl.to_numpy(), ds_raw.tsl.to_numpy())
    res = retrieve(loss, m["Length_km"].to_numpy(), m["Frequency_GHz"].to_numpy(),
                   m["Polarization"].to_numpy(), cfg)

    out = xr.Dataset(
        {"R": (("time", "sublink"), res["R"]),
         "A_rain": (("time", "sublink"), res["A_obs"]),
         "wet": (("time", "sublink"), res["wet"])},
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


def raw_to_opensense(ds_raw: xr.Dataset, meta: pd.DataFrame) -> xr.Dataset:
    """The archive's flat (time, sublink) layout -> OpenSense-1.0.

    Every link has exactly two sublinks, directions A and B, which become
    ``sublink_id``. Geometry is taken from direction A (its Near site is
    ``site_0``). Needed by anything that works per link rather than per
    sublink - the nearby-link wet/dry method compares neighbouring *links*,
    and would count a link's own reverse direction as a neighbour otherwise.
    """
    m = meta.reset_index()
    m = m[m.Sublink.isin(ds_raw.sublink.values.astype(int))]
    pairs = m.pivot(index="Link", columns="Direction", values="Sublink")
    pairs = pairs.dropna().astype(int)                   # both directions present
    order = [pairs[d].to_numpy() for d in ("A", "B")]

    def stack(var):
        return np.stack([ds_raw[var].sel(sublink=s).to_numpy() for s in order],
                        axis=-1).astype(float)           # (time, cml_id, sublink_id)

    a = meta.loc[order[0]]
    freq = np.stack([meta.loc[s, "Frequency_GHz"].to_numpy() for s in order], -1)
    pol = np.stack([meta.loc[s, "Polarization"].to_numpy() for s in order], -1)
    dims = ("time", "cml_id", "sublink_id")
    ds = xr.Dataset(
        {"tsl": (dims, stack("tsl")), "rsl": (dims, stack("rsl"))},
        coords={"time": ds_raw.time.to_numpy(), "cml_id": pairs.index.to_numpy(),
                "sublink_id": ["A", "B"],
                "site_0_lat": ("cml_id", a.NearLatitude_DecDeg.to_numpy()),
                "site_0_lon": ("cml_id", a.NearLongitude_DecDeg.to_numpy()),
                "site_1_lat": ("cml_id", a.FarLatitude_DecDeg.to_numpy()),
                "site_1_lon": ("cml_id", a.FarLongitude_DecDeg.to_numpy()),
                "length": ("cml_id", a.Length_km.to_numpy(), {"units": "km"}),
                "frequency": (("cml_id", "sublink_id"), freq, {"units": "GHz"}),
                "polarization": (("cml_id", "sublink_id"), pol)})
    from core.opensense.example_data import normalize_cml
    return normalize_cml(ds, CRS)


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
    ds = xr.open_dataset(RADAR_NC).sel(time=slice(start, end))

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
    values = pd.read_csv(GAUGES / "city/CityGauges-2015JJA.csv",
                         parse_dates=["Time_UTC"]).set_index("Time_UTC")
    values = values.loc[start:end]
    meta = pd.read_csv(GAUGES / "city/CityGauges-metadata.csv")

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
                cache: bool = True, baseline_margin_h: float = 4.0,
                retrieval: str = "default") -> tuple:
    """Return (radar, cml, gauges) for one event, all on a common time axis.

    Everything is resampled to the radar's native 5-minute step, which is the
    coarsest of the three and the grid the merge runs on.

    ``retrieval="improved"`` uses ``retrieval.retrieve_improved`` (nearby-link
    wet/dry + Leijnse 2008) instead of the default chain, with a 24-hour
    margin because the nearby-link reference level needs a day of history.
    Cached separately, as ``<event>_improved_*.nc``.
    """
    if retrieval not in ("default", "improved"):
        raise ValueError(f"retrieval must be 'default' or 'improved', got {retrieval!r}")
    if retrieval == "improved":
        baseline_margin_h = max(baseline_margin_h, 24.0)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    stem = event.key if retrieval == "default" else f"{event.key}_improved"
    paths = {n: PROCESSED / f"{stem}_{n}.nc"
             for n in ("radar", "cml", "gauge")}
    if cache and all(p.exists() for p in paths.values()):
        print(f"  [cache] {stem}")
        return tuple(xr.open_dataset(paths[n]) for n in ("radar", "cml", "gauge"))

    print(f"  radar   {event.start} .. {event.end}")
    rad = load_radar(event.start, event.end)

    # The baseline estimator needs dry weather on both sides of the event to
    # anchor on, so the raw window is padded and trimmed back afterwards.
    pad = pd.Timedelta(hours=baseline_margin_h)
    print(f"  cml     loading raw TSL/RSL (+/-{baseline_margin_h} h for baseline)")
    raw = xr.open_dataset(CML_NC).sel(
        time=slice(str(pd.Timestamp(event.start) - pad),
                   str(pd.Timestamp(event.end) + pad))).load()
    meta = load_cml_metadata()
    print(f"          {raw.sizes['time']} samples x {raw.sizes['sublink']} sublinks")
    if retrieval == "default":
        sub = retrieve_cml_rain(raw, meta)
        sub = sub.sel(time=slice(event.start, event.end))
        cml = sublinks_to_cmls(sub)
    else:
        ds = raw_to_opensense(raw, meta)
        res = retrieve_improved(ds)
        print(f"          {res.attrs['wet_dry']}")
        cml = cv.project_cml(combine_sublinks(res), CRS)
        cml = cml.sel(time=slice(event.start, event.end))
        cml.coords["length"] = cml.length_km
        cml.coords["frequency"] = cml.frequency_ghz
        cml.R.attrs["units"] = "mm h-1"

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
                        license="CC-BY-SA-4.0", retrieval=retrieval)
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
