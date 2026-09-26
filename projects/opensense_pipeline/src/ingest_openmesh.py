"""
OpenMesh NYC -> the same (radar, cml, gauge) triple the other ingests return.

Three things make this dataset different from OpenMRG and OpenRainER, and all
three matter:

**RSL only.** OpenMesh publishes received power with no transmitted power, so
path loss is ``-RSL`` rather than ``TSL - RSL``. Anything that moves the
transmitter - automatic power control, a hardware change - enters as if it
were rain. See ``retrieval.total_loss_from``.

**Up to three sublinks per link, in different bands.** Roughly 5-6, 24 and
58-69 GHz. In the 20-day subset most links report a single sublink (51 of
75), 20 report two - usually both directions of one band - and only 4 report
three. Where bands differ they are not redundant measurements: a 60 GHz
sublink saturates in rain that a 5.7 GHz one barely registers, so sublinks
are retrieved separately and combined by taking the band that is actually
responsive.

**No radar in the dataset.** It comes from KOKX through
``core/radar/nexrad.py``.

**And it snows.** The record covers the 2023-24 winter. ITU-R P.838-3 is a
*rain* relation; applying it to snow is wrong, and the module says so rather
than quietly returning a number. See ``PHASE_NOTE``.

    python -m ingest_openmesh --event rain_0113
    python -m ingest_openmesh --list
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from core.opensense import conventions as cv  # noqa: E402
from core.opensense.retrieval import link_metadata, retrieve_dataset  # noqa: E402

PROCESSED = REPO_ROOT / "dataset/open_datasets/OpenMesh_NYC/processed"

# UTM 18N covers New York City.
CRS = "EPSG:32618"

PHASE_NOTE = (
    "ITU-R P.838-3 relates specific attenuation to *rain* rate. On a snow day "
    "the retrieved value is not a snowfall rate: dry snow scatters far less "
    "per mm/h of melted water than rain, while wet snow scatters more than "
    "either. The retrieval is reported as 'rain-equivalent' and should be "
    "read as an attenuation proxy, not a precipitation rate."
)


@dataclass(frozen=True)
class Event:
    key: str
    label: str
    date: str
    kind: str          # rain | snow | mixed, from METAR present weather
    note: str = ""


# All inside the OpenMesh 20-day subset, so CML and radar cover the same hours.
EVENTS = {
    "rain_0113": Event("rain_0113", "13 January 2024", "2024-01-13", "rain",
                       "26.1 mm, zero frozen observations - unambiguous rain"),
    "snow_0119": Event("snow_0119", "19 January 2024", "2024-01-19", "snow",
                       "100% frozen observations - unambiguous snow"),
    "mixed_0116": Event("mixed_0116", "16 January 2024", "2024-01-16", "snow",
                        "70% frozen - the large mid-January storm, both phases"),
    "rain_0128": Event("rain_0128", "28 January 2024", "2024-01-28", "rain",
                       "22.7 mm, second-wettest liquid day in the window"),
}


# --------------------------------------------------------------------------
def _band_sensitivity(freq_ghz: np.ndarray, length_km: np.ndarray,
                      reference_rain_mm_h: float = 5.0) -> np.ndarray:
    """Attenuation each sublink would show at a reference rain rate, in dB.

    The usability test for a band. A sublink whose answer is below the
    receiver's quantization step cannot measure rain at that rate, however
    well the retrieval is written.
    """
    k, alpha = cv.itu_coefficients(freq_ghz.ravel(),
                                   np.full(freq_ghz.size, "vertical"))
    k = k.reshape(freq_ghz.shape)
    alpha = alpha.reshape(freq_ghz.shape)
    return k * reference_rain_mm_h**alpha * length_km


def load_cml(event: Event, subset: str = "20d", pad_hours: float = 6.0,
             min_sensitivity_db: float = 0.6) -> xr.Dataset:
    """Retrieve rain-equivalent rate per link from RSL alone.

    ``min_sensitivity_db`` is the attenuation a sublink must develop at
    5 mm/h to be used at all - twice the 0.3 dB quantization step.
    """
    from core.opensense import example_data
    import pandas as pd

    cml = example_data.load("openmesh", subset)["cml"]
    start = pd.Timestamp(event.date)
    window = cml.sel(time=slice(str(start - pd.Timedelta(hours=pad_hours)),
                                str(start + pd.Timedelta(hours=24 + pad_hours))))

    # RSL only: retrieve_dataset sees no tsl and uses -RSL as the loss.
    per_band = retrieve_dataset(window)
    rain = per_band.R.transpose("time", "cml_id", "sublink_id").values
    n_c = rain.shape[1]
    length, freq, _ = link_metadata(
        window, window.rsl.isel(time=0, drop=True).transpose("cml_id", "sublink_id"))

    # Pick a band per link by whether it can physically see the rain, rather
    # than averaging across all three.
    #
    # This is not a refinement, it is the difference between a result and
    # noise. A 5.7 GHz sublink on a 2 km path develops 0.029 dB at 10 mm/h -
    # about thirty times *below* the 0.3 dB quantization step of the reported
    # RSL. Inverting the power law there divides by k*L ~ 8e-4, so a single
    # quantization step of noise comes back as ~40 mm/h. Taking a median over
    # three bands with one like that returns the noise.
    usable = _band_sensitivity(freq, length) >= min_sensitivity_db
    rain_masked = np.where(usable[None, :, :], rain, np.nan)

    # Among usable bands prefer the most sensitive, which is the one whose
    # attenuation sits furthest above the quantization floor.
    sensitivity = np.where(usable, _band_sensitivity(freq, length), -np.inf)
    best = np.argmax(sensitivity, axis=1)
    rows = np.arange(n_c)
    combined = rain_masked[:, rows, best]
    combined[:, ~usable.any(axis=1)] = np.nan

    ds = xr.Dataset({"R": (("time", "cml_id"), combined)},
                    coords={"time": window.time, "cml_id": window.cml_id})
    for c in ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon",
              "site_0_x", "site_0_y", "site_1_x", "site_1_y", "x", "y",
              "length_km", "frequency_ghz"):
        if c in cml.coords:
            da = cml[c]
            if "sublink_id" in da.dims:
                da = da.isel(sublink_id=0, drop=True)
            ds.coords[c] = ("cml_id", np.asarray(da))
    ds.coords["length"] = ds.length_km
    ds.coords["frequency"] = ds.frequency_ghz
    ds.R.attrs.update(units="mm h-1", note=PHASE_NOTE,
                      retrieval="RSL-only (no TSL published)")
    # keep the per-band retrieval for anyone who wants to look at phase
    ds["R_bands"] = (("time", "cml_id", "sublink_id"), rain)
    ds.coords["sublink_id"] = window.sublink_id
    ds.coords["band_used_ghz"] = ("cml_id", freq[rows, best])
    ds.coords["band_sensitivity_db"] = ("cml_id", sensitivity[rows, best])
    ds.attrs["usable_links"] = int(usable.any(axis=1).sum())
    ds.attrs["min_sensitivity_db"] = min_sensitivity_db
    return ds


def load_radar(event: Event) -> xr.Dataset:
    """KOKX reflectivity for the day, as a rate, on a projected grid."""
    from core.radar import nexrad

    ds = nexrad.build_day(event.date, event.kind)
    out = cv.project_grid(xr.Dataset({"R": ds.R}), CRS)
    out.R.attrs.update(units="mm h-1", phase=event.kind,
                       note=f"Z-S relation used for snow; {ds.R.attrs.get('note','')}")
    return out


def load_pws(event: Event, subset: str = "20d") -> xr.Dataset:
    """Personal weather stations as rain rate."""
    from core.opensense import example_data
    import pandas as pd

    pws = example_data.load("openmesh", subset)["pws"]
    start = pd.Timestamp(event.date)
    w = pws.sel(time=slice(str(start), str(start + pd.Timedelta(hours=24))))

    var = "rainfall_rate" if "rainfall_rate" in w.data_vars else "rainfall_amount"
    da = w[var].transpose("time", "id")
    if var == "rainfall_amount":
        interval = float(np.diff(w.time.values[:2])
                         .astype("timedelta64[s]").astype(float)[0])
        da = da * (3600.0 / interval)

    ds = xr.Dataset({"R": da})
    for c in ("lat", "lon", "x", "y"):
        if c in w.coords:
            ds.coords[c] = ("id", np.asarray(w[c]))
    ds.R.attrs["units"] = "mm h-1"
    return ds


def build_event(event: Event, resample: str = "15min",
                cache: bool = True) -> tuple:
    """(radar, cml, gauge) on a common axis, matching the other ingests."""
    import pandas as pd

    PROCESSED.mkdir(parents=True, exist_ok=True)
    paths = {n: PROCESSED / f"{event.key}_{n}.nc"
             for n in ("radar", "cml", "gauge")}
    if cache and all(p.exists() for p in paths.values()):
        print(f"  [cache] {event.key}")
        return tuple(xr.open_dataset(paths[n])
                     for n in ("radar", "cml", "gauge"))

    print(f"  cml     retrieving from RSL ({event.kind})")
    cml = load_cml(event)
    print(f"          {cml.sizes['cml_id']} links x {cml.sizes['sublink_id']} bands")
    print("  radar   KOKX")
    rad = load_radar(event)
    print(f"          grid {rad.sizes['y']} x {rad.sizes['x']}")
    print("  pws     loading")
    gauge = load_pws(event)

    day = pd.Timestamp(event.date)
    sl = slice(str(day), str(day + pd.Timedelta(hours=24)))
    cml = cml.sel(time=sl).resample(time=resample).mean()
    rad = rad.sel(time=sl).resample(time=resample).mean()
    gauge = gauge.resample(time=resample).mean()

    times = np.intersect1d(np.intersect1d(rad.time, cml.time), gauge.time)
    rad, cml, gauge = (d.sel(time=times) for d in (rad, cml, gauge))

    for name, ds in (("radar", rad), ("cml", cml), ("gauge", gauge)):
        ds.attrs.update(event=event.key, label=event.label, phase=event.kind,
                        source="OpenMesh NYC + NEXRAD KOKX",
                        doi="10.5281/zenodo.15268340", note=PHASE_NOTE)
        if cache:
            ds.to_netcdf(paths[name])
    return rad, cml, gauge


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", choices=sorted(EVENTS), default="rain_0113")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.list:
        for e in EVENTS.values():
            print(f"  {e.key:12s} {e.date}  {e.kind:6s}  {e.note}")
        raise SystemExit

    ev = EVENTS[args.event]
    print(f"{ev.label}  ({ev.kind}) - {ev.note}")
    rad, cml, gauge = build_event(ev, cache=not args.no_cache)
    print(f"\nradar  {dict(rad.sizes)}  max {float(rad.R.max()):.1f} mm/h")
    print(f"cml    {dict(cml.sizes)}  max {float(cml.R.max()):.1f} mm/h")
    print(f"gauge  {dict(gauge.sizes)}  max {float(gauge.R.max()):.1f} mm/h")
