"""
OpenRainER (Emilia-Romagna, Italy) raw archive -> OpenSense-1.0 datasets.

Where OpenMRG ships raw instrument data in a bespoke layout, OpenRainER is
already published to the OpenSense convention:

* ``CML_YYYYMM*.nc`` - (cml_id, sublink_id, time) RSL/TSL at 1 min, with
  ``site_0_lat/lon``, ``site_1_lat/lon``, ``frequency``, ``polarization`` and
  ``length`` already attached as coordinates. 151 links.
* ``RADrain_YYYYMM.nc`` - 15-minute accumulated rain on a 290 x 373 lat/lon
  grid, in mm.
* ``AWS_YYYYMM.nc`` - 319 automatic weather stations at 15 min, rain in mm.

So the ingest here is mostly projection and unit handling, and the CML
retrieval is the same chain as OpenMRG - which is the point: if the retrieval
and merge code only worked on the dataset it was written against, it would not
be a pipeline.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import poligrain as plg
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.opensense import conventions as cv  # noqa: E402
from core.opensense.retrieval import (RetrievalConfig, combine_sublinks,  # noqa: E402
                                      retrieve_dataset, retrieve_improved)

RAW = REPO_ROOT / "dataset/open_datasets/OpenRainER_Italy/raw"
EXTRACTED = RAW / "extracted"
PROCESSED = REPO_ROOT / "dataset/open_datasets/OpenRainER_Italy/processed"

# UTM 32N covers Emilia-Romagna.
CRS = "EPSG:32632"

# The AWS and RADrain 15-minute accumulations are stamped at the END of their
# interval: the value at 12:15 is the rain of 12:00-12:15. Not stated in the
# dataset README; established by lagging the 1-minute CML signals against
# both (CML-gauge r 0.35 at lag 0, 0.69 one step back, and radar-gauge peak
# at lag 0). The CML must be aggregated the same way or it sits one whole
# step out of line with both references.
ACCUMULATION_LABEL = "end"


@dataclass(frozen=True)
class Event:
    key: str
    label: str
    start: str
    end: str
    month: str
    note: str = ""


# Ranked from the AWS record by 3-hour domain-mean rainfall over ~294 stations.
EVENTS = {
    "sep26": Event("sep26", "26 September 2021", "2021-09-26T09:00",
                   "2021-09-26T18:00", "202109",
                   "wettest 3 h of the two-year record, 16.5 mm domain mean"),
    "aug18": Event("aug18", "18 August 2022", "2022-08-18T06:00",
                   "2022-08-18T15:00", "202208",
                   "summer convective event, 14.1 mm domain mean"),
}


def _ensure_extracted(member: str, archive: str) -> Path:
    """Extract one month from a tar, and gunzip it, if not already done."""
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    target = EXTRACTED / member
    if target.exists():
        return target

    gz = EXTRACTED / f"{member}.gz"
    if not gz.exists():
        tar = RAW / archive
        if not tar.exists():
            raise FileNotFoundError(
                f"{tar} missing - run fetch.py --dataset openrainer first")
        print(f"  extracting {member}.gz from {archive}")
        subprocess.run(["tar", "-xf", str(tar), "-C", str(EXTRACTED),
                        f"{member}.gz"], check=True)

    print(f"  gunzip {member}.gz")
    with gzip.open(gz, "rb") as fin, target.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    return target



def _cml_file(month: str) -> Path:
    """CML files are named by their full time span, so glob for the month."""
    matches = sorted(EXTRACTED.glob(f"CML_{month}*.nc"))
    if matches:
        return matches[0]
    gzs = sorted(EXTRACTED.glob(f"CML_{month}*.nc.gz"))
    if gzs:
        return _ensure_extracted(gzs[0].name[:-3], "CML.tar")
    # Ask the tar what the member is actually called.
    listing = subprocess.run(["tar", "-tf", str(RAW / "CML.tar")],
                             capture_output=True, text=True, check=True)
    member = next(line[:-3] for line in listing.stdout.split()
                  if line.startswith(f"CML_{month}"))
    return _ensure_extracted(member, "CML.tar")


# --------------------------------------------------------------------------
def _per_link(da: xr.DataArray) -> xr.DataArray:
    """First sublink's value of per-sublink metadata; attrs (units) kept."""
    return da.isel(sublink_id=0, drop=True) if "sublink_id" in da.dims else da


def load_cml(event: Event, cfg: RetrievalConfig | None = None,
             retrieval: str = "default") -> xr.Dataset:
    """Retrieve path rain rate from raw RSL/TSL for one event window.

    The files are already OpenSense-1.0, so this is ``retrieve_dataset`` on
    a padded window - the baseline needs dry weather either side of the event
    to anchor on - then the two sublinks averaged per link.

    Unit trap: OpenRainER declares length in metres and frequency in MHz,
    where OpenMRG uses km and GHz. Fed straight into ``k*L`` and the ITU-R
    table they retrieve zero rain everywhere, silently. ``retrieve_dataset``
    reads the declared units through ``conventions.py``.

    ``retrieval="improved"`` runs ``retrieve_improved`` on a 24-hour margin
    instead, since the nearby-link reference level needs a day of history.
    """
    path = _cml_file(event.month)
    pad = pd.Timedelta(hours=24 if retrieval == "improved" else 4)
    with xr.open_dataset(path) as raw:
        ds = raw.sel(time=slice(str(pd.Timestamp(event.start) - pad),
                                str(pd.Timestamp(event.end) + pad))).load()

    # 1-minute sampling here against 10 s for OpenMRG, so the windows that
    # mean 5 minutes and 3 hours are scaled from the data, not hardcoded.
    # The raw files skip minutes; the rolling windows count samples.
    step = pd.Timedelta(minutes=1)
    ds = ds.reindex(time=pd.date_range(ds.time.values[0], ds.time.values[-1],
                                       freq=step))
    if retrieval == "improved":
        res = retrieve_improved(ds, cfg)
        print(f"          {res.attrs['wet_dry']}")
    else:
        res = retrieve_dataset(ds, cfg)
    out = combine_sublinks(res)
    out = cv.project_cml(out, CRS)
    # downstream (mergeplg, the synthetic benchmark) reads km and GHz per link
    out.coords["length"] = ("cml_id", cv.to_km(_per_link(ds.length)))
    out.coords["frequency"] = ("cml_id", cv.to_ghz(_per_link(ds.frequency)))
    out.R.attrs["units"] = "mm h-1"
    return out.sel(time=slice(event.start, event.end))


def load_radar(event: Event, bbox_pad_km: float = 15.0,
               cml_ds: xr.Dataset | None = None) -> xr.Dataset:
    """Radar rain rate on a projected grid, cropped to the CML network."""
    path = _ensure_extracted(f"RADrain_{event.month}.nc", "RADrain.tar")
    ds = xr.open_dataset(path).sel(time=slice(event.start, event.end))

    # 15-minute accumulations in mm -> mm/h
    rain = ds.rainfall_amount * 4.0

    if cml_ds is not None:
        pad = bbox_pad_km / 111.0
        lon0 = float(min(cml_ds.site_0_lon.min(), cml_ds.site_1_lon.min())) - pad
        lon1 = float(max(cml_ds.site_0_lon.max(), cml_ds.site_1_lon.max())) + pad
        lat0 = float(min(cml_ds.site_0_lat.min(), cml_ds.site_1_lat.min())) - pad
        lat1 = float(max(cml_ds.site_0_lat.max(), cml_ds.site_1_lat.max())) + pad
        rain = rain.sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1))
        if rain.sizes["lat"] == 0:       # descending latitude axis
            rain = (ds.rainfall_amount * 4.0).sel(
                lon=slice(lon0, lon1), lat=slice(lat1, lat0))

    out = cv.project_grid(xr.Dataset({"R": rain}), CRS)
    out.R.attrs["units"] = "mm h-1"
    return out


def load_gauges(event: Event) -> xr.Dataset:
    """AWS rain gauges as rain rate on a projected grid."""
    path = _ensure_extracted(f"AWS_{event.month}.nc", "AWS.tar")
    ds = xr.open_dataset(path).sel(time=slice(event.start, event.end))

    rain = (ds.rainfall_amount * 4.0).transpose("time", "id")   # mm/15min -> mm/h
    out = xr.Dataset({"R": rain})
    out.coords["lat"] = ("id", np.asarray(ds.latitude))
    out.coords["lon"] = ("id", np.asarray(ds.longitude))
    x, y = plg.spatial.project_point_coordinates(out.lon, out.lat, CRS)
    out.coords["x"], out.coords["y"] = ("id", np.asarray(x)), ("id", np.asarray(y))
    out.R.attrs["units"] = "mm h-1"
    return out


def aggregate_like_references(cml: xr.Dataset, freq: str = "15min") -> xr.Dataset:
    """CML rain averaged over the same windows the references accumulate."""
    from core.opensense.evaluation import aggregate

    out = xr.Dataset({"R": aggregate(cml.R, freq, label=ACCUMULATION_LABEL)})
    for name, c in cml.coords.items():
        if "time" not in c.dims:
            out.coords[name] = c
    out.R.attrs.update(cml.R.attrs)
    return out


def build_event(event: Event, resample: str = "15min",
                cache: bool = True, retrieval: str = "default") -> tuple:
    """Return (radar, cml, gauges) on a common 15-minute axis.

    Radar and gauges arrive as 15-minute accumulations stamped at interval
    end; the 1-minute CML retrieval is averaged over the same (t - 15, t]
    windows so that every timestamp means the same quarter hour for all
    three.

    ``retrieval="improved"`` uses ``retrieve_improved``; cached separately
    as ``<event>_improved_*.nc``.
    """
    if retrieval not in ("default", "improved"):
        raise ValueError(f"retrieval must be 'default' or 'improved', got {retrieval!r}")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    stem = event.key if retrieval == "default" else f"{event.key}_improved"
    paths = {n: PROCESSED / f"{stem}_{n}.nc"
             for n in ("radar", "cml", "gauge")}
    if cache and all(p.exists() for p in paths.values()):
        print(f"  [cache] {stem}")
        return tuple(xr.open_dataset(paths[n])
                     for n in ("radar", "cml", "gauge"))

    print(f"  cml     {event.start} .. {event.end}")
    cml = load_cml(event, retrieval=retrieval)
    print(f"          {cml.sizes['cml_id']} links")
    print("  radar   loading")
    rad = load_radar(event, cml_ds=cml)
    print(f"          grid {rad.sizes['y']} x {rad.sizes['x']}")
    print("  gauges  loading")
    gauge = load_gauges(event)

    cml = aggregate_like_references(cml, resample)
    gauge = gauge.resample(time=resample).mean()
    rad = rad.resample(time=resample).mean()

    times = np.intersect1d(np.intersect1d(rad.time, cml.time), gauge.time)
    rad, cml, gauge = (d.sel(time=times) for d in (rad, cml, gauge))

    for name, ds in (("radar", rad), ("cml", cml), ("gauge", gauge)):
        ds.attrs.update(event=event.key, label=event.label,
                        source="OpenRainER", doi="10.5281/zenodo.22829808",
                        license="CC-BY-4.0", retrieval=retrieval,
                        time_label=f"{ACCUMULATION_LABEL} of {resample} interval")
        if cache:
            ds.to_netcdf(paths[name])
    return rad, cml, gauge


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", choices=sorted(EVENTS), default="sep26")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    ev = EVENTS[args.event]
    print(f"{ev.label}  ({ev.note})")
    rad, cml, gauge = build_event(ev, cache=not args.no_cache)
    print(f"\nradar  {dict(rad.sizes)}  max {float(rad.R.max()):.1f} mm/h")
    print(f"cml    {dict(cml.sizes)}  max {float(cml.R.max()):.1f} mm/h")
    print(f"gauge  {dict(gauge.sizes)}  max {float(gauge.R.max()):.1f} mm/h")
