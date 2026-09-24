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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import conventions as cv                                             # noqa: E402
from ingest_openmrg import _baseline_from_dry, _wet_dry_rolling_std  # noqa: E402
from core.itu_p838 import get_k_alpha                                # noqa: E402

RAW = REPO_ROOT / "dataset/open_datasets/OpenRainER_Italy/raw"
EXTRACTED = RAW / "extracted"
PROCESSED = REPO_ROOT / "dataset/open_datasets/OpenRainER_Italy/processed"

# UTM 32N covers Emilia-Romagna.
CRS = "EPSG:32632"


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
def load_cml(event: Event) -> xr.Dataset:
    """Retrieve path rain rate from raw RSL/TSL for one event window."""
    path = _cml_file(event.month)
    pad = pd.Timedelta(hours=4)
    ds = xr.open_dataset(path).sel(
        time=slice(str(pd.Timestamp(event.start) - pad),
                   str(pd.Timestamp(event.end) + pad))).load()

    # (cml_id, sublink_id, time) -> total loss, averaged over the two sublinks
    total_loss = (ds.tsl - ds.rsl).transpose("time", "cml_id", "sublink_id")
    loss = np.asarray(total_loss).astype(float)
    n_time, n_cml, n_sub = loss.shape

    flat = loss.reshape(n_time, n_cml * n_sub)
    flat = pd.DataFrame(flat).ffill().bfill().to_numpy()

    # 1-minute sampling here, against 10 s for OpenMRG, so the window lengths
    # that mean 5 minutes and 3 hours differ by a factor of six.
    wet = _wet_dry_rolling_std(flat, window=5, threshold_db=0.8)
    baseline = _baseline_from_dry(flat, wet, window=180)
    a_rain = np.clip(flat - baseline, 0.0, None)
    a_rain[a_rain < 0.1] = 0.0
    a_rain = a_rain.reshape(n_time, n_cml, n_sub)

    # Unit trap: OpenRainER declares length in metres and frequency in MHz,
    # where OpenMRG uses km and GHz. Feeding metres into k*L, or MHz into the
    # ITU-R table, silently retrieves zero rain everywhere. conventions.py
    # reads the declared units, and falls back on magnitude when a source
    # ships no units attribute at all.
    length = cv.to_km(ds.length)
    freq = cv.to_ghz(ds.frequency)
    pol = cv.normalize_polarization(ds.polarization.values)

    # frequency/polarization may be per (cml_id, sublink_id) or per cml_id
    if freq.ndim == 1:
        freq = np.repeat(freq[:, None], n_sub, axis=1)
    if pol.size != freq.size:
        pol = np.full(freq.shape, "vertical")
    pol = pol.reshape(freq.shape)
    if length.ndim == 1:
        length = np.repeat(length[:, None], n_sub, axis=1)

    k = np.empty(freq.shape)
    alpha = np.empty(freq.shape)
    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            p = pol[i, j]
            k[i, j], alpha[i, j] = get_k_alpha(
                float(freq[i, j]),
                p if p in ("vertical", "horizontal") else "vertical")

    waa_max, waa_rate = 0.5, 0.28
    rain = np.zeros_like(a_rain)
    for _ in range(8):
        waa = waa_max * (1.0 - np.exp(-waa_rate * rain))
        rain = (np.clip(a_rain - waa, 0.0, None)
                / (k[None] * length[None])) ** (1.0 / alpha[None])
    rain[~np.isfinite(rain)] = np.nan

    with np.errstate(invalid="ignore"):
        rain_cml = np.nanmean(rain, axis=2)          # average the sublinks

    out = xr.Dataset(
        {"R": (("time", "cml_id"), rain_cml)},
        coords={"time": np.asarray(ds.time), "cml_id": np.asarray(ds.cml_id)},
    )
    for name in ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"):
        out.coords[name] = ("cml_id", np.asarray(ds[name]).reshape(n_cml, -1)[:, 0]
                            if np.asarray(ds[name]).ndim > 1
                            else np.asarray(ds[name]))
    out.coords["length"] = ("cml_id", length[:, 0])
    out.coords["frequency"] = ("cml_id", freq[:, 0])

    x0, y0 = plg.spatial.project_point_coordinates(
        out.site_0_lon, out.site_0_lat, CRS)
    x1, y1 = plg.spatial.project_point_coordinates(
        out.site_1_lon, out.site_1_lat, CRS)
    for nm, v in (("site_0_x", x0), ("site_0_y", y0),
                  ("site_1_x", x1), ("site_1_y", y1)):
        out.coords[nm] = ("cml_id", np.asarray(v))
    out.coords["x"] = ("cml_id", (np.asarray(x0) + np.asarray(x1)) / 2)
    out.coords["y"] = ("cml_id", (np.asarray(y0) + np.asarray(y1)) / 2)
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

    lon2d, lat2d = np.meshgrid(np.asarray(rain.lon), np.asarray(rain.lat))
    out = xr.Dataset({"R": rain.rename({"lat": "y", "lon": "x"})})
    out.coords["longitudes"] = (("y", "x"), lon2d)
    out.coords["latitudes"] = (("y", "x"), lat2d)

    xs, ys = plg.spatial.project_point_coordinates(
        out.longitudes, out.latitudes, CRS)
    out.coords["x_grid"], out.coords["y_grid"] = xs, ys
    xv, yv = np.asarray(xs), np.asarray(ys)
    out.coords["x"] = ("x", xv[xv.shape[0] // 2, :])
    out.coords["y"] = ("y", yv[:, xv.shape[1] // 2])
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


def build_event(event: Event, resample: str = "15min",
                cache: bool = True) -> tuple:
    """Return (radar, cml, gauges) on a common 15-minute axis."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    paths = {n: PROCESSED / f"{event.key}_{n}.nc"
             for n in ("radar", "cml", "gauge")}
    if cache and all(p.exists() for p in paths.values()):
        print(f"  [cache] {event.key}")
        return tuple(xr.open_dataset(paths[n])
                     for n in ("radar", "cml", "gauge"))

    print(f"  cml     {event.start} .. {event.end}")
    cml = load_cml(event)
    print(f"          {cml.sizes['cml_id']} links")
    print("  radar   loading")
    rad = load_radar(event, cml_ds=cml)
    print(f"          grid {rad.sizes['y']} x {rad.sizes['x']}")
    print("  gauges  loading")
    gauge = load_gauges(event)

    cml = cml.resample(time=resample).mean()
    gauge = gauge.resample(time=resample).mean()
    rad = rad.resample(time=resample).mean()

    times = np.intersect1d(np.intersect1d(rad.time, cml.time), gauge.time)
    rad, cml, gauge = (d.sel(time=times) for d in (rad, cml, gauge))

    for name, ds in (("radar", rad), ("cml", cml), ("gauge", gauge)):
        ds.attrs.update(event=event.key, label=event.label,
                        source="OpenRainER", doi="10.5281/zenodo.22829808",
                        license="CC-BY-4.0")
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
