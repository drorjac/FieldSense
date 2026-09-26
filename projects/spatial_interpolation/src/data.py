"""
OpenMRG inputs on one 15-minute axis: radar, gauges, links, per-link rain.

Parts 1 and 2.2 of ``advanced_models_colab_v2.ipynb``, reproduced exactly:
the same radar conversion, the same 15-minute right-closed aggregation, the
same nearest-in-8-minutes alignment. The Colab/Drive setup and the v1
migration code are not needed here - OpenMRG is read from the repository's
archive through ``core.scientific_packages.pynncml_compat``.

Everything expensive is cached under ``cfg.path(...)``.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import xarray as xr

from config import NowcastConfig

ALIGN_TOLERANCE = pd.Timedelta("8min")


def _cached(path, build):
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = build()
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def link_dataset(cfg: NowcastConfig, which: str = "full"):
    """PyNNcml's OpenMRG ``LinkDataset`` for the full period or one split."""
    import pynncml as pnc

    from core.scientific_packages import pynncml_compat

    pynncml_compat.apply()
    sl = cfg.period(which)

    def build():
        return pynncml_compat.quietly(pnc.datasets.loader_open_mrg_dataset,
                                      data_path=pynncml_compat.openmrg_data_path(),
                                      time_slice=sl)
    return _cached(cfg.path(f"openmrg_{which}.pkl"), build)


def radar_15min(cfg: NowcastConfig) -> dict:
    """Radar rain rate on 15-minute steps, with its lon/lat grid.

    As the notebook: raw bytes x 0.4 - 30 (read without xarray's scaling),
    255 missing, capped at 53 dBZ against hail, Z = 200 R^1.6, zero below
    0 dBZ, then the mean over each 15 minutes, right-closed and labelled.
    (The file itself declares b = 1.5; the notebook used 1.6.)
    """
    def build():
        with xr.open_dataset(cfg.radar_nc, mask_and_scale=False) as ds:
            ds = ds.sel(time=slice(str(pd.Timestamp(cfg.full.start) - pd.Timedelta("15min")),
                                   str(pd.Timestamp(cfg.full.stop) + pd.Timedelta(days=1))))
            raw = ds["data"].values
            dbz = raw.astype(np.float32) * 0.4 - 30.0
            dbz[raw == 255] = np.nan
            dbz = np.minimum(dbz, 53.0)
            rain = np.power(np.power(10.0, dbz / 10.0) / 200.0, 1.0 / 1.6)
            rain = np.where(np.isnan(dbz) | (dbz <= 0.0), 0.0, rain).astype(np.float32)
            da = xr.DataArray(rain, dims=("time", "y", "x"),
                              coords={"time": ds.time.values})
            r15 = da.resample(time="15min", label="right", closed="right").mean(skipna=True)
            lat, lon = ds["lat"].values, ds["lon"].values
        if lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        r15 = r15.sel(time=slice(cfg.full.start, str(pd.Timestamp(cfg.full.stop) + pd.Timedelta(days=1))))
        return {"R": r15.fillna(0.0).values.astype(np.float32),
                "times": pd.DatetimeIndex(r15.time.values), "lon": lon, "lat": lat}
    return _cached(cfg.path("radar_15min.pkl"), build)


def crop_bounds(lon: np.ndarray, lat: np.ndarray, lat_range, lon_range) -> tuple:
    """(y_lo, y_hi, x_lo, x_hi) of the smallest grid window covering a box."""
    m = ((lat >= lat_range[0]) & (lat <= lat_range[1]) &
         (lon >= lon_range[0]) & (lon <= lon_range[1]))
    y, x = np.where(m)
    if y.size == 0:
        raise ValueError(f"crop {lat_range} x {lon_range} contains no grid cells")
    return int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1


def _to_15min(values: np.ndarray, times, columns, radar_times) -> pd.DataFrame:
    return (pd.DataFrame(values, index=pd.DatetimeIndex(times), columns=columns)
            .resample("15min", label="right", closed="right").mean()
            .reindex(radar_times, method="nearest", tolerance=ALIGN_TOLERANCE)
            .fillna(0.0).astype(np.float32))


def _decode_times(v, n, start):
    if v is None or len(v) != n:
        return pd.date_range(start=start, periods=n, freq="1min")
    arr = np.asarray(v).ravel()
    if arr.dtype.kind in "MO":
        return pd.to_datetime(arr)
    mag = np.median(np.abs(arr.astype(np.int64)))
    unit = "s" if mag < 1e10 else "ms" if mag < 1e13 else "us" if mag < 1e16 else "ns"
    return pd.to_datetime(arr.astype(np.int64), unit=unit)


def gauges_15min(dataset, radar_times, start) -> dict:
    """Gauge rain rate on the radar's 15-minute axis, and gauge positions."""
    names, lons, lats, series = [], [], [], []
    for i, p in enumerate(dataset.point_set):
        v = np.asarray(p.data_array, dtype=np.float32).ravel()
        t = _decode_times(getattr(p, "time_array", None), len(v), start)
        names.append(str(getattr(p, "name", f"gauge_{i}")))
        lons.append(float(np.ravel(p.lon)[0]))
        lats.append(float(np.ravel(p.lat)[0]))
        series.append(pd.Series(v, index=t, name=names[-1]))
    frame = pd.concat(series, axis=1)
    aligned = _to_15min(frame.values, frame.index, names, radar_times)
    return {"R": aligned, "lon": np.array(lons), "lat": np.array(lats), "names": names}


def link_midpoints(dataset) -> tuple:
    """Midpoint lon/lat of every link (what the IDW maps interpolate from)."""
    lons, lats = [], []
    for lk in dataset.link_set:
        s0, s1 = lk.meta_data.lon_lat_site_zero, lk.meta_data.lon_lat_site_one
        lons.append((float(s0[0]) + float(s1[0])) / 2)
        lats.append((float(s0[1]) + float(s1[1])) / 2)
    return np.array(lons), np.array(lats)


def links_physical(dataset, radar_times, start) -> pd.DataFrame:
    """Per-link rain from PyNNcml's one-step dynamic baseline, 15-min.

    ``one_step_dynamic_baseline(MAX, r_min=0.1, window=60, wet=0.5)``, as in
    the notebook's Part 2.2 - the "Model 1" physical estimator.
    """
    import pynncml as pnc

    est = pnc.scm.rain_estimation.one_step_dynamic_baseline(
        pnc.scm.power_law.PowerLawType.MAX, 0.1, 60, 0.5)
    rains, times = [], None
    for lk in dataset.link_set:
        out = est(lk.attenuation(), lk.meta_data)
        rain = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy().squeeze()
        rains.append(rain.astype(np.float32))
        if times is None and hasattr(lk, "time_array"):
            times = lk.time_array
    n = min(len(r) for r in rains)
    t = _decode_times(None if times is None else np.asarray(times)[:n], n, start)
    return _to_15min(np.stack([r[:n] for r in rains], axis=1), t, None, radar_times)
