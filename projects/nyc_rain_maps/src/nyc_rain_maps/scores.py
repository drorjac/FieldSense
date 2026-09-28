"""Scores of an estimated rain field against a reference (radar or gauges).

Follows the scoring rules in the project README: resample estimate and reference to a common grid and
accumulation interval first; score only where both are valid and report how many
samples that was; report per event as well as pooled. The headline metric is **NRMSE**
(RMSE normalised by the reference mean) so storms of different magnitude compare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid
from core.radar.mrms import path_average, to_grid


def scores(est, ref, wet_threshold: float = 0.1) -> dict:
    """Continuous and categorical scores over all jointly valid samples.

    Returns ``n, mean_ref, mean_est, bias, rel_bias, mae, rmse, nrmse, corr`` and the
    detection scores ``pod, far, csi`` for "wet" = value >= ``wet_threshold``.
    """
    e = np.asarray(est, dtype=float).ravel()
    r = np.asarray(ref, dtype=float).ravel()
    ok = np.isfinite(e) & np.isfinite(r)
    e, r = e[ok], r[ok]
    n = int(ok.sum())
    if n == 0:
        return {"n": 0}
    err = e - r
    mean_ref = float(r.mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    hit = np.sum((e >= wet_threshold) & (r >= wet_threshold))
    miss = np.sum((e < wet_threshold) & (r >= wet_threshold))
    false = np.sum((e >= wet_threshold) & (r < wet_threshold))
    corr = float(np.corrcoef(e, r)[0, 1]) if n > 2 and e.std() > 0 and r.std() > 0 else np.nan
    return {
        "n": n,
        "mean_ref": mean_ref,
        "mean_est": float(e.mean()),
        "bias": float(err.mean()),
        "rel_bias": float(e.sum() / r.sum() - 1) if r.sum() > 0 else np.nan,
        "mae": float(np.abs(err).mean()),
        "rmse": rmse,
        "nrmse": rmse / mean_ref if mean_ref > 0 else np.nan,
        "corr": corr,
        "pod": float(hit / (hit + miss)) if hit + miss else np.nan,
        "far": float(false / (hit + false)) if hit + false else np.nan,
        "csi": float(hit / (hit + miss + false)) if hit + miss + false else np.nan,
    }


def align_to_reference(est: xr.DataArray, ref: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Put ``ref`` on ``est``'s grid (block mean / nearest) and keep common times."""
    grid = Grid(est.lat.values, est.lon.values)
    ref_g = to_grid(ref, grid)
    times = np.intersect1d(est.time.values, ref_g.time.values)
    return est.sel(time=times), ref_g.sel(time=times)


def compare_maps(est_hourly: xr.DataArray, radar_hourly: xr.DataArray, wet_threshold: float = 0.1,
                 only_wet_hours: bool = False) -> tuple[pd.DataFrame, dict]:
    """Hourly map-vs-radar scores: ``(per_hour DataFrame, pooled dict)``.

    Both inputs are hourly accumulations (mm) with hour-ENDING labels. The radar is
    resampled onto the estimate's grid. ``only_wet_hours`` drops hours where the radar
    domain mean is below ``wet_threshold``.
    """
    e, r = align_to_reference(est_hourly, radar_hourly)
    if only_wet_hours:
        keep = r.mean(("lat", "lon")) >= wet_threshold
        e, r = e.sel(time=keep), r.sel(time=keep)
    rows = [{"time": pd.Timestamp(t), **scores(e.sel(time=t), r.sel(time=t), wet_threshold)}
            for t in e.time.values]
    per_hour = pd.DataFrame(rows).set_index("time") if rows else pd.DataFrame()
    pooled = scores(e, r, wet_threshold)
    joint = e.notnull() & r.notnull()
    tot_e = e.where(joint).sum("time", min_count=1)
    tot_r = r.where(joint).sum("time", min_count=1)
    pooled["event_total_scores"] = scores(tot_e, tot_r, wet_threshold)
    pooled["hours"] = int(e.sizes["time"])
    return per_hour, pooled


def compare_links(link_hourly: xr.DataArray, radar_hourly: xr.DataArray, links: pd.DataFrame,
                  wet_threshold: float = 0.1) -> pd.DataFrame:
    """Per-link scores of hourly link rain (mm) vs radar averaged along each link path.

    This is the fairest CML-vs-radar comparison: no interpolation error is involved.
    """
    radar_path = path_average(radar_hourly, links.loc[list(link_hourly.link.values)])
    radar_path = radar_path.rename(link="link")
    times = np.intersect1d(link_hourly.time.values, radar_path.time.values)
    rows = []
    for lk in link_hourly.link.values:
        e = link_hourly.sel(link=lk, time=times)
        r = radar_path.sel(link=lk, time=times)
        s = scores(e, r, wet_threshold)
        ok = e.notnull() & r.notnull()
        s.update(link=str(lk), total_est=float(e.where(ok).sum()), total_radar=float(r.where(ok).sum()))
        rows.append(s)
    return pd.DataFrame(rows).set_index("link")
