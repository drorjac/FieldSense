"""The map metrics a learned-mapping study reports, on top of ``core.maps.scores``.

``scores.scores`` gives the pooled continuous and categorical scores (RMSE, MAE, bias,
correlation, POD/FAR/CSI) of any two arrays. A map study also needs:

``pcc_per_step``          Pearson correlation of each time step's field (spatial pattern)
``event_totals``          each storm's total, estimate against reference (hydrological bias)
``cumulative_event_error`` the summary of those totals over events
``detection``             POD, FAR, CSI at any rain threshold
``sample_at_points``      a map at point locations (the cell containing each point)
``gauge_point_error``     the scores at independent gauges
``skill_table``           all of the above for several maps, on one common sample

Every function scores only where estimate and reference are both finite, inside an
optional ``mask`` of cells; ``skill_table`` further restricts every map to the samples
*all* maps have, so no method improves its score by leaving gaps.

    from core.maps.map_skill import skill_table
    table = skill_table({"idw": idw, "gmz": gmz}, radar, event=ds.event_id, mask=ds.observable,
                        gauges=ds.gauges, threshold=0.1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.maps.scores import scores


def _cells(a, mask=None) -> np.ndarray:
    """``(time, lat, lon)`` (array or DataArray) -> ``(time, cells)`` float, cells in ``mask``."""
    if isinstance(a, xr.DataArray):
        a = a.transpose("time", ...).values
    a = np.asarray(a, dtype=float)
    a = a.reshape(a.shape[0], -1)
    if mask is not None:
        m = np.asarray(mask.values if isinstance(mask, xr.DataArray) else mask).astype(bool).ravel()
        a = a[:, m]
    return a


def _times(a, n):
    return pd.DatetimeIndex(a.time.values) if isinstance(a, xr.DataArray) and "time" in a.dims \
        else pd.RangeIndex(n)


def pcc_per_step(est, ref, mask=None, min_samples: int = 3) -> pd.Series:
    """Spatial Pearson correlation of every time step (NaN when either field is constant
    or fewer than ``min_samples`` cells are jointly valid)."""
    e, r = _cells(est, mask), _cells(ref, mask)
    out = np.full(e.shape[0], np.nan)
    for t in range(e.shape[0]):
        ok = np.isfinite(e[t]) & np.isfinite(r[t])
        if ok.sum() >= min_samples and e[t, ok].std() > 0 and r[t, ok].std() > 0:
            out[t] = np.corrcoef(e[t, ok], r[t, ok])[0, 1]
    return pd.Series(out, index=_times(ref, e.shape[0]), name="pcc")


def pcc_pooled(est, ref, mask=None) -> float:
    """Pearson correlation over all jointly valid cell-steps (space and time pooled)."""
    return scores(_cells(est, mask), _cells(ref, mask)).get("corr", np.nan)


def detection(est, ref, threshold: float = 0.1, mask=None) -> dict:
    """POD, FAR and CSI for "wet" = value >= ``threshold``, with the hit/miss counts' base ``n``."""
    s = scores(_cells(est, mask), _cells(ref, mask), wet_threshold=threshold)
    return {"threshold": threshold, "n": s.get("n", 0), **{k: s.get(k, np.nan) for k in ("pod", "far", "csi")}}


def event_totals(est, ref, event, mask=None) -> pd.DataFrame:
    """Each event's total, estimate vs reference, one row per event.

    ``event`` gives the event of each time step. Totals are summed over the steps where
    both maps have a value at a cell, then averaged over the cells (``mm`` for maps in
    mm per step). Also the cell-wise RMSE of the event-total maps.
    """
    e, r = _cells(est, mask), _cells(ref, mask)
    ev = np.asarray(event.values if isinstance(event, xr.DataArray) else event).astype(str)
    rows = []
    for k in dict.fromkeys(ev):
        sel = ev == k
        ok = np.isfinite(e[sel]) & np.isfinite(r[sel])
        te = np.where(ok, e[sel], 0.0).sum(0)
        tr = np.where(ok, r[sel], 0.0).sum(0)
        cells = ok.any(0)
        if not cells.any():
            rows.append({"event": k, "steps": int(sel.sum()), "cells": 0})
            continue
        te, tr = te[cells], tr[cells]
        rows.append({"event": k, "steps": int(sel.sum()), "cells": int(cells.sum()),
                     "total_est": float(te.mean()), "total_ref": float(tr.mean()),
                     "error": float(te.mean() - tr.mean()),
                     "rel_error": float(te.mean() / tr.mean() - 1) if tr.mean() > 0 else np.nan,
                     "rmse_cells": float(np.sqrt(np.mean((te - tr) ** 2)))})
    return pd.DataFrame(rows).set_index("event")


def cumulative_event_error(est, ref, event, mask=None) -> dict:
    """Summary of :func:`event_totals` over events: mean error (bias, mm), mean absolute
    error, mean absolute relative error, and the mean cell-wise RMSE of event totals."""
    t = event_totals(est, ref, event, mask)
    if "error" not in t or t.error.notna().sum() == 0:
        return {"events": 0}
    return {"events": int(t.error.notna().sum()), "event_bias": float(t.error.mean()),
            "event_mae": float(t.error.abs().mean()),
            "event_abs_rel_error": float(t.rel_error.abs().mean()),
            "event_rmse_cells": float(t.rmse_cells.mean())}


def sample_at_points(field: xr.DataArray, lat, lon, station=None) -> xr.DataArray:
    """The map in the grid cell containing each point: ``(station, time)``."""
    s = field.sel(lat=xr.DataArray(np.asarray(lat), dims="station"),
                  lon=xr.DataArray(np.asarray(lon), dims="station"), method="nearest")
    s = s.drop_vars(["lat", "lon"], errors="ignore")
    if station is not None:
        s = s.assign_coords(station=np.asarray(station))
    return s.transpose("station", ...)


def _gauge_coords(gauges: xr.DataArray):
    lat = gauges["station_lat"] if "station_lat" in gauges.coords else gauges["lat"]
    lon = gauges["station_lon"] if "station_lon" in gauges.coords else gauges["lon"]
    return lat.values, lon.values


def gauge_point_error(field: xr.DataArray, gauges: xr.DataArray, wet_threshold: float = 0.1) -> dict:
    """Scores (``core.maps.scores.scores``) of the map at the gauges, against the gauges.

    ``gauges(station, time)`` needs ``lat``/``lon`` (or ``station_lat``/``station_lon``)
    and the map's times."""
    lat, lon = _gauge_coords(gauges)
    est = sample_at_points(field.sel(time=gauges.time), lat, lon)
    return scores(est.values, gauges.transpose("station", "time").values, wet_threshold)


def skill_table(estimates: dict, ref, event=None, mask=None, gauges: xr.DataArray | None = None,
                threshold: float = 0.1, common: bool = True) -> pd.DataFrame:
    """One row per map: the continuous, correlation, event-total and detection scores
    against ``ref``, and the scores at ``gauges``.

    ``estimates``: ``{name: (time, lat, lon)}`` on ``ref``'s grid and times. ``common``:
    score every map on the cell-steps (and station-steps) that all of them and the
    reference have. Columns: ``n, rmse, mae, bias, pcc, pcc_step`` (mean of the
    per-step correlations), ``event_bias, event_abs_rel_error, pod, far, csi`` and,
    with gauges, ``gauge_n, gauge_rmse, gauge_mae, gauge_bias, gauge_pcc``.
    """
    names = list(estimates)
    r = _cells(ref, mask)
    E = {k: _cells(v, mask) for k, v in estimates.items()}
    joint = np.isfinite(r)
    if common:
        for v in E.values():
            joint &= np.isfinite(v)
    if gauges is not None:
        lat, lon = _gauge_coords(gauges)
        G = gauges.transpose("station", "time").values.astype(float)
        P = {k: sample_at_points(v if isinstance(v, xr.DataArray) else ref.copy(data=v), lat, lon)
             .sel(time=gauges.time).values for k, v in estimates.items()}
        gjoint = np.isfinite(G)
        if common:
            for v in P.values():
                gjoint &= np.isfinite(v)
    rows = []
    for k in names:
        ok = joint & np.isfinite(E[k])
        e = np.where(ok, E[k], np.nan)
        rr = np.where(ok, r, np.nan)
        s = scores(e, rr, threshold)
        step = pcc_per_step(e, rr).values
        row = {"method": k, "n": s.get("n", 0), "rmse": s.get("rmse"), "mae": s.get("mae"),
               "bias": s.get("bias"), "pcc": s.get("corr"),
               "pcc_step": float(np.nanmean(step)) if np.isfinite(step).any() else np.nan}
        if event is not None:
            c = cumulative_event_error(e, rr, event)
            row.update({"event_bias": c.get("event_bias"), "event_abs_rel_error": c.get("event_abs_rel_error")})
        row.update({"pod": s.get("pod"), "far": s.get("far"), "csi": s.get("csi")})
        if gauges is not None:
            gok = gjoint & np.isfinite(P[k])
            gs = scores(np.where(gok, P[k], np.nan), np.where(gok, G, np.nan), threshold)
            row.update({"gauge_n": gs.get("n", 0), "gauge_rmse": gs.get("rmse"), "gauge_mae": gs.get("mae"),
                        "gauge_bias": gs.get("bias"), "gauge_pcc": gs.get("corr")})
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")
