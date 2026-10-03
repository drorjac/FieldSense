"""
Every FieldSense method on a synthetic case, scored against the truth.

With real data the truth is unknown: maps are scored against gauges that are
themselves point samples, motion against nothing at all. On a ``SyntheticCase``
the rain and its motion are known everywhere, so this module runs the
repository's methods exactly as the real-data projects do and scores them
against what really fell:

``maps(case)``
    rain maps per interval from every sensor combination: the radar alone; IDW
    of gauges, of links (midpoints), of both and of everything; GMZ of the links
    (``core.maps``); the radar adjusted with gauges, links or both by mean-field
    bias, additive and multiplicative IDW (``core.maps.merge``) and by mergeplg's
    additive / multiplicative IDW, ordinary kriging and KED
    (``core.maps.mergeplg_methods``).
``score_maps``
    NRMSE, bias, correlation, CSI against the truth (``core.maps.scores``), per
    interval and on hourly totals.
``motion(rate, meta)`` / ``score_motion``
    LK, VET, DARTS, Proesmans (``core.nowcast.methods``) from any product, and
    their error against the true flow.
``nowcasts``
    persistence, extrapolation along each product's own motion, along the true
    motion (the advection ceiling) and along any extra motion field given (e.g.
    a learned one), S-PROG, STEPS (ensemble mean), optionally LINDA - scored
    against the truth with ``core.nowcast.verify``.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from core.maps import scores as ms
from core.simulation.scenario import SyntheticCase

IDW_FULL = {"power": 2.0, "radius_m": None, "nnear": None}      # cover the whole domain
ADJUST = ("mfb", "add", "mul")
MERGEPLG = ("idw_add", "idw_mul", "okrig_add", "ked")


# --------------------------------------------------------------------------
# maps
# --------------------------------------------------------------------------
def maps(case: SyntheticCase, sensor_sets=("gauges", "links", "links+gauges"),
         with_mergeplg: bool = True, with_gmz: bool = True, timings: Optional[dict] = None) -> dict:
    """``{name: (time, lat, lon) mm}`` for every method and sensor combination available."""
    from core.maps import merge
    from core.maps.idw import idw_map
    timings = {} if timings is None else timings
    out = {}

    def timed(name, fn):
        t = time.perf_counter()
        try:
            out[name] = fn()
        except Exception as e:                       # a method that cannot run is reported, not fatal
            out[name] = None
            timings[name + ":error"] = repr(e)
        timings[name] = time.perf_counter() - t

    grid = case.geo_grid
    if case.radar is not None:
        out["radar"] = case.radar
    sources = {"links": case.links, "gauges": case.gauges, "pws": case.pws}
    sources = {k: v for k, v in sources.items() if v is not None and v.size}

    if "links" in sources:
        timed("idw links", lambda: idw_map(sources["links"], grid, radius_m=None))
        if with_gmz:
            timed("gmz links", lambda: _gmz(sources["links"], grid))
    if "gauges" in sources:
        obs_g, _ = merge.observations(gauges=sources["gauges"])
        timed("idw gauges", lambda: merge.merge_idw(obs_g, grid, IDW_FULL))
    if {"links", "gauges"} <= set(sources):
        obs_lg, _ = merge.observations(cml=sources["links"], gauges=sources["gauges"])
        timed("idw links+gauges", lambda: merge.merge_idw(obs_lg, grid, IDW_FULL))
    if len(sources) == 3:
        obs_all, _ = merge.observations(cml=sources["links"], gauges=sources["gauges"], pws=sources["pws"])
        timed("idw links+gauges+pws", lambda: merge.merge_idw(obs_all, grid, IDW_FULL))

    if case.radar is None:
        return out
    for sset in sensor_sets:
        parts = sset.split("+")
        if not all(p in sources for p in parts):
            continue
        kw = {("cml" if p == "links" else p): sources[p] for p in parts}
        obs, rad = merge.observations(case.radar, **kw)
        for m in ADJUST:
            timed(f"{m} [{sset}]", lambda m=m: merge.adjust(case.radar, obs, rad, method=m,
                                                            idw=IDW_FULL, min_radar_mm=0.05))
        if with_mergeplg and "pws" not in parts:            # mergeplg takes links and gauges only
            from core.maps.mergeplg_methods import Merger
            try:
                mg = Merger(case.radar, links=sources.get("links") if "links" in parts else None,
                            gauges=sources.get("gauges") if "gauges" in parts else None)
            except Exception as e:
                timings[f"mergeplg [{sset}]:error"] = repr(e)
                continue
            for m in MERGEPLG:
                timed(f"{m} [{sset}]", lambda m=m: mg.field(m))
    return out


def _gmz(links, grid):
    from core.maps.gmz import gmz_map
    return gmz_map(links, grid, radius_m=1e9)


def hourly(da: xr.DataArray, interval_min: float) -> np.ndarray:
    """Sum of consecutive intervals into hours (incomplete last hour dropped)."""
    k = int(round(60 / interval_min))
    a = da.transpose("time", "lat", "lon").values
    n = a.shape[0] // k
    return a[:n * k].reshape(n, k, *a.shape[1:]).sum(1)


def score_maps(case: SyntheticCase, products: dict) -> pd.DataFrame:
    """One row per method: scores per interval (threshold 0.1 mm) and on hourly totals (1 mm)."""
    truth_h = hourly(case.truth, case.interval_min)
    rows = []
    for name, da in products.items():
        if da is None:
            continue
        da = da.transpose("time", "lat", "lon").sel(time=case.truth.time.values)
        s5 = ms.scores(da.values, case.truth.values, wet_threshold=0.1)
        sh = ms.scores(hourly(da, case.interval_min), truth_h, wet_threshold=1.0) if truth_h.size \
            else {"n": 0}
        rows.append({"method": name, "sensors": _sensors_of(name), "family": _family_of(name),
                     **{f"{k}": v for k, v in sh.items()},
                     **{f"{k}_5min": s5[k] for k in ("nrmse", "corr", "rel_bias", "csi") if k in s5},
                     "coverage": float(np.isfinite(da.values).mean())})
    return pd.DataFrame(rows)


def _sensors_of(name: str) -> str:
    if name == "radar":
        return "radar"
    if "[" in name:
        return "radar+" + name.split("[")[1].rstrip("]")
    return name.split(" ", 1)[1]


def _family_of(name: str) -> str:
    if name == "radar":
        return "radar only"
    if "[" in name:
        return "radar adjusted"
    return "no radar"


# --------------------------------------------------------------------------
# motion
# --------------------------------------------------------------------------
MOTION = ("LK", "VET", "DARTS", "proesmans")


def nowcast_meta(case: SyntheticCase) -> dict:
    from core.nowcast.grid import metadata
    return metadata(case.geo_grid, int(case.interval_min), case.truth.time.values)


def motion(rate: np.ndarray, meta: dict, methods=MOTION) -> dict:
    """``{method: (2, y, x)}`` pixels per step from the fields up to the last one."""
    from core.nowcast import methods as nm
    out = {}
    for m in methods:
        try:
            out[m] = nm.motion(rate, meta, m)
        except Exception:
            out[m] = None
    return out


def score_motion(est: np.ndarray, true: np.ndarray, mask: Optional[np.ndarray] = None,
                 px_to_kmh: float = 1.0) -> dict:
    """Mean endpoint error, mean speed error and direction error, over ``mask`` (default all)."""
    if est is None:
        return {"epe_kmh": np.nan, "speed_err_kmh": np.nan, "dir_err_deg": np.nan, "vec_corr": np.nan}
    m = np.ones(true.shape[1:], bool) if mask is None else mask
    d = (est - true)[:, m] * px_to_kmh
    se = (np.hypot(*est[:, m]) - np.hypot(*true[:, m])) * px_to_kmh
    ang = np.degrees(np.arctan2(est[1, m], est[0, m]) - np.arctan2(true[1, m], true[0, m]))
    ang = (ang + 180) % 360 - 180
    a, b = est[:, m].ravel(), true[:, m].ravel()
    return {"epe_kmh": float(np.hypot(*d).mean()), "speed_err_kmh": float(se.mean()),
            "dir_err_deg": float(np.abs(ang).mean()),
            "vec_corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 else np.nan}


# --------------------------------------------------------------------------
# nowcasting
# --------------------------------------------------------------------------
DETERMINISTIC = ("persistence", "extrapolation", "sprog")


def nowcasts(case: SyntheticCase, product_rates: dict, issue_idx, n_leads: int = 12,
             methods=DETERMINISTIC, steps_members: int = 0, with_linda: bool = False,
             extra_motion: Optional[dict] = None, motion_method: str = "LK",
             collector=None):
    """Nowcast every product from every issue time; accumulate scores against the truth.

    ``product_rates``: ``{product: (time, y, x) mm/h}``. ``extra_motion``:
    ``{label: fn(rate_history) -> (2, y, x)}`` for motion fields from elsewhere
    (the learned estimator); each adds an ``extrapolation (label)`` method.
    Returns a ``core.nowcast.verify.Collector`` keyed by
    ``(product, method, lead)``; cells whose rain must have come from outside the
    domain (by the true motion) are not scored, for any method.
    """
    from core.nowcast import methods as nm
    from core.nowcast.verify import Collector, Deterministic
    meta = nowcast_meta(case)
    truth = case.rate
    true_v = case.velocity
    col = collector or Collector(lambda: Deterministic(case.pixel_km))
    reach = nm.reachable(true_v, truth.shape[1:], n_leads)
    for t in issue_idx:
        obs = truth[t + 1:t + 1 + n_leads]
        if obs.shape[0] < n_leads:
            continue
        obs = np.where(reach, obs, np.nan)
        for prod, rate in product_rates.items():
            hist = rate[:t + 1]
            v = nm.motion(hist, meta, motion_method)
            fc = {}
            for m in methods:
                fc[m] = nm.deterministic(m, hist, meta, v, n_leads)
            fc["extrapolation (true motion)"] = nm.deterministic("extrapolation", hist, meta, true_v, n_leads)
            for label, fn in (extra_motion or {}).items():
                fc[f"extrapolation ({label})"] = nm.deterministic("extrapolation", hist, meta, fn(hist), n_leads)
            if with_linda:
                fc["linda"] = nm.deterministic("linda", hist, meta, v, n_leads)
            if steps_members:
                ens = nm.ensemble("steps", hist, meta, v, n_leads, n_members=steps_members, seed=int(t))
                fc["steps (mean)"] = np.nanmean(ens, axis=0)
            for m, f in fc.items():
                f = np.where(np.isfinite(f), f, 0.0)
                for k in range(n_leads):
                    col[(prod, m, (k + 1) * case.interval_min)].add(f[k], obs[k])
    return col


# --------------------------------------------------------------------------
# accuracy by spatial and temporal scale
# --------------------------------------------------------------------------
SPACE_KM = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)
TIME_MIN = (5, 15, 30, 60)


def _to_scale(a: np.ndarray, dx_from: float, s_km: float) -> np.ndarray:
    """``(t, y, x)`` from pixels of ``dx_from`` km to pixels of ``s_km``: block mean, or each
    pixel repeated (an estimate read at a finer pixel holds its own value)."""
    from core.simulation.sensors import block_mean
    r = s_km / dx_from
    if r >= 1:
        k = int(round(r))
        n = a.shape[-1] // k * k
        return block_mean(a[..., :n, :n], k)
    k = int(round(1 / r))
    return np.kron(a, np.ones((1, k, k)))


def _sum_time(a: np.ndarray, k: int) -> np.ndarray:
    n = a.shape[0] // k
    return a[:n * k].reshape(n, k, *a.shape[1:]).sum(1)


def score_scales(case: SyntheticCase, products: dict, space_km=SPACE_KM, time_min=TIME_MIN,
                 wet_mm_h: float = 1.0, region: Optional[np.ndarray] = None,
                 min_region_frac: float = 0.5) -> pd.DataFrame:
    """Every product against the truth at every pair of spatial and temporal scales.

    The truth is aggregated from its own grid (``case.truth_fine``, e.g. 100 m); products
    from the analysis grid, and read pixel by pixel at scales finer than it - the value a
    street-level user gets from that map. ``region`` (bool on the truth grid) keeps only the
    blocks at least ``min_region_frac`` inside it. CSI at ``wet_mm_h`` (scaled to the
    accumulation period).
    """
    T = case.truth_fine
    rows = []
    for m in time_min:
        k = int(round(m / case.interval_min))
        if k < 1 or T.shape[0] // k == 0:
            continue
        Tt = _sum_time(T, k)
        thr = wet_mm_h * m / 60.0
        for s in space_km:
            if s < case.dx_km - 1e-9:
                continue
            truth_s = _to_scale(Tt, case.dx_km, s)
            keep = None
            if region is not None:
                frac = _to_scale(region[None].astype(float), case.dx_km, s)[0]
                keep = frac >= min_region_frac
                if not keep.any():
                    continue
            for name, da in products.items():
                if da is None:
                    continue
                a = _sum_time(da.transpose("time", "lat", "lon").values.astype(float), k)
                est = _to_scale(a, case.pixel_km, s)
                n = min(est.shape[-1], truth_s.shape[-1])
                e, t = est[..., :n, :n], truth_s[..., :n, :n]
                if keep is not None:
                    e, t = e[:, keep[:n, :n]], t[:, keep[:n, :n]]
                sc = ms.scores(e, t, wet_threshold=thr)
                rows.append({"method": name, "space_km": s, "time_min": m,
                             **{k_: sc.get(k_, np.nan) for k_ in ("nrmse", "corr", "rel_bias", "csi", "n")}})
    return pd.DataFrame(rows)


def link_distance_km(case: SyntheticCase) -> np.ndarray:
    """Distance from every truth-grid cell to the nearest link path (km), ``(n, n)``."""
    net = case.network
    n = case.truth_fine.shape[-1]
    c = (np.arange(n) + 0.5) * case.dx_km
    X, Y = np.meshgrid(c, c)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    best = np.full(P.shape[0], np.inf)
    for xa, ya, xb, yb in zip(net.xa, net.ya, net.xb, net.yb):
        d = np.array([xb - xa, yb - ya])
        t = np.clip(((P - [xa, ya]) @ d) / max(d @ d, 1e-12), 0, 1)
        q = np.array([xa, ya]) + t[:, None] * d
        best = np.minimum(best, np.hypot(*(P - q).T))
    return best.reshape(n, n)


def maps_with(case: SyntheticCase, radar_name: str, prefix: str, **kw) -> dict:
    """``maps`` with another of the case's radars as the radar, names prefixed."""
    import dataclasses
    other = dataclasses.replace(case, radar=case.radars[radar_name])
    out = maps(other, **kw)
    return {f"{prefix}{k}": v for k, v in out.items() if k == "radar" or "[" in k}
