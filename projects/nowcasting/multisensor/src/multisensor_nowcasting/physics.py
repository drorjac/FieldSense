"""Physics nowcasts of every product with pysteps (``core.nowcast``), scored per issue time.

Issue times: every 30 minutes of the test split (the RNN's test weeks, :func:`links.split`)
when more than 5% of the radar's domain has rain above 0.1 mm/h, with 30 minutes of
history and the next hour present.

For each issue time and product (radar ``R``; links ``C``; PWS ``P``; links and PWS ``CP``;
radar adjusted with links ``RC``, PWS ``RP`` or both ``RCP``):

``<p>|persistence``             the last field, unchanged
``<p>|extrapolation``           semi-Lagrangian advection of the last field (dBR) along the
                                product's own Lucas-Kanade motion
``<p>|sprog``                   S-PROG along the product's own motion
``<p>|extrapolation_Rmotion``   the same along the *radar's* motion (sensor products only)
``<p>|sprog_Rmotion``           S-PROG along the radar's motion
``<p>|steps_mean``              the mean of a 10-member STEPS ensemble (R and RCP with their
                                own motion, CP with the radar's); the ensemble's CRPS is kept

Every forecast is compared with the radar at each lead (``scoring.stats``) on the cells
the radar's motion can reach, and sampled at the independent gauges. The radar's motion
also gives the verification mask, so all forecasts are scored on the same cells.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from core.nowcast import methods as M
from core.nowcast.grid import metadata

from . import scoring
from .cube import Cube, cell_index
from .settings import (CACHE_DIR, ENSEMBLE_PRODUCTS, ISSUE_EVERY, MIN_WET_FRACTION, N_HISTORY, N_LEADS,
                       N_MEMBERS, PHYSICS_METHODS, PRODUCTS, SEED, STEP_MIN)

log = logging.getLogger(__name__)


def issue_times(cube: Cube) -> np.ndarray:
    """Indices of the issue times (see the module docstring)."""
    t = cube.times
    ok = cube.contiguous
    n = t.size
    # history and leads all contiguous
    run = np.zeros(n, int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if ok[i] else 0
    need_hist = run >= N_HISTORY - 1
    fut = np.zeros(n, bool)
    fut[:n - N_LEADS] = run[N_LEADS:] >= N_LEADS
    on_grid = (t == t.floor(ISSUE_EVERY))
    test = cube.split == 2
    cand = np.flatnonzero(need_hist & fut & on_grid & test)
    R = cube["R"]
    wet = np.array([np.nanmean(R[i] > 0.1) if np.isfinite(R[i]).any() else 0.0 for i in cand])
    return cand[wet > MIN_WET_FRACTION]


def forecast_ids() -> list:
    ids = []
    for p in PRODUCTS:
        ids += [f"{p}|{m}" for m in PHYSICS_METHODS]
        if p != "R":
            ids += [f"{p}|extrapolation_Rmotion", f"{p}|sprog_Rmotion"]
    ids += [f"{p}|steps_mean" for p in ENSEMBLE_PRODUCTS]
    return ids


# ------------------------------------------------------------------ worker

_W = {}


def _init(network: str):
    import warnings
    warnings.filterwarnings("ignore")
    c = Cube(network)
    g = c.gauges["g5"]
    _W.update(cube=c, meta=metadata(c.grid, STEP_MIN, product="multisensor_nowcasting"),
              area=c.area, gcells=cell_index(c.grid, g.lat.values, g.lon.values))


def history(cube: Cube, ch: str, i: int) -> np.ndarray:
    return np.nan_to_num(np.asarray(cube[ch][i - N_HISTORY + 1:i + 1], dtype=float), nan=0.0)


def masks_for(cube: Cube, i: int, meta: dict, area: np.ndarray):
    """Radar motion and the verification masks of issue ``i``."""
    v_r = M.motion(history(cube, "R", i), meta, "LK")
    reach = M.reachable(v_r, cube.grid.shape, N_LEADS)
    return v_r, {"domain": reach, "area": reach & area[None]}


def _one(i: int) -> dict:
    cube, meta, area, gcells = _W["cube"], _W["meta"], _W["area"], _W["gcells"]
    t0 = time.time()
    obs = np.asarray(cube["R"][i + 1:i + 1 + N_LEADS], dtype=float)
    v_r, masks = masks_for(cube, i, meta, area)
    out = {"i": i, "stats": {}, "gauge": {}, "crps": {}, "crps_g": {}, "motion": {}}

    def keep(fid, f):
        f = np.clip(np.nan_to_num(f, nan=0.0), 0, None)
        out["stats"][fid] = scoring.stats(f, obs, masks)
        out["gauge"][fid] = f.reshape(N_LEADS, -1)[:, gcells].astype("float32")

    wet = obs[0] > 0.1
    for p in PRODUCTS:
        hist = history(cube, p, i)
        try:
            v_own = v_r if p == "R" else M.motion(hist, meta, "LK")
        except Exception:
            v_own = np.zeros_like(v_r)
        if p != "R":
            d = np.hypot(*(v_own - v_r)) * 2.0 * 60 / STEP_MIN
            out["motion"][p] = float(d[wet].mean()) if wet.any() else np.nan
        motions = [("", v_own)] + ([("_Rmotion", v_r)] if p != "R" else [])
        for m in PHYSICS_METHODS:
            for suffix, v in (motions if m != "persistence" else [("", v_own)]):
                fid = f"{p}|{m}{suffix}"
                try:
                    keep(fid, M.deterministic(m, hist, meta, v, N_LEADS))
                except Exception as exc:
                    log.debug("%s at %d: %r", fid, i, exc)
        if p in ENSEMBLE_PRODUCTS:
            v = v_r if p == "CP" else v_own
            try:
                e = M.ensemble("steps", hist, meta, v, N_LEADS, n_members=N_MEMBERS, seed=SEED + i)
                e = np.clip(np.nan_to_num(e, nan=0.0), 0, None)
                keep(f"{p}|steps_mean", e.mean(0))
                cr = scoring.crps_members(e, obs)
                c = np.zeros((N_LEADS, len(scoring.REGIONS), 2))
                for li in range(N_LEADS):
                    for ri, r in enumerate(scoring.REGIONS):
                        mk = masks[r][li] & np.isfinite(obs[li])
                        c[li, ri] = cr[li][mk].sum(), mk.sum()
                out["crps"][p] = c
                out["crps_g"][p] = e.reshape(N_MEMBERS, N_LEADS, -1)[:, :, gcells].astype("float32")
            except Exception as exc:
                log.debug("steps %s at %d: %r", p, i, exc)
    out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------ driver


def save_issues(cube: Cube, idx: np.ndarray) -> dict:
    """Issue times, their verification masks and the gauge observations, shared by every method."""
    path = CACHE_DIR / cube.network / "scores" / "issues.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata(cube.grid, STEP_MIN)
    area = cube.area
    masks = np.zeros((idx.size, N_LEADS) + cube.grid.shape, bool)
    for k, i in enumerate(idx):
        masks[k] = masks_for(cube, int(i), meta, area)[1]["domain"]
    g = cube.gauges["g5"]
    G = g.transpose("time", "station").values
    gobs = np.stack([G[i + 1:i + 1 + N_LEADS] for i in idx]).astype("float32")
    times = cube.times[idx]
    np.savez_compressed(path, idx=idx, times=times.values, days=times.floor("D").values,
                        reach=masks, gauges=gobs, stations=g.station.values.astype(str))
    return dict(np.load(path, allow_pickle=True))


def load_issues(network: str) -> dict:
    return dict(np.load(CACHE_DIR / network / "scores" / "issues.npz", allow_pickle=True))


def run(network: str, workers: int = 7, refresh: bool = False) -> None:
    import multiprocessing as mp
    out = CACHE_DIR / network / "scores" / "physics.npz"
    if out.exists() and not refresh:
        log.info("physics scores exist: %s", out)
        return
    cube = Cube(network)
    idx = issue_times(cube)
    log.info("%s: %d issue times", network, idx.size)
    save_issues(cube, idx)
    ids = forecast_ids()
    n_g = cube.gauges["g5"].sizes["station"]
    S = np.full((idx.size, len(ids), N_LEADS, len(scoring.REGIONS), len(scoring.STATS)), np.nan)
    Gv = np.full((idx.size, len(ids), N_LEADS, n_g), np.nan, dtype="float32")
    C = np.full((idx.size, len(ENSEMBLE_PRODUCTS), N_LEADS, len(scoring.REGIONS), 2), np.nan)
    Cg = np.full((idx.size, len(ENSEMBLE_PRODUCTS), N_MEMBERS, N_LEADS, n_g), np.nan, dtype="float32")
    motion = []
    pos = {int(i): k for k, i in enumerate(idx)}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(network,)) as pool:
        for n_done, r in enumerate(pool.imap_unordered(_one, [int(i) for i in idx], chunksize=2), 1):
            k = pos[r["i"]]
            for fid, s in r["stats"].items():
                S[k, ids.index(fid)] = s
                Gv[k, ids.index(fid)] = r["gauge"][fid]
            for p, c in r["crps"].items():
                C[k, ENSEMBLE_PRODUCTS.index(p)] = c
                Cg[k, ENSEMBLE_PRODUCTS.index(p)] = r["crps_g"][p]
            motion.append({"issue": k, **r["motion"]})
            if n_done % 50 == 0:
                log.info("%s physics: %d/%d issues (%.0f s)", network, n_done, idx.size, time.time() - t0)
    np.savez_compressed(out, ids=np.array(ids), stats=S, gauge=Gv, crps=C, crps_gauge=Cg,
                        ens_products=np.array(ENSEMBLE_PRODUCTS))
    pd.DataFrame(motion).sort_values("issue").to_csv(out.with_name("motion_vs_radar.csv"), index=False)
    log.info("%s physics done in %.0f s", network, time.time() - t0)
