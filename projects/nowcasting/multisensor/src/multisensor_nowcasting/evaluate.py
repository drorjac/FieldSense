"""Physics and learned nowcasts in one table, against the radar and at the independent gauges.

Reads ``scores/physics.npz`` (:mod:`.physics`) and ``scores/learned.npz`` (:func:`learn.score_all`),
both on the same issue times, and writes to ``results/``:

``scores_<network>.csv``       every forecast x lead x region: CSI, POD, FAR (0.5, 1, 5 mm/h),
                               FSS (2, 10, 30 km at 1 mm/h), MAE, RMSE, bias, correlation
``gauges_<network>.csv``       every forecast at the gauges: 5-min rate at each lead, and the
                               next-hour accumulation (mean rate over the 12 leads)
``headline_<network>.csv``     differences to the radar's extrapolation nowcast with 95%
                               intervals from resampling storm days (paired bootstrap)
``crps_<network>.csv``         STEPS ensembles: CRPS against the radar and at the gauges
``training_<network>.csv``     the trained models: samples, weights, epochs, validation loss
``motion_<network>.csv``       how far each product's motion is from the radar's
"""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pandas as pd

from . import scoring
from .physics import load_issues
from .settings import CACHE_DIR, HEADLINE_LEADS, N_BOOT, N_LEADS, RESULTS_DIR, STEP_MIN

log = logging.getLogger(__name__)

REFERENCE = "R|extrapolation"
HEADLINE_METRICS = ("CSI_1", "FSS_10km", "MAE")


def _fallback(ids: list, S: np.ndarray, G: np.ndarray) -> pd.DataFrame:
    """S-PROG fails on nearly dry fields (a sensor map with rain in a few cells); there its
    forecast is the extrapolation along the same motion (S-PROG of a field with no
    predictable scales left). Returns how often each forecast fell back."""
    rows = []
    for j, fid in enumerate(ids):
        if "|sprog" not in fid:
            continue
        k = ids.index(fid.replace("|sprog", "|extrapolation"))
        bad = ~np.isfinite(S[:, j, 0, 0, scoring.IDX["n"]])
        S[bad, j], G[bad, j] = S[bad, k], G[bad, k]
        rows.append({"forecast": fid, "fallback_to": ids[k], "issues": int(bad.sum()), "of": int(bad.size)})
    return pd.DataFrame(rows)


def load(network: str) -> dict:
    d = CACHE_DIR / network / "scores"
    phys = np.load(d / "physics.npz", allow_pickle=True)
    ids = list(phys["ids"])
    s, g = phys["stats"].copy(), phys["gauge"].copy()
    fallback = _fallback(ids, s, g)
    S, G = [s], [g]
    if (d / "learned.npz").exists():
        ln = np.load(d / "learned.npz", allow_pickle=True)
        ids += list(ln["ids"])
        S.append(ln["stats"])
        G.append(ln["gauge"])
    return {"ids": ids, "stats": np.concatenate(S, axis=1), "gauge": np.concatenate(G, axis=1),
            "phys": phys, "issues": load_issues(network), "fallback": fallback}


def _common(S: np.ndarray) -> np.ndarray:
    """Issues where every forecast exists (a failed pysteps run drops that issue for all)."""
    ok = np.all(np.isfinite(S[..., scoring.IDX["n"]]), axis=(1, 2, 3))
    return ok


def score_table(D: dict, network: str) -> pd.DataFrame:
    S = D["stats"]
    ok = _common(S)
    rows = []
    for j, fid in enumerate(D["ids"]):
        sc = scoring.pooled(S[ok, j])
        for li in range(N_LEADS):
            for ri, region in enumerate(scoring.REGIONS):
                rows.append({"network": network, "forecast": fid, "lead_min": (li + 1) * STEP_MIN,
                             "region": region, "n_issues": int(ok.sum()),
                             **{k: float(v[li, ri]) for k, v in sc.items()}})
    return pd.DataFrame(rows)


def gauge_table(D: dict, network: str) -> pd.DataFrame:
    ok = _common(D["stats"])
    obs = D["issues"]["gauges"][ok]                         # (issue, lead, gauge)
    rows = []
    for j, fid in enumerate(D["ids"]):
        f = D["gauge"][ok, j]
        for li in range(N_LEADS):
            lm = (li + 1) * STEP_MIN
            if lm in HEADLINE_LEADS:
                rows.append({"network": network, "forecast": fid, "what": f"rate at {lm} min",
                             **scoring.point_scores(f[:, li].ravel(), obs[:, li].ravel())})
        fh = f.mean(1)
        oh = np.where(np.isfinite(obs).all(1), obs.mean(1), np.nan)
        rows.append({"network": network, "forecast": fid, "what": "next-hour total",
                     **scoring.point_scores(fh.ravel(), oh.ravel())})
    return pd.DataFrame(rows)


def _gauge_hour_bootstrap(D: dict, j: int, j_ref: int, days, n_boot=N_BOOT, seed=0):
    """Paired bootstrap (days) of the next-hour RMSE difference at the gauges."""
    ok = _common(D["stats"])
    obs = D["issues"]["gauges"][ok]
    oh = np.where(np.isfinite(obs).all(1), obs.mean(1), np.nan)
    a, b = D["gauge"][ok, j].mean(1), D["gauge"][ok, j_ref].mean(1)
    valid = np.isfinite(oh) & np.isfinite(a) & np.isfinite(b)
    ea = np.where(valid, (a - oh) ** 2, 0).sum(1)
    eb = np.where(valid, (b - oh) ** 2, 0).sum(1)
    n = valid.sum(1)
    ud, inv = np.unique(days, return_inverse=True)
    A = np.bincount(inv, ea, ud.size)
    B = np.bincount(inv, eb, ud.size)
    N = np.bincount(inv, n, ud.size)

    def val(ix):
        return np.sqrt(A[ix].sum() / N[ix].sum()) - np.sqrt(B[ix].sum() / N[ix].sum())

    rng = np.random.default_rng(seed)
    draws = [val(rng.integers(0, ud.size, ud.size)) for _ in range(n_boot)]
    return val(np.arange(ud.size)), *np.percentile(draws, [2.5, 97.5])


def headline(D: dict, network: str) -> pd.DataFrame:
    S = D["stats"]
    ok = _common(S)
    days = D["issues"]["days"][ok]
    j_ref = D["ids"].index(REFERENCE)
    rows = []
    for j, fid in enumerate(D["ids"]):
        for metric in HEADLINE_METRICS:
            if fid == REFERENCE:
                est, lo, hi = scoring.bootstrap(S[ok, j], days, metric)
            else:
                est, lo, hi = scoring.bootstrap(S[ok, j], days, metric, S_ref=S[ok, j_ref])
            for lm in HEADLINE_LEADS:
                li = lm // STEP_MIN - 1
                for ri, region in enumerate(scoring.REGIONS):
                    rows.append({"network": network, "forecast": fid, "metric": metric, "lead_min": lm,
                                 "region": region, "vs": "value" if fid == REFERENCE else REFERENCE,
                                 "estimate": est[li, ri], "low": lo[li, ri], "high": hi[li, ri]})
        if fid != REFERENCE:
            e, lo, hi = _gauge_hour_bootstrap(D, j, j_ref, days)
            rows.append({"network": network, "forecast": fid, "metric": "gauge next-hour RMSE", "lead_min": 60,
                         "region": "gauges", "vs": REFERENCE, "estimate": e, "low": lo, "high": hi})
    return pd.DataFrame(rows)


def crps_table(D: dict, network: str) -> pd.DataFrame:
    phys = D["phys"]
    ok = _common(D["stats"])
    C, Cg = phys["crps"][ok], phys["crps_gauge"][ok]
    obs = D["issues"]["gauges"][ok]
    rows = []
    for k, p in enumerate(phys["ens_products"]):
        for li in range(N_LEADS):
            lm = (li + 1) * STEP_MIN
            if lm not in HEADLINE_LEADS:
                continue
            for ri, region in enumerate(scoring.REGIONS):
                c = np.nansum(C[:, k, li, ri, 0]) / np.nansum(C[:, k, li, ri, 1])
                rows.append({"network": network, "product": str(p), "lead_min": lm, "where": region, "CRPS": c})
            m = Cg[:, k, :, li, :]                           # (issue, member, gauge)
            o = obs[:, li, :]
            good = np.isfinite(o) & np.all(np.isfinite(m), axis=1)
            cr = scoring.crps_members(np.moveaxis(m, 1, 0), np.nan_to_num(o))
            rows.append({"network": network, "product": str(p), "lead_min": lm, "where": "gauges",
                         "CRPS": float(cr[good].mean()) if good.any() else np.nan})
    return pd.DataFrame(rows)


def run(network: str, refresh: bool = False) -> None:
    from . import learn
    learn.score_all(network, refresh=refresh)
    D = load(network)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    score_table(D, network).to_csv(RESULTS_DIR / f"scores_{network}.csv", index=False, float_format="%.4f")
    gauge_table(D, network).to_csv(RESULTS_DIR / f"gauges_{network}.csv", index=False, float_format="%.4f")
    headline(D, network).to_csv(RESULTS_DIR / f"headline_{network}.csv", index=False, float_format="%.4f")
    crps_table(D, network).to_csv(RESULTS_DIR / f"crps_{network}.csv", index=False, float_format="%.4f")
    d = CACHE_DIR / network / "scores"
    for src, dst in (("training.csv", f"training_{network}.csv"), ("motion_vs_radar.csv", f"motion_{network}.csv")):
        if (d / src).exists():
            shutil.copy(d / src, RESULTS_DIR / dst)
    D["fallback"].assign(network=network).to_csv(RESULTS_DIR / f"sprog_fallback_{network}.csv", index=False)
    iss = D["issues"]
    pd.DataFrame({"issue": pd.DatetimeIndex(iss["times"]), "day": pd.DatetimeIndex(iss["days"]),
                  "scored": _common(D["stats"])}).to_csv(RESULTS_DIR / f"issues_{network}.csv", index=False)
    log.info("%s evaluated: %d forecasts, %d issue times", network, len(D["ids"]), int(_common(D["stats"]).sum()))
