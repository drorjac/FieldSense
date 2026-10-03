"""Scores as additive sums, so every forecast - physics or learned - is scored the same way.

For each forecast, issue time and lead time, :func:`stats` reduces the forecast and the
radar it is compared with to sums over the verified cells: the contingency table at each
threshold, the sums behind MAE, RMSE, bias and correlation, and the numerator and
denominator of the fractions skill score at each scale. Pooled scores and their bootstrap
confidence intervals (resampling storm days) are then functions of these sums
(:func:`pooled`, :func:`bootstrap`), and two forecasts are compared on exactly the same
issue times, cells and leads.

Regions: ``domain`` - every cell the radar's motion can reach from inside the domain
(rain advected in from outside is unknown to every method; one mask for all methods, from
the radar's own motion) and where the radar has data; ``area`` - the same, within 10 km of
a link or PWS, where the sensors can know the rain.

The values at the independent gauges are kept as they are (``(issue, lead, gauge)``), for
point scores at 5 minutes and for the next-hour accumulation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .settings import CAT_THRESHOLDS, FSS_SCALES_KM, N_BOOT, PIXEL_KM

REGIONS = ("domain", "area")
FSS_THRESHOLD = 1.0
STATS = ([f"{k}_{t:g}" for t in CAT_THRESHOLDS for k in ("hit", "miss", "fa", "cn")]
         + ["n", "sf", "so", "sae", "sse", "sff", "soo", "sfo"]
         + [f"fss{k}_{s}" for s in FSS_SCALES_KM for k in ("num", "den")])
IDX = {s: i for i, s in enumerate(STATS)}


def stats(forecast: np.ndarray, obs: np.ndarray, masks: dict) -> np.ndarray:
    """``(lead, region, stat)`` sums for one forecast ``(lead, y, x)`` against ``obs``."""
    n_lead = forecast.shape[0]
    out = np.zeros((n_lead, len(REGIONS), len(STATS)))
    for li in range(n_lead):
        f_all, o_all = forecast[li], obs[li]
        for ri, region in enumerate(REGIONS):
            m = masks[region][li] & np.isfinite(o_all) & np.isfinite(f_all)
            if not m.any():
                continue
            f, o = f_all[m], o_all[m]
            row = out[li, ri]
            for t in CAT_THRESHOLDS:
                fy, oy = f >= t, o >= t
                row[IDX[f"hit_{t:g}"]] = np.sum(fy & oy)
                row[IDX[f"miss_{t:g}"]] = np.sum(~fy & oy)
                row[IDX[f"fa_{t:g}"]] = np.sum(fy & ~oy)
                row[IDX[f"cn_{t:g}"]] = np.sum(~fy & ~oy)
            d = f - o
            row[IDX["n"]] = f.size
            row[IDX["sf"]], row[IDX["so"]] = f.sum(), o.sum()
            row[IDX["sae"]], row[IDX["sse"]] = np.abs(d).sum(), (d * d).sum()
            row[IDX["sff"]], row[IDX["soo"]], row[IDX["sfo"]] = (f * f).sum(), (o * o).sum(), (f * o).sum()
            w = m.astype(float)
            for s in FSS_SCALES_KM:
                size = max(1, int(round(s / PIXEL_KM)))
                cover = ndimage.uniform_filter(w, size, mode="constant")
                inside = m & (cover > 0.5)
                if not inside.any():
                    continue
                pf = ndimage.uniform_filter((np.nan_to_num(f_all) >= FSS_THRESHOLD) * w, size, mode="constant") / np.maximum(cover, 1e-9)
                po = ndimage.uniform_filter((np.nan_to_num(o_all) >= FSS_THRESHOLD) * w, size, mode="constant") / np.maximum(cover, 1e-9)
                row[IDX[f"fssnum_{s}"]] = np.sum((pf - po)[inside] ** 2)
                row[IDX[f"fssden_{s}"]] = np.sum(pf[inside] ** 2 + po[inside] ** 2)
    return out


def crps_members(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS per cell for an ensemble ``(member, ...)`` (the energy form), same shape as ``obs``."""
    m = np.sort(members, axis=0)
    n = m.shape[0]
    t1 = np.mean(np.abs(m - obs[None]), axis=0)
    i = np.arange(1, n + 1).reshape((n,) + (1,) * obs.ndim)
    t2 = np.sum((2 * i - n - 1) * m, axis=0) / (n * n)
    return t1 - t2


def scores_from(S: np.ndarray) -> dict:
    """Scores from summed stats ``(..., stat)`` -> ``{name: array(...)}``."""
    g = lambda k: S[..., IDX[k]]
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for t in CAT_THRESHOLDS:
            h, mi, fa = g(f"hit_{t:g}"), g(f"miss_{t:g}"), g(f"fa_{t:g}")
            out[f"CSI_{t:g}"] = h / (h + mi + fa)
            out[f"POD_{t:g}"] = h / (h + mi)
            out[f"FAR_{t:g}"] = fa / (h + fa)
        n = g("n")
        out["MAE"] = g("sae") / n
        out["RMSE"] = np.sqrt(g("sse") / n)
        out["bias"] = g("sf") / g("so") - 1.0
        cov = g("sfo") / n - (g("sf") / n) * (g("so") / n)
        vf = g("sff") / n - (g("sf") / n) ** 2
        vo = g("soo") / n - (g("so") / n) ** 2
        out["corr"] = cov / np.sqrt(vf * vo)
        for s in FSS_SCALES_KM:
            out[f"FSS_{s}km"] = 1.0 - g(f"fssnum_{s}") / g(f"fssden_{s}")
    return out


def pooled(S: np.ndarray) -> dict:
    """``S (issue, lead, region, stat)`` -> scores ``(lead, region)`` pooled over issues."""
    return scores_from(S.sum(axis=0))


def bootstrap(S: np.ndarray, days: np.ndarray, metric: str, S_ref: np.ndarray | None = None,
              n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """Percentile 95% CI of ``metric`` (or of its difference to ``S_ref``, paired) resampling days.

    Returns ``(estimate, low, high)`` arrays of shape ``(lead, region)``.
    """
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(days, return_inverse=True)
    per_day = np.stack([S[inv == k].sum(0) for k in range(ud.size)])
    per_ref = None if S_ref is None else np.stack([S_ref[inv == k].sum(0) for k in range(ud.size)])

    def value(idx):
        v = scores_from(per_day[idx].sum(0))[metric]
        if per_ref is not None:
            v = v - scores_from(per_ref[idx].sum(0))[metric]
        return v

    est = value(np.arange(ud.size))
    draws = np.stack([value(rng.integers(0, ud.size, ud.size)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return est, lo, hi


def point_scores(f: np.ndarray, o: np.ndarray) -> dict:
    """RMSE, correlation and bias of paired point values (NaN pairs dropped)."""
    ok = np.isfinite(f) & np.isfinite(o)
    f, o = f[ok], o[ok]
    if f.size < 3:
        return {"n": int(f.size), "RMSE": np.nan, "corr": np.nan, "bias": np.nan}
    return {"n": int(f.size), "RMSE": float(np.sqrt(np.mean((f - o) ** 2))),
            "corr": float(np.corrcoef(f, o)[0, 1]) if f.std() > 0 and o.std() > 0 else np.nan,
            "bias": float(f.sum() / o.sum() - 1.0) if o.sum() > 0 else np.nan}
