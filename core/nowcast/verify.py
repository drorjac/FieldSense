"""Verification, pooled over issue times with pysteps' own accumulators.

The session's scores (blocks 04 and 04a), each accumulated over every issue time of a
study and computed once at the end, so a score is a pooled one (one contingency table,
one set of sums), not an average of per-forecast scores:

* categorical, per threshold (0.5, 1, 5 mm/h): POD, FAR, CSI, ETS, frequency bias
  (``pysteps.verification.detcatscores``);
* continuous: ME, MAE, RMSE, Pearson correlation (``detcontscores``);
* spatial: fractions skill score at 2-50 km for 1 mm/h (``spatialscores.fss``), and
  SAL (Wernli et al. 2008) - :func:`core.nowcast.scores.sal_score`, on ``scipy.ndimage`` with pysteps' object
  definition (threshold 1/15 of the 95th percentile of wet pixels), because pysteps' SAL
  needs scikit-image;
* ensemble: CRPS, ROC area and reliability for 1 mm/h, rank histogram
  (``probscores``, ``ensscores``).

Only cells where both fields are finite count; categorical and continuous scores take the
valid cells as flat arrays (pysteps counts a NaN pair as a correct negative otherwise).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .scores import sal_score

CAT_THRESHOLDS = (0.5, 1.0, 5.0)          # mm/h
FSS_THRESHOLD = 1.0
FSS_SCALES_KM = (2, 10, 30, 50)
PROB_THRESHOLD = 1.0

CAT_SCORES = ("POD", "FAR", "CSI", "ETS", "BIAS")
CONT_SCORES = ("ME", "MAE", "RMSE", "corr_p")


class Deterministic:
    """Pooled deterministic scores of one (product, method, reference, lead) stream."""

    def __init__(self, pixel_km: float, gridded: bool = True, thresholds=CAT_THRESHOLDS,
                 fss_scales_km=FSS_SCALES_KM, fss_threshold: float = FSS_THRESHOLD):
        from pysteps.verification import detcatscores as dc, detcontscores as dk, spatialscores as ss
        self.cat = {t: dc.det_cat_fct_init(t) for t in thresholds}
        self.cont = dk.det_cont_fct_init()
        self.fss = {s: ss.fss_init(fss_threshold, max(1, int(round(s / pixel_km)))) for s in fss_scales_km} \
            if gridded else {}
        self.sal = []
        self.n = 0

    def add(self, forecast: np.ndarray, obs: np.ndarray, sal: bool = False):
        from pysteps.verification import detcatscores as dc, detcontscores as dk, spatialscores as ss
        ok = np.isfinite(forecast) & np.isfinite(obs)
        if not ok.any():
            return
        f, o = forecast[ok], obs[ok]
        for t, c in self.cat.items():
            dc.det_cat_fct_accum(c, f, o)
        dk.det_cont_fct_accum(self.cont, f, o)
        if self.fss:
            F, O = np.where(ok, forecast, 0.0), np.where(ok, obs, 0.0)
            for s, acc in self.fss.items():
                ss.fss_accum(acc, F, O)
            if sal:
                v = sal_score(F, O)
                if v is not None:
                    self.sal.append(v)
        self.n += 1

    def compute(self) -> dict:
        from pysteps.verification import detcatscores as dc, detcontscores as dk, spatialscores as ss
        out = {"n_forecasts": self.n}
        if self.n == 0:
            return out
        for t, c in self.cat.items():
            r = dc.det_cat_fct_compute(c, scores=list(CAT_SCORES))
            out.update({f"{k}_{t:g}": float(v) for k, v in r.items()})
        r = dk.det_cont_fct_compute(self.cont, scores=list(CONT_SCORES))
        out.update({("corr" if k == "corr_p" else k): float(v) for k, v in r.items()})
        for s, acc in self.fss.items():
            out[f"FSS_{s}km"] = float(ss.fss_compute(acc))
        if self.sal:
            s = np.array(self.sal)
            out.update(SAL_S=float(np.mean(s[:, 0])), SAL_A=float(np.mean(s[:, 1])),
                       SAL_L=float(np.mean(s[:, 2])), SAL_n=len(s))
        return out


class Ensemble:
    """Pooled ensemble scores: CRPS, ROC area and reliability at 1 mm/h, rank histogram."""

    def __init__(self, n_members: int, prob_threshold: float = PROB_THRESHOLD):
        from pysteps import verification as v
        self.thr = prob_threshold
        self.crps = v.probscores.CRPS_init()
        self.roc = v.probscores.ROC_curve_init(prob_threshold, n_prob_thrs=10)
        self.rel = v.probscores.reldiag_init(prob_threshold, n_bins=10)
        self.rank = v.ensscores.rankhist_init(n_members, 0.1)
        self.mean = Deterministic(1.0, gridded=False)
        self.n = 0

    def add(self, members: np.ndarray, obs: np.ndarray):
        from pysteps import verification as v
        ok = np.isfinite(obs) & np.all(np.isfinite(members), axis=0)
        if not ok.any():
            return
        m, o = members[:, ok], obs[ok]
        v.probscores.CRPS_accum(self.crps, m, o)
        p = (m >= self.thr).mean(axis=0)          # exceedance probability
        v.probscores.ROC_curve_accum(self.roc, p, o)
        v.probscores.reldiag_accum(self.rel, p, o)
        v.ensscores.rankhist_accum(self.rank, m, o)
        self.mean.add(m.mean(axis=0), o)
        self.n += 1

    def compute(self) -> dict:
        from pysteps import verification as v
        if self.n == 0:
            return {"n_forecasts": 0}
        pod, pofd, area = v.probscores.ROC_curve_compute(self.roc, compute_area=True)
        rel = v.probscores.reldiag_compute(self.rel)
        rank = v.ensscores.rankhist_compute(self.rank)
        mean = {f"ensmean_{k}": val for k, val in self.mean.compute().items()
                if k in ("MAE", "RMSE", "corr", "CSI_1")}
        return {"n_forecasts": self.n, "CRPS": float(v.probscores.CRPS_compute(self.crps)),
                "ROC_area": float(area), "rel_prob": [float(x) for x in rel[0]],
                "rel_freq": [float(x) for x in rel[1]], "rank_hist": [float(x) for x in rank],
                **mean}


def at_points(fields: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """``(..., y, x)`` fields sampled at flat cell indices -> ``(..., n_points)``."""
    flat = fields.reshape(fields.shape[:-2] + (-1,))
    return flat[..., cells]


class Collector:
    """Accumulators keyed by any hashable (e.g. product, method, reference, lead)."""

    def __init__(self, factory):
        self.acc = defaultdict(factory)

    def __getitem__(self, key):
        return self.acc[key]

    def rows(self, names: tuple) -> list:
        return [{**dict(zip(names, k)), **a.compute()} for k, a in self.acc.items()]
