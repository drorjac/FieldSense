"""
Scoring forecasts against gauges (point to pixel) and radar (grid to grid).

Part 6 of ``advanced_models_colab_v2.ipynb``. Metrics are the notebook's:
RMSE, MAE, bias, Pearson CC, and HSS / F1 / precision / recall of the wet
(> 0.1 mm/h) decision.

Every forecast is ``{h: (N_h, H, W)}`` over the test split, sample ``i``
aligned with truth frame ``lookback - 1 + h + i`` (see ``forecasting``).
The truth is sliced here, in one place, so no model can be off by a step
without every model being off by the same step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix


def metrics(truth, pred, threshold: float = 0.1) -> dict:
    """Continuous and wet/dry skill over all finite pairs."""
    t = np.asarray(truth, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if t.size == 0:
        return {k: np.nan for k in ("RMSE", "MAE", "Bias", "CC", "HSS", "F1",
                                    "Precision", "Recall")}
    tb, pb = (t > threshold).astype(int), (p > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(tb, pb, labels=[0, 1]).ravel()
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return {"RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
            "MAE": float(np.mean(np.abs(p - t))),
            "Bias": float(np.mean(p - t)),
            "CC": float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 and t.std() > 1e-9 else np.nan,
            "HSS": float(2 * (tp * tn - fp * fn) / den) if den else np.nan,
            "F1": float(2 * precision * recall / (precision + recall))
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan,
            "Precision": float(precision), "Recall": float(recall)}


def gauge_pixels(lon, lat, gauge_lon, gauge_lat) -> np.ndarray:
    """Flat index of the grid cell nearest each gauge (in lon/lat, as the notebook)."""
    tree = cKDTree(np.column_stack([np.ravel(lon), np.ravel(lat)]))
    return tree.query(np.column_stack([gauge_lon, gauge_lat]))[1]


def truth_frames(test: np.ndarray, lookback: int, h: int, n: int) -> np.ndarray:
    """The ``n`` frames forecasts for horizon ``h`` are scored against."""
    start = lookback - 1 + h
    return test[start:start + n]


def score(forecasts: dict, radar_test: np.ndarray, gauge_test: np.ndarray,
          pixels: np.ndarray, lookback: int, crop: tuple, threshold: float = 0.1):
    """Two tables, vs gauges and vs radar in the crop, one row per model x horizon.

    ``forecasts`` is ``{model: {h: (N_h, H, W)}}``; ``gauge_test`` is
    (T_test, n_gauges) on the same axis as ``radar_test``.
    """
    y0, y1, x0, x1 = crop
    gauge_rows, radar_rows = [], []
    for name, by_h in forecasts.items():
        for h, pred in by_h.items():
            n = len(pred)
            g = truth_frames(gauge_test, lookback, h, n)
            r = truth_frames(radar_test, lookback, h, n)
            if len(g) != n:
                continue
            at_gauges = pred.reshape(n, -1)[:, pixels]
            gauge_rows.append({"model": name, "horizon_min": 15 * h,
                               **metrics(g, at_gauges, threshold)})
            radar_rows.append({"model": name, "horizon_min": 15 * h,
                               **metrics(r[:, y0:y1, x0:x1], pred[:, y0:y1, x0:x1], threshold)})
    return pd.DataFrame(gauge_rows), pd.DataFrame(radar_rows)


def ranking(table: pd.DataFrame, metric: str = "HSS", higher_is_better: bool = True):
    """Mean over horizons per model, best first."""
    return (table.groupby("model")[["RMSE", "CC", "HSS", "F1"]].mean()
            .sort_values(metric, ascending=not higher_is_better))


def compare(faithful: pd.DataFrame, corrected: pd.DataFrame, metric: str = "HSS"):
    """Mean-over-horizons ``metric`` per model, both modes and the change."""
    a = faithful.groupby("model")[metric].mean().rename("faithful")
    b = corrected.groupby("model")[metric].mean().rename("corrected")
    out = pd.concat([a, b], axis=1)
    out["change"] = out.corrected - out.faithful
    return out.sort_values("change", ascending=False)
