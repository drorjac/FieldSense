"""
Turning per-link path averages back into a 2-D rainfall field, and scoring it.

Two interpolators:

``idw_midpoint``
    Inverse-distance weighting from the link midpoint. The standard baseline,
    and the standard criticism: it throws away the fact that a CML measures a
    line, collapsing a 5 km path to a single point.

``idw_path``
    Inverse-distance weighting from points spread along each path, with each
    link's total weight normalized so a long link does not outvote a short one
    purely by contributing more samples.

The error decomposition is the point of the exercise. Reconstructing from the
*true* path averages isolates what is lost to network geometry - sparse,
non-uniform, line-integral sampling. Reconstructing from the *retrieved* path
averages adds what the sensor chain costs on top.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cml_network import CMLNetwork
from rain_fields import WET_THRESHOLD_MM_H, Grid


def idw_midpoint(net: CMLNetwork, values: np.ndarray, grid: Grid,
                 power: float = 2.0, smoothing_km: float = 0.4) -> np.ndarray:
    """Inverse-distance weighting from link midpoints."""
    mx, my = net.midpoints
    xx, yy = grid.meshgrid()
    return _idw(xx, yy, mx, my, values, power, smoothing_km)


def idw_path(net: CMLNetwork, values: np.ndarray, grid: Grid,
             power: float = 2.0, smoothing_km: float = 0.4,
             n_samples: int = 24) -> np.ndarray:
    """Inverse-distance weighting from points spread along each path.

    Each link contributes ``n_samples`` pseudo-observations carrying that
    link's path-averaged value, weighted by ``1 / n_samples`` so every link
    carries one link's worth of influence regardless of its length.
    """
    px, py = net.path_points(n_samples)
    xx, yy = grid.meshgrid()
    vals = np.repeat(values[:, None], n_samples, axis=1).ravel()
    per_link = np.full(vals.shape, 1.0 / n_samples)
    return _idw(xx, yy, px.ravel(), py.ravel(), vals, power, smoothing_km,
                extra_weight=per_link)


def _idw(xx, yy, sx, sy, values, power, smoothing_km,
         extra_weight: Optional[np.ndarray] = None) -> np.ndarray:
    dx = xx[..., None] - sx[None, None, :]
    dy = yy[..., None] - sy[None, None, :]
    d2 = dx * dx + dy * dy + smoothing_km**2
    w = d2 ** (-power / 2.0)
    if extra_weight is not None:
        w = w * extra_weight[None, None, :]
    return (w * values[None, None, :]).sum(axis=-1) / w.sum(axis=-1)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(truth: np.ndarray, estimate: np.ndarray) -> dict:
    """Field-level agreement between a reconstruction and the truth."""
    t = truth.ravel()
    e = estimate.ravel()
    err = e - t

    denom = t.std() * e.std()
    corr = float(((t - t.mean()) * (e - e.mean())).mean() / denom) if denom > 0 else float("nan")

    war_t = float((truth >= WET_THRESHOLD_MM_H).mean())
    war_e = float((estimate >= WET_THRESHOLD_MM_H).mean())

    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "corr": corr,
        "imf_true": float(t.mean()),
        "imf_est": float(e.mean()),
        "war_true": war_t,
        "war_est": war_e,
        "peak_true": float(t.max()),
        "peak_est": float(e.max()),
        "peak_ratio": float(e.max() / t.max()) if t.max() > 0 else float("nan"),
    }


def decompose(truth: np.ndarray, ideal: np.ndarray, full: np.ndarray) -> dict:
    """Split reconstruction error into sampling and sensor contributions.

    ``ideal`` is reconstructed from true path averages, ``full`` from retrieved
    ones. The two error fields are not guaranteed orthogonal, so the sensor
    share is reported as the *increment* in RMSE rather than as an
    independently attributable variance.
    """
    s_ideal = score(truth, ideal)
    s_full = score(truth, full)
    increment = s_full["rmse"] - s_ideal["rmse"]
    return {
        "rmse_sampling": s_ideal["rmse"],
        "rmse_total": s_full["rmse"],
        "rmse_sensor_increment": increment,
        "sampling_share": (s_ideal["rmse"] / s_full["rmse"]
                           if s_full["rmse"] > 0 else float("nan")),
    }
