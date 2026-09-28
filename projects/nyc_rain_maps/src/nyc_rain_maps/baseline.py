"""Baselines and wet/dry classification (numpy re-implementations of PyNNcml's).

The baseline is the attenuation a link would show without rain (free-space loss,
antenna misalignment, humidity, ...). Rain attenuation = total loss - baseline.

These functions reproduce PyNNcml's torch modules exactly (verified in the tests),
including edge handling and NaN propagation, but work on plain arrays of shape
``(n_links, n_time)`` and run without torch.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def trailing_min(x: np.ndarray, window: int, skipna: bool = False) -> np.ndarray:
    """``min(x[i-window+1 .. i])`` along the last axis (shorter window at the start).

    By default NaN propagates - any NaN in the window gives NaN, as ``torch.min`` does, so
    one missing minute blanks the next ``window`` minutes. ``skipna=True`` ignores NaN
    (NaN only where the whole window is NaN).
    """
    x = np.asarray(x, dtype=float)
    if skipna:
        out = trailing_min(np.where(np.isnan(x), np.inf, x), window)
        return np.where(np.isinf(out), np.nan, out)
    pad = np.full(x.shape[:-1] + (window - 1,), np.inf)
    padded = np.concatenate([pad, x], axis=-1)
    return np.min(sliding_window_view(padded, window, axis=-1), axis=-1)


def dynamic_baseline(attenuation: np.ndarray, window: int = 200, quantization_delta: float = 1.0,
                     skipna: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One-step dynamic baseline (Ostrometzky & Messer 2018), PyNNcml ``OneStepDynamic``.

    ``bl = trailing_min(A + qd/2, window)`` and rain attenuation ``A - bl - qd/2``,
    i.e. ``A - min(A over the last `window` samples) - qd``. The quantization delta
    removes the bias of subtracting a *minimum* of 1-dB-quantized samples.
    Returns ``(rain_attenuation, baseline)``.
    """
    half = quantization_delta / 2.0
    bl = trailing_min(np.asarray(attenuation, float) + half, window, skipna=skipna)
    return attenuation - bl - half, bl


def std_wet_dry(attenuation: np.ndarray, window: int = 240, threshold: float = 1.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-std wet/dry (Schleiss & Berne 2010), PyNNcml ``STDWetDry``.

    sigma_i = population std of A over the trailing window ending at i, for
    i >= window-1; the series is then shifted so the window is (nearly) centred, with
    zeros padded at both ends - the first ``(window-1)//2`` and last
    ``window-1-(window-1)//2`` samples are always dry. wet = clip(round(sigma / (2 th)), 0, 1)
    (round half to even), which equals ``sigma > th`` except exactly at the boundary.
    Returns ``(wet in {0, 1, NaN}, sigma)``.
    """
    A = np.atleast_2d(np.asarray(attenuation, float))
    n = A.shape[-1]
    if n < window:
        sig = np.zeros_like(A)
    else:
        core = np.std(sliding_window_view(A, window, axis=-1), axis=-1)   # (links, n-window+1)
        lead = (window - 1) // 2
        sig = np.concatenate([np.zeros(A.shape[:-1] + (lead,)), core,
                              np.zeros(A.shape[:-1] + (window - 1 - lead,))], axis=-1)
    wet = np.clip(np.round(sig / (2.0 * threshold)), 0.0, 1.0)
    return wet, sig


def constant_baseline(attenuation: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Constant baseline (Schleiss & Berne 2010), PyNNcml ``ConstantBaseLine``.

    While wet, hold the last value; while dry, follow the attenuation. The first sample
    is always the attenuation. NaN in ``wet`` counts as wet (held), as in PyNNcml.
    """
    A = np.atleast_2d(np.asarray(attenuation, float))
    W = np.atleast_2d(np.asarray(wet, float))
    bl = np.empty_like(A)
    bl[:, 0] = A[:, 0]
    hold = (W != 0)          # NaN != 0 -> True
    for i in range(1, A.shape[1]):
        bl[:, i] = np.where(hold[:, i], bl[:, i - 1], A[:, i])
    return bl


def median_dry_baseline(total_loss: np.ndarray, dry: np.ndarray) -> np.ndarray:
    """Per-link median of total loss over samples flagged ``dry`` (manual method)."""
    TL = np.atleast_2d(np.asarray(total_loss, float))
    D = np.atleast_2d(np.asarray(dry, bool))
    return np.array([np.nanmedian(t[d]) if d.any() else np.nan for t, d in zip(TL, D)])
