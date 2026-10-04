"""Advection-based temporal interpolation for accumulations (session block 03a).

An hourly total built from 5-minute radar scans treats each scan as the rate for its
whole interval, so a moving cell leaves a string of beads. Interpolating between
consecutive scans along the motion field - each intermediate frame a time-weighted mean
of the earlier scan advected forward and the later one advected back - gives the smooth
swath the rain actually laid down. This is the session's ``interpolate`` function,
vectorised, on a motion field from Lucas-Kanade on the hour's scans in dBR.

It is meant for *instantaneous* scans (OpenMRG's 5-minute reflectivity). OpenRainER's
radar product is already a 15-minute rain depth - an integral over time - so there the
plain sum is the right total and interpolation only blurs it; it is not applied.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def interpolate_pair(r1: np.ndarray, r2: np.ndarray, velocity: np.ndarray, n_sub: int,
                     order: int = 1) -> list:
    """Frames strictly between ``r1`` and ``r2`` (one step apart), ``n_sub`` sub-steps per step."""
    ny, nx = r1.shape
    x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    out = []
    for k in range(1, n_sub):
        a = k / n_sub
        w1 = map_coordinates(r1, (y - a * velocity[1], x - a * velocity[0]), order=order, mode="nearest")
        w2 = map_coordinates(r2, (y + (1 - a) * velocity[1], x + (1 - a) * velocity[0]), order=order,
                             mode="nearest")
        out.append((1 - a) * w1 + a * w2)
    return out


def hourly_total(rates: np.ndarray, velocity: np.ndarray | None = None, n_sub: int = 5) -> np.ndarray:
    """Hour total (mm) from the scans ``rates`` (mm/h) spanning the hour, both ends included
    (13 scans for 5-minute data), as in the session: the mean of all frames times 1 h.
    With a ``velocity`` field ``(2, y, x)`` in pixels per scan step, ``n_sub - 1`` advected
    frames are added between each pair; without one, the plain mean is returned."""
    rates = np.where(np.isfinite(rates), rates, 0.0)
    if velocity is None:
        return rates.mean(axis=0)
    frames = list(rates)
    for i in range(len(rates) - 1):
        frames.extend(interpolate_pair(rates[i], rates[i + 1], velocity, n_sub))
    return np.mean(frames, axis=0)
