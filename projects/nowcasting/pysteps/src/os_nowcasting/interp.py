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

The interpolation itself is in :mod:`core.nowcast.advection`; this module adds the motion
field from :func:`os_nowcasting.nowcast.motion`.
"""

from __future__ import annotations

import numpy as np

from core.nowcast import advection
from core.nowcast.advection import interpolate_pair  # noqa: F401  (re-exported)

from .nowcast import motion


def hourly_total(rates: np.ndarray, meta: dict, n_sub: int = 5, interpolate: bool = True) -> np.ndarray:
    """Hour total (mm) from the scans ``rates`` (mm/h); with ``interpolate``, advected frames
    are added between scans along Lucas-Kanade motion (see :func:`core.nowcast.advection.hourly_total`)."""
    if not interpolate:
        return advection.hourly_total(rates)
    v = motion(np.where(np.isfinite(rates), rates, 0.0), meta, "LK")
    return advection.hourly_total(rates, v, n_sub)
