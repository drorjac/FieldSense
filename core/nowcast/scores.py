"""Verification scores pysteps does not provide without optional dependencies.

``sal_score``  structure, amplitude and location of Wernli et al. (2008), on
               ``scipy.ndimage`` (pysteps' own SAL needs scikit-image)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def sal_score(forecast: np.ndarray, obs: np.ndarray, thr_factor: float = 1 / 15,
              wet: float = 0.1) -> tuple | None:
    """Structure, amplitude, location (Wernli et al. 2008); None if either field is dry."""
    def objects(R):
        w = R[R > wet]
        if w.size < 4:
            return None
        lab, n = ndimage.label(R >= thr_factor * np.quantile(w, 0.95))
        return lab, n

    def scaled_volume(R, lab, n):
        idx = np.arange(1, n + 1)
        tot = ndimage.sum(R, lab, idx)
        mx = ndimage.maximum(R, lab, idx)
        return float(np.sum(tot * tot / mx) / np.sum(tot)) if n else 0.0

    def com(R, lab=None, n=0):
        yy, xx = np.indices(R.shape)
        c = np.array([np.sum(yy * R), np.sum(xx * R)]) / np.sum(R)
        if lab is None:
            return c
        idx = np.arange(1, n + 1)
        tot = ndimage.sum(R, lab, idx)
        cs = np.array(ndimage.center_of_mass(R, lab, idx)).reshape(-1, 2)
        r = np.sum(tot * np.hypot(*(cs - c).T)) / np.sum(tot) if n else 0.0
        return c, r

    of, oo = objects(forecast), objects(obs)
    if of is None or oo is None or of[1] == 0 or oo[1] == 0:
        return None
    vf, vo = scaled_volume(forecast, *of), scaled_volume(obs, *oo)
    S = (vf - vo) / (0.5 * (vf + vo))
    df, do = forecast.mean(), obs.mean()
    A = (df - do) / (0.5 * (df + do))
    d = np.hypot(*forecast.shape)
    (cf, rf), (co, ro) = com(forecast, *of), com(obs, *oo)
    L = np.hypot(*(cf - co)) / d + 2 * abs(rf - ro) / d
    return float(S), float(A), float(L)
