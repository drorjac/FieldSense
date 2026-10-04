"""
1-D fields: rain along a line, rain at a point in time.

A link sees rain along a line; a gauge sees it as a time series at a point.
Both are 1-D fields, and their statistics are what the sensors' errors depend
on: path-averaging bias depends on the variability along the path,
tipping-bucket error on the intermittency in time. This module simulates both
directly, without a 2-D field:

``metagaussian_1d``   a 1-D GRF (any covariance of ``random_fields``) transformed to rain
                      with an exact wet fraction - a transect or a time series
``cascade_1d``        beta-lognormal cascade in 1-D (Over & Gupta 1996)
``bartlett_lewis``    Bartlett-Lewis rectangular pulses (Rodriguez-Iturbe, Cox & Isham
                      1987): Poisson storms, each a Poisson train of rectangular cells
``neyman_scott``      Neyman-Scott rectangular pulses (Cowpertwait 1991): Poisson storms,
                      each with a Poisson number of cells displaced exponentially
``transect``          the 1-D profile of a 2-D field along a segment (what a link sees)

Times in hours, lengths in km, intensities in mm/h.
"""

from __future__ import annotations

import numpy as np

from core.simulation import random_fields as rf


def metagaussian_1d(n: int, dx: float, rng: np.random.Generator, cov: str = "exponential",
                    length: float = 5.0, nu: float = 1.0, beta: float = 2.0, war: float = 0.4,
                    dist: str = "lognormal", mean_wet: float = 3.0, cv_wet: float = 1.2) -> np.ndarray:
    g = rf.grf_1d(n, dx, rng, cov, length, nu, beta)
    return rf.to_rain(g, war, dist, mean_wet, cv_wet)


def cascade_1d(n_levels: int, rng: np.random.Generator, beta: float = 0.2, sigma2: float = 0.3,
               mean: float = 1.0) -> np.ndarray:
    """``2**n_levels`` values; each level splits a cell in two with weights ``B * Y``."""
    f = np.array([mean])
    p_wet = 2.0 ** (-beta)
    s = np.sqrt(sigma2)
    for _ in range(n_levels):
        f = np.repeat(f, 2)
        b = np.where(rng.random(f.size) < p_wet, 1.0 / p_wet, 0.0)
        f = f * b * np.exp(s * rng.standard_normal(f.size) - sigma2 / 2)
    return f


def _pulses_to_series(starts, durations, intensities, duration_h, dt_h):
    """Sum rectangular pulses on a regular grid, exactly (fractional overlap per step)."""
    n = int(round(duration_h / dt_h))
    edges = np.arange(n + 1) * dt_h
    out = np.zeros(n)
    for s, d, x in zip(starts, durations, intensities):
        e = s + d
        i0, i1 = int(np.floor(s / dt_h)), int(np.ceil(e / dt_h))
        for i in range(max(i0, 0), min(i1, n)):
            overlap = min(e, edges[i + 1]) - max(s, edges[i])
            if overlap > 0:
                out[i] += x * overlap / dt_h
    return out


def bartlett_lewis(duration_h: float, dt_min: float, rng: np.random.Generator,
                   storm_rate_per_h: float = 0.02, cell_rate_per_h: float = 2.0,
                   storm_activity_h: float = 6.0, cell_duration_h: float = 0.3,
                   cell_intensity_mm_h: float = 4.0) -> np.ndarray:
    """Original Bartlett-Lewis rectangular pulses model, mean rate per step (mm/h).

    Storms arrive as a Poisson process (``storm_rate_per_h``); within a storm cells
    arrive at ``cell_rate_per_h`` for an exponential activity period of mean
    ``storm_activity_h`` (the first cell at the storm origin); each cell is a
    rectangular pulse with exponential duration and intensity.
    """
    starts, durs, ints = [], [], []
    t = rng.exponential(1 / storm_rate_per_h)
    while t < duration_h:
        active = rng.exponential(storm_activity_h)
        c = 0.0
        while c <= active:
            starts.append(t + c)
            durs.append(rng.exponential(cell_duration_h))
            ints.append(rng.exponential(cell_intensity_mm_h))
            c += rng.exponential(1 / cell_rate_per_h)
        t += rng.exponential(1 / storm_rate_per_h)
    return _pulses_to_series(starts, durs, ints, duration_h, dt_min / 60.0)


def neyman_scott(duration_h: float, dt_min: float, rng: np.random.Generator,
                 storm_rate_per_h: float = 0.02, mean_cells: float = 8.0, cell_delay_h: float = 1.0,
                 cell_duration_h: float = 0.3, cell_intensity_mm_h: float = 4.0) -> np.ndarray:
    """Neyman-Scott rectangular pulses model, mean rate per step (mm/h).

    Storm origins are Poisson; each storm has ``Poisson(mean_cells)`` cells whose
    starts are exponentially delayed after the origin (mean ``cell_delay_h``).
    """
    starts, durs, ints = [], [], []
    t = rng.exponential(1 / storm_rate_per_h)
    while t < duration_h:
        for _ in range(rng.poisson(mean_cells)):
            starts.append(t + rng.exponential(cell_delay_h))
            durs.append(rng.exponential(cell_duration_h))
            ints.append(rng.exponential(cell_intensity_mm_h))
        t += rng.exponential(1 / storm_rate_per_h)
    return _pulses_to_series(starts, durs, ints, duration_h, dt_min / 60.0)


def transect(field: np.ndarray, dx_km: float, x0: float, y0: float, x1: float, y1: float,
             n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """``(distance_km, values)`` of ``field`` along the segment (bilinear)."""
    from scipy.ndimage import map_coordinates
    s = np.linspace(0, 1, n)
    x, y = x0 + s * (x1 - x0), y0 + s * (y1 - y0)
    v = map_coordinates(field, [y / dx_km - 0.5, x / dx_km - 0.5], order=1, mode="nearest")
    return s * np.hypot(x1 - x0, y1 - y0), v


def series_stats(r: np.ndarray, dt_min: float) -> dict:
    """Wet fraction, mean, lag-1 autocorrelation and the mean wet/dry spell lengths (min)."""
    wet = r > 0.1
    runs = np.diff(np.concatenate([[0], wet.astype(int), [0]]))
    wet_spells = (np.flatnonzero(runs == -1) - np.flatnonzero(runs == 1)) * dt_min
    dry = ~wet
    druns = np.diff(np.concatenate([[0], dry.astype(int), [0]]))
    dry_spells = (np.flatnonzero(druns == -1) - np.flatnonzero(druns == 1)) * dt_min
    a = r - r.mean()
    return {"wet_fraction": float(wet.mean()), "mean_mm_h": float(r.mean()),
            "acf1": float((a[1:] * a[:-1]).mean() / a.var()) if a.var() > 0 else np.nan,
            "mean_wet_spell_min": float(wet_spells.mean()) if wet_spells.size else 0.0,
            "mean_dry_spell_min": float(dry_spells.mean()) if dry_spells.size else 0.0}
