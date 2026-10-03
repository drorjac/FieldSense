"""Simulated links over simulated rain: the exact path integral against the linear law.

A link of length ``L`` at frequency ``f`` in a rain field ``R(x)`` attenuates by

    A_exact  = integral over the path of a R(x)^b dx  =  a L mean(R^b)
    A_linear = a L mean(R)^b                           (the law every retrieval inverts)

with ``(a, b)`` from ITU-R P.838-3. The two agree when the rain is uniform along the
path or ``b = 1``; otherwise Jensen's inequality puts ``A_exact`` above ``A_linear``
for ``b > 1`` and below for ``b < 1`` (Berne & Uijlenhoet 2007). Everything here
calls ``core.simulation``: the fields come from ``generators``, the integral and the
linear law from ``cml_network.forward_model``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.itu_p838 import get_k_alpha
from core.simulation import cml_network as cn
from core.simulation import generators as gen
from core.simulation.rain_fields import Grid

# rain regimes, as presets of core.simulation.generators: from smooth and widespread
# to small intense cells
REGIMES = ("stratiform_matern", "scale_free", "banded_anisotropic",
           "clustered_storms", "squall_line", "convective_cells")
# backhaul bands from 10 GHz (b ~ 1.25) to 80 GHz (b ~ 0.7)
FREQS_GHZ = (10.0, 15.0, 18.0, 23.0, 26.0, 32.0, 38.0, 60.0, 80.0)
LENGTH_BINS_KM = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0)

# the clean chain: no wet antenna, baseline error, noise or quantization
CLEAN = cn.SensorConfig(waa_max_db=0.0, waa_max_db_assumed=0.0, baseline_sigma_db=0.0,
                        noise_sigma_db=0.0, quantization_db=0.0)


def random_links(grid: Grid, n_links: int, length_km=(0.3, 20.0), freqs_ghz=FREQS_GHZ,
                 seed: int = 0) -> cn.CMLNetwork:
    """``n_links`` straight links inside ``grid``: lengths log-uniform in ``length_km``,
    random orientation, frequency drawn from ``freqs_ghz``, polarization at random."""
    rng = np.random.default_rng(seed)
    L = np.exp(rng.uniform(*np.log(length_km), n_links))
    theta = rng.uniform(0.0, np.pi, n_links)
    half = 0.5 * L[:, None] * np.c_[np.cos(theta), np.sin(theta)]
    margin = np.abs(half) + grid.dx_km
    mid = rng.uniform(margin, grid.size_km - margin)
    a, b = mid - half, mid + half
    freqs = rng.choice(np.asarray(freqs_ghz, float), n_links)
    pols = np.where(rng.random(n_links) < 0.5, "vertical", "horizontal")
    ka = np.array([get_k_alpha(float(f), str(p)) for f, p in zip(freqs, pols)])
    return cn.CMLNetwork(xa=a[:, 0], ya=a[:, 1], xb=b[:, 0], yb=b[:, 1], freq_ghz=freqs,
                         pol=pols, k=ka[:, 0], alpha=ka[:, 1])


def path_law_table(net: cn.CMLNetwork, grid: Grid, regimes=REGIMES, n_fields: int = 4,
                   seed: int = 0, wet_mm_h: float = 0.1) -> pd.DataFrame:
    """One row per (regime, field, link) with path-mean rain of at least ``wet_mm_h``.

    Columns: ``regime``, ``field``, ``link``, ``L_km``, ``f_ghz``, ``pol``, ``a``, ``b``,
    ``R_bar`` (path-mean rain, mm/h), ``cv_path`` (its coefficient of variation along
    the path), ``A_exact`` and ``A_linear`` (dB), and ``R_linear``, the rain the linear
    law retrieves from ``A_exact``.
    """
    rows = []
    for regime in regimes:
        for i in range(n_fields):
            rain = gen.make(regime, seed=seed + i).build(grid)
            out = cn.forward_model(rain, grid, net, CLEAN)
            s = out["samples"]
            r = out["R_path_true"]
            with np.errstate(invalid="ignore", divide="ignore"):
                cv = np.where(r > 0, s.std(axis=1) / r, np.nan)
            rows.append(pd.DataFrame({
                "regime": regime, "field": i, "link": np.arange(net.n_links),
                "L_km": net.length_km, "f_ghz": net.freq_ghz, "pol": net.pol,
                "a": net.k, "b": net.alpha, "R_bar": r, "cv_path": cv,
                "A_exact": out["A_rain"], "A_linear": out["A_uniform"],
                "R_linear": out["R_retrieved_clean"]})[r >= wet_mm_h])
    table = pd.concat(rows, ignore_index=True)
    table["regime"] = pd.Categorical(table.regime, categories=list(regimes))
    return table


def length_bin(L_km, bins=LENGTH_BINS_KM) -> pd.Categorical:
    """Link length (km) as a labelled bin, ``"2-4 km"``."""
    labels = [f"{lo:g}-{hi:g} km" for lo, hi in zip(bins[:-1], bins[1:])]
    return pd.cut(np.asarray(L_km), bins, labels=labels)


def gap_summary(table: pd.DataFrame, by=("regime",), bins=LENGTH_BINS_KM) -> pd.DataFrame:
    """The linear law's error in rain, ``R_linear / R_bar - 1``, by length bin (and ``by``).

    Columns: ``n``, ``median``, ``p10``, ``p90`` and ``median_abs`` of the relative
    rain error, and ``within_5pct``, the fraction of cases the linear law gets within 5 %.
    """
    t = table.assign(gap=table.R_linear / table.R_bar - 1.0, length=length_bin(table.L_km, bins))
    g = t.groupby(list(by) + ["length"], observed=True).gap
    return pd.DataFrame({"n": g.size(), "median": g.median(), "p10": g.quantile(0.1),
                         "p90": g.quantile(0.9), "median_abs": g.apply(lambda x: float(x.abs().median())),
                         "within_5pct": g.apply(lambda x: float((x.abs() < 0.05).mean()))})


def score_path_law(table: pd.DataFrame, A_hat, bins=LENGTH_BINS_KM) -> pd.DataFrame:
    """Score a path law ``A_hat`` (one value per row of ``table``) against ``A_exact``.

    By length bin: ``rmse_db``, the median relative error and the fraction within 5 %.
    The linear law, ``A_hat = table.A_linear``, is the baseline.
    """
    t = table.assign(err=np.asarray(A_hat, float) - table.A_exact,
                     length=length_bin(table.L_km, bins))
    t["rel"] = t.err / t.A_exact
    g = t.groupby("length", observed=True)
    return pd.DataFrame({"n": g.size(), "rmse_db": g.err.apply(lambda e: float(np.sqrt((e**2).mean()))),
                         "median_rel_error": g.rel.median(),
                         "within_5pct": g.rel.apply(lambda x: float((x.abs() < 0.05).mean()))})
