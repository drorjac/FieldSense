"""
How to combine a physics branch and a data-driven branch, on synthetic links.

The physics branch inverts the ITU-R power law ``A = k R^alpha L``; the data
branch is a small MLP from attenuation to rain rate. Five ways of producing
one estimate are compared, all fitted on the training split and scored on
the test split:

``physics``   the power-law inversion alone
``mlp``       the MLP alone
``scalar``    ``w r_phys + (1 - w) r_mlp``, one weight grid-searched
``gate``      ``w(A)`` a logistic function of attenuation, fitted to the
              per-sample ideal weights
``fc``        a tiny MLP on ``[r_phys, r_mlp]``
``residual``  physics plus an MLP trained on the physics residual

Consolidated from the original working notebook ``Simulation_MBML.ipynb``,
which wrote the generator, the mixers and the sweep out three times. Its
coefficient table matched ITU-R P.838-3 *horizontal*; here they come from
``core.itu_p838`` with the polarization a parameter.

    from mixing import sweep
    table = sweep(frequencies=(5, 23, 60, 70), noise_db=np.linspace(0.001, 1, 20))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from core.itu_p838 import get_k_alpha

STRATEGIES = ("physics", "mlp", "scalar", "gate", "fc", "residual")


@dataclass(frozen=True)
class Link:
    """One synthetic link: the power law and how the data are drawn."""

    k: float
    alpha: float
    length_km: float = 1.0
    rain: str = "uniform"          # uniform on [0.1, 30] or gamma(0.8, 4) clipped

    @classmethod
    def itu(cls, freq_ghz: float, pol: str = "horizontal", **kw) -> "Link":
        k, alpha = get_k_alpha(freq_ghz, pol)
        return cls(k, alpha, **kw)


def make_split(link: Link, noise_db: float, n: int, rng: np.random.Generator):
    """Attenuation (n, 1) and rain (n,), split 75/25; attenuation kept > 0."""
    if link.rain == "gamma":
        rain = np.clip(rng.gamma(0.8, 4.0, n), 0.1, 80.0)
    else:
        rain = rng.uniform(0.1, 30.0, n)
    att = link.k * rain ** link.alpha * link.length_km + rng.normal(0, noise_db, n)
    att = np.clip(att, 1e-4, None).reshape(-1, 1)
    return train_test_split(att, rain, test_size=0.25, random_state=0)


def physics_inverse(att: np.ndarray, link: Link) -> np.ndarray:
    """R = (A / (k L))^(1/alpha)."""
    return np.clip(np.ravel(att) / (link.k * link.length_km), 1e-8, None) ** (1 / link.alpha)


def _mlp(hidden=(32, 16), max_iter=800):
    return MLPRegressor(hidden, max_iter=max_iter, random_state=0)


def scalar_weight(r_phys, r_mlp, y) -> float:
    """Grid-searched w in [0, 1] minimizing MSE of the blend."""
    grid = np.linspace(0, 1, 51)
    return float(grid[np.argmin([mean_squared_error(y, w * r_phys + (1 - w) * r_mlp)
                                 for w in grid])])


def gate_weight(att, r_phys, r_mlp, y, eps: float = 1e-6):
    """w(A) = sigmoid(w0 + w1 A), fitted to the per-sample ideal weights."""
    att = np.ravel(att)
    diff = r_phys - r_mlp
    ok = np.abs(diff) > 1e-6
    ideal = np.clip((y - r_mlp)[ok] / diff[ok], eps, 1 - eps)
    w0, w1 = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), att[ok]]),
                             np.log(ideal / (1 - ideal)), rcond=None)[0]
    return lambda a: 1 / (1 + np.exp(-(w0 + w1 * np.ravel(a))))


def run_once(link: Link, noise_db: float, n: int = 100,
             rng: np.random.Generator | None = None) -> dict:
    """Test MSE of every strategy, plus the fitted weights, for one draw."""
    rng = rng or np.random.default_rng()
    x_tr, x_te, y_tr, y_te = make_split(link, noise_db, n, rng)
    rp_tr, rp_te = physics_inverse(x_tr, link), physics_inverse(x_te, link)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp = _mlp().fit(x_tr, y_tr)
        rn_tr, rn_te = mlp.predict(x_tr), mlp.predict(x_te)
        fc = MLPRegressor((8,), max_iter=400, random_state=0).fit(
            np.c_[rp_tr, rn_tr], y_tr)
        res = _mlp().fit(x_tr, y_tr - rp_tr)

    w = scalar_weight(rp_tr, rn_tr, y_tr)
    g = gate_weight(x_tr, rp_tr, rn_tr, y_tr)
    estimates = {"physics": rp_te, "mlp": rn_te,
                 "scalar": w * rp_te + (1 - w) * rn_te,
                 "gate": g(x_te) * rp_te + (1 - g(x_te)) * rn_te,
                 "fc": fc.predict(np.c_[rp_te, rn_te]),
                 "residual": rp_te + res.predict(x_te)}
    out = {s: mean_squared_error(y_te, e) for s, e in estimates.items()}
    out.update(scalar_weight=w, gate_weight_mean=float(np.mean(g(x_te))))
    return out


def sweep(frequencies=(5, 23, 60, 70), noise_db=np.linspace(0.001, 1.0, 20),
          repeats: int = 5, n: int = 100, pol: str = "horizontal",
          length_km: float = 1.0, rain: str = "uniform", seed: int = 123) -> pd.DataFrame:
    """Every strategy over frequency x noise x repeat; one row per draw."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in frequencies:
        link = Link.itu(f, pol, length_km=length_km, rain=rain)
        for sigma in noise_db:
            for r in range(repeats):
                rows.append({"freq_ghz": f, "noise_db": float(sigma), "repeat": r,
                             **run_once(link, float(sigma), n, rng)})
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean test MSE per strategy and frequency, and how often each wins."""
    mean = table.groupby("freq_ghz")[list(STRATEGIES)].mean()
    wins = (table[list(STRATEGIES)].idxmin(axis=1).groupby(table.freq_ghz)
            .value_counts(normalize=True).unstack(fill_value=0.0))
    return pd.concat({"mean MSE": mean, "share of draws won": wins}, axis=1)
