"""
Gaussian random fields in 1-D and 2-D, and the transforms that turn them into rain.

Nearly every stochastic rainfall simulator in the literature starts from a
correlated Gaussian field and bends it into rain with a non-linear transform:
the meta-Gaussian (or "trans-Gaussian") construction of turning-bands
simulators, STREAP (Paschalis et al. 2013), SAMPO (Leblois & Creutin 2013) and
the STEPS noise generator (Bowler et al. 2006) all do. This module provides
both halves, separately, so the generators in ``generators`` can mix them:

``covariance spectra``
    ``powerlaw`` (``P(k) ~ k^-beta``, scale-free, the pysteps/STEPS default),
    ``exponential``, ``gaussian`` and ``matern`` (with smoothness ``nu``;
    ``nu = 0.5`` is exponential, ``nu -> inf`` Gaussian), each with an
    anisotropy ratio and direction.

``grf`` / ``grf_1d``
    A zero-mean, unit-variance field with that spectrum, by FFT (periodic).

``to_rain``
    The meta-Gaussian transform: the top ``war`` share of the field is wet,
    and the wet values follow a chosen distribution (gamma, lognormal,
    exponential, Weibull) with a given mean and coefficient of variation; the
    wet fraction and the wet-value distribution are therefore exact by
    construction, whatever the covariance.

``truncated_power``
    The older Allcroft & Glasbey (2003) transform ``R = a (g - t)^b`` that
    ``rain_fields`` already uses.

Spatial lengths are in km and frequencies in cycles per km.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

COVARIANCES = ("powerlaw", "exponential", "gaussian", "matern")
WET_DISTRIBUTIONS = ("gamma", "lognormal", "exponential", "weibull")


# --------------------------------------------------------------------------
# spectra
# --------------------------------------------------------------------------
def spectral_density(k: np.ndarray, cov: str = "matern", length_km: float = 5.0,
                     nu: float = 1.0, beta: float = 3.0, dim: int = 2) -> np.ndarray:
    """Power spectral density (unnormalised) at radial frequency ``k`` (cycles/km).

    ``length_km`` is the correlation length of the covariance (``C(r) = exp(-r/L)``
    for ``exponential``, ``exp(-(r/L)^2)`` for ``gaussian``, the Matern scale for
    ``matern``); ``powerlaw`` ignores it and uses ``beta``.
    """
    if cov == "powerlaw":
        with np.errstate(divide="ignore"):
            s = np.where(k > 0, k, np.inf) ** (-beta)
        return s
    if cov == "exponential":
        cov, nu = "matern", 0.5
    if cov == "gaussian":
        return np.exp(-(np.pi * k * length_km) ** 2)
    if cov == "matern":
        # Matern spectral density in d dimensions with scale L:
        # S(k) ~ (2 nu / L^2 + 4 pi^2 k^2)^-(nu + d/2)
        return (2.0 * nu / length_km ** 2 + 4.0 * np.pi ** 2 * k ** 2) ** (-(nu + dim / 2.0))
    raise ValueError(f"cov must be one of {COVARIANCES}, got {cov!r}")


def _wavenumbers(n: int, dx_km: float, anisotropy: float, angle_deg: float):
    """Radial frequency on an ``n x n`` FFT grid, stretched for anisotropy.

    ``anisotropy`` > 1 makes structures longer along ``angle_deg`` (degrees
    counter-clockwise from the x axis) by that factor.
    """
    kx = np.fft.fftfreq(n, d=dx_km)
    kxx, kyy = np.meshgrid(kx, kx, indexing="xy")
    if anisotropy == 1.0:
        return np.hypot(kxx, kyy)
    a = np.deg2rad(angle_deg)
    k_par = kxx * np.cos(a) + kyy * np.sin(a)
    k_perp = -kxx * np.sin(a) + kyy * np.cos(a)
    return np.hypot(k_par * anisotropy, k_perp)


def grf(n: int, dx_km: float, rng: np.random.Generator, cov: str = "matern",
        length_km: float = 5.0, nu: float = 1.0, beta: float = 3.0,
        anisotropy: float = 1.0, angle_deg: float = 0.0) -> np.ndarray:
    """Zero-mean unit-variance periodic Gaussian random field, ``(n, n)``.

    Complex white noise filtered by ``sqrt(S(k))``: unlike a random-phase field
    with a fixed amplitude it is Gaussian at every scale, which matters when the
    field is later cut into scale bands (``spacetime`` cascade evolution).
    """
    k = _wavenumbers(n, dx_km, anisotropy, angle_deg)
    amp = np.sqrt(spectral_density(k, cov, length_km, nu, beta, dim=2))
    amp[0, 0] = 0.0
    noise = np.fft.fft2(rng.standard_normal((n, n)))
    f = np.fft.ifft2(noise * amp).real
    return _standardise(f)


def grf_1d(n: int, dx: float, rng: np.random.Generator, cov: str = "matern",
           length: float = 5.0, nu: float = 1.0, beta: float = 2.0) -> np.ndarray:
    """1-D version of :func:`grf` (``dx`` and ``length`` in any common unit)."""
    k = np.abs(np.fft.fftfreq(n, d=dx))
    amp = np.sqrt(spectral_density(k, cov, length, nu, beta, dim=1))
    amp[0] = 0.0
    f = np.fft.ifft(np.fft.fft(rng.standard_normal(n)) * amp).real
    return _standardise(f)


def _standardise(f: np.ndarray) -> np.ndarray:
    s = f.std()
    return (f - f.mean()) / s if s > 0 else f - f.mean()


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def wet_distribution(dist: str, mean: float, cv: float):
    """A frozen ``scipy.stats`` distribution of wet rain rates with ``mean`` and ``cv``."""
    if dist == "gamma":
        shape = 1.0 / cv ** 2
        return stats.gamma(a=shape, scale=mean / shape)
    if dist == "lognormal":
        s2 = np.log1p(cv ** 2)
        return stats.lognorm(s=np.sqrt(s2), scale=mean * np.exp(-s2 / 2))
    if dist == "exponential":
        return stats.expon(scale=mean)
    if dist == "weibull":
        from scipy.optimize import brentq
        from scipy.special import gamma as G
        # cv of a Weibull depends only on its shape c
        c = brentq(lambda c: np.sqrt(G(1 + 2 / c) / G(1 + 1 / c) ** 2 - 1) - cv, 0.1, 50.0)
        return stats.weibull_min(c=c, scale=mean / G(1 + 1 / c))
    raise ValueError(f"dist must be one of {WET_DISTRIBUTIONS}, got {dist!r}")


def to_rain(g: np.ndarray, war: float = 0.5, dist: str = "lognormal", mean_wet: float = 3.0,
            cv_wet: float = 1.2, exact: bool = True, threshold_mm_h: float = 0.1) -> np.ndarray:
    """Meta-Gaussian anamorphosis of a standard Gaussian field into rain (mm/h).

    The ``war`` highest values are wet; within them, the quantile of each value
    is mapped onto ``dist`` with the given mean and CV. With ``exact`` the
    quantiles come from ranks, so the realised wet fraction is exactly ``war``
    whatever the field's own histogram; otherwise from ``Phi(g)``, which keeps
    the transform pointwise (a fixed function of ``g``) - needed when the same
    transform must be applied to every frame of an evolving latent field.
    """
    if war <= 0:
        return np.zeros_like(g, dtype=float)
    if exact:
        u = (stats.rankdata(g.ravel()) - 0.5) / g.size
        u = u.reshape(g.shape)
    else:
        u = stats.norm.cdf(g)
    t = 1.0 - min(war, 1.0)
    wet = u > t
    q = np.where(wet, (u - t) / max(1.0 - t, 1e-12), 0.0)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    rain = np.where(wet, _fast_ppf(wet_distribution(dist, mean_wet, cv_wet), q), 0.0)
    # values that come out below the detection threshold are dry, as observed
    rain[rain < threshold_mm_h] = 0.0
    return rain


_Q_NODES = np.unique(np.concatenate([np.logspace(-6, -2, 200), np.linspace(0.01, 0.99, 2000),
                                      1 - np.logspace(-2, -6, 200)]))


def _fast_ppf(frozen, q: np.ndarray) -> np.ndarray:
    """``frozen.ppf(q)`` by interpolation in a table of 2,400 quantiles (scipy's gamma ppf is
    slow on a large grid); exact at the nodes, and within 0.1% between them."""
    return np.interp(q, _Q_NODES, frozen.ppf(_Q_NODES))


def truncated_power(g: np.ndarray, threshold: float, exponent: float, scale: float) -> np.ndarray:
    """Allcroft & Glasbey (2003): ``R = scale * (g - threshold)^exponent`` where ``g > threshold``."""
    return scale * np.clip(g - threshold, 0.0, None) ** exponent


def normal_scores(x: np.ndarray, tiebreak: np.ndarray | None = None) -> np.ndarray:
    """Values -> standard normal scores by rank.

    Dry pixels all tie at zero. Breaking the ties by pixel order would stripe
    the scores row by row, so they are broken by ``tiebreak`` - pass a smooth
    random field so the dry area gets a smooth latent field too (otherwise by
    pixel order).
    """
    tiebreak = np.arange(x.size) if tiebreak is None else np.asarray(tiebreak).ravel()
    ranks = np.lexsort((tiebreak, x.ravel())).argsort()
    return stats.norm.ppf((ranks + 0.5) / x.size).reshape(x.shape)
