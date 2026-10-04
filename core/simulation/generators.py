"""
2-D rainfall field generators: one interface, many models.

Every generator is a dataclass with a ``seed`` and ``build(grid) -> (n, n)``
rain rate in mm/h, so any of them can be moved in time by
``core.simulation.spacetime.simulate`` (or ``moving_fields.sequence``), sampled
by the simulated sensors in ``sensors``, and swapped in a benchmark by name:

    from core.simulation import generators as gen
    model = gen.make("gaussian_cells", n_cells=25, profile="hycell", seed=3)
    rain = model.build(Grid(n=128, dx_km=0.5))

=====================  ===============================================================
key                    model
=====================  ===============================================================
``gaussian_cells``     superposition of elliptical rain cells ("multi-Gaussian"):
                       Gaussian, exponential or HyCell profiles (Feral et al. 2003);
                       cells uniform or clustered in storms (Neyman-Scott, Cox &
                       Isham 1988; Northrop 1998); optional stratiform background
``metagaussian``       meta-Gaussian field: Matern / exponential / Gaussian / power-law
                       covariance, anisotropy, exact wet fraction, gamma / lognormal /
                       exponential / Weibull wet intensities (turning-bands family;
                       the spatial model of STREAP, Paschalis et al. 2013)
``rainfarm``           RainFARM (Rebora et al. 2006): exponential of a power-law GRF,
                       optionally constrained to a coarse field (stochastic downscaling)
``cascade``            discrete beta-lognormal multiplicative cascade (Over & Gupta
                       1996): intermittency ``beta``, log-variance ``sigma2``
``multifractal``       universal multifractal FIF (Schertzer & Lovejoy 1987): ``alpha``,
                       ``C1``, ``H``, thresholded to a wet fraction
``stratiform``         ``rain_fields.StratiformField``
``convective``         ``rain_fields.ConvectiveField``
``frontal``            ``rain_fields.FrontalBandField``
=====================  ===============================================================

Non-rain fields (temperature, humidity, pressure, wind) are in ``met_fields``,
and a physically based warm-rain cloud model that evolves in time in
``cloud_model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Tuple

import numpy as np

from core.simulation import random_fields as rf
from core.simulation.rain_fields import (WET_THRESHOLD_MM_H, ConvectiveField, FrontalBandField, Grid,
                                         StratiformField)


def _periodic_delta(a, b, period):
    d = a - b
    return d - period * np.round(d / period)


# --------------------------------------------------------------------------
# multi-Gaussian / HyCell cells
# --------------------------------------------------------------------------
PROFILES = ("gaussian", "exponential", "hycell")


@dataclass
class GaussianCells:
    """Rain as a sum (or max) of elliptical cells.

    Cells are drawn either uniformly (``n_storms = 0``) or as a Neyman-Scott
    cluster process: ``n_storms`` storm centres, each with ``Poisson(n_cells /
    n_storms)`` cells scattered around it with a Gaussian spread of
    ``storm_radius_km``. Peaks are gamma distributed (``peak_mean_mm_h``,
    ``peak_cv``), major axes gamma distributed around ``radius_km``, aspect
    ratios uniform in ``aspect``, orientations ``orientation_deg`` +/-
    ``orientation_spread_deg``.

    ``profile``
        ``gaussian`` ``P exp(-r^2/2)``; ``exponential`` ``P exp(-r)``;
        ``hycell`` the HyCell shape: a Gaussian convective core plus an
        exponential stratiform skirt of ``skirt_frac * P`` that decays over
        ``skirt_scale`` core radii, whichever is larger.
    ``combine``
        ``sum`` (overlapping cells add up) or ``max``.
    ``background_war``
        wet fraction of a weak meta-Gaussian background (0 for none).
    ``texture_strength``
        a multiplicative log-normal texture with a power-law spectrum (0 for none).
    """

    name: str = "Gaussian cells"
    key: str = "gaussian_cells"
    n_cells: float = 20
    poisson: bool = True
    n_storms: int = 0
    storm_radius_km: float = 6.0
    peak_mean_mm_h: float = 25.0
    peak_cv: float = 0.7
    peak_max_mm_h: float = 150.0
    radius_km: float = 2.0
    radius_cv: float = 0.5
    aspect: Tuple[float, float] = (0.5, 1.0)
    orientation_deg: float = 0.0
    orientation_spread_deg: float = 180.0
    profile: str = "gaussian"
    skirt_frac: float = 0.12
    skirt_scale: float = 2.5
    edge_mm_h: float = 1.0
    cutoff_mm_h: float = WET_THRESHOLD_MM_H
    combine: str = "sum"
    background_war: float = 0.0
    background_mean_mm_h: float = 0.8
    background_length_km: float = 8.0
    texture_strength: float = 0.0     # sd of a multiplicative log-normal small-scale texture
    texture_beta: float = 2.4         # its spectral slope (power law: structure at every scale)
    advection_kmh: Tuple[float, float] = (20.0, 5.0)
    periodic: bool = True
    seed: int = 0

    # ---------------------------------------------------------------- cells
    def sample_cells(self, grid: Grid, rng: np.random.Generator, n: Optional[float] = None,
                     domain_km: Optional[Tuple[float, float, float, float]] = None) -> dict:
        """Draw cell parameters as a dict of arrays (positions in km)."""
        x0, x1, y0, y1 = domain_km or (0.0, grid.size_km, 0.0, grid.size_km)
        mean_n = self.n_cells if n is None else n
        count = int(rng.poisson(mean_n)) if self.poisson else int(round(mean_n))
        if self.n_storms > 0:
            sx = rng.uniform(x0, x1, self.n_storms)
            sy = rng.uniform(y0, y1, self.n_storms)
            owner = rng.integers(0, self.n_storms, count)
            x = sx[owner] + rng.normal(0, self.storm_radius_km, count)
            y = sy[owner] + rng.normal(0, self.storm_radius_km, count)
            if self.periodic and domain_km is None:
                x, y = np.mod(x, grid.size_km), np.mod(y, grid.size_km)
        else:
            x, y = rng.uniform(x0, x1, count), rng.uniform(y0, y1, count)
        shape_p = 1.0 / self.peak_cv ** 2
        peak = np.minimum(rng.gamma(shape_p, self.peak_mean_mm_h / shape_p, count), self.peak_max_mm_h)
        shape_r = 1.0 / self.radius_cv ** 2
        major = np.maximum(rng.gamma(shape_r, self.radius_km / shape_r, count), 0.5 * grid.dx_km)
        minor = major * rng.uniform(*self.aspect, count)
        theta = np.deg2rad(self.orientation_deg
                           + rng.uniform(-0.5, 0.5, count) * self.orientation_spread_deg)
        return dict(x=x, y=y, peak=peak, major=major, minor=minor, theta=theta)

    def profile_fn(self, r: np.ndarray) -> np.ndarray:
        """Normalised shape (1 at the centre) at normalised radius ``r``."""
        if self.profile == "gaussian":
            return np.exp(-0.5 * r ** 2)
        if self.profile == "exponential":
            return np.exp(-r)
        if self.profile == "hycell":
            return np.maximum(np.exp(-0.5 * r ** 2), self.skirt_frac * np.exp(-r / self.skirt_scale))
        raise ValueError(f"profile must be one of {PROFILES}")

    def render(self, cells: dict, grid: Grid, origin_km: Tuple[float, float] = (0.0, 0.0),
               scale: Optional[np.ndarray] = None) -> np.ndarray:
        """Rain field of ``cells`` on ``grid`` whose lower-left corner is at ``origin_km``.

        ``scale`` multiplies each cell's peak (the lifecycle envelope).
        """
        xx, yy = grid.meshgrid()
        xx, yy = xx + origin_km[0], yy + origin_km[1]
        out = np.zeros(grid.shape)
        peaks = cells["peak"] * (1.0 if scale is None else scale)
        for i in np.flatnonzero(peaks > self.cutoff_mm_h):
            # only the window the cell can reach: 6 major radii covers every profile
            reach = cells["major"][i] * (6.0 if self.profile == "gaussian" else 12.0)
            if self.periodic and origin_km == (0.0, 0.0):
                dx = _periodic_delta(xx, cells["x"][i], grid.size_km)
                dy = _periodic_delta(yy, cells["y"][i], grid.size_km)
            else:
                dx, dy = xx - cells["x"][i], yy - cells["y"][i]
            near = (np.abs(dx) < reach) & (np.abs(dy) < reach)
            if not near.any():
                continue
            c, s = np.cos(cells["theta"][i]), np.sin(cells["theta"][i])
            u = dx[near] * c + dy[near] * s
            v = -dx[near] * s + dy[near] * c
            r = np.hypot(u / cells["major"][i], v / cells["minor"][i])
            val = peaks[i] * self.profile_fn(r)
            if self.profile != "gaussian":
                val[val < self.edge_mm_h] = 0.0
            if self.combine == "max":
                out[near] = np.maximum(out[near], val)
            else:
                out[near] += val
        out[out < self.cutoff_mm_h] = 0.0
        return out

    def background(self, grid: Grid, rng: np.random.Generator) -> np.ndarray:
        if self.background_war <= 0:
            return np.zeros(grid.shape)
        g = rf.grf(grid.n, grid.dx_km, rng, cov="matern", length_km=self.background_length_km, nu=1.5)
        return rf.to_rain(g, self.background_war, "gamma", self.background_mean_mm_h, 0.8)

    def texture(self, grid: Grid, rng: np.random.Generator) -> np.ndarray:
        """Mean-one multiplicative texture (1 without ``texture_strength``): rain cells are
        ragged down to the street scale, not smooth ellipses."""
        if self.texture_strength <= 0:
            return np.ones(grid.shape)
        g = rf.grf(grid.n, grid.dx_km, rng, cov="powerlaw", beta=self.texture_beta)
        s = self.texture_strength
        return np.exp(s * g - s ** 2 / 2)

    def build(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        self.cells = self.sample_cells(grid, rng)
        rain = (self.render(self.cells, grid) + self.background(grid, rng)) * self.texture(grid, rng)
        rain[rain < WET_THRESHOLD_MM_H] = 0.0
        return rain


# --------------------------------------------------------------------------
# meta-Gaussian
# --------------------------------------------------------------------------
@dataclass
class MetaGaussian:
    """Correlated Gaussian field transformed to rain with an exact wet fraction.

    The covariance (``cov``, ``length_km``, ``nu``, ``beta``, ``anisotropy``,
    ``angle_deg``) sets the spatial structure; ``war``, ``dist``,
    ``mean_wet_mm_h`` and ``cv_wet`` the marginal distribution. See
    ``random_fields`` for the choices.
    """

    name: str = "Meta-Gaussian"
    key: str = "metagaussian"
    cov: str = "matern"
    length_km: float = 6.0
    nu: float = 1.0
    beta: float = 3.0
    anisotropy: float = 1.0
    angle_deg: float = 0.0
    war: float = 0.5
    dist: str = "lognormal"
    mean_wet_mm_h: float = 3.0
    cv_wet: float = 1.2
    advection_kmh: Tuple[float, float] = (25.0, 10.0)
    periodic: bool = True
    seed: int = 0

    def latent(self, grid: Grid, rng: np.random.Generator) -> np.ndarray:
        return rf.grf(grid.n, grid.dx_km, rng, self.cov, self.length_km, self.nu, self.beta,
                      self.anisotropy, self.angle_deg)

    def transform(self, g: np.ndarray, exact: bool = True) -> np.ndarray:
        return rf.to_rain(g, self.war, self.dist, self.mean_wet_mm_h, self.cv_wet, exact=exact)

    def build(self, grid: Grid) -> np.ndarray:
        return self.transform(self.latent(grid, np.random.default_rng(self.seed)))


# --------------------------------------------------------------------------
# RainFARM
# --------------------------------------------------------------------------
@dataclass
class RainFARM:
    """RainFARM: ``R ~ exp(sigma * g)`` with ``g`` a power-law GRF (Rebora et al. 2006).

    ``slope`` is the spectral slope of the *2-D* power spectrum of ``g``.
    The field is scaled to ``mean_mm_h``; ``war`` < 1 zeroes the lowest
    values to make dry areas (the original method has none). With
    ``coarse`` set, every ``factor x factor`` block is rescaled to the coarse
    field's value - the stochastic downscaling RainFARM was designed for.
    """

    name: str = "RainFARM"
    key: str = "rainfarm"
    slope: float = 3.4
    sigma: float = 1.0
    mean_mm_h: float = 2.0
    war: float = 0.6
    coarse: Optional[np.ndarray] = dc_field(default=None, repr=False)
    advection_kmh: Tuple[float, float] = (25.0, -5.0)
    periodic: bool = True
    seed: int = 0

    def build(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        g = rf.grf(grid.n, grid.dx_km, rng, cov="powerlaw", beta=self.slope)
        r = np.exp(self.sigma * g)
        if self.war < 1:
            r[r < np.quantile(r, 1 - self.war)] = 0.0
        if self.coarse is not None:
            return downscale_to(r, self.coarse)
        r *= self.mean_mm_h / r.mean()
        r[r < WET_THRESHOLD_MM_H] = 0.0
        return r


def downscale_to(fine: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """Rescale each block of ``fine`` so its mean equals the matching ``coarse`` cell."""
    f = fine.shape[0] // coarse.shape[0]
    blocks = fine.reshape(coarse.shape[0], f, coarse.shape[1], f).mean(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(blocks > 0, coarse / blocks, 0.0)
    out = fine * np.kron(ratio, np.ones((f, f)))
    out[out < WET_THRESHOLD_MM_H] = 0.0
    return out


# --------------------------------------------------------------------------
# multiplicative cascades
# --------------------------------------------------------------------------
@dataclass
class BetaLognormalCascade:
    """Discrete beta-lognormal random cascade (Over & Gupta 1996).

    At each of ``log2(n)`` levels every cell splits into four with weights
    ``W = B * Y``: ``B`` is 0 with probability ``1 - 2^-beta`` (else
    ``2^beta``) and makes the field intermittent; ``Y = exp(sigma Z - sigma^2/2)``
    is a mean-one lognormal that makes it heavy tailed. ``E[W] = 1``, so the
    field mean is ``mean_mm_h`` in expectation. ``smooth_cells`` blurs the
    block edges the cascade leaves.
    """

    name: str = "Beta-lognormal cascade"
    key: str = "cascade"
    beta: float = 0.15
    sigma2: float = 0.25
    mean_mm_h: float = 2.0
    smooth_cells: float = 1.0
    advection_kmh: Tuple[float, float] = (20.0, 10.0)
    periodic: bool = True
    seed: int = 0

    def build(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        levels = int(np.ceil(np.log2(grid.n)))
        f = np.full((1, 1), self.mean_mm_h)
        p_wet = 2.0 ** (-self.beta)
        sigma = np.sqrt(self.sigma2)
        for _ in range(levels):
            f = np.kron(f, np.ones((2, 2)))
            b = np.where(rng.random(f.shape) < p_wet, 1.0 / p_wet, 0.0)
            y = np.exp(sigma * rng.standard_normal(f.shape) - self.sigma2 / 2)
            f = f * b * y
        f = f[:grid.n, :grid.n]
        if self.smooth_cells > 0:
            from scipy.ndimage import gaussian_filter
            f = gaussian_filter(f, self.smooth_cells, mode="wrap")
        f[f < WET_THRESHOLD_MM_H] = 0.0
        return f


@dataclass
class UniversalMultifractal:
    """Universal multifractal field by fractionally integrated flux (FIF).

    ``alpha`` (0-2, multifractality index; 2 = lognormal), ``C1`` (mean
    intermittency, codimension of the mean) and ``H`` (degree of fractional
    integration, smoothness) are the three universal parameters (Schertzer &
    Lovejoy 1987; Pecknold et al. 1993). Rain observed by radar typically has
    ``alpha ~ 1.5-1.8``, ``C1 ~ 0.05-0.2``, ``H ~ 0-0.5``. Steps:

    1. extremal Levy-stable noise of index ``alpha`` (maximally skewed to the left);
    2. its fractional integral of order ``d/alpha'`` in Fourier space - the
       singular generator ``Gamma``, scaled by ``(C1/|alpha-1|)^(1/alpha)``;
    3. the conservative flux ``exp(Gamma)``, normalised to mean one;
    4. a fractional integration ``k^-H``.

    The flux has no zeros, so the field is thresholded to ``war``.
    """

    name: str = "Universal multifractal"
    key: str = "multifractal"
    alpha: float = 1.6
    C1: float = 0.1
    H: float = 0.3
    war: float = 0.5
    mean_wet_mm_h: float = 4.0
    advection_kmh: Tuple[float, float] = (20.0, 0.0)
    periodic: bool = True
    seed: int = 0

    def build(self, grid: Grid) -> np.ndarray:
        from scipy.stats import levy_stable
        rng = np.random.default_rng(self.seed)
        n, d = grid.n, 2
        a = self.alpha if abs(self.alpha - 1.0) > 1e-3 else 1.001
        # extremal (beta = -1) stable noise with unit scale; for alpha = 2 a Gaussian of variance 2
        noise = levy_stable.rvs(a, -1.0, size=(n, n), random_state=rng)
        noise = np.clip(noise, -1e3, 1e3)           # the extremes of a finite sample
        k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n)))
        k[0, 0] = 1.0
        # FIF (Schertzer & Lovejoy 1987; Pecknold et al. 1993): the noise is convolved with
        # |x|^(-d/alpha) in real space, i.e. multiplied by |k|^(-d/alpha') in Fourier space,
        # with 1/alpha + 1/alpha' = 1.
        a_conj = a / (a - 1.0)
        kernel = k ** (-d / a_conj)
        kernel[0, 0] = 0.0
        gen = np.fft.ifft2(np.fft.fft2(noise - noise.mean()) * kernel).real
        # A weighted sum of stable variables is stable with scale (sum |w|^alpha)^(1/alpha).
        # Scale the generator so that log <exp(q Gamma)> = C1/(alpha-1) (q^alpha - q) ln(lambda)
        # at the grid's scale ratio lambda = n: the moment scaling function of a UM cascade.
        # For scipy's extremal stable law of scale s, log <exp(q X)> = s^alpha q^alpha /
        # |cos(pi alpha / 2)|, hence the cosine factor.
        w = np.fft.ifft2(kernel).real
        scale = np.sum(np.abs(w) ** a) ** (1 / a)
        cos_term = abs(np.cos(np.pi * a / 2)) if abs(a - 2.0) > 1e-9 else 1.0
        gen = gen / scale * (self.C1 * np.log(n) * cos_term / abs(a - 1)) ** (1 / a)
        flux = np.exp(gen - gen.max())
        flux /= flux.mean()
        if self.H > 0:
            kh = k ** (-self.H)
            kh[0, 0] = 1.0
            flux = np.fft.ifft2(np.fft.fft2(flux) * kh).real
        flux = flux - flux.min()
        cut = np.quantile(flux, 1 - self.war) if self.war < 1 else 0.0
        rain = np.clip(flux - cut, 0, None)
        wet = rain > 0
        if wet.any():
            rain *= self.mean_wet_mm_h / rain[wet].mean()
        rain[rain < WET_THRESHOLD_MM_H] = 0.0
        return rain


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
REGISTRY = {
    "gaussian_cells": GaussianCells,
    "metagaussian": MetaGaussian,
    "rainfarm": RainFARM,
    "cascade": BetaLognormalCascade,
    "multifractal": UniversalMultifractal,
    "stratiform": StratiformField,
    "convective": ConvectiveField,
    "frontal": FrontalBandField,
}

# ready-made regimes, each a (key, params) pair: typical values from the
# references above, tuned to a 64 km domain
PRESETS = {
    "convective_cells": ("gaussian_cells", dict(n_cells=18, radius_km=1.8, peak_mean_mm_h=30,
                                                background_war=0.08)),
    "clustered_storms": ("gaussian_cells", dict(n_cells=30, n_storms=3, storm_radius_km=5,
                                                radius_km=1.5, profile="hycell", peak_mean_mm_h=35)),
    "squall_line": ("gaussian_cells", dict(n_cells=25, n_storms=1, storm_radius_km=3,
                                           radius_km=2.0, profile="hycell", peak_mean_mm_h=40,
                                           aspect=(0.3, 0.6), orientation_deg=60,
                                           orientation_spread_deg=20, combine="max",
                                           background_war=0.15)),
    "stratiform_matern": ("metagaussian", dict(cov="matern", length_km=12, nu=1.5, war=0.85,
                                               dist="gamma", mean_wet_mm_h=2.0, cv_wet=0.8)),
    "banded_anisotropic": ("metagaussian", dict(cov="matern", length_km=6, nu=1.0, anisotropy=4,
                                                angle_deg=35, war=0.45, mean_wet_mm_h=4.0)),
    "scale_free": ("metagaussian", dict(cov="powerlaw", beta=3.2, war=0.5, mean_wet_mm_h=3.0)),
    "rainfarm": ("rainfarm", {}),
    "cascade": ("cascade", {}),
    "multifractal": ("multifractal", {}),
}


def make(key: str, **params):
    """A generator by registry key or preset name, with parameters overridden."""
    if key in PRESETS:
        base, defaults = PRESETS[key]
        return REGISTRY[base](**{**defaults, **params})
    if key not in REGISTRY:
        raise KeyError(f"unknown model {key!r}; registry: {sorted(REGISTRY)}, presets: {sorted(PRESETS)}")
    return REGISTRY[key](**params)
