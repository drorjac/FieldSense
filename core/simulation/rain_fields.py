"""
Three synthetic rainfall field models on a common 2-D grid.

The three cover the regimes that behave differently under sparse line-integral
sampling, which is the point of the CML experiment downstream:

``StratiformField``
    Widespread, low intensity, smooth. High wet-area ratio, long decorrelation
    length. Built as a spectrally-shaped Gaussian random field pushed through a
    threshold-and-power transform.

``ConvectiveField``
    Isolated intense cells over a weak background. Low wet-area ratio, short
    decorrelation length, heavy-tailed intensities. Built as an explicit
    superposition of anisotropic cells.

``FrontalBandField``
    An organized band: narrow intense leading line with a broad trailing
    stratiform region. Intermediate statistics, strongly anisotropic.

All fields are defined on a periodic domain so that advection is an exact
circular shift and introduces no interpolation artifacts.

Intensities are rain rates in mm/h. ``WET_THRESHOLD_MM_H`` is the conventional
0.1 mm/h cut used to separate wet from dry pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Tuple

import numpy as np

WET_THRESHOLD_MM_H = 0.1


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """Regular square grid with periodic wrap-around."""

    n: int = 128
    dx_km: float = 0.25

    @property
    def size_km(self) -> float:
        return self.n * self.dx_km

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.n)

    @property
    def x_km(self) -> np.ndarray:
        return (np.arange(self.n) + 0.5) * self.dx_km

    @property
    def y_km(self) -> np.ndarray:
        return (np.arange(self.n) + 0.5) * self.dx_km

    @property
    def extent_km(self) -> Tuple[float, float, float, float]:
        return (0.0, self.size_km, 0.0, self.size_km)

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x_km, self.y_km, indexing="xy")


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def spectral_grf(grid: Grid, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean unit-variance Gaussian random field with a P(k) ~ k^-beta spectrum.

    Larger ``beta`` concentrates variance at low wavenumbers, giving a smoother
    field. Built by FFT, so the result is exactly periodic.
    """
    kx = np.fft.fftfreq(grid.n, d=grid.dx_km)
    ky = np.fft.fftfreq(grid.n, d=grid.dx_km)
    kxx, kyy = np.meshgrid(kx, ky, indexing="xy")
    k = np.hypot(kxx, kyy)
    k[0, 0] = 1.0

    amplitude = k ** (-beta / 2.0)
    amplitude[0, 0] = 0.0  # drop the DC term; mean is set afterwards

    phase = rng.uniform(0.0, 2.0 * np.pi, size=grid.shape)
    spectrum = amplitude * np.exp(1j * phase)
    f = np.fft.ifft2(spectrum).real

    std = f.std()
    if std == 0.0:
        return f
    return (f - f.mean()) / std


def _transform_to_rain(g: np.ndarray, threshold: float, exponent: float,
                       mean_wet: float) -> np.ndarray:
    """Threshold-and-power transform of a Gaussian field into a rain rate field.

    Clipping at ``threshold`` produces the dry areas; the power ``exponent``
    controls how heavy the intensity tail is; the result is rescaled so the mean
    over wet pixels is ``mean_wet``.
    """
    excess = np.clip(g - threshold, 0.0, None)
    rain = excess**exponent
    wet = rain > 0.0
    if wet.any():
        rain = rain * (mean_wet / rain[wet].mean())
    rain[rain < WET_THRESHOLD_MM_H] = 0.0
    return rain


def _solve_threshold_for_war(g: np.ndarray, target_war: float, exponent: float,
                             mean_wet: float) -> float:
    """Find the Gaussian-field threshold that realizes a target wet-area ratio.

    The transform zeroes anything below 0.1 mm/h after rescaling, so the
    realized wet fraction is not simply the quantile of the threshold. Bisection
    on the threshold is cheap and exact enough.
    """
    lo, hi = float(g.min()) - 1.0, float(g.max())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        war = float((_transform_to_rain(g, mid, exponent, mean_wet) > 0.0).mean())
        if war > target_war:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _periodic_delta(a: np.ndarray, b: float, period: float) -> np.ndarray:
    """Minimum-image separation ``a - b`` on a periodic axis."""
    d = a - b
    return d - period * np.round(d / period)


# --------------------------------------------------------------------------
# the three models
# --------------------------------------------------------------------------
@dataclass
class StratiformField:
    """Widespread low-intensity rain: smooth, high wet fraction."""

    name: str = "Stratiform"
    key: str = "stratiform"
    beta: float = 3.2
    target_war: float = 0.86
    mean_wet_mm_h: float = 2.1
    exponent: float = 1.4
    advection_kmh: Tuple[float, float] = (14.0, 5.0)
    seed: int = 11

    description: str = (
        "Spectral GRF, beta=3.2, threshold-and-power transform"
    )

    def build(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        g = spectral_grf(grid, self.beta, rng)
        t = _solve_threshold_for_war(g, self.target_war, self.exponent,
                                     self.mean_wet_mm_h)
        return _transform_to_rain(g, t, self.exponent, self.mean_wet_mm_h)


@dataclass
class ConvectiveField:
    """Isolated intense cells on a weak stratiform background."""

    name: str = "Convective cells"
    key: str = "convective"
    n_cells: int = 12
    peak_shape: float = 2.5          # gamma shape for cell peak intensity
    peak_scale: float = 12.0         # gamma scale, mm/h
    peak_clip: Tuple[float, float] = (6.0, 90.0)
    radius_shape: float = 2.0        # gamma shape for the major axis, km
    radius_scale: float = 0.50
    radius_floor_km: float = 0.30
    profile_exponent: float = 1.5    # 2.0 = Gaussian; < 2 sharpens the core
    cutoff_radii: float = 1.6        # cell support, in normalized radii
    texture_beta: float = 2.2        # sub-cell structure
    texture_strength: float = 0.24
    background_war: float = 0.05
    background_mean_mm_h: float = 0.7
    background_beta: float = 2.6
    advection_kmh: Tuple[float, float] = (22.0, -8.0)
    seed: int = 23

    description: str = (
        "12 compact anisotropic cells + weak GRF background"
    )

    cells: list = dc_field(default_factory=list, repr=False)

    def build(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        xx, yy = grid.meshgrid()
        period = grid.size_km
        rain = np.zeros(grid.shape)
        self.cells = []

        for _ in range(self.n_cells):
            x0 = rng.uniform(0.0, period)
            y0 = rng.uniform(0.0, period)
            peak = float(np.clip(rng.gamma(self.peak_shape, self.peak_scale),
                                 *self.peak_clip))
            major = self.radius_floor_km + rng.gamma(self.radius_shape,
                                                     self.radius_scale)
            minor = major * rng.uniform(0.45, 1.0)
            theta = rng.uniform(0.0, np.pi)

            dx = _periodic_delta(xx, x0, period)
            dy = _periodic_delta(yy, y0, period)
            u = dx * np.cos(theta) + dy * np.sin(theta)
            v = -dx * np.sin(theta) + dy * np.cos(theta)

            r2 = (u / major) ** 2 + (v / minor) ** 2
            # Compact support: a real cell has an edge. Subtracting the value at
            # the cutoff radius takes the profile smoothly to zero instead of
            # leaving an infinite tail that floods the domain with drizzle.
            shape_fn = np.exp(-0.5 * r2 ** (self.profile_exponent / 2.0))
            pedestal = np.exp(-0.5 * self.cutoff_radii**self.profile_exponent)
            rain += peak * np.clip(shape_fn - pedestal, 0.0, None) / (1.0 - pedestal)
            self.cells.append(dict(x_km=x0, y_km=y0, peak_mm_h=peak,
                                   major_km=major, minor_km=minor,
                                   theta_rad=theta))

        # Real cells are ragged, not smooth ellipses. A multiplicative GRF
        # breaks up the analytic profile without moving the cell centres.
        texture = spectral_grf(grid, self.texture_beta, rng)
        rain = rain * np.clip(1.0 + self.texture_strength * texture, 0.25, 1.45)

        g = spectral_grf(grid, self.background_beta, rng)
        t = _solve_threshold_for_war(g, self.background_war, 1.3,
                                     self.background_mean_mm_h)
        rain = rain + _transform_to_rain(g, t, 1.3, self.background_mean_mm_h)

        rain[rain < WET_THRESHOLD_MM_H] = 0.0
        return rain


@dataclass
class FrontalBandField:
    """Squall-line geometry: narrow convective line, broad trailing stratiform."""

    name: str = "Frontal band"
    key: str = "frontal"
    orientation_deg: float = 35.0    # normal direction of travel
    centre_frac: float = 0.50        # band position along the normal, 0-1
    offset_km: float = 0.0           # and moved this far along the normal
    conv_peak_mm_h: float = 30.0
    conv_width_km: float = 1.4
    strat_peak_mm_h: float = 3.2
    strat_width_km: float = 4.5
    lead_edge_km: float = 0.7        # sharp cut ahead of the line
    along_band_beta: float = 3.0
    texture_beta: float = 2.4
    texture_strength: float = 0.30
    advection_kmh: Tuple[float, float] = (30.0, 21.0)
    seed: int = 37
    # a straight band at an arbitrary angle cannot tile the torus: shifting
    # this field wraps a cut band into view, so movers pad it instead
    periodic: bool = False

    description: str = (
        "Analytic cross-band profile, GRF along-band modulation"
    )

    def build(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        xx, yy = grid.meshgrid()
        theta = np.deg2rad(self.orientation_deg)

        centre = self.centre_frac * grid.size_km
        u = (xx - centre) * np.cos(theta) + (yy - centre) * np.sin(theta) - self.offset_km

        # Cross-band profile. u > 0 is ahead of the line (sharp edge),
        # u < 0 is the trailing stratiform region.
        line = self.conv_peak_mm_h * np.exp(-0.5 * (u / self.conv_width_km) ** 2)
        ahead_cut = np.exp(-0.5 * np.clip(u, 0.0, None) ** 2
                           / self.lead_edge_km**2)
        trail = np.where(u < 0.0,
                         self.strat_peak_mm_h * np.exp(u / self.strat_width_km),
                         0.0)
        profile = line * ahead_cut + trail

        # Along-band inhomogeneity: the line is never uniform in intensity.
        modulation = spectral_grf(grid, self.along_band_beta, rng)
        modulation = 1.0 + 0.55 * modulation
        modulation = np.clip(modulation, 0.25, 1.7)

        texture = spectral_grf(grid, self.texture_beta, rng)
        texture = np.clip(1.0 + self.texture_strength * texture, 0.2, 1.6)

        rain = profile * modulation * texture
        rain[rain < WET_THRESHOLD_MM_H] = 0.0
        return rain


MODELS = (StratiformField(), ConvectiveField(), FrontalBandField())


# --------------------------------------------------------------------------
# advection
# --------------------------------------------------------------------------
def advect(field: np.ndarray, grid: Grid, velocity_kmh: Tuple[float, float],
           minutes: float) -> np.ndarray:
    """Translate a field by ``velocity_kmh`` over ``minutes``.

    Uses a circular shift on the periodic grid, rounded to whole cells, so no
    interpolation smoothing is introduced. Sub-cell residuals are dropped -
    which quantizes the velocity; ``moving_fields.shift`` has the exact
    (spectral) version and ``moving_fields.sequence`` builds whole moving,
    growing and evolving sequences.
    """
    from core.simulation.moving_fields import shift
    return shift(field, grid.dx_km, velocity_kmh, minutes / 60.0, method="integer")


# --------------------------------------------------------------------------
# field statistics - the "proportions"
# --------------------------------------------------------------------------
def field_stats(rain: np.ndarray, grid: Grid) -> dict:
    """Standard descriptors of a rainfall field.

    ``war``
        Wet-area ratio: the proportion of the domain that is raining at or
        above 0.1 mm/h. This is the "proportion" that drives how well a sparse
        sensor network can see the field.
    ``imf``
        Image mean flux: domain-mean rain rate including dry pixels (mm/h).
    ``cmf``
        Conditional mean flux: mean over wet pixels only (mm/h). imf = war * cmf.
    ``decorrelation_km``
        Lag at which the isotropic spatial autocorrelation first falls below
        1/e.
    """
    wet = rain >= WET_THRESHOLD_MM_H
    war = float(wet.mean())
    imf = float(rain.mean())
    cmf = float(rain[wet].mean()) if wet.any() else 0.0
    wet_vals = rain[wet]

    return {
        "war": war,
        "imf": imf,
        "cmf": cmf,
        "max": float(rain.max()),
        "p99_wet": float(np.percentile(wet_vals, 99)) if wet.any() else 0.0,
        "p50_wet": float(np.percentile(wet_vals, 50)) if wet.any() else 0.0,
        "cv_wet": float(wet_vals.std() / wet_vals.mean()) if wet.any() else 0.0,
        "decorrelation_km": decorrelation_length(rain, grid),
        "frac_flux_top5pct": _flux_concentration(rain, 0.05),
    }


def _flux_concentration(rain: np.ndarray, top_frac: float) -> float:
    """Share of total water volume delivered by the wettest ``top_frac`` of pixels."""
    total = rain.sum()
    if total <= 0.0:
        return 0.0
    flat = np.sort(rain.ravel())[::-1]
    n_top = max(1, int(round(top_frac * flat.size)))
    return float(flat[:n_top].sum() / total)


def radial_autocorrelation(rain: np.ndarray, grid: Grid,
                           max_lag_km: float = 12.0):
    """Isotropic spatial autocorrelation via the Wiener-Khinchin theorem.

    Valid without windowing here because the fields are genuinely periodic.
    """
    f = rain - rain.mean()
    power = np.abs(np.fft.fft2(f)) ** 2
    acf = np.fft.ifft2(power).real
    acf = np.fft.fftshift(acf)
    acf /= acf.max()

    n = grid.n
    cy = cx = n // 2
    yy, xx = np.indices((n, n))
    lag_km = np.hypot(xx - cx, yy - cy) * grid.dx_km

    n_bins = int(max_lag_km / grid.dx_km)
    bins = np.linspace(0.0, max_lag_km, n_bins + 1)
    idx = np.digitize(lag_km.ravel(), bins) - 1
    valid = (idx >= 0) & (idx < n_bins)

    sums = np.bincount(idx[valid], weights=acf.ravel()[valid], minlength=n_bins)
    counts = np.bincount(idx[valid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    centres = 0.5 * (bins[:-1] + bins[1:])
    return centres, profile


def decorrelation_length(rain: np.ndarray, grid: Grid) -> float:
    """First lag where the isotropic autocorrelation drops below 1/e."""
    lags, acf = radial_autocorrelation(rain, grid)
    below = np.where(acf < np.exp(-1.0))[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(lags[0])
    # linear interpolation between the bracketing lags
    a0, a1 = acf[i - 1], acf[i]
    l0, l1 = lags[i - 1], lags[i]
    if a0 == a1:
        return float(l1)
    w = (a0 - np.exp(-1.0)) / (a0 - a1)
    return float(l0 + w * (l1 - l0))


def exceedance_curve(rain: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """P(R > level) over the whole domain, dry pixels included."""
    return np.array([float((rain > lv).mean()) for lv in levels])
