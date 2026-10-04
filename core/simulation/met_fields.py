"""
Meteorological fields other than rain, on the same grid and interface.

The sensors FieldSense works with see more than rain: links respond to
humidity (water-vapour absorption) and temperature, weather stations report
temperature, humidity, pressure and wind. Each class here has ``build(grid)``
like the rain generators, so ``spacetime.simulate`` can move it and a point
sensor in ``sensors`` can sample it.

=================  ==========================================================
``Temperature``    mean + large-scale gradient + Matern GRF anomaly (deg C);
                   ``rain`` cools it where it rains (cold pool)
``Humidity``       relative humidity in (0, 100) %: a logistic transform of a
                   GRF, so it never leaves its bounds; moister where it rains
``Pressure``       smooth (Gaussian-covariance) field around a mean, plus a
                   gradient in the direction of a geostrophic wind (hPa)
``Wind``           mean wind + divergence-free random part (u, v in m/s), from
                   a GRF stream function - also a ``flows.RandomFlow``
``CloudCover``     cloud fraction (0-1), a smoothed threshold of a GRF that
                   contains every raining pixel
=================  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Tuple

import numpy as np

from core.simulation import random_fields as rf
from core.simulation.rain_fields import Grid


@dataclass
class Temperature:
    name: str = "Temperature"
    key: str = "temperature"
    units: str = "degC"
    mean_c: float = 20.0
    gradient_c_per_100km: Tuple[float, float] = (0.0, -1.5)
    anomaly_sd_c: float = 1.0
    length_km: float = 20.0
    cold_pool_c_per_mm_h: float = 0.25
    cold_pool_max_c: float = 6.0
    rain: Optional[np.ndarray] = dc_field(default=None, repr=False)
    advection_kmh: Tuple[float, float] = (20.0, 5.0)
    periodic: bool = True
    seed: int = 0

    def build(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        xx, yy = grid.meshgrid()
        c = grid.size_km / 2
        t = self.mean_c + (self.gradient_c_per_100km[0] * (xx - c) + self.gradient_c_per_100km[1] * (yy - c)) / 100
        t = t + self.anomaly_sd_c * rf.grf(grid.n, grid.dx_km, rng, "matern", self.length_km, nu=1.5)
        if self.rain is not None:
            t = t - np.minimum(self.cold_pool_c_per_mm_h * self.rain, self.cold_pool_max_c)
        return t


@dataclass
class Humidity:
    name: str = "Relative humidity"
    key: str = "humidity"
    units: str = "%"
    mean_pct: float = 70.0
    spread: float = 0.8           # sd of the logit
    length_km: float = 15.0
    rain_moistening: float = 1.5  # logit units added where it rains
    rain: Optional[np.ndarray] = dc_field(default=None, repr=False)
    advection_kmh: Tuple[float, float] = (20.0, 5.0)
    periodic: bool = True
    seed: int = 0

    def build(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        m = np.log(self.mean_pct / (100 - self.mean_pct))
        z = m + self.spread * rf.grf(grid.n, grid.dx_km, rng, "matern", self.length_km, nu=1.5)
        if self.rain is not None:
            from scipy.ndimage import gaussian_filter
            wet = gaussian_filter((self.rain > 0.1).astype(float), 1.5 / grid.dx_km, mode="wrap")
            z = z + self.rain_moistening * wet
        return 100.0 / (1.0 + np.exp(-z))


@dataclass
class Pressure:
    name: str = "Surface pressure"
    key: str = "pressure"
    units: str = "hPa"
    mean_hpa: float = 1010.0
    gradient_hpa_per_100km: Tuple[float, float] = (0.0, -2.0)
    anomaly_sd_hpa: float = 0.5
    length_km: float = 60.0
    advection_kmh: Tuple[float, float] = (20.0, 5.0)
    periodic: bool = True
    seed: int = 0

    def build(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        xx, yy = grid.meshgrid()
        c = grid.size_km / 2
        p = self.mean_hpa + (self.gradient_hpa_per_100km[0] * (xx - c)
                             + self.gradient_hpa_per_100km[1] * (yy - c)) / 100
        return p + self.anomaly_sd_hpa * rf.grf(grid.n, grid.dx_km, rng, "gaussian", self.length_km)


@dataclass
class Wind:
    """Wind vector ``(2, n, n)`` in m/s; ``build`` returns the speed, ``vector`` both components."""

    name: str = "Wind speed"
    key: str = "wind"
    units: str = "m/s"
    mean_ms: Tuple[float, float] = (6.0, 2.0)
    rms_ms: float = 2.5
    length_km: float = 15.0
    advection_kmh: Tuple[float, float] = (20.0, 5.0)
    periodic: bool = True
    seed: int = 0

    def vector(self, grid: Grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        psi = rf.grf(grid.n, grid.dx_km, rng, "gaussian", self.length_km)
        dpsi_dy, dpsi_dx = np.gradient(psi, grid.dx_km)
        u, v = -dpsi_dy, dpsi_dx
        s = np.sqrt(np.mean(u ** 2 + v ** 2))
        return np.stack([self.mean_ms[0] + u / s * self.rms_ms, self.mean_ms[1] + v / s * self.rms_ms])

    def build(self, grid: Grid) -> np.ndarray:
        return np.hypot(*self.vector(grid))


@dataclass
class CloudCover:
    name: str = "Cloud cover"
    key: str = "cloud"
    units: str = "fraction"
    cover: float = 0.6
    length_km: float = 10.0
    rain: Optional[np.ndarray] = dc_field(default=None, repr=False)
    advection_kmh: Tuple[float, float] = (20.0, 5.0)
    periodic: bool = True
    seed: int = 0

    def build(self, grid: Grid) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        from scipy.stats import norm
        rng = np.random.default_rng(self.seed)
        g = rf.grf(grid.n, grid.dx_km, rng, "matern", self.length_km, nu=1.5)
        c = norm.cdf((g - norm.ppf(1 - self.cover)) / 0.3)        # soft edges
        if self.rain is not None:
            c = np.maximum(c, gaussian_filter((self.rain > 0.1).astype(float), 2.0, mode="wrap") * 2)
        return np.clip(c, 0, 1)


MET_FIELDS = {"temperature": Temperature, "humidity": Humidity, "pressure": Pressure,
              "wind": Wind, "cloud": CloudCover}
