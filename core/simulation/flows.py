"""
Velocity fields that carry simulated rain, and the advection that moves it.

Real rain does not move as one rigid block: a mesoscale vortex turns it, a
jet shears it, a front deforms it. Motion estimators (Lucas-Kanade, VET,
DARTS) and every advection nowcast assume a smooth motion field, so the test
needs fields where that motion is known exactly and is not always uniform.

All flows are 2-D, in km/h, ``u`` towards +x (east) and ``v`` towards +y
(north, row index increasing), and divergence-free unless stated:

=================  ==============================================================
``UniformFlow``    constant ``(u, v)``
``RotationFlow``   solid-body rotation at ``omega_deg_h`` about ``centre_km``, plus a mean
``ShearFlow``      ``u = u0 + shear_per_h * (y - yc)``, ``v = v0``
``DeformationFlow``  pure strain ``u = d (x - xc)``, ``v = -d (y - yc)``, plus a mean
``RandomFlow``     mean wind plus a random divergence-free part from a Gaussian stream
                   function (rms ``rms_kmh``, correlation length ``length_km``)
=================  ==============================================================

``centre_km = None`` puts the centre in the middle of the visible domain.
Coordinates are km with the visible domain's lower-left corner at (0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from core.simulation.rain_fields import Grid


@dataclass
class Flow:
    """Base class: ``at(x, y)`` gives ``(u, v)`` km/h at points in km."""

    centre_km: Optional[Tuple[float, float]] = None

    uniform = False

    def prepare(self, domain_km: float, sim_grid: Grid = None, origin_km=(0.0, 0.0)) -> "Flow":
        """Fix the centre (and any random part) for a visible domain of ``domain_km``."""
        if self.centre_km is None:
            self.centre_km = (domain_km / 2.0, domain_km / 2.0)
        return self

    def at(self, x, y):
        raise NotImplementedError

    def field(self, grid: Grid, origin_km=(0.0, 0.0)) -> np.ndarray:
        """``(2, n, n)`` velocity at the cell centres of ``grid`` placed at ``origin_km``."""
        xx, yy = grid.meshgrid()
        u, v = self.at(xx + origin_km[0], yy + origin_km[1])
        return np.stack([np.broadcast_to(u, xx.shape), np.broadcast_to(v, xx.shape)]).astype(float)

    @property
    def mean_kmh(self) -> Tuple[float, float]:
        return getattr(self, "mean", (0.0, 0.0))

    def describe(self) -> str:
        return type(self).__name__


@dataclass
class UniformFlow(Flow):
    mean: Tuple[float, float] = (25.0, 10.0)
    uniform = True

    def at(self, x, y):
        return np.full(np.shape(x), self.mean[0]), np.full(np.shape(x), self.mean[1])

    def describe(self):
        return f"uniform {np.hypot(*self.mean):.0f} km/h"


@dataclass
class RotationFlow(Flow):
    omega_deg_h: float = 60.0
    mean: Tuple[float, float] = (0.0, 0.0)

    def at(self, x, y):
        w = np.deg2rad(self.omega_deg_h)
        xc, yc = self.centre_km
        return self.mean[0] - w * (y - yc), self.mean[1] + w * (x - xc)

    def describe(self):
        return f"rotation {self.omega_deg_h:.0f} deg/h"


@dataclass
class ShearFlow(Flow):
    mean: Tuple[float, float] = (20.0, 5.0)
    shear_per_h: float = 0.8

    def at(self, x, y):
        yc = self.centre_km[1]
        return self.mean[0] + self.shear_per_h * (y - yc), np.full(np.shape(x), self.mean[1])

    def describe(self):
        return f"shear {self.shear_per_h:.1f}/h"


@dataclass
class DeformationFlow(Flow):
    mean: Tuple[float, float] = (15.0, 0.0)
    rate_per_h: float = 0.5

    def at(self, x, y):
        xc, yc = self.centre_km
        return self.mean[0] + self.rate_per_h * (x - xc), self.mean[1] - self.rate_per_h * (y - yc)

    def describe(self):
        return f"deformation {self.rate_per_h:.1f}/h"


@dataclass
class RandomFlow(Flow):
    """Mean wind plus a smooth random divergence-free perturbation."""

    mean: Tuple[float, float] = (20.0, 5.0)
    rms_kmh: float = 10.0
    length_km: float = 30.0
    seed: int = 0

    def prepare(self, domain_km, sim_grid=None, origin_km=(0.0, 0.0)):
        from core.simulation.random_fields import grf
        super().prepare(domain_km)
        if sim_grid is None:
            sim_grid = Grid(n=64, dx_km=domain_km / 64)
        rng = np.random.default_rng(self.seed)
        psi = grf(sim_grid.n, sim_grid.dx_km, rng, cov="gaussian", length_km=self.length_km)
        dpsi_dy, dpsi_dx = np.gradient(psi, sim_grid.dx_km)
        u, v = -dpsi_dy, dpsi_dx
        s = np.sqrt(np.mean(u ** 2 + v ** 2))
        self._u, self._v = u / s * self.rms_kmh, v / s * self.rms_kmh
        self._grid, self._origin = sim_grid, origin_km
        return self

    def at(self, x, y):
        g = self._grid
        col = (np.asarray(x) - self._origin[0]) / g.dx_km - 0.5
        row = (np.asarray(y) - self._origin[1]) / g.dx_km - 0.5
        coords = [np.ravel(row), np.ravel(col)]
        u = map_coordinates(self._u, coords, order=1, mode="grid-wrap").reshape(np.shape(x))
        v = map_coordinates(self._v, coords, order=1, mode="grid-wrap").reshape(np.shape(x))
        return self.mean[0] + u, self.mean[1] + v

    def describe(self):
        return f"random rms {self.rms_kmh:.0f} km/h"


FLOWS = {"uniform": UniformFlow, "rotation": RotationFlow, "shear": ShearFlow,
         "deformation": DeformationFlow, "random": RandomFlow}


def make_flow(kind: str, **params) -> Flow:
    return FLOWS[kind](**params)


# --------------------------------------------------------------------------
# advection
# --------------------------------------------------------------------------
def departure_points(flow: Flow, grid: Grid, origin_km, hours: float):
    """Backward trajectories over ``hours`` from every cell centre (two-step midpoint).

    Returns the departure points ``(x, y)`` in km.
    """
    xx, yy = grid.meshgrid()
    x, y = xx + origin_km[0], yy + origin_km[1]
    u, v = flow.at(x, y)
    xm, ym = x - 0.5 * hours * u, y - 0.5 * hours * v
    u, v = flow.at(xm, ym)
    return x - hours * u, y - hours * v


def sample(field: np.ndarray, grid: Grid, origin_km, x, y, order: int = 3,
           mode: str = "grid-wrap", cval: float = np.nan) -> np.ndarray:
    """``field`` (on ``grid`` at ``origin_km``) at points ``(x, y)`` km."""
    col = (x - origin_km[0]) / grid.dx_km - 0.5
    row = (y - origin_km[1]) / grid.dx_km - 0.5
    return map_coordinates(field, [row.ravel(), col.ravel()], order=order, mode=mode,
                           cval=cval).reshape(x.shape)


def advect(field: np.ndarray, flow: Flow, grid: Grid, origin_km, hours: float,
           order: int = 3) -> np.ndarray:
    """Move ``field`` with ``flow`` for ``hours`` (one semi-Lagrangian step).

    A uniform flow on a periodic grid is moved exactly by a Fourier phase ramp.
    """
    if flow.uniform:
        from core.simulation.moving_fields import shift
        return shift(field, grid.dx_km, flow.mean, hours)
    x, y = departure_points(flow, grid, origin_km, hours)
    return sample(field, grid, origin_km, x, y, order=order)


def velocity_px(velocity_kmh: np.ndarray, dx_km: float, dt_min: float) -> np.ndarray:
    """km/h -> pixels per time step (the units of pysteps' motion fields)."""
    return np.asarray(velocity_kmh) * (dt_min / 60.0) / dx_km
