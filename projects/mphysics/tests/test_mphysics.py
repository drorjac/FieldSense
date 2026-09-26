"""N-body simulation and discovery, and the PINN losses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import nbody  # noqa: E402


def test_vectorized_accelerations_match_the_pairwise_sum():
    rng = np.random.default_rng(0)
    pos, m = rng.normal(size=(4, 3)), rng.uniform(0.5, 2, 4)
    loop = np.zeros_like(pos)
    for i in range(4):
        for j in range(4):
            if i != j:
                d = pos[j] - pos[i]
                loop[i] += nbody.G * m[j] * d / np.linalg.norm(d) ** 3
    np.testing.assert_allclose(nbody.accelerations(pos, m), loop, rtol=1e-12)


def test_circular_orbit_keeps_its_radius_and_conserves():
    traj = nbody.simulate(*nbody.two_body(eccentricity=0.0, m_planet=1e-9), t_end=12.6,
                          n_samples=800)
    r, _ = nbody.planet_radial(traj)
    assert r.max() - r.min() < 1e-6
    m = nbody.measure(traj)
    assert m["energy_drift"] < 1e-6 and m["angular_momentum_drift"] < 1e-6


def test_figure8_returns_to_its_start_after_one_period():
    """Regression: the original velocities (vy = 0.4662) broke the orbit."""
    traj = nbody.simulate(*nbody.three_body("figure8"), t_end=6.3259, n_samples=2000)
    assert np.abs(traj.pos[-1] - traj.pos[0]).max() < 1e-3
    assert nbody.measure(traj)["dist"].min() > 0.5         # no close encounters


def test_sindy_with_the_gravity_library_recovers_newton():
    pytest.importorskip("pysindy")
    traj = nbody.simulate(*nbody.two_body(eccentricity=0.3), t_end=30, n_samples=2000)
    model, x, t = nbody.discover_dynamics(traj, library="gravity")
    coef = dict(zip(model.get_feature_names(), model.coefficients()[2]))   # (vx)'
    assert coef["x/r^3"] == pytest.approx(-1.001, rel=2e-3)                # -G(M+m)
    assert np.isfinite(nbody.rollout(model, x[0], t)).all()


def test_rollout_marks_divergence_as_nan():
    class Explodes:
        def predict(self, s):
            return s * 10.0
    out = nbody.rollout(Explodes(), np.ones(2), np.linspace(0, 5, 50))
    assert np.isnan(out[-1]).all() and np.isfinite(out[0]).all()


def test_pinn_residuals_vanish_on_the_true_trajectory():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(SRC / "gravity"))
    from gravity.pinn import Throw, physics_loss

    throw = Throw()

    class Exact(torch.nn.Module):
        def forward(self, t):
            return throw.h0 + throw.v0 * t - 0.5 * throw.g * t ** 2

    t = torch.linspace(0, 2, 30).view(-1, 1)
    for residual in ("velocity", "acceleration"):
        assert float(physics_loss(Exact(), t, throw.v0, throw.g, residual).detach()) < 1e-8
    with pytest.raises(ValueError):
        physics_loss(Exact(), t, throw.v0, throw.g, "jerk")
