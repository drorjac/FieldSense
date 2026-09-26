"""
N-body gravity: simulate, measure, and try to rediscover the law.

The code behind ``notebooks/archive/03_nbody_full_pipeline.ipynb``, which kept all of it inline:

``simulate``           integrate Newtonian gravity (DOP853, rtol 1e-10, atol 1e-12)
``two_body``           a planet on a circular or elliptical orbit
``three_body``         triangle, line or figure-8 initial conditions
``measure``            distances, speeds, numerical accelerations, energy
                       and angular momentum, and their conservation drift
``discover_force_law`` PySR on (r, |a|) - does 1/r^2 fall out?
``discover_dynamics``  SINDy on (x, y, vx, vy), with a library chosen by name

The library is the whole question for SINDy. The original notebook used
polynomials up to degree 2, in which gravity (``-GM x / r^3``) cannot be
written - so it could not succeed, and its summary blamed chaos.
``library="gravity"`` adds the inverse-cube terms and recovers the law.

    from nbody import two_body, simulate, measure
    traj = simulate(*two_body(eccentricity=0.3), t_end=30, n_samples=2000)
    m = measure(traj)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

G = 1.0                               # normalized units


@dataclass
class Trajectory:
    """A simulation: time, and per-body positions and velocities."""

    t: np.ndarray                     # (n_samples,)
    pos: np.ndarray                   # (n_samples, n_bodies, 3)
    vel: np.ndarray                   # (n_samples, n_bodies, 3)
    masses: np.ndarray
    label: str = ""


def accelerations(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Newtonian accelerations, (n_bodies, 3), vectorized over pairs."""
    diff = pos[None, :, :] - pos[:, None, :]             # r_j - r_i
    dist3 = np.linalg.norm(diff, axis=-1) ** 3
    np.fill_diagonal(dist3, np.inf)
    return G * np.sum(masses[None, :, None] * diff / dist3[:, :, None], axis=1)


def simulate(masses, pos0, vel0, t_end: float, n_samples: int,
             rtol: float = 1e-10, atol: float = 1e-12, label: str = "") -> Trajectory:
    """Integrate the n-body problem from (pos0, vel0) over [0, t_end].

    ``atol`` matters as much as ``rtol``: solve_ivp's default absolute
    tolerance is 1e-6, which caps the accuracy whatever ``rtol`` says (the
    original notebook set only ``rtol``).
    """
    masses = np.asarray(masses, dtype=float)
    n = masses.size

    def rhs(_t, s):
        p, v = s[:3 * n].reshape(n, 3), s[3 * n:]
        return np.concatenate([v, accelerations(p, masses).ravel()])

    t_eval = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(rhs, (0.0, t_end), np.concatenate([np.ravel(pos0), np.ravel(vel0)]),
                    t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return Trajectory(sol.t, sol.y[:3 * n].T.reshape(-1, n, 3),
                      sol.y[3 * n:].T.reshape(-1, n, 3), masses, label)


def two_body(eccentricity: float = 0.3, r0: float = 1.0, m_star: float = 1.0,
             m_planet: float = 1e-3):
    """(masses, pos0, vel0) for a planet starting at perihelion.

    Perihelion speed ``v = sqrt(GM (1 + e) / r0)``; ``e = 0`` is circular.
    """
    v0 = np.sqrt(G * m_star * (1 + eccentricity) / r0)
    pos = np.array([[0.0, 0.0, 0.0], [r0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [0.0, v0, 0.0]])
    return np.array([m_star, m_planet]), pos, vel


def three_body(config: str = "triangle", size: float = 1.0, speed: float = 0.4):
    """(masses, pos0, vel0), equal masses, total momentum removed."""
    masses = np.ones(3)
    if config == "triangle":
        pos = size * np.array([[-0.5, 0, 0], [0.5, 0, 0], [0, np.sqrt(3) / 2, 0]])
        vel = speed * np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0]], dtype=float)
    elif config == "line":
        pos = np.array([[-1.0, 0, 0], [0, 0, 0], [1, 0, 0]])
        vel = speed * np.array([[0, 0.5, 0], [0, -1, 0], [0, 0.5, 0]])
    elif config == "figure8":
        # Chenciner & Montgomery (2000), period ~6.3259. The original notebook
        # set both velocity components to 0.4662, which is not this solution:
        # vy is 0.4324, and with the wrong value the choreography breaks up
        # into chaotic close encounters.
        pos = np.array([[-0.97000436, 0.24308753, 0], [0.97000436, -0.24308753, 0], [0, 0, 0]])
        v = np.array([0.46620368, 0.43236573, 0.0])
        vel = np.array([v, v, -2 * v])
    else:
        raise ValueError(f"unknown config {config!r}")
    vel = vel - np.sum(masses[:, None] * vel, axis=0) / masses.sum()
    return masses, pos, vel


def measure(traj: Trajectory) -> dict:
    """Derived quantities and conservation drift.

    Acceleration is differentiated numerically from velocity, as it would
    be from real tracking data - the discovery methods see that, not the
    simulator's exact force.
    """
    t, pos, vel, m = traj.t, traj.pos, traj.vel, traj.masses
    dt = t[1] - t[0]
    acc = np.gradient(vel, dt, axis=0)
    speed = np.linalg.norm(vel, axis=-1)                 # (n_samples, n_bodies)
    i, j = np.triu_indices(m.size, 1)
    dist = np.linalg.norm(pos[:, j] - pos[:, i], axis=-1)   # (n_samples, n_pairs)
    kinetic = 0.5 * np.sum(m * speed ** 2, axis=1)
    potential = -G * np.sum(m[i] * m[j] / dist, axis=1)
    energy = kinetic + potential
    lz = np.sum(m * (pos[..., 0] * vel[..., 1] - pos[..., 1] * vel[..., 0]), axis=1)

    def drift(x):
        return float((x.max() - x.min()) / (abs(x.mean()) + 1e-12))

    return {"t": t, "acc": acc, "speed": speed, "dist": dist, "pairs": list(zip(i, j)),
            "kinetic": kinetic, "potential": potential, "energy": energy,
            "angular_momentum": lz, "energy_drift": drift(energy),
            "angular_momentum_drift": drift(lz)}


def planet_radial(traj: Trajectory, body: int = 1, star: int = 0) -> tuple:
    """(r, |a|) of one body about another - the data for a force law."""
    r = np.linalg.norm(traj.pos[:, body] - traj.pos[:, star], axis=-1)
    a = np.linalg.norm(np.gradient(traj.vel[:, body], traj.t[1] - traj.t[0], axis=0), axis=-1)
    return r, a


def discover_force_law(r, a, n: int = 500, niterations: int = 50, seed: int = 0, **kw):
    """PySR on (r, |a|); returns the fitted regressor (``.sympy()`` for the law)."""
    from pysr import PySRRegressor

    idx = np.linspace(0, len(r) - 1, n, dtype=int)
    model = PySRRegressor(niterations=niterations, binary_operators=["+", "-", "*", "/"],
                          unary_operators=["square", "inv", "sqrt"], populations=20,
                          population_size=40, maxsize=15, verbosity=0, progress=False,
                          random_state=seed, deterministic=True, parallelism="serial",
                          **kw)
    model.fit(r[idx].reshape(-1, 1), a[idx], variable_names=["r"])
    return model


def discover_dynamics(traj: Trajectory, body: int = 1, library: str = "gravity",
                      threshold: float = 0.05, n: int = 1000):
    """SINDy on one body's planar state (x, y, vx, vy).

    ``library="polynomial"`` is degree-2 polynomials, the original notebook's
    choice, which cannot express 1/r^3. ``library="gravity"`` is linear terms
    plus ``x/r^3`` and ``y/r^3`` - the form Newton's law takes.
    """
    import pysindy as ps

    idx = np.linspace(0, len(traj.t) - 1, n, dtype=int)
    rel = traj.pos[idx, body, :2] - traj.pos[idx, 0, :2]
    x = np.column_stack([rel, traj.vel[idx, body, :2] - traj.vel[idx, 0, :2]])
    names = ["x", "y", "vx", "vy"]
    if library == "polynomial":
        lib = ps.PolynomialLibrary(degree=2, include_bias=True)
    elif library == "gravity":
        # two-argument terms, fed only (x, y): x/r^3 and y/r^3
        inverse_cube = ps.CustomLibrary(
            library_functions=[lambda x, y: x / (x ** 2 + y ** 2) ** 1.5,
                               lambda x, y: y / (x ** 2 + y ** 2) ** 1.5],
            function_names=[lambda x, y: f"{x}/r^3", lambda x, y: f"{y}/r^3"])
        lib = ps.GeneralizedLibrary(
            [ps.PolynomialLibrary(degree=1, include_bias=False), inverse_cube],
            inputs_per_library=[[0, 1, 2, 3], [0, 1]])
    else:
        raise ValueError(f"library must be 'polynomial' or 'gravity', got {library!r}")
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold), feature_library=lib)
    model.fit(x, t=traj.t[idx], feature_names=names)
    return model, x, traj.t[idx]



def rollout(model, x0: np.ndarray, t: np.ndarray, blowup: float = 1e3) -> np.ndarray:
    """Integrate a fitted SINDy model from ``x0``; NaN after it diverges.

    pysindy's own ``simulate`` uses LSODA, which prints C-level errors when a
    wrong model diverges. Here the integration stops cleanly once any state
    component passes ``blowup`` and the rest of the series is NaN.
    """
    def rhs(_t, s):
        return model.predict(s[None, :])[0]

    def escaped(_t, s):
        return blowup - np.max(np.abs(s))
    escaped.terminal = True

    with np.errstate(all="ignore"):
        sol = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, events=escaped, rtol=1e-8)
    out = np.full((t.size, x0.size), np.nan)
    out[:sol.y.shape[1]] = sol.y.T
    return out
