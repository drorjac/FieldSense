"""
Space-time rainfall: any generator, any flow, four kinds of evolution.

``moving_fields.sequence`` moves a field at one uniform velocity. This module
generalises it to the non-uniform flows of ``flows`` and to evolution
mechanisms taken from the stochastic simulators in the literature:

=============  ==================================================================
``frozen``     the field is carried by the flow and nothing else (Taylor's
               hypothesis). For a non-uniform flow the departure-point map is
               accumulated and the initial field interpolated once per frame, so
               no numerical diffusion builds up.
``ar1``        the latent Gaussian field evolves as an AR(1) process in the moving
               frame, ``g_t = rho A(g_{t-1}) + sqrt(1 - rho^2) e_t`` with
               ``rho = exp(-dt / tau)``, ``A`` the advection and ``e_t`` a fresh
               realisation of the same model; mapped back onto the initial field's
               values, so the wet fraction and intensities are preserved (as in
               ``moving_fields``; the temporal model of STREAP and SAMPO).
``cascade``    as ``ar1`` but per scale: the latent field is split into octave bands
               and each band ``j`` decorrelates with its own
               ``tau_j = tau * (L_j / 10 km)^scale_exponent`` - small features die
               fast, large ones live long (the lifetime-scale relation behind
               S-PROG and STEPS; Venugopal et al. 1999, Seed 2003).
``lifecycle``  for ``GaussianCells``: every cell is born, grows, peaks and decays
               (``sin`` envelope over a gamma-distributed lifetime) while it is
               carried by the flow; new cells are born as a Poisson process so the
               mean number alive stays ``n_cells``. Cells are rendered analytically
               each step (Neyman-Scott storm centres move with the flow too).
=============  ==================================================================

On top of any of these, ``growth_per_h`` scales intensity by ``exp(growth t)``
and ``imf_sigma`` / ``imf_tau_min`` add a random AR(1) modulation of the
domain-mean intensity (STREAP's areal-mean process).

The field is simulated on a domain padded by ``pad_frac`` on every side and
cropped, so rain enters the visible domain from outside as it would in reality
(and a non-periodic model never wraps into view).

    from core.simulation import generators as gen, flows, spacetime as st
    seq = st.simulate(gen.make("convective_cells"), Grid(n=128, dx_km=0.5), n_steps=36,
                      dt_min=5, flow=flows.RotationFlow(omega_deg_h=40, mean=(20, 0)),
                      evolution="lifecycle", lifetime_min=40)
    seq.frames, seq.velocity          # (36, 128, 128) mm/h; (2, 128, 128) km/h
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

from core.simulation import flows as fl
from core.simulation import random_fields as rf
from core.simulation.rain_fields import WET_THRESHOLD_MM_H, Grid

EVOLUTIONS = ("frozen", "ar1", "cascade", "lifecycle")


@dataclass
class FieldSequence:
    """A simulated sequence with the truth needed to score anything estimated from it."""

    frames: np.ndarray              # (T, n, n) rain rate, mm/h
    grid: Grid
    dt_min: float
    velocity: np.ndarray            # (2, n, n) true flow, km/h, (u east, v north)
    flow: fl.Flow
    model: object
    evolution: str
    params: dict = dc_field(default_factory=dict)
    extras: dict = dc_field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return self.frames.shape[0]

    @property
    def times_min(self) -> np.ndarray:
        return np.arange(self.n_steps) * self.dt_min

    def velocity_px(self, dt_min: Optional[float] = None, dx_km: Optional[float] = None) -> np.ndarray:
        """True motion in pixels per step, pysteps' convention (``V[0]`` along x / columns)."""
        return fl.velocity_px(self.velocity, dx_km or self.grid.dx_km, dt_min or self.dt_min)

    def lagrangian_persistence(self, t: int, h: int) -> np.ndarray:
        """Frame ``t`` carried ``h`` steps on by the true flow; NaN where it came from outside."""
        hours = h * self.dt_min / 60.0
        if self.flow.uniform:
            x, y = self.grid.meshgrid()
            u, v = self.flow.mean
            x, y = x - u * hours, y - v * hours
        else:
            # integrate the backward trajectory in sub-steps of at most 5 minutes
            n_sub = max(1, int(np.ceil(h * self.dt_min / 5.0)))
            x, y = self.grid.meshgrid()
            for _ in range(n_sub):
                x, y = _step_back(self.flow, x, y, hours / n_sub)
        out = fl.sample(self.frames[t], self.grid, (0.0, 0.0), x, y, order=1,
                        mode="constant", cval=np.nan)
        return np.clip(out * np.exp(self.params.get("growth_per_h", 0.0) * hours), 0.0, None)

    def predictability(self, h: int) -> float:
        """Correlation of frame ``t + h`` with frame ``t`` carried by the true flow, pooled.

        1 for a frozen field; it falls with evolution and bounds what any
        advection forecast can reach at that horizon.
        """
        n = self.n_steps - h
        if n < 1:
            return float("nan")
        a = np.concatenate([self.lagrangian_persistence(t, h).ravel() for t in range(n)])
        b = self.frames[h:h + n].ravel()
        ok = np.isfinite(a)
        a, b = a[ok], b[ok]
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def _step_back(flow, x, y, hours):
    u, v = flow.at(x, y)
    u2, v2 = flow.at(x - 0.5 * hours * u, y - 0.5 * hours * v)
    return x - hours * u2, y - hours * v2


def _step_forward(flow, x, y, hours):
    return _step_back(flow, x, y, -hours)


def simulate(model, grid: Grid, n_steps: int, dt_min: float = 5.0, flow: Optional[fl.Flow] = None,
             evolution: str = "ar1", tau_min: float = 60.0, scale_exponent: float = 1.0,
             n_levels: int = 6, lifetime_min: float = 45.0, lifetime_cv: float = 0.5,
             growth_per_h: float = 0.0, imf_sigma: float = 0.0, imf_tau_min: float = 60.0,
             pad_frac: float = 0.25, evolve_every: int = 1, seed: int = 0) -> FieldSequence:
    """Simulate ``n_steps`` frames of ``model`` at ``dt_min`` (see the module docstring).

    ``flow`` defaults to a uniform flow at the model's ``advection_kmh``. ``evolve_every``
    (``ar1`` / ``cascade``): mix in fresh structure every that many steps, with the AR
    coefficient of that interval, and only advect in between - the same decorrelation at a
    fraction of the cost when the time step is much shorter than ``tau``.
    """
    if evolution not in EVOLUTIONS:
        raise ValueError(f"evolution must be one of {EVOLUTIONS}")
    rng = np.random.default_rng(seed)
    if flow is None:
        flow = fl.UniformFlow(mean=tuple(getattr(model, "advection_kmh", (20.0, 0.0))))
    pad = int(round(pad_frac * grid.n)) if pad_frac > 0 else 0
    if flow.uniform and getattr(model, "periodic", True) and evolution != "lifecycle":
        pad = 0                     # exact periodic shifts need no margin
    sim = Grid(n=grid.n + 2 * pad, dx_km=grid.dx_km)
    origin = (-pad * grid.dx_km, -pad * grid.dx_km)
    flow.prepare(grid.size_km, sim, origin)
    hours = dt_min / 60.0

    if evolution == "lifecycle":
        frames, extras = _lifecycle(model, sim, origin, grid, n_steps, hours, flow, rng,
                                    lifetime_min, lifetime_cv)
    else:
        frames, extras = _latent_evolution(model, sim, origin, n_steps, hours, flow, rng, evolution,
                                           tau_min, scale_exponent, n_levels, max(1, evolve_every))
        frames = frames[:, pad:pad + grid.n, pad:pad + grid.n]

    t_h = np.arange(n_steps) * hours
    factor = np.exp(growth_per_h * t_h)
    if imf_sigma > 0:
        a = np.exp(-dt_min / imf_tau_min)
        z = np.zeros(n_steps)
        for i in range(1, n_steps):
            z[i] = a * z[i - 1] + np.sqrt(1 - a ** 2) * rng.standard_normal()
        factor = factor * np.exp(imf_sigma * z - imf_sigma ** 2 / 2)
    frames = np.clip(frames * factor[:, None, None], 0.0, None)
    frames[frames < WET_THRESHOLD_MM_H] = 0.0

    params = dict(tau_min=tau_min, scale_exponent=scale_exponent, lifetime_min=lifetime_min,
                  growth_per_h=growth_per_h, imf_sigma=imf_sigma, pad=pad, seed=seed)
    return FieldSequence(frames.astype(np.float32), grid, dt_min, flow.field(grid), flow, model,
                         evolution, params, extras)


# --------------------------------------------------------------------------
# latent-field evolutions
# --------------------------------------------------------------------------
def _octave_bands(sim: Grid, n_levels: int):
    """Fourier weights of ``n_levels`` octave bands that sum to one, and their scales (km)."""
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(sim.n, sim.dx_km), np.fft.fftfreq(sim.n, sim.dx_km)))
    k_max = 0.5 / sim.dx_km
    centres = k_max / 2.0 ** np.arange(n_levels)              # finest first
    logk = np.log2(np.where(k > 0, k, centres[-1] / 4))
    w = np.stack([np.exp(-0.5 * ((logk - np.log2(c)) / 0.5) ** 2) for c in centres])
    w[-1] = np.where(k <= centres[-1], 1.0, w[-1])            # everything larger goes to the last band
    w /= w.sum(0, keepdims=True)
    return w, 1.0 / centres


def _latent_evolution(model, sim, origin, n_steps, hours, flow, rng, evolution, tau_min,
                      scale_exponent, n_levels, every=1):
    base = model.build(sim)
    values = np.sort(base.ravel())

    def latent_of(rain):
        tie = rf.grf(sim.n, sim.dx_km, rng, cov="gaussian", length_km=4.0 * sim.dx_km + 2.0)
        return rf.normal_scores(rain, tie)

    def to_rain(g):
        ranks = np.argsort(np.argsort(g.ravel()))
        return values[ranks].reshape(g.shape)

    # A model with a fixed geometry (the frontal band, ``periodic = False``) is not
    # stationary: a fresh realisation has its band where the first one started, so it
    # is carried to where the flow has taken the field by now before it is mixed in.
    stationary = getattr(model, "periodic", True)
    xx, yy = sim.meshgrid()
    x0, y0 = xx + origin[0], yy + origin[1]
    disp = {"x": np.zeros_like(xx), "y": np.zeros_like(yy), "step": 0}

    def carry(step):
        if flow.uniform:
            disp["step"] = step
            return
        xd, yd = fl.departure_points(flow, sim, origin, hours)
        disp["x"] = fl.sample(disp["x"], sim, origin, xd, yd, order=1, mode="nearest") + (xd - x0)
        disp["y"] = fl.sample(disp["y"], sim, origin, xd, yd, order=1, mode="nearest") + (yd - y0)

    def moved(field):
        if flow.uniform:
            from core.simulation.moving_fields import shift
            return shift(field, sim.dx_km, flow.mean, disp["step"] * hours)
        return fl.sample(field, sim, origin, x0 + disp["x"], y0 + disp["y"], order=3)

    def fresh():
        f = latent_of(dataclasses.replace(model, seed=int(rng.integers(2 ** 31))).build(sim))
        return f if stationary else moved(f)

    frames = [base]
    if evolution == "frozen":
        if flow.uniform:
            from core.simulation.moving_fields import shift
            for i in range(1, n_steps):
                frames.append(np.clip(shift(base, sim.dx_km, flow.mean, i * hours), 0, None))
        else:
            # accumulate the departure map (the displacement already accumulated at the
            # departure point, plus this step's) and interpolate the initial field once
            for i in range(1, n_steps):
                carry(i)
                frames.append(np.clip(moved(base), 0, None))
        return np.stack(frames), {}

    rho = float(np.exp(-every * hours * 60.0 / tau_min))
    g = latent_of(base)
    if evolution == "ar1":
        for i in range(1, n_steps):
            if not stationary:
                carry(i)
            g = fl.advect(g, flow, sim, origin, hours)
            if i % every == 0:
                g = rho * g + np.sqrt(1 - rho ** 2) * fresh()
            frames.append(to_rain(g))
        return np.stack(frames), {"rho": rho}

    # cascade: one AR(1) per octave band
    w, scales = _octave_bands(sim, n_levels)
    taus = tau_min * (scales / 10.0) ** scale_exponent
    rhos = np.exp(-every * hours * 60.0 / taus)
    G = np.fft.fft2(g)
    bands = [np.fft.ifft2(G * wj).real for wj in w]
    for i in range(1, n_steps):
        if not stationary:
            carry(i)
        bands = [fl.advect(b, flow, sim, origin, hours) for b in bands]
        if i % every == 0:
            E = np.fft.fft2(fresh())
            bands = [r * b + np.sqrt(1 - r ** 2) * np.fft.ifft2(E * wj).real
                     for b, r, wj in zip(bands, rhos, w)]
        frames.append(to_rain(np.sum(bands, axis=0)))
    return np.stack(frames), {"band_scales_km": scales, "band_tau_min": taus}


# --------------------------------------------------------------------------
# cell lifecycle
# --------------------------------------------------------------------------
def _lifecycle(model, sim, origin, grid, n_steps, hours, flow, rng, lifetime_min, lifetime_cv):
    if not hasattr(model, "sample_cells"):
        raise TypeError("evolution='lifecycle' needs a cell model (generators.GaussianCells)")
    model = dataclasses.replace(model, periodic=False)
    lo, hi = origin[0], origin[0] + sim.size_km
    box = (lo, hi, lo, hi)
    area_ratio = (sim.size_km / grid.size_km) ** 2
    n_alive = model.n_cells * area_ratio
    life_h = lifetime_min / 60.0
    shape_l = 1.0 / lifetime_cv ** 2

    storms = None
    if model.n_storms > 0:
        storms = np.stack([rng.uniform(lo, hi, model.n_storms), rng.uniform(lo, hi, model.n_storms)])

    def new_cells(n_mean):
        if storms is None:
            c = model.sample_cells(sim, rng, n=n_mean, domain_km=box)
        else:
            c = dataclasses.replace(model, n_storms=0).sample_cells(sim, rng, n=n_mean, domain_km=box)
            k = c["x"].size
            owner = rng.integers(0, storms.shape[1], k)
            c["x"] = storms[0, owner] + rng.normal(0, model.storm_radius_km, k)
            c["y"] = storms[1, owner] + rng.normal(0, model.storm_radius_km, k)
        c["life"] = rng.gamma(shape_l, life_h / shape_l, c["x"].size)
        c["age"] = np.zeros(c["x"].size)
        return c

    cells = new_cells(n_alive)
    cells["age"] = rng.uniform(0, 1, cells["x"].size) * cells["life"]       # a population in progress
    births_per_step = n_alive * hours / life_h
    bg = model.background(sim, rng)
    tex = model.texture(sim, rng) if hasattr(model, "texture") else None
    if tex is not None and model.texture_strength <= 0:
        tex = None
    moving = model.background_war > 0 or tex is not None
    bg_model = dataclasses.replace(model, background_war=0.0)

    def carried(field):
        """A static field (background, texture) where the flow has taken it by step i."""
        if flow.uniform:
            from core.simulation.moving_fields import shift
            return shift(field, sim.dx_km, flow.mean, i * hours)
        return fl.sample(field, sim, origin, x0 + dx_acc, y0 + dy_acc, order=3)
    frames, counts = [], []
    xx, yy = sim.meshgrid()
    x0, y0 = xx + origin[0], yy + origin[1]
    dx_acc, dy_acc = np.zeros_like(xx), np.zeros_like(yy)
    for i in range(n_steps):
        env = np.sin(np.pi * np.clip(cells["age"] / cells["life"], 0, 1))
        shown = dict(cells)
        grow = 0.6 + 0.4 * env                                  # cells widen as they mature
        shown["major"], shown["minor"] = cells["major"] * grow, cells["minor"] * grow
        rain = bg_model.render(shown, sim, origin_km=origin, scale=env)
        if model.background_war > 0:
            rain = rain + np.clip(carried(bg), 0, None)
        if tex is not None:
            rain = rain * np.clip(carried(tex), 0, None)
        p = int(round(-origin[0] / sim.dx_km))
        frames.append(rain[p:p + grid.n, p:p + grid.n])
        counts.append(int(((cells["x"] >= 0) & (cells["x"] < grid.size_km)
                           & (cells["y"] >= 0) & (cells["y"] < grid.size_km)).sum()))
        # move, age, die, be born
        cells["x"], cells["y"] = _step_forward(flow, cells["x"], cells["y"], hours)
        if storms is not None:
            storms = np.stack(_step_forward(flow, storms[0], storms[1], hours))
            storms[0] = lo + np.mod(storms[0] - lo, hi - lo)    # storms leaving re-enter upstream
            storms[1] = lo + np.mod(storms[1] - lo, hi - lo)
        if moving and not flow.uniform:
            xd, yd = fl.departure_points(flow, sim, origin, hours)
            dx_acc = fl.sample(dx_acc, sim, origin, xd, yd, order=1, mode="nearest") + (xd - x0)
            dy_acc = fl.sample(dy_acc, sim, origin, xd, yd, order=1, mode="nearest") + (yd - y0)
        cells["age"] = cells["age"] + hours
        alive = (cells["age"] < cells["life"]) & (cells["x"] > lo - 10) & (cells["x"] < hi + 10) \
            & (cells["y"] > lo - 10) & (cells["y"] < hi + 10)
        cells = {k: v[alive] for k, v in cells.items()}
        born = new_cells(births_per_step)
        cells = {k: np.concatenate([cells[k], born[k]]) for k in cells}
    return np.stack(frames), {"cells_in_view": np.array(counts)}
