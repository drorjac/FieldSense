"""
Moving rainfall fields with a known truth: translation, growth, evolution.

One generator for every project that needs rain which moves:
``physics_ml`` recovers the velocity from the sequence, ``spatial_interpolation``
checks its forecasters against it, ``rainfall_field_sim`` samples it with a
link network. The fields are the models in ``rain_fields``; this module adds
time.

``shift``      translate a periodic field by any distance - exactly, by a
               phase ramp in Fourier space, or by whole cells
``sequence``   a field moving at a known velocity, optionally growing or
               decaying, and optionally evolving (its structure decorrelating
               in the moving frame, as real rain does)

Three levels of difficulty, in the order a forecaster should meet them:

1. pure translation - Lagrangian persistence is exact, and anything that
   cannot match it has not learned advection
2. with growth/decay - intensity changes along the path
3. with evolution - the pattern itself changes; only the part the past
   predicts is forecastable, and ``MovingSequence.predictability`` says how
   much that is

    from core.simulation import moving_fields as mf
    from core.simulation.rain_fields import Grid, StratiformField
    seq = mf.sequence(StratiformField(), Grid(n=64, dx_km=1.0), n_steps=24,
                      dt_min=15, velocity_kmh=(20, 5), evolve_tau_min=120)
    seq.frames                  # (24, 64, 64) mm/h
    seq.lagrangian_persistence(8, h=2)  # frame 8 moved 30 min on at the true velocity
    seq.predictability(2)       # the correlation any forecaster can hope for at 30 min
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from core.simulation.rain_fields import WET_THRESHOLD_MM_H, Grid, spectral_grf


def shift(field: np.ndarray, dx_km: float, velocity_kmh, hours: float,
          method: str = "spectral") -> np.ndarray:
    """Translate a periodic field by ``velocity_kmh * hours``.

    ``spectral`` is exact for any displacement (a phase ramp in Fourier
    space) but rings slightly at the hard edges of intermittent rain, so
    callers clip negative values. On an even grid the Nyquist mode cannot
    move by a fraction of a cell, so fractional shifts damp it and do not
    compose exactly; whole-cell shifts are exact rolls. ``integer`` rolls by
    whole cells - no ringing, but the velocity is quantized: at 0.25 km cells
    and 5-minute steps, 14 km/h becomes 15 km/h.
    """
    ux, uy = velocity_kmh
    if method == "integer":
        sx = round(ux * hours / dx_km)
        sy = round(uy * hours / dx_km)
        return np.roll(np.roll(field, sy, axis=0), sx, axis=1)
    if method != "spectral":
        raise ValueError(f"method must be 'spectral' or 'integer', got {method!r}")
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx, d=dx_km)[None, :]
    ky = np.fft.fftfreq(ny, d=dx_km)[:, None]
    phase = np.exp(-2j * np.pi * (kx * ux * hours + ky * uy * hours))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * phase))


@dataclass
class MovingSequence:
    """A generated sequence and everything needed to score a forecast of it."""

    frames: np.ndarray          # (T, n, n) rain rate, mm/h
    grid: Grid
    dt_min: float
    velocity_kmh: tuple
    growth_per_h: float
    evolve_tau_min: float | None
    extended: np.ndarray | None = None   # non-periodic models: frames on the padded domain
    pad: int = 0                         # cells of padding on each side of ``extended``

    @property
    def times_min(self) -> np.ndarray:
        return np.arange(len(self.frames)) * self.dt_min

    def lagrangian_persistence(self, t: int, h: int) -> np.ndarray:
        """Frame ``t`` moved ``h`` steps on at the true velocity.

        The best forecast that knows the motion but not the future growth or
        evolution - the ceiling for an advection-only method.
        """
        hours = h * self.dt_min / 60.0
        src = self.frames[t] if self.extended is None else self.extended[t]
        out = shift(src, self.grid.dx_km, self.velocity_kmh, hours)
        if self.extended is not None:
            out = _crop(out, self.pad, self.grid.n)
        return np.clip(out * np.exp(self.growth_per_h * hours), 0.0, None)

    def predictability(self, h: int) -> float:
        """Correlation between frame t+h and frame t moved h steps on.

        1 for pure translation; it falls with evolution, and bounds what any
        forecaster can achieve at that horizon.
        """
        # pooled over all (t, t+h) pairs, so frames the rain has mostly left
        # weigh in by how much rain they hold, not one vote each
        n = len(self.frames) - h
        if n < 1:
            return float("nan")
        a = np.concatenate([self.lagrangian_persistence(t, h).ravel() for t in range(n)])
        b = self.frames[h:h + n].ravel()
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def sequence(model, grid: Grid, n_steps: int, dt_min: float = 5.0,
             velocity_kmh=None, growth_per_h: float = 0.0,
             evolve_tau_min: float | None = None, method: str = "spectral",
             seed: int = 0) -> MovingSequence:
    """A rain field from ``model`` moving over ``n_steps`` of ``dt_min``.

    ``velocity_kmh`` defaults to the model's own ``advection_kmh``.
    ``growth_per_h`` scales intensity by ``exp(growth * t)`` (negative for
    decay). ``evolve_tau_min`` makes the pattern decorrelate in the moving
    frame with that e-folding time: each step the field's latent Gaussian
    scores are mixed, AR(1) style, with those of a fresh realization of the
    same model - new cells for convective rain, a new along-band texture for a
    front - and mapped back onto the original values, so wet fraction and
    intensities are preserved while the pattern reshapes. ``None`` is frozen structure (pure translation).

    A model with ``periodic = False`` (the frontal band) is generated on a
    domain padded by the distance travelled and cropped, so nothing wraps.

    ``model`` is any ``rain_fields`` model, or ``"smooth"`` for a
    differentiable Gaussian field (5 +/- 2 mm/h, no dry areas), which is what
    derivative-based methods such as SINDy need.
    """
    rng = np.random.default_rng(seed)
    velocity = tuple(velocity_kmh if velocity_kmh is not None
                     else getattr(model, "advection_kmh", (0.0, 0.0)))
    hours = dt_min / 60.0
    smooth = isinstance(model, str) and model == "smooth"
    target = grid
    pad = 0
    if not smooth and not getattr(model, "periodic", True):
        # build on a domain larger by the whole path, move there, crop the
        # middle: the wrap-around stays in the margin, out of view
        travel_km = np.hypot(*velocity) * n_steps * hours
        pad = int(np.ceil(travel_km / grid.dx_km)) + 1
        if pad > 4 * grid.n:
            raise ValueError(f"{type(model).__name__} is not periodic and would travel "
                             f"{travel_km:.0f} km; use fewer steps or a periodic model")
        grid = Grid(n=grid.n + 2 * pad, dx_km=grid.dx_km)

    if smooth:
        def realization(_):
            return spectral_grf(grid, beta=3.2, rng=rng)

        def to_rain(g):
            return 5.0 + 2.0 * g
        latent = realization(0)
    else:
        base = model.build(grid)
        sorted_rain = np.sort(base.ravel())

        def realization(i):
            # a fresh draw of the same regime (its cells, its band), in the
            # latent Gaussian space, moved to where the flow has carried it
            fresh = dataclasses.replace(model, seed=int(rng.integers(2 ** 31))).build(grid)
            return shift(_latent(fresh, grid, rng), grid.dx_km, velocity, i * hours)

        def to_rain(g):
            # the base field's own values, re-ranked: wet fraction and the
            # intensity distribution are preserved exactly
            ranks = np.argsort(np.argsort(g.ravel()))
            return sorted_rain[ranks].reshape(g.shape)
        latent = _latent(base, grid, rng)

    frames = []
    if evolve_tau_min is None:                           # frozen structure
        field0 = to_rain(latent) if smooth else base
        for i in range(n_steps):
            f = shift(field0, grid.dx_km, velocity, i * hours, method)
            frames.append(f * np.exp(growth_per_h * i * hours))
    else:
        rho = float(np.exp(-dt_min / evolve_tau_min))
        g = latent
        for i in range(n_steps):
            if i:
                g = rho * shift(g, grid.dx_km, velocity, hours) \
                    + np.sqrt(1 - rho ** 2) * realization(i)
            frames.append(to_rain(g) * np.exp(growth_per_h * i * hours))
    frames = np.clip(np.stack(frames), 0.0, None).astype(np.float32)
    if not pad:
        return MovingSequence(frames, grid, dt_min, velocity, growth_per_h, evolve_tau_min)
    return MovingSequence(_crop(frames, pad, target.n), target, dt_min, velocity,
                          growth_per_h, evolve_tau_min, extended=frames, pad=pad)


def _crop(a: np.ndarray, pad: int, n: int) -> np.ndarray:
    return a[..., pad:pad + n, pad:pad + n]


def _latent(rain: np.ndarray, grid: Grid, rng: np.random.Generator) -> np.ndarray:
    """Rain -> standard normal scores by rank.

    Dry pixels all tie at zero; ranking them by pixel order would stripe the
    latent field row by row, so ties are broken by a smooth random field.
    """
    from scipy.stats import norm

    tiebreak = spectral_grf(grid, beta=3.2, rng=rng).ravel()
    ranks = np.lexsort((tiebreak, rain.ravel())).argsort()
    return norm.ppf((ranks + 0.5) / ranks.size).reshape(rain.shape)


def wet_fraction(frames: np.ndarray, threshold: float = WET_THRESHOLD_MM_H) -> np.ndarray:
    """Share of wet pixels per frame."""
    return (frames >= threshold).mean(axis=tuple(range(1, frames.ndim)))
