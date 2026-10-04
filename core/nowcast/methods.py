"""Motion, extrapolation and the pysteps nowcasting methods, as in the training school.

Everything takes rain rates in mm/h, ``(time, y, x)`` with the most recent field last,
and the pysteps metadata of :func:`core.nowcast.grid.metadata`; nowcasts come back in
mm/h, ``(lead, y, x)`` or ``(member, lead, y, x)``.

Motion (session block 03): ``LK`` (Lucas-Kanade, 4 fields), ``VET`` (3), ``DARTS`` (8),
``proesmans`` (2), computed on dBR (threshold 0.1 mm/h, -15 dBR fill) unless
``transform=False``.

Nowcasts (blocks 04, 04a):

=================  ===========================================================
``persistence``    Eulerian persistence - the last field, unchanged
``extrapolation``  Lagrangian persistence: semi-Lagrangian advection of the
                   last field in dBR
``sprog``          S-PROG (Seed 2003): cascade (6 levels), AR(2) per level, dBR
``anvil``          ANVIL (Pulkkinen et al. 2020) on rain rate, AR(2), 50-pixel
                   AR window (the radar's VIL is not available)
``linda``          LINDA deterministic (Pulkkinen et al. 2021), up to 15 Shi-Tomasi
                   features (OpenCV; pysteps' default blob detector needs
                   scikit-image), no perturbations
``steps``          STEPS ensemble (Bowler et al. 2006, Seed et al. 2013):
                   nonparametric noise, CDF probability matching, incremental
                   masking, no velocity perturbation (the session's settings)
``linda_p``        LINDA-P, the ensemble version
=================  ===========================================================
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np

from .grid import from_dbr, to_dbr

N_PAST = {"LK": 4, "VET": 3, "DARTS": 8, "proesmans": 2}   # fields each motion method uses

MOTION_METHODS = ("LK", "VET", "DARTS", "proesmans")
LINDA_WORKERS = 4


@contextlib.contextmanager
def _quiet():
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


def motion(rate: np.ndarray, meta: dict, method: str = "LK", transform: bool = True) -> np.ndarray:
    """Advection field ``(2, y, x)`` in pixels per step from the last ``N_PAST[method]`` fields."""
    from pysteps import motion as pm
    n = N_PAST[method]
    past = np.asarray(rate[-n:], dtype=float)       # Proesmans' Cython kernel takes float64 only
    if transform:
        past, _ = to_dbr(past, meta)
    past = np.where(np.isfinite(past), past, np.nanmin(past))
    kwargs = {"verbose": False} if method in ("LK", "VET", "DARTS") else {}
    if method == "DARTS":           # Fourier terms must fit the grid (default 50 each way)
        ny, nx = past.shape[1:]
        kwargs.update(N_x=min(50, nx // 3), N_y=min(50, ny // 3))
    if method == "proesmans":
        past = past[-2:]
    with _quiet():
        v = pm.get_method(method)(past, **kwargs)
    return np.asarray(v[:2] if v.ndim == 3 else v[0][:2], dtype=float)


def extrapolate(field: np.ndarray, velocity: np.ndarray, timesteps) -> np.ndarray:
    """Semi-Lagrangian extrapolation of ``field``; ``timesteps`` an int or fractional steps."""
    from pysteps.extrapolation.semilagrangian import extrapolate as sl
    with _quiet():
        return sl(field, velocity, timesteps, allow_nonfinite_values=True)


def cascade_levels(shape) -> int:
    """Cascade levels for S-PROG/STEPS: 6 (the session's S-PROG) unless the grid is too
    small to hold that many scales (OpenMRG is 44 x 35 pixels)."""
    return int(min(6, max(3, np.floor(np.log2(min(shape))) - 1)))


def deterministic(method: str, rate: np.ndarray, meta: dict, velocity: np.ndarray,
                  n_leadtimes: int) -> np.ndarray:
    """One deterministic nowcast, mm/h ``(lead, y, x)``."""
    from pysteps import nowcasts
    if method == "persistence":
        return np.repeat(rate[-1:], n_leadtimes, axis=0)
    km = meta["xpixelsize"] / 1000.0
    with _quiet():
        if method == "extrapolation":
            dbr, mdb = to_dbr(rate[-1:], meta)
            f = nowcasts.get_method("extrapolation")(
                dbr[-1], velocity, n_leadtimes,
                extrap_kwargs={"allow_nonfinite_values": True, "interp_order": 1})
            return from_dbr(f, mdb)
        if method == "sprog":
            dbr, mdb = to_dbr(rate[-3:], meta)
            f = nowcasts.get_method("sprog")(dbr, velocity, n_leadtimes,
                                             n_cascade_levels=cascade_levels(rate.shape[1:]),
                                             precip_thr=mdb["threshold"])
            return from_dbr(f, mdb)
        if method == "anvil":
            f = nowcasts.get_method("anvil")(rate[-4:], velocity, n_leadtimes, ar_order=2,
                                             ar_window_radius=min(50, min(rate.shape[1:]) // 2))
            return np.clip(np.nan_to_num(f), 0, None)
        if method == "linda":
            f = nowcasts.get_method("linda")(rate[-3:], velocity, n_leadtimes,
                                             feature_method="shitomasi", max_num_features=15,
                                             add_perturbations=False, kmperpixel=km,
                                             timestep=meta["accutime"], num_workers=LINDA_WORKERS)
            return np.clip(np.nan_to_num(f), 0, None)
    raise ValueError(method)


def ensemble(method: str, rate: np.ndarray, meta: dict, velocity: np.ndarray, n_leadtimes: int,
             n_members: int = 12, seed: int = 1234, num_workers: int = 1) -> np.ndarray:
    """An ensemble nowcast, mm/h ``(member, lead, y, x)``."""
    from pysteps import nowcasts
    km = meta["xpixelsize"] / 1000.0
    with _quiet():
        if method == "steps":
            dbr, mdb = to_dbr(rate[-3:], meta)
            f = nowcasts.get_method("steps")(
                dbr, velocity, n_leadtimes, n_ens_members=n_members,
                n_cascade_levels=cascade_levels(rate.shape[1:]),
                kmperpixel=km, precip_thr=mdb["threshold"], timestep=meta["accutime"],
                noise_method="nonparametric", vel_pert_method=None, probmatching_method="cdf",
                mask_method="incremental", seed=seed, num_workers=num_workers)
            return from_dbr(f, mdb)
        if method == "linda_p":
            f = nowcasts.get_method("linda")(
                rate[-3:], velocity, n_leadtimes, feature_method="shitomasi", max_num_features=15,
                add_perturbations=True, n_ens_members=n_members, kmperpixel=km,
                timestep=meta["accutime"], seed=seed, num_workers=LINDA_WORKERS)
            return np.clip(np.nan_to_num(f), 0, None)
    raise ValueError(method)


def reachable(velocity: np.ndarray, shape, n_leadtimes: int) -> np.ndarray:
    """``(lead, y, x)`` True where the semi-Lagrangian backward trajectory stays in the domain.

    Rain advected in from outside the domain is unknown to every method; scoring those cells
    as forecast zeros penalises the moving methods and favours persistence.
    """
    return np.isfinite(extrapolate(np.ones(shape), velocity, n_leadtimes))
