"""
Can SINDy recover how a rainfall field moves, from the field alone?

This is the temporal counterpart to ``discover_itu.py``. That one asks
whether symbolic regression recovers the *static* physics a CML retrieval
inverts - ``gamma = k R^alpha``. This asks whether sparse regression recovers
the *dynamics* a rainfall field obeys as it crosses a region.

To first order a rain field is advected by the wind:

    dR/dt = -u dR/dx - v dR/dy

so fitting a library of spatial derivatives against the time derivative
should return the advection velocity in its coefficients - the coefficient
on ``R_x`` is ``-u``, on ``R_y`` is ``-v``. That is field estimation, not a
toy: it is exactly the step a nowcast needs, and ``projects/
spatial_interpolation`` benchmarks POD-SINDy against a Transformer and a
Mamba-style SSM for precisely this.

Why this lives here rather than beside the Lorenz notebook: the Lorenz system
demonstrates the *method*. This demonstrates the method **on the quantity
this repository measures**, against a velocity that is known exactly.

Three experiments, in the order that makes the result interpretable:

``clean``
    ``rainfall_field_sim`` fields translated at a velocity we choose.
    Ground truth is exact, so this establishes what recovery looks like.
``noisy``
    the same, with observation noise. Spatial derivatives amplify noise
    badly, so this is where the method's real tolerance shows.
``sampled``
    the field observed only where the CML network is, then interpolated
    back to a grid - what a nowcast actually receives.

    python -m discover_advection --experiment clean
    python -m discover_advection --all
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from core.data_paths import REPO_ROOT  # noqa: E402

from core.simulation import moving_fields as mf
from core.simulation.rain_fields import Grid, StratiformField, spectral_grf

RESULTS = HERE.parent / "results"


# --------------------------------------------------------------------------
# exact sub-pixel advection
# --------------------------------------------------------------------------
def shift_exact(field: np.ndarray, dx_km: float,
                u_kmh: float, v_kmh: float, hours: float) -> np.ndarray:
    """Translate a periodic field by an arbitrary, non-integer distance.

    ``rain_fields.advect`` rounds to whole cells, which quantizes the velocity
    - at 0.25 km cells and 5-minute steps, 14 km/h becomes 15 km/h. That would
    make this experiment measure the rounding, not the method. The exact
    (spectral) shift lives in ``core.simulation.moving_fields``.
    """
    return mf.shift(field, dx_km, (u_kmh, v_kmh), hours, method="spectral")


def make_sequence(velocity=(14.0, 5.0), n_steps: int = 40,
                  dt_min: float = 1.0, grid: Grid | None = None,
                  seed: int = 3, field: str = "smooth") -> tuple:
    """A rainfall field advected at a known velocity.

    Stratiform is used deliberately: it is smooth, so its spatial derivatives
    are well conditioned. A convective field is mostly zeros with sharp edges,
    which is a much harder derivative problem.

    ``dt_min`` has to keep the per-step displacement well under one grid cell
    or the finite-difference time derivative is under-resolved and every
    estimator reads low - at 5-minute steps and 1 km cells, 14 km/h moves the
    field 1.17 cells per step and least squares returns 12.2 km/h on data
    with no noise at all. One minute puts it at 0.23 cells.
    """
    grid = grid or Grid(n=64, dx_km=1.0)
    if field == "smooth":
        # A pure Gaussian random field: everywhere differentiable, so the
        # finite differences the method depends on are well posed.
        rng = np.random.default_rng(seed)
        base = 5.0 + 2.0 * spectral_grf(grid, beta=3.2, rng=rng)
    else:
        # A real rainfall field is intermittent: the threshold transform puts
        # hard zeros next to wet pixels, and a spatial derivative across that
        # edge is not defined. This is the honest case, and it is harder.
        base = StratiformField(seed=seed).build(grid)
    hours = dt_min / 60.0
    frames = [shift_exact(base, grid.dx_km, *velocity, hours * i)
              for i in range(n_steps)]
    return np.stack(frames), grid, hours


# --------------------------------------------------------------------------
# the two estimators
# --------------------------------------------------------------------------
def _spectral_derivative(seq: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Exact derivative along one axis of a periodic field."""
    n = seq.shape[axis]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=spacing)
    shape = [1] * seq.ndim
    shape[axis] = n
    return np.real(np.fft.ifft(
        np.fft.fft(seq, axis=axis) * (1j * k).reshape(shape), axis=axis))


def least_squares_velocity(seq: np.ndarray, dx_km: float, dt_h: float,
                           spectral: bool = True) -> dict:
    """Direct solve of ``R_t = a R_x + b R_y``; the reference answer.

    Spatial derivatives are taken spectrally, which matters more than it
    sounds. A central difference computes ``sin(k dx)/dx`` instead of ``k``,
    so it under-reads high-wavenumber content - and it under-reads the
    *spatial* derivative more than the temporal one, because the field moves
    only a fraction of a cell per timestep. Their ratio is the velocity, so
    the error does not cancel: on exact, noise-free fields, finite
    differences return 16.8 km/h for a true 14.0. Spectral derivatives are
    exact for a periodic field and return 13.6.

    The time derivative stays a finite difference. The sequence is *not*
    periodic in time - the field translates out of one edge and into the
    other - so a spectral time derivative picks up Gibbs error and reads
    12.8 instead.
    """
    r_t = np.gradient(seq, dt_h, axis=0)
    if spectral:
        r_y = _spectral_derivative(seq, dx_km, axis=1)
        r_x = _spectral_derivative(seq, dx_km, axis=2)
    else:
        r_y = np.gradient(seq, dx_km, axis=1)
        r_x = np.gradient(seq, dx_km, axis=2)

    A = np.column_stack([r_x.ravel(), r_y.ravel()])
    b = r_t.ravel()
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ coef
    ss = 1.0 - ((b - pred) ** 2).sum() / ((b - b.mean()) ** 2).sum()
    return {"u": float(-coef[0]), "v": float(-coef[1]), "r2": float(ss)}


def sindy_velocity(seq: np.ndarray, dx_km: float, dt_h: float,
                   threshold: float = 0.5, derivative_order: int = 2) -> dict:
    """SINDy with a PDE library over spatial derivatives of the field.

    The library deliberately includes second derivatives and quadratic terms
    the true dynamics do not use. Recovering advection means *not* selecting
    them - sparsity is the claim being tested, not just the fit.

    ``threshold`` is in the units of the coefficients, which here are km/h.
    It has to be set against the expected velocity: at 1e-3 every term in the
    library survives and the "discovery" is a dense regression wearing a
    sparse label.
    """
    import pysindy as ps

    n_t, n_y, n_x = seq.shape
    x = np.transpose(seq, (2, 1, 0))[:, :, :, None]     # (x, y, t, 1)
    spatial_grid = np.zeros((n_x, n_y, 2))
    spatial_grid[:, :, 0] = (np.arange(n_x) * dx_km)[:, None]
    spatial_grid[:, :, 1] = (np.arange(n_y) * dx_km)[None, :]

    # Spectral spatial derivatives, for the reason in least_squares_velocity.
    lib = ps.PDELibrary(
        function_library=ps.PolynomialLibrary(degree=1, include_bias=False),
        derivative_order=derivative_order,
        spatial_grid=spatial_grid,
        include_bias=False,
        differentiation_method=ps.SpectralDerivative,
    )
    model = ps.SINDy(feature_library=lib,
                     optimizer=ps.STLSQ(threshold=threshold, alpha=1e-6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, t=dt_h, feature_names=["R"])

    names = model.get_feature_names()
    coefs = model.coefficients()[0]
    terms = {n.replace(" ", ""): c for n, c in zip(names, coefs)}

    def pick(*candidates):
        for c in candidates:
            if c in terms:
                return terms[c]
        return 0.0

    u = -pick("R_1", "x0_1", "R_x")
    v = -pick("R_2", "x0_2", "R_y")
    active = {n: float(c) for n, c in zip(names, coefs) if abs(c) > 1e-8}
    return {"u": float(u), "v": float(v), "n_active": len(active),
            "terms": active, "feature_names": list(names)}


# --------------------------------------------------------------------------
def run(experiment: str, velocity=(14.0, 5.0), noise: float = 0.05,
        n_links: int = 90, field: str = "smooth") -> dict:
    seq, grid, dt_h = make_sequence(velocity, field=field)
    label = {"clean": f"exact {field} fields",
             "noisy": f"fields + {noise:.0%} multiplicative noise",
             "sampled": f"observed by {n_links} CML paths, IDW back to grid"}[experiment]

    if experiment == "noisy":
        rng = np.random.default_rng(1)
        seq = seq * np.exp(rng.normal(0.0, noise, seq.shape))
    elif experiment == "sampled":
        seq = _cml_sampled(seq, grid, n_links)

    print(f"\n{'=' * 68}\n{experiment}: {label}\n{'=' * 68}")
    print(f"  true velocity            u = {velocity[0]:6.2f}   "
          f"v = {velocity[1]:6.2f}  km/h")

    fd = least_squares_velocity(seq, grid.dx_km, dt_h, spectral=False)
    print(f"  least sq, finite diff    u = {fd['u']:6.2f}   "
          f"v = {fd['v']:6.2f}   R2 = {fd['r2']:.4f}")
    ls = least_squares_velocity(seq, grid.dx_km, dt_h)
    print(f"  least sq, spectral       u = {ls['u']:6.2f}   "
          f"v = {ls['v']:6.2f}   R2 = {ls['r2']:.4f}")

    sy = sindy_velocity(seq, grid.dx_km, dt_h)
    print(f"  SINDy                    u = {sy['u']:6.2f}   "
          f"v = {sy['v']:6.2f}   {sy['n_active']} active term(s)")
    print(f"  SINDy error              du = {sy['u'] - velocity[0]:+6.2f}  "
          f"dv = {sy['v'] - velocity[1]:+6.2f}  km/h")
    if sy["terms"]:
        print("  selected terms:")
        for n, c in sorted(sy["terms"].items(), key=lambda kv: -abs(kv[1])):
            print(f"      {n:<12} {c:+.4f}")

    return {"experiment": experiment, "label": label,
            "true_u": velocity[0], "true_v": velocity[1],
            "field": field,
            "lstsq_findiff": fd, "lstsq": ls, "sindy": {k: v for k, v in sy.items()
                                   if k != "feature_names"}}


def _cml_sampled(seq: np.ndarray, grid: Grid, n_links: int) -> np.ndarray:
    """Observe each frame along CML paths, then interpolate back to the grid.

    This is the field a nowcast actually sees: not the rain, but an
    interpolation of path averages over a sparse network.
    """
    from core.simulation.cml_network import sample_along_paths, synthesize_network
    from core.simulation.reconstruct import idw_path

    net = synthesize_network(grid, n_links=n_links)
    out = []
    for frame in seq:
        path_mean = sample_along_paths(frame, grid, net).mean(axis=1)
        out.append(np.asarray(idw_path(net, path_mean, grid)))
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["clean", "noisy", "sampled"],
                    default="clean")
    ap.add_argument("--u", type=float, default=14.0)
    ap.add_argument("--v", type=float, default=5.0)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--field", choices=["smooth", "rain"], default="smooth",
                    help="smooth: differentiable GRF. rain: intermittent, "
                         "with hard zeros (realistic, and harder)")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    todo = ["clean", "noisy", "sampled"] if args.all else [args.experiment]
    out = [run(e, (args.u, args.v), args.noise, field=args.field)
           for e in todo]

    import json
    path = RESULTS / "discover_advection.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")

    if len(out) > 1:
        fig_path = RESULTS / "discover_advection.png"
        figure(out, fig_path)
        print(f"wrote {fig_path.relative_to(REPO_ROOT)}")



# --------------------------------------------------------------------------
def figure(runs: list[dict], path: Path) -> None:
    """Recovered velocity per experiment, against the truth."""
    import matplotlib.pyplot as plt
    from core import viz_style as vs
    vs.use_style()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    names = [r["experiment"] for r in runs]
    y = np.arange(len(names))
    h = 0.26

    for ax, comp, truth in zip(axes, ("u", "v"),
                               (runs[0]["true_u"], runs[0]["true_v"])):
        fd = [r["lstsq_findiff"][comp] for r in runs]
        ls = [r["lstsq"][comp] for r in runs]
        sy = [r["sindy"][comp] for r in runs]

        ax.barh(y - h, fd, h, color=vs.INK_MUTED, label="least sq, finite diff")
        ax.barh(y, ls, h, color="#eb6834", label="least sq, spectral")
        ax.barh(y + h, sy, h, color="#2a78d6", label="SINDy")
        ax.axvline(truth, color=vs.STATUS_GOOD, lw=2, ls="--",
                   label=f"true {comp} = {truth:g} km/h")

        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(f"recovered {comp} (km h$^{{-1}}$)")
        ax.set_title(f"{comp} component", fontsize=10.5)
        ax.grid(axis="y", visible=False)
        if comp == "u":
            ax.legend(fontsize=7.8, loc="lower right")

    fig.suptitle("SINDy on rainfall-field dynamics: recovering advection "
                 "as the observation degrades",
                 fontsize=12.5, color=vs.INK_PRIMARY, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
