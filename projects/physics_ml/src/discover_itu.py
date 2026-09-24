"""
Can symbolic regression rediscover the ITU-R power law from CML data?

ITU-R P.838-3 says specific attenuation is a power law in rain rate::

    gamma [dB/km] = k * R**alpha

with ``k`` and ``alpha`` tabulated per frequency and polarization. Every
retrieval in this repository inverts that relation. This asks the opposite
question - given attenuation and rain rate, does PySR find the power law, and
does it recover the tabulated coefficients?

Three experiments, deliberately in this order:

``synthetic``
    ITU-generated pairs with noise, so the answer is known exactly. If PySR
    cannot recover ``k`` and ``alpha`` here, nothing downstream means
    anything.
``reference``
    Real OpenMRG attenuation against the *reference* rain rate OpenSense
    publishes. This is partly circular - that reference was itself retrieved
    through an assumed power law - which makes it a useful diagnostic rather
    than a physical test: it reverse-engineers the coefficients the reference
    retrieval used.
``gauge``
    Real OpenMRG attenuation against *rain gauges*. Not circular, and
    therefore the honest test, but noisier: a gauge is a point and a link is
    a path average, and the two disagree for reasons that have nothing to do
    with the power law.

    python -m discover_itu --experiment synthetic
    python -m discover_itu --experiment reference --band 23
    python -m discover_itu --all
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "projects/opensense_pipeline/src"))

from core.itu_p838 import get_k_alpha            # noqa: E402

RESULTS = HERE.parent / "results"

# Narrow frequency bands so k and alpha are near-constant within one. These
# are the bands the OpenMRG network actually populates - it clusters hard
# around 30 GHz rather than the 23 GHz a generic CML network might suggest.
BANDS = {15: (13.0, 17.0), 20: (18.0, 22.0), 30: (28.0, 32.0),
         35: (33.0, 36.5), 38: (37.0, 39.5)}


def _per_link(da) -> np.ndarray:
    """Collapse a per-sublink coordinate to one value per link.

    Dimension order is not consistent between these files - OpenMRG's example
    stores frequency as (sublink_id, cml_id) while length is (cml_id,) - so
    the sublink axis is selected by name, never by position.
    """
    import xarray as xr

    if isinstance(da, xr.DataArray) and "sublink_id" in da.dims:
        da = da.isel(sublink_id=0, drop=True)
    return np.asarray(da)


def _regressor(niterations: int, seed: int = 0):
    """A PySR run configured to be able to express a power law.

    ``pow`` has to be in the operator set or the search cannot reach
    ``k * R**alpha`` at all - it will spend its budget on polynomial
    approximations instead, which fit acceptably and explain nothing.
    """
    from pysr import PySRRegressor

    return PySRRegressor(
        niterations=niterations,
        binary_operators=["*", "+", "-", "^"],
        unary_operators=["log", "exp"],
        constraints={"^": (-1, 3)},      # keep exponents simple
        model_selection="best",
        maxsize=14,
        progress=False,
        verbosity=0,
        deterministic=True,
        parallelism="serial",
        random_state=seed,
        temp_equation_file=True,
    )


def effective_power_law(R: np.ndarray, pred: np.ndarray) -> dict:
    """The power law a candidate equation is *equivalent to*.

    PySR returns expressions in whatever algebraic form the search landed on -
    ``(0.1189 * x0) ^ 0.96`` is a power law with k = 0.1295, but that is not
    readable off the string. Fitting the candidate's own predictions in
    log-log space recovers the (k, alpha) it actually implements, which is
    what can be compared against the ITU table.
    """
    ok = (R > 0) & (pred > 0) & np.isfinite(pred)
    if ok.sum() < 10:
        return {"k": float("nan"), "alpha": float("nan"), "r2": float("nan")}
    lr, lp = np.log(R[ok]), np.log(pred[ok])
    slope, intercept = np.polyfit(lr, lp, 1)
    resid = lp - (slope * lr + intercept)
    ss = 1.0 - resid.var() / lp.var() if lp.var() > 0 else float("nan")
    return {"k": float(np.exp(intercept)), "alpha": float(slope),
            "r2": float(ss)}


def fit_power_law(R: np.ndarray, gamma: np.ndarray) -> dict:
    """Least-squares power law through log-log space, as the reference answer.

    ``log gamma = log k + alpha log R`` is linear, so this is the estimate
    symbolic regression has to match or beat.
    """
    ok = (R > 0) & (gamma > 0) & np.isfinite(R) & np.isfinite(gamma)
    slope, intercept = np.polyfit(np.log(R[ok]), np.log(gamma[ok]), 1)
    return {"k": float(np.exp(intercept)), "alpha": float(slope),
            "n": int(ok.sum())}


# --------------------------------------------------------------------------
def data_synthetic(freq_ghz: float = 7.0, n: int = 1500,
                   noise_db_km: float = 0.05, seed: int = 0) -> tuple:
    """ITU pairs with additive measurement noise. Truth is known."""
    k, alpha = get_k_alpha(freq_ghz, "vertical")
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.2, 40.0, n)
    gamma = k * R**alpha + rng.normal(0.0, noise_db_km, n)
    keep = gamma > 0
    return R[keep], gamma[keep], {"k": k, "alpha": alpha}


def _openmrg_pairs(band: int, target: str, min_rain: float = 0.5):
    """Specific attenuation and rain rate from the OpenMRG example subset.

    Returns ``(R, gamma)`` for wet samples on links inside one frequency
    band. ``gamma`` is baseline-subtracted rain attenuation divided by path
    length, which is the quantity ITU-R P.838-3 actually describes.
    """
    import example_data
    import conventions as cv
    from retrieval import RetrievalConfig, baseline_from_dry, wet_dry_rolling_std

    ds = example_data.load("openmrg", "8d")
    cml = ds["cml"].transpose("time", "cml_id", "sublink_id")

    lo, hi = BANDS[band]
    freq_da = ds["cml"].frequency
    freq = cv.to_ghz(freq_da)
    if freq.ndim > 1:
        axis = list(freq_da.dims).index("sublink_id")
        freq = np.take(freq, 0, axis=axis)
    sel = (freq >= lo) & (freq <= hi)
    if sel.sum() == 0:
        raise SystemExit(f"no links in the {band} GHz band")

    loss = np.asarray(cml.tsl - cml.rsl, dtype=float)[:, sel, :]
    n_t, n_c, n_s = loss.shape
    flat = loss.reshape(n_t, n_c * n_s)

    cfg = RetrievalConfig.for_interval(10.0)
    wet = wet_dry_rolling_std(flat, cfg.wet_window, cfg.wet_threshold_db)
    base = baseline_from_dry(pd.DataFrame(flat).ffill().bfill().to_numpy(),
                             wet, cfg.baseline_window)
    a_rain = np.clip(flat - base, 0.0, None).reshape(n_t, n_c, n_s)

    length = cv.to_km(_per_link(ds["cml"].length))[sel]
    gamma = np.nanmean(a_rain, axis=2) / length[None, :]      # dB/km

    if target == "reference":
        R = np.nanmean(np.asarray(cml.R)[:, sel, :], axis=2)
    else:
        R = _gauge_rain_at_links(ds, sel, cml)

    g, r = gamma.ravel(), R.ravel()
    ok = np.isfinite(g) & np.isfinite(r) & (r >= min_rain) & (g > 0.01)
    return r[ok], g[ok], {"band_ghz": band, "n_links": int(sel.sum())}


def _gauge_rain_at_links(ds, sel, cml) -> np.ndarray:
    """Nearest municipal gauge rain rate, broadcast onto each link and time."""
    g = ds["gauge_municipal"]
    interval = float(np.diff(g.time.values[:2])
                     .astype("timedelta64[s]").astype(float)[0])
    g_rate = np.asarray(g.rainfall_amount.transpose("time", "id")) \
        * (3600.0 / interval)

    # gauges are 1-minute, CMLs 10-second: index each CML step to its gauge
    idx = np.searchsorted(g.time.values, cml.time.values).clip(
        0, g_rate.shape[0] - 1)
    g_rate = g_rate[idx]

    lx, ly = np.asarray(ds["cml"].x)[sel], np.asarray(ds["cml"].y)[sel]
    gx, gy = np.asarray(g.x), np.asarray(g.y)
    nearest = [int(np.argmin((gx - x) ** 2 + (gy - y) ** 2))
               for x, y in zip(lx, ly)]
    return g_rate[:, nearest]


# --------------------------------------------------------------------------
def run(experiment: str, band: int, niterations: int,
        max_points: int = 3000, noise: float = 0.02) -> dict:
    if experiment == "synthetic":
        R, gamma, truth = data_synthetic(freq_ghz=float(band),
                                         noise_db_km=noise)
        label = (f"synthetic ITU pairs, {band} GHz vertical, "
                 f"noise {noise} dB/km")
    else:
        R, gamma, meta = _openmrg_pairs(band, experiment)
        k, alpha = get_k_alpha(float(band), "vertical")
        truth = {"k": k, "alpha": alpha}
        label = (f"OpenMRG {band} GHz band vs "
                 f"{'reference retrieval' if experiment == 'reference' else 'rain gauges'}"
                 f" ({meta['n_links']} links)")

    if R.size > max_points:
        idx = np.random.default_rng(0).choice(R.size, max_points, replace=False)
        R, gamma = R[idx], gamma[idx]

    fit = fit_power_law(R, gamma)
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"  samples                     {R.size:,}")
    print(f"  ITU-R P.838-3 tabulated     k = {truth['k']:.4f}   "
          f"alpha = {truth['alpha']:.3f}")
    print(f"  log-log least squares       k = {fit['k']:.4f}   "
          f"alpha = {fit['alpha']:.3f}")

    print("  running PySR ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _regressor(niterations)
        model.fit(R.reshape(-1, 1), gamma)

    # Every Pareto candidate, reduced to the power law it is equivalent to.
    print("\n  Pareto front, each reduced to its effective k * R^alpha:")
    print(f"    {'cplx':>4} {'k':>9} {'alpha':>7} {'fit':>7}  equation")
    rows = []
    for _, row in model.equations_.iterrows():
        cplx = int(row["complexity"])
        if cplx > 12:
            continue
        pred = np.asarray(model.predict(R.reshape(-1, 1), index=row.name),
                          dtype=float)
        eff = effective_power_law(R, pred)
        err = (abs(eff["alpha"] - truth["alpha"])
               if np.isfinite(eff["alpha"]) else np.inf)
        rows.append({"complexity": cplx, "equation": str(row["equation"]),
                     "loss": float(row["loss"]), **eff, "alpha_err": err})
        mark = ""
        print(f"    {cplx:>4} {eff['k']:>9.4f} {eff['alpha']:>7.3f} "
              f"{eff['r2']:>7.3f}  {str(row['equation'])[:52]}{mark}")

    closest = min((r for r in rows if np.isfinite(r["alpha_err"])),
                  key=lambda r: r["alpha_err"], default=None)
    if closest:
        print(f"\n  closest to ITU: complexity {closest['complexity']}  ->  "
              f"k = {closest['k']:.4f} (ITU {truth['k']:.4f}),  "
              f"alpha = {closest['alpha']:.3f} (ITU {truth['alpha']:.3f})")
        print(f"                  k off by {100*(closest['k']/truth['k']-1):+.1f}%, "
              f"alpha off by {closest['alpha']-truth['alpha']:+.3f}")

    best = model.get_best()
    return {"experiment": experiment, "label": label, "truth": truth,
            "loglog": fit, "pareto": rows,
            "pysr_best": str(best["equation"]),
            "pysr_best_complexity": int(best["complexity"]),
            "closest_to_itu": closest}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["synthetic", "reference", "gauge"],
                    default="synthetic")
    ap.add_argument("--band", type=int, default=30, choices=sorted(BANDS))
    ap.add_argument("--niterations", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.02,
                    help="synthetic measurement noise, dB/km")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    todo = (["synthetic", "reference", "gauge"] if args.all
            else [args.experiment])
    if args.all:
        print(f"running all three experiments in the {args.band} GHz band")
    out = [run(e, args.band, args.niterations, noise=args.noise)
           for e in todo]

    import json
    path = RESULTS / "discover_itu.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")

    if len(out) > 1:
        fig_path = RESULTS / "discover_itu.png"
        figure(out, fig_path)
        print(f"wrote {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
def figure(runs: list[dict], path: Path) -> None:
    """Recovered power laws against the ITU table, one panel per experiment."""
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO_ROOT / "projects/rainfall_field_sim/src"))
    import viz_style as vs
    vs.use_style()

    fig, axes = plt.subplots(1, len(runs) + 1,
                             figsize=(4.4 * (len(runs) + 1), 4.4))
    axes = np.atleast_1d(axes)

    R = np.logspace(np.log10(0.5), np.log10(40), 200)
    for ax, run_ in zip(axes, runs):
        t = run_["truth"]
        ax.plot(R, t["k"] * R ** t["alpha"], lw=2.4, color=vs.INK_SECONDARY,
                label=f"ITU-R  k={t['k']:.3f}, α={t['alpha']:.3f}")

        ll = run_["loglog"]
        ax.plot(R, ll["k"] * R ** ll["alpha"], lw=1.8, ls="--",
                color="#eb6834",
                label=f"log-log LS  k={ll['k']:.3f}, α={ll['alpha']:.3f}")

        c = run_.get("closest_to_itu")
        if c:
            ax.plot(R, c["k"] * R ** c["alpha"], lw=1.8, color="#2a78d6",
                    label=f"PySR (c={c['complexity']})  "
                          f"k={c['k']:.3f}, α={c['alpha']:.3f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rain rate R (mm h$^{-1}$)")
        ax.set_ylabel("specific attenuation γ (dB km$^{-1}$)")
        ax.set_title(run_["experiment"], fontsize=10.5)
        ax.legend(fontsize=7.2, loc="upper left")
        ax.grid(alpha=0.3, which="both")

    # summary: how close each experiment got on alpha
    ax = axes[-1]
    names = [r["experiment"] for r in runs]
    errs = [abs(r["closest_to_itu"]["alpha"] - r["truth"]["alpha"])
            if r.get("closest_to_itu") else np.nan for r in runs]
    colors = ["#1baf7a" if e < 0.05 else "#eda100" if e < 0.15 else "#d03b3b"
              for e in errs]
    ax.barh(np.arange(len(names)), errs, color=colors, height=0.55)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("|α recovered − α ITU|")
    ax.set_title("Exponent recovery error", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    for i, e in enumerate(errs):
        ax.text(e + max(errs) * 0.02, i, f"{e:.3f}", va="center", fontsize=8.5,
                color=vs.INK_SECONDARY)

    fig.suptitle("Recovering ITU-R P.838-3 from data with symbolic regression",
                 fontsize=12.5, color=vs.INK_PRIMARY, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
