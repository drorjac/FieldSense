"""
Run the full demonstration: three rainfall fields, a CML network over them,
the sensor chain, and the reconstructed fields.

    python projects/simulation/regimes/src/run_demo.py

Writes six figures and a summary table to ``projects/simulation/regimes/results/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from core.data_paths import REPO_ROOT  # noqa: E402
sys.path.insert(0, str(HERE))        # this project's modules

import figures                                            # noqa: E402
from core import viz_style as vs  # noqa: E402
from core.simulation import moving_fields as mf  # noqa: E402
from core.simulation.cml_network import (SensorConfig, forward_model,  # noqa: E402
                                         path_averaging_bias, synthesize_network)
from core.simulation.rain_fields import (MODELS, FrontalBandField, Grid,  # noqa: E402
                                         field_stats)
from core.simulation.reconstruct import decompose, idw_path, score  # noqa: E402

RESULTS = HERE.parent / "results"
SWEEP_DENSITIES = (25, 50, 90, 160, 280)
SWEEP_REALIZATIONS = 6
MOVING_TAUS_MIN = (None, 120, 60)       # frozen, then evolving with that e-folding time
MOVING_DT_MIN, MOVING_STEPS = 10, 7     # 0 to 60 min


def build_fields(grid):
    fields, stats = {}, {}
    for m in MODELS:
        fields[m.key] = m.build(grid)
        stats[m.key] = field_stats(fields[m.key], grid)
    return fields, stats


def run_sensors(fields, grid, net, cfg):
    return {m.key: forward_model(fields[m.key], grid, net, cfg) for m in MODELS}


def run_reconstruction(fields, grid, net, results):
    out = {}
    for m in MODELS:
        res = results[m.key]
        ideal = idw_path(net, res["R_path_true"], grid)
        full = idw_path(net, res["R_retrieved"], grid)
        out[m.key] = {
            "ideal": ideal,
            "full": full,
            "score_ideal": score(fields[m.key], ideal),
            "score_full": score(fields[m.key], full),
            "decomp": decompose(fields[m.key], ideal, full),
        }
    return out


def run_density_sweep(fields, grid, cfg, densities=SWEEP_DENSITIES,
                      n_realizations=SWEEP_REALIZATIONS):
    """Repeat the whole chain across network densities.

    Averaged over several independent topologies per density. A single
    realization is dominated by whether its particular links happen to fall on
    the rain cells, which swamps the density trend entirely.
    """
    sweep = {m.key: {"n_links": list(densities), "rmse_sampling": [],
                     "rmse_total": [], "sampling_spread": []}
             for m in MODELS}

    for n in densities:
        per_model = {m.key: {"samp": [], "tot": []} for m in MODELS}
        for r in range(n_realizations):
            # Node count scales with the link target so topology stays plausible.
            net = synthesize_network(grid, n_links=n,
                                     n_nodes=max(12, int(n * 0.55)),
                                     seed=1000 + 17 * r)
            for m in MODELS:
                res = forward_model(fields[m.key], grid, net, cfg)
                ideal = idw_path(net, res["R_path_true"], grid)
                full = idw_path(net, res["R_retrieved"], grid)
                d = decompose(fields[m.key], ideal, full)
                per_model[m.key]["samp"].append(d["rmse_sampling"])
                per_model[m.key]["tot"].append(d["rmse_total"])

        for m in MODELS:
            samp = np.array(per_model[m.key]["samp"])
            tot = np.array(per_model[m.key]["tot"])
            sweep[m.key]["rmse_sampling"].append(float(samp.mean()))
            sweep[m.key]["rmse_total"].append(float(tot.mean()))
            sweep[m.key]["sampling_spread"].append(float(samp.std()))
    return sweep


def run_moving(grid, taus=MOVING_TAUS_MIN):
    """Each regime moving at its own velocity for an hour, frozen or evolving.

    Returns ``{key: {tau: MovingSequence}}`` and prints how much of the field
    at +30 and +60 min the true motion alone predicts.
    """
    out = {}
    print("\nmoving fields: correlation of truth with the field moved at the true velocity")
    print(f"  {'':<18}" + "".join(f"{'frozen' if t is None else f'tau {t} min':>14}" for t in taus))
    for m in MODELS:
        if isinstance(m, FrontalBandField):
            # start the band upstream by half the hour's travel, so it
            # crosses the middle of the domain at +30 min instead of leaving
            half_km = np.hypot(*m.advection_kmh) * (MOVING_STEPS - 1) * MOVING_DT_MIN / 120
            m = dataclasses.replace(m, offset_km=-half_km)
        out[m.key] = {t: mf.sequence(m, grid, MOVING_STEPS, MOVING_DT_MIN,
                                     evolve_tau_min=t, seed=1) for t in taus}
        cells = [f"{out[m.key][t].predictability(3):.2f} / {out[m.key][t].predictability(6):.2f}"
                 for t in taus]
        print(f"  {m.name:<18}" + "".join(f"{c:>14}" for c in cells) + "   (+30 / +60 min)")
    return out


def print_summary(net, stats, results, recon, cfg):
    line = "-" * 78
    print(f"\n{line}\nFIELD STATISTICS\n{line}")
    print(f"{'model':<18}{'wet area':>10}{'mean':>9}{'wet mean':>10}"
          f"{'peak':>8}{'decorr':>9}{'top 5% of area':>16}")
    print(f"{'':<18}{'(%)':>10}{'(mm/h)':>9}{'(mm/h)':>10}{'(mm/h)':>8}"
          f"{'(km)':>9}{'holds (% water)':>16}")
    for m in MODELS:
        s = stats[m.key]
        print(f"{m.name:<18}{s['war']*100:>10.1f}{s['imf']:>9.2f}{s['cmf']:>10.2f}"
              f"{s['max']:>8.1f}{s['decorrelation_km']:>9.2f}"
              f"{s['frac_flux_top5pct']*100:>16.1f}")

    print(f"\n{line}\nCML SENSOR CHAIN  ({net.n_links} links, "
          f"{net.summary()['total_path_km']:.0f} km of path)\n{line}")
    print(f"{'model':<18}{'wet links':>11}{'path CV':>9}{'Jensen bias':>13}"
          f"{'  (a>1)':>9}{'  (a<1)':>9}{'link RMSE':>11}")
    for m in MODELS:
        res = results[m.key]
        b = path_averaging_bias(res, net)
        Rt = res["R_path_true"]
        wet = Rt > 0.5
        cv = (res["samples"].std(axis=1) / np.maximum(Rt, 1e-9))[wet]
        err = res["R_retrieved"] - Rt
        print(f"{m.name:<18}{int(wet.sum()):>11d}{np.median(cv):>9.2f}"
              f"{b['median_rel_bias']*100:>12.1f}%"
              f"{b['median_rel_bias_alpha_gt1']*100:>8.1f}%"
              f"{b['median_rel_bias_alpha_lt1']*100:>8.1f}%"
              f"{np.sqrt((err**2).mean()):>11.2f}")

    print(f"\n{line}\nFIELD RECONSTRUCTION (path-aware IDW)\n{line}")
    print(f"{'model':<18}{'RMSE geom':>11}{'RMSE full':>11}{'sensor':>9}"
          f"{'corr':>8}{'peak kept':>11}{'wet area':>11}")
    print(f"{'':<18}{'(mm/h)':>11}{'(mm/h)':>11}{'share':>9}{'':>8}"
          f"{'(%)':>11}{'est. (%)':>11}")
    for m in MODELS:
        d = recon[m.key]["decomp"]
        s = recon[m.key]["score_full"]
        sensor_pct = (d["rmse_sensor_increment"] / d["rmse_total"] * 100
                      if d["rmse_total"] > 0 else float("nan"))
        print(f"{m.name:<18}{d['rmse_sampling']:>11.2f}{d['rmse_total']:>11.2f}"
              f"{sensor_pct:>8.1f}%{s['corr']:>8.2f}"
              f"{s['peak_ratio']*100:>11.1f}{s['war_est']*100:>11.1f}")
    print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--links", type=int, default=90,
                    help="links in the operating network (default 90)")
    ap.add_argument("--grid", type=int, default=128,
                    help="grid cells per side (default 128)")
    ap.add_argument("--dx", type=float, default=0.25,
                    help="grid spacing in km (default 0.25)")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the network-density sweep (much faster)")
    args = ap.parse_args()

    vs.use_style()
    RESULTS.mkdir(parents=True, exist_ok=True)

    grid = Grid(n=args.grid, dx_km=args.dx)
    cfg = SensorConfig()

    print(f"domain {grid.size_km:g} x {grid.size_km:g} km "
          f"at {grid.dx_km:g} km ({grid.n} x {grid.n} cells)")

    fields, stats = build_fields(grid)
    net = synthesize_network(grid, n_links=args.links)
    results = run_sensors(fields, grid, net, cfg)
    recon = run_reconstruction(fields, grid, net, results)

    print_summary(net, stats, results, recon, cfg)

    print("\nwriting figures...")
    figures.fig_fields(MODELS, fields, stats, grid, RESULTS / "fig1_fields.png")
    figures.fig_proportions(MODELS, fields, stats, grid,
                            RESULTS / "fig2_proportions.png")
    figures.fig_network(MODELS, fields, results, net, grid,
                        RESULTS / "fig3_network.png")
    figures.fig_sensor_physics(MODELS, results, net, cfg,
                               RESULTS / "fig4_sensor_physics.png")
    figures.fig_reconstruction(MODELS, fields, recon, grid,
                               RESULTS / "fig5_reconstruction.png")

    if not args.no_sweep:
        print(f"running network-density sweep "
          f"({len(SWEEP_DENSITIES)} densities x {SWEEP_REALIZATIONS} topologies)...")
        sweep = run_density_sweep(fields, grid, cfg)
        figures.fig_error_budget(MODELS, recon, sweep,
                                 RESULTS / "fig6_error_budget.png",
                                 n_links=args.links)

    moving = run_moving(grid)
    figures.fig_moving(MODELS, moving, grid, RESULTS / "fig7_moving_fields.png")

    for p in sorted(RESULTS.glob("*.png")):
        print(f"  {p.relative_to(REPO_ROOT)}  ({p.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
