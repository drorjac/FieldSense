"""
Run the whole pipeline: ingest -> benchmark -> merge -> maps -> validation.

    python projects/opensense_pipeline/src/run_pipeline.py            # everything
    python projects/opensense_pipeline/src/run_pipeline.py --skip-benchmark
    python projects/opensense_pipeline/src/run_pipeline.py --dataset openmrg

Writes figures and a results table to ``projects/opensense_pipeline/results/``.
Assumes ``fetch.py`` has already pulled the archives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
# Order matters: this project's modules must win over same-named modules in
# rainfall_field_sim (both have plotting helpers), so HERE is inserted last
# and therefore searched first.
sys.path.insert(0, str(HERE))

import plots                                                # noqa: E402
import ingest_openmrg as omrg                               # noqa: E402
import ingest_openrainer as orain                           # noqa: E402
from core import viz_style as vs  # noqa: E402
from merging import API_TAG, METHODS, run, score            # noqa: E402
from synthetic_benchmark import run_benchmark               # noqa: E402

RESULTS = HERE.parent / "results"


def peak_timestep(ds_cml: xr.Dataset) -> int:
    """The timestep with the most rain on the link network."""
    total = np.nansum(np.asarray(ds_cml.R), axis=1)
    return int(np.argmax(total))


def sample_at_gauges(field: np.ndarray, ds_rad, ds_gauge) -> np.ndarray:
    """Nearest-pixel value of a gridded field at each gauge."""
    xg = np.asarray(ds_rad.x_grid)
    yg = np.asarray(ds_rad.y_grid)
    out = []
    for x, y in zip(np.asarray(ds_gauge.x), np.asarray(ds_gauge.y)):
        i, j = np.unravel_index(np.argmin((xg - x) ** 2 + (yg - y) ** 2),
                                xg.shape)
        out.append(field[i, j])
    return np.asarray(out)


def apply_methods(ds_rad, ds_cml, ds_gauge, t_index: int,
                  hold_out_gauges: bool = True) -> dict:
    """Run every method at one timestep.

    ``hold_out_gauges`` keeps the gauges out of the merge so they stay an
    independent check. Feeding them in and then scoring against them would
    measure how well each method reproduces its own input.
    """
    da_rad = ds_rad.R.isel(time=t_index)
    da_cml = ds_cml.R.isel(time=t_index)
    da_gauge = ds_gauge.R.isel(time=t_index)

    # Links with no retrieval at this step carry no information.
    good = np.isfinite(np.asarray(da_cml))
    da_cml = da_cml.isel(cml_id=good)

    fields = {}
    for method in METHODS:
        gauges_in = None if hold_out_gauges else da_gauge
        fields[method.key] = run(method, da_rad, da_cml, gauges_in)
    return fields, da_gauge


def coarsen_radar(ds_rad, max_cells: int):
    """Coarsen a radar grid until it is under ``max_cells``.

    Block kriging costs scale with grid cells times links times discretization
    points. OpenRainER's 160 x 285 grid over 151 links takes ~100 s per
    timestep per kriging method, which makes a 20-timestep validation a
    35-minute job - and the extra resolution buys nothing, because a 151-link
    network cannot constrain a 1 km field over a 300 km domain anyway.

    The same coarsening is applied to every method, so the ranking is
    unaffected. The published maps are still produced at full resolution.
    """
    cells = ds_rad.sizes["y"] * ds_rad.sizes["x"]
    if cells <= max_cells:
        return ds_rad, 1
    factor = int(np.ceil(np.sqrt(cells / max_cells)))
    out = ds_rad.coarsen(y=factor, x=factor, boundary="trim").mean()
    # coarsen drops the 2-D geometry coords' alignment; rebuild them
    out.coords["x_grid"] = ds_rad.x_grid.coarsen(
        y=factor, x=factor, boundary="trim").mean()
    out.coords["y_grid"] = ds_rad.y_grid.coarsen(
        y=factor, x=factor, boundary="trim").mean()
    return out, factor


def validate_over_time(ds_rad, ds_cml, ds_gauge, n_times: int = 20,
                       max_cells: int = 6000, select_by: str = "cml") -> list[dict]:
    """Score every method against the gauges, pooled over several timesteps.

    A single timestep gives one value per gauge - ten numbers for OpenMRG -
    which is far too few to separate methods; the correlation of a
    ten-point sample swings wildly. Pooling the wettest ``n_times`` steps
    gives a few hundred paired values and a stable ranking.

    The gauges are held out of every merge, so this stays an independent
    check rather than a measure of how well each method reproduces its own
    input.

    ``select_by`` picks what "wettest" is judged by. ``cml`` (the default,
    and what the published results use) ranks timesteps by the CML
    network's own retrieved rain - so two retrievals get scored on
    different timesteps, and always where the CML reads highest. To compare
    retrievals, use ``gauge``: the ranking then comes from the held-out
    reference and is identical for every retrieval.
    """
    ds_rad, factor = coarsen_radar(ds_rad, max_cells)
    if factor > 1:
        print(f"    grid coarsened {factor}x for validation -> "
              f"{ds_rad.sizes['y']} x {ds_rad.sizes['x']}")

    if select_by not in ("cml", "gauge"):
        raise ValueError(f"select_by must be 'cml' or 'gauge', got {select_by!r}")
    source = ds_cml if select_by == "cml" else ds_gauge
    wetness = np.nansum(np.asarray(source.R), axis=1)
    order = np.argsort(wetness)[::-1]
    chosen = sorted(int(t) for t in order[:n_times] if wetness[order[0]] > 0)

    pooled = {m.key: {"est": [], "obs": []} for m in METHODS}
    for t in chosen:
        fields, _ = apply_methods(ds_rad, ds_cml, ds_gauge, t)
        obs = np.asarray(ds_gauge.R.isel(time=t))
        for method in METHODS:
            est = sample_at_gauges(np.asarray(fields[method.key]), ds_rad,
                                   ds_gauge)
            pooled[method.key]["est"].append(est)
            pooled[method.key]["obs"].append(obs)

    rows = []
    for method in METHODS:
        est = np.concatenate(pooled[method.key]["est"])
        obs = np.concatenate(pooled[method.key]["obs"])
        rows.append({"method": method.label, "method_key": method.key,
                     "family": method.family, "n_timesteps": len(chosen),
                     **score(obs, est)})
    return rows, pooled, chosen


# --------------------------------------------------------------------------
def run_dataset(name: str, module, event_key: str,
                n_val: int = 20, max_cells: int = 6000,
                source: str = "raw", offline: bool = False,
                retrieval: str = "default", select_by: str = "cml") -> dict:
    """Ingest one event, apply every method, write maps and validation.

    ``source`` picks where the data comes from - the full local Zenodo
    archives (``raw``) or the curated example subsets (``example``). Both
    produce the same (radar, cml, gauge) triple, so nothing downstream cares.

    ``retrieval`` picks the CML chain: ``default`` or ``improved``
    (``retrieval.retrieve_improved``). Only the CML input changes; radar,
    gauges, merge methods and the validation are identical.
    """
    # Results from the two sources, and the two retrievals, must not
    # overwrite each other.
    tag = ("" if source == "raw" else f"_{source}") + \
          ("" if retrieval == "default" else f"_{retrieval}") + \
          ("" if select_by == "cml" else f"_by{select_by}") + API_TAG

    if source == "example":
        key = name.lower()
        _, subset, _, label = EXAMPLE_EVENTS[key]
        print(f"\n{'=' * 74}\n{name}: {label}\n"
              f"  curated example subset '{subset}'\n{'=' * 74}")
        ds_rad, ds_cml, ds_gauge = build_from_example(key, offline=offline,
                                                      retrieval=retrieval)
        event_key_used, event_label = subset, label
    else:
        event = module.EVENTS[event_key]
        print(f"\n{'=' * 74}\n{name}: {event.label}\n  {event.note}\n{'=' * 74}")
        # The ingest modules already emit gauges on an "id" dimension carrying
        # x/y and lon/lat, which is what mergeplg and poligrain need.
        ds_rad, ds_cml, ds_gauge = module.build_event(event, retrieval=retrieval)
        event_key_used, event_label = event.key, event.label

    t = peak_timestep(ds_cml)
    stamp = str(ds_cml.time.values[t])[:16].replace("T", " ")
    print(f"  peak timestep {t}: {stamp} UTC")
    print(f"  radar {dict(ds_rad.sizes)}  cml {ds_cml.sizes['cml_id']} links  "
          f"gauges {ds_gauge.sizes['id']}")

    plots.fig_sensors(
        ds_rad, ds_cml, ds_gauge, t,
        f"{name} - {event_label}, {stamp} UTC: three sensor geometries",
        RESULTS / f"{name.lower()}_1_sensors{tag}.png")

    fields, _ = apply_methods(ds_rad, ds_cml, ds_gauge, t)
    labels = {m.key: m.label for m in METHODS}
    plots.fig_merged_maps(
        ds_rad, fields, labels,
        f"{name} - {event_label}, {stamp} UTC: rainfall maps by method",
        RESULTS / f"{name.lower()}_2_maps{tag}.png")

    print(f"  validating against gauges over the {n_val} wettest timesteps")
    rows, pooled, chosen = validate_over_time(ds_rad, ds_cml, ds_gauge,
                                              n_val, max_cells, select_by)
    label_map = {m.key: {"label": m.label, "family": m.family} for m in METHODS}
    plots.fig_gauge_validation(
        {k: np.concatenate(v["est"]) for k, v in pooled.items()},
        label_map, np.concatenate(pooled[METHODS[0].key]["obs"]),
        RESULTS / f"{name.lower()}_3_gauge_validation{tag}.png",
        f"{name}: methods vs held-out gauges, "
        f"{len(chosen)} wettest timesteps")

    n_pairs = rows[0]["n_valid"]
    print(f"\n  {'method':38s} {'RMSE':>8} {'bias':>8} {'corr':>7}   "
          f"(n={n_pairs} gauge-timesteps)")
    for r in sorted(rows, key=lambda z: (np.isnan(z['rmse']), z["rmse"])):
        print(f"  {r['method']:38s} {r['rmse']:8.2f} {r['bias']:+8.2f} "
              f"{r['corr']:7.3f}")

    coverage = coverage_analysis(pooled, ds_cml, ds_gauge, len(chosen))
    if coverage:
        plots.fig_coverage(coverage, RESULTS / f"{name.lower()}_4_coverage{tag}.png",
                           f"{name}: does merge gain depend on proximity "
                           f"to the link network?")
        dist = distance_to_network(ds_cml, np.asarray(ds_gauge.x),
                                   np.asarray(ds_gauge.y))
        print(f"\n  gauge distance to nearest link: median "
              f"{np.median(dist):.1f} km, max {dist.max():.1f} km")
        bands = list(dict.fromkeys(r["band"] for r in coverage))
        print(f"  {'band':>12} {'n':>6}  best method (RMSE)"
              f"{'':>18} radar-only RMSE")
        for b in bands:
            sel = [r for r in coverage if r["band"] == b]
            best = min(sel, key=lambda z: z["rmse"])
            radar = next(r for r in sel if r["method_key"] == "radar_only")
            print(f"  {b:>12} {best['n_gauges']:6d}  {best['method']:34s} "
                  f"{best['rmse']:6.2f}   {radar['rmse']:6.2f}")

    return {"dataset": name, "event": event_key_used,
            "label": event_label, "retrieval": retrieval,
            "val_select": select_by,
            "val_timesteps": [str(ds_cml.time.values[t])[:16] for t in chosen],
            "timestep": stamp, "gauge_validation": rows,
            "coverage_analysis": coverage}


# --------------------------------------------------------------------------
# the example-subset source
# --------------------------------------------------------------------------
EXAMPLE_EVENTS = {
    "openmrg": ("openmrg", "8d", "gauge_municipal",
                "OpenMRG 8-day subset, 2015-07-22 .. 07-29"),
    "openrainer": ("openrainer", "8d", "gauge",
                   "OpenRainER 8-day subset"),
}

# The common time step per example dataset: the coarsest native step of its
# three sensors, so no reference is upsampled into gaps.
EXAMPLE_STEP = {"openmrg": "5min", "openrainer": "15min"}


def build_from_example(dataset: str, resample: str | None = None,
                       offline: bool = False, retrieval: str = "default") -> tuple:
    """(radar, cml, gauge) from a curated example subset.

    The same triple ``ingest_*.build_event`` returns from the full Zenodo
    archives, so everything downstream is unchanged. The CML retrieval is the
    shared chain in ``retrieval.py``, run on the subset's own tsl/rsl.

    Radar needs no Z-R step: ``example_data`` returns it as a rate on a
    projected grid. (OpenRainER's subset ships 15-minute accumulations under
    the name ``R``; before that conversion moved into ``example_data`` this
    function read them as mm/h, four times too low.)
    """
    from core.opensense import example_data
    from core.opensense.retrieval import (combine_sublinks, retrieve_dataset,
                                          retrieve_improved, sampling_interval_s)

    key, subset, gauge_component, label = EXAMPLE_EVENTS[dataset]
    cache = example_data.sample_dir(key)
    if offline and not cache.exists():
        raise SystemExit(
            f"--offline but no cached subset at {cache}.\n"
            f"Run: python -m core.opensense.example_data "
            f"--dataset {key} --subset {subset}")

    data = example_data.load(key, subset)

    # --- CML: the shared retrieval chain on the subset's raw signals ---
    chain = retrieve_improved if retrieval == "improved" else retrieve_dataset
    cml = combine_sublinks(chain(data["cml"]))
    cml.coords["length"] = cml.length_km
    cml.coords["frequency"] = cml.frequency_ghz

    # --- radar: example_data already made it a rate on a projected grid ---
    rad = data["radar"][["R"]]

    # --- gauges: accumulations per sampling step -> mm/h ---
    g_raw = data[gauge_component]
    g_interval = sampling_interval_s(g_raw.time)
    gauge = xr.Dataset({"R": g_raw.rainfall_amount.transpose("time", "id")
                        * (3600.0 / g_interval)})
    for c in ("x", "y", "lon", "lat"):
        if c in g_raw.coords:
            gauge.coords[c] = ("id", np.asarray(g_raw[c]))

    # The CML is averaged over the windows the references accumulate over -
    # OpenRainER stamps its 15-minute accumulations at interval end - and
    # at their native step. This function used to resample OpenRainER to
    # 5 minutes with start labels, which left the CML one bin out of line.
    from core.opensense.evaluation import aggregate
    resample = resample or EXAMPLE_STEP[key]
    label = example_data.DATASETS[key].accumulation_label
    cml = xr.Dataset({"R": aggregate(cml.R, resample, label=label)},
                     coords={c: v for c, v in cml.coords.items()
                             if "time" not in v.dims})
    gauge = gauge.resample(time=resample).mean()
    rad = rad.resample(time=resample).mean()
    times = np.intersect1d(np.intersect1d(rad.time, cml.time), gauge.time)
    rad, cml, gauge = (x.sel(time=times) for x in (rad, cml, gauge))
    for ds_ in (rad, cml, gauge):
        ds_.attrs.update(source=f"{key} example subset {subset}", label=label)
    return rad, cml, gauge


# --------------------------------------------------------------------------
# coverage analysis
# --------------------------------------------------------------------------
def distance_to_network(ds_cml, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Shortest distance (km) from each gauge to any CML path."""
    from core.opensense.evaluation import distance_to_network as dist
    return dist(ds_cml, gx, gy)


def coverage_analysis(pooled: dict, ds_cml, ds_gauge, n_times: int,
                      bands=((0.0, 2.0), (2.0, 5.0), (5.0, 10.0),
                             (10.0, np.inf))) -> list[dict]:
    """Score each method by how far its gauges sit from the link network.

    This is what reconciles the synthetic ranking with the real one. Merging
    can only add information where the links are; scored at gauges far outside
    the network, a merge is degrading a radar field with an extrapolation.
    """
    dist = distance_to_network(ds_cml, np.asarray(ds_gauge.x),
                               np.asarray(ds_gauge.y))
    tiled = np.tile(dist, n_times)

    rows = []
    for method in METHODS:
        est = np.concatenate(pooled[method.key]["est"])
        obs = np.concatenate(pooled[method.key]["obs"])
        for lo, hi in bands:
            sel = (tiled >= lo) & (tiled < hi)
            if sel.sum() < 20:
                continue
            label = f"{lo:g}-{hi:g} km" if np.isfinite(hi) else f">{lo:g} km"
            rows.append({"method": method.label, "method_key": method.key,
                         "family": method.family, "band": label,
                         "band_lo": lo, "n_gauges": int(sel.sum()),
                         **score(obs[sel], est[sel])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["openmrg", "openrainer", "both"],
                    default="both")
    ap.add_argument("--openmrg-event", default="aug25",
                    choices=sorted(omrg.EVENTS))
    ap.add_argument("--openrainer-event", default="sep26",
                    choices=sorted(orain.EVENTS))
    ap.add_argument("--source", choices=["raw", "example"], default="raw",
                    help="raw: the full Zenodo archives already on disk "
                         "(default). example: the small curated subsets, "
                         "downloaded on first use (~57 MB).")
    ap.add_argument("--offline", action="store_true",
                    help="never download; fail if the data is not local")
    ap.add_argument("--retrieval", choices=["default", "improved"],
                    default="default",
                    help="CML chain: the default, or retrieve_improved "
                         "(nearby-link wet/dry + Leijnse 2008). Outputs are "
                         "tagged _improved and never overwrite the default's.")
    ap.add_argument("--val-select", choices=["cml", "gauge"], default="cml",
                    help="rank the validation timesteps by CML rain (default, "
                         "as published) or by gauge rain - the same timesteps "
                         "for every retrieval, so use this to compare them")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--val-timesteps", type=int, default=20,
                    help="wettest timesteps pooled for gauge validation")
    ap.add_argument("--val-max-cells", type=int, default=6000,
                    help="coarsen the grid below this many cells for the "
                         "pooled validation (maps stay full resolution)")
    args = ap.parse_args()

    vs.use_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {"source": args.source, "retrieval": args.retrieval}
    print(f"data source: {args.source}"
          + ("  (offline)" if args.offline else ""))

    if not args.skip_benchmark:
        print(f"\n{'=' * 74}\nSynthetic benchmark on the real OpenMRG "
              f"geometry\n{'=' * 74}")
        if args.source == "example":
            rad, cml, gauge = build_from_example("openmrg", offline=args.offline)
        else:
            rad, cml, gauge = omrg.build_event(omrg.EVENTS[args.openmrg_event])
        rows = run_benchmark(rad, cml, gauge, n_seeds=args.seeds)
        plots.fig_benchmark(rows, RESULTS / ("benchmark_ranking.png" if args.source == "raw"
                                else f"benchmark_ranking_{args.source}.png"))
        summary["benchmark"] = rows

        for regime in dict.fromkeys(r["regime"] for r in rows):
            sel = sorted((r for r in rows if r["regime"] == regime),
                         key=lambda z: z["rmse"])
            print(f"\n  {regime}")
            for r in sel:
                flag = "  BLOW-UP" if r["frac_implausible"] > 0.001 else ""
                print(f"    {r['method']:38s} RMSE {r['rmse']:10.2f}  "
                      f"corr {r['corr']:6.3f}{flag}")

    runs = []
    if args.dataset in ("openmrg", "both"):
        runs.append(run_dataset("OpenMRG", omrg, args.openmrg_event,
                                args.val_timesteps, args.val_max_cells,
                                args.source, args.offline, args.retrieval,
                                args.val_select))
    if args.dataset in ("openrainer", "both"):
        runs.append(run_dataset("OpenRainER", orain, args.openrainer_event,
                                args.val_timesteps, args.val_max_cells,
                                args.source, args.offline, args.retrieval,
                                args.val_select))
    summary["applications"] = runs

    suffix = ("" if args.source == "raw" else f"_{args.source}") + \
             ("" if args.retrieval == "default" else f"_{args.retrieval}") + \
             ("" if args.val_select == "cml" else f"_by{args.val_select}") + API_TAG
    out = RESULTS / f"summary{suffix}.json"
    if out.exists() and args.dataset != "both":
        # a one-dataset run replaces that dataset's entry and keeps the other's
        old = json.loads(out.read_text())
        done = {r["dataset"] for r in runs}
        summary["applications"] = [a for a in old.get("applications", [])
                                   if a.get("dataset") not in done] + runs
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    for p in sorted(RESULTS.glob("*.png")):
        print(f"  {p.relative_to(REPO_ROOT)}  ({p.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
