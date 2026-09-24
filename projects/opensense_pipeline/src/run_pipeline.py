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
sys.path.insert(0, str(REPO_ROOT / "projects/rainfall_field_sim/src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import plots                                                # noqa: E402
import ingest_openmrg as omrg                               # noqa: E402
import ingest_openrainer as orain                           # noqa: E402
import viz_style as vs                                      # noqa: E402
from merging import METHODS, run, score                     # noqa: E402
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
                       max_cells: int = 6000) -> list[dict]:
    """Score every method against the gauges, pooled over several timesteps.

    A single timestep gives one value per gauge - ten numbers for OpenMRG -
    which is far too few to separate methods; the correlation of a
    ten-point sample swings wildly. Pooling the wettest ``n_times`` steps
    gives a few hundred paired values and a stable ranking.

    The gauges are held out of every merge, so this stays an independent
    check rather than a measure of how well each method reproduces its own
    input.
    """
    ds_rad, factor = coarsen_radar(ds_rad, max_cells)
    if factor > 1:
        print(f"    grid coarsened {factor}x for validation -> "
              f"{ds_rad.sizes['y']} x {ds_rad.sizes['x']}")

    wetness = np.nansum(np.asarray(ds_cml.R), axis=1)
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
                source: str = "raw", offline: bool = False) -> dict:
    """Ingest one event, apply every method, write maps and validation.

    ``source`` picks where the data comes from - the full local Zenodo
    archives (``raw``) or the curated example subsets (``example``). Both
    produce the same (radar, cml, gauge) triple, so nothing downstream cares.
    """
    # Results from the two sources must not overwrite each other: the example
    # subsets cover a different period from the curated events.
    tag = "" if source == "raw" else f"_{source}"

    if source == "example":
        key = name.lower()
        _, subset, _, label = EXAMPLE_EVENTS[key]
        print(f"\n{'=' * 74}\n{name}: {label}\n"
              f"  curated example subset '{subset}'\n{'=' * 74}")
        ds_rad, ds_cml, ds_gauge = build_from_example(key, offline=offline)
        event_key_used, event_label = subset, label
    else:
        event = module.EVENTS[event_key]
        print(f"\n{'=' * 74}\n{name}: {event.label}\n  {event.note}\n{'=' * 74}")
        # The ingest modules already emit gauges on an "id" dimension carrying
        # x/y and lon/lat, which is what mergeplg and poligrain need.
        ds_rad, ds_cml, ds_gauge = module.build_event(event)
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
                                              n_val, max_cells)
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
            "label": event_label,
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


def build_from_example(dataset: str, resample: str = "5min",
                       offline: bool = False) -> tuple:
    """(radar, cml, gauge) from a curated example subset.

    The same triple ``ingest_*.build_event`` returns from the full Zenodo
    archives, so everything downstream is unchanged. The CML retrieval is the
    shared chain in ``retrieval.py``, run on the subset's own tsl/rsl.

    Radar in these subsets is already rain rate on a lat/lon grid, so it needs
    projecting but not a Z-R step.
    """
    import example_data
    from retrieval import RetrievalConfig, retrieve

    key, subset, gauge_component, label = EXAMPLE_EVENTS[dataset]
    cache = example_data.CACHE / example_data.DATASETS[key].folder
    if offline and not cache.exists():
        raise SystemExit(
            f"--offline but no cached subset at {cache}.\n"
            f"Run: python projects/opensense_pipeline/src/example_data.py "
            f"--dataset {key} --subset {subset}")

    data = example_data.load(key, subset)
    crs = example_data.DATASETS[key].crs
    cml_raw = data["cml"]

    # --- CML: run the retrieval chain on the subset's raw signals ---
    d = cml_raw.transpose("time", "cml_id", "sublink_id")
    loss = np.asarray(d.tsl - d.rsl, dtype=float)
    n_t, n_c, n_s = loss.shape
    length = np.asarray(cml_raw.length_km)
    if length.ndim == 1:
        length = np.repeat(length[:, None], n_s, axis=1)
    freq = np.asarray(cml_raw.frequency_ghz)
    if freq.ndim == 1:
        freq = np.repeat(freq[:, None], n_s, axis=1)
    pol = np.asarray(cml_raw.polarization)
    pol = pol.reshape(freq.shape) if pol.size == freq.size \
        else np.full(freq.shape, "vertical")

    interval = float(np.diff(cml_raw.time.values[:2])
                     .astype("timedelta64[s]").astype(float)[0])
    cfg = RetrievalConfig.for_interval(interval)
    rain = retrieve(loss.reshape(n_t, n_c * n_s), length.ravel(),
                    freq.ravel(), pol.ravel(), cfg)["R"]
    with np.errstate(invalid="ignore"):
        rain = np.nanmean(rain.reshape(n_t, n_c, n_s), axis=2)

    keep = ("site_0_x", "site_0_y", "site_1_x", "site_1_y", "x", "y",
            "length_km", "frequency_ghz")
    cml = xr.Dataset({"R": (("time", "cml_id"), rain)},
                     coords={"time": cml_raw.time, "cml_id": cml_raw.cml_id})
    for c in keep:
        if c not in cml_raw.coords:
            continue
        da = cml_raw[c]
        # Per-link metadata is sometimes stored per sublink, and the dim order
        # is not consistent between files - select by name, never by position.
        if "sublink_id" in da.dims:
            da = da.isel(sublink_id=0, drop=True)
        cml.coords[c] = ("cml_id", np.asarray(da))
    cml.coords["length"] = cml.length_km
    cml.coords["frequency"] = cml.frequency_ghz

    # --- radar: already rain rate, just needs projected grid coordinates ---
    rad_raw = data["radar"]
    var = "R" if "R" in rad_raw.data_vars else "rainfall_amount"
    rad = xr.Dataset({"R": rad_raw[var]})
    lon2d, lat2d = np.asarray(rad_raw.lon), np.asarray(rad_raw.lat)
    if lon2d.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon2d, lat2d)
    rad.coords["longitudes"] = (("y", "x"), lon2d)
    rad.coords["latitudes"] = (("y", "x"), lat2d)
    import poligrain as plg
    xs, ys = plg.spatial.project_point_coordinates(
        rad.longitudes, rad.latitudes, crs)
    rad.coords["x_grid"], rad.coords["y_grid"] = xs, ys
    xv, yv = np.asarray(xs), np.asarray(ys)
    rad.coords["x"] = ("x", xv[xv.shape[0] // 2, :])
    rad.coords["y"] = ("y", yv[:, xv.shape[1] // 2])

    # --- gauges: accumulations per sampling step -> mm/h ---
    g_raw = data[gauge_component]
    g_interval = float(np.diff(g_raw.time.values[:2])
                       .astype("timedelta64[s]").astype(float)[0])
    gauge = xr.Dataset({"R": g_raw.rainfall_amount.transpose("time", "id")
                        * (3600.0 / g_interval)})
    for c in ("x", "y", "lon", "lat"):
        if c in g_raw.coords:
            gauge.coords[c] = ("id", np.asarray(g_raw[c]))

    cml = cml.resample(time=resample).mean()
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
    """Shortest distance (km) from each point to any CML path.

    Point-to-segment, not point-to-midpoint: a CML measures along its whole
    line, so a gauge beside the middle of a long link is well covered even
    though both endpoints are far away.
    """
    x0 = np.asarray(ds_cml.site_0_x)
    y0 = np.asarray(ds_cml.site_0_y)
    x1 = np.asarray(ds_cml.site_1_x)
    y1 = np.asarray(ds_cml.site_1_y)
    vx, vy = x1 - x0, y1 - y0
    len2 = vx * vx + vy * vy

    out = np.empty(gx.size)
    for i, (px, py) in enumerate(zip(gx, gy)):
        t = np.clip(np.where(len2 > 0, ((px - x0) * vx + (py - y0) * vy)
                             / np.maximum(len2, 1e-9), 0.0), 0.0, 1.0)
        out[i] = np.min(np.hypot(px - (x0 + t * vx), py - (y0 + t * vy)))
    return out / 1000.0


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
    summary = {"source": args.source}
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
                                args.source, args.offline))
    if args.dataset in ("openrainer", "both"):
        runs.append(run_dataset("OpenRainER", orain, args.openrainer_event,
                                args.val_timesteps, args.val_max_cells,
                                args.source, args.offline))
    summary["applications"] = runs

    out = RESULTS / ("summary.json" if args.source == "raw"
                     else f"summary_{args.source}.json")
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    for p in sorted(RESULTS.glob("*.png")):
        print(f"  {p.relative_to(REPO_ROOT)}  ({p.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
