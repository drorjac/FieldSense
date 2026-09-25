"""
Radar against CML-derived rainfall maps, through precipitation events.

Both sensors claim to measure the same field and disagree, and the useful
question is not which is right but *where* and *when* they part company.

Each event is reduced to two maps on the **same grid**: the radar as
published, and a CML map interpolated onto that grid with a line-aware method
so link geometry is preserved. Everything is then compared per timestep and in
accumulation.

What is being asked:

* do they agree on **where** the rain is (spatial correlation per timestep)
* do they agree on **how much** (bias, ratio of accumulations)
* does the answer depend on **precipitation type** - a convective cell, a
  mixed event and a widespread frontal day are all in the OpenMRG record
* does agreement fall off **away from the link network**, which would say the
  disagreement is coverage rather than physics

    python -m compare_radar_cml --event aug25
    python -m compare_radar_cml --all
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

RESULTS = HERE.parent / "results"

# (module, event key, human label, regime)
EVENTS = [
    ("openmrg", "torslanda", "Torslanda 2015-07-28", "convective cell"),
    ("openmrg", "aug25", "25 August 2015", "mixed"),
    ("openmrg", "jun17", "17 June 2015", "widespread frontal"),
    ("openrainer", "sep26", "26 September 2021", "widespread, Italy"),
    ("openmesh", "rain_0113", "NYC 13 Jan 2024", "rain"),
    ("openmesh", "rain_0128", "NYC 28 Jan 2024", "rain"),
    ("openmesh", "mixed_0116", "NYC 16 Jan 2024", "snow + rain"),
    ("openmesh", "snow_0119", "NYC 19 Jan 2024", "snow"),
]

# NYC events are the phase experiment: the same links and the same radar,
# through liquid and frozen precipitation.
NYC_EVENTS = [e for e in EVENTS if e[0] == "openmesh"]


def load_event(source: str, key: str):
    import ingest_openmesh as omesh
    import ingest_openmrg as omrg
    import ingest_openrainer as orain

    module = {"openmrg": omrg, "openrainer": orain,
              "openmesh": omesh}[source]
    return module.build_event(module.EVENTS[key])


def cml_on_radar_grid(ds_rad: xr.Dataset, ds_cml: xr.Dataset,
                      t_index: int) -> np.ndarray:
    """Interpolate the CML observations onto the radar grid for one timestep.

    Line-aware IDW, so a link contributes along its path rather than at its
    midpoint. Links with no retrieval at this step are dropped rather than
    treated as zero - a missing link is not a dry link.
    """
    from merging import conform_radar
    from mergeplg import interpolate

    da_rad = conform_radar(ds_rad.R.isel(time=t_index))
    da_cml = ds_cml.R.isel(time=t_index)
    good = np.isfinite(np.asarray(da_cml))
    if good.sum() < 5:
        return np.full(da_rad.shape, np.nan)
    da_cml = da_cml.isel(cml_id=good)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = interpolate.InterpolateIDW(min_observations=1)
        model.update(da_cml=da_cml)
        out = model.interpolate(da_rad, da_cml=da_cml, p=2,
                                idw_method="radolan", nnear=8,
                                max_distance=30000)
    return np.clip(np.asarray(out), 0.0, None)


def distance_to_network(ds_cml, xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
    """Distance (km) from every grid cell to the nearest CML path."""
    from core.opensense.evaluation import distance_to_network as dist
    return dist(ds_cml, xg, yg)


# --------------------------------------------------------------------------
def cml_vs_gauge(ds_cml, ds_gauge, radius_m: float = 3000.0) -> dict:
    """CML against co-located point sensors - the control for the radar.

    If CML and radar disagree, either could be at fault. Scoring the CML
    against gauges within ``radius_m`` separates the two: agreement here with
    disagreement against radar points at the radar, and disagreement with both
    points at the retrieval.
    """
    R = np.asarray(ds_cml.R)
    G = np.asarray(ds_gauge.R)
    cx, cy = np.asarray(ds_cml.x), np.asarray(ds_cml.y)
    gx, gy = np.asarray(ds_gauge.x), np.asarray(ds_gauge.y)

    corrs, ratios = [], []
    for i in range(gx.size):
        near = np.hypot(cx - gx[i], cy - gy[i]) < radius_m
        if near.sum() < 1:
            continue
        c = np.nanmean(R[:, near], axis=1)
        g = G[:, i]
        ok = np.isfinite(c) & np.isfinite(g)
        if ok.sum() < 10 or g[ok].sum() <= 0:
            continue
        ratios.append(c[ok].sum() / g[ok].sum())
        if c[ok].std() > 0 and g[ok].std() > 0:
            corrs.append(float(np.corrcoef(c[ok], g[ok])[0, 1]))

    return {"n_pairs": len(ratios),
            "corr": float(np.nanmedian(corrs)) if corrs else np.nan,
            "ratio": float(np.median(ratios)) if ratios else np.nan}


def compare(source: str, key: str, label: str, regime: str,
            max_timesteps: int = 40, wet_threshold: float = 0.1) -> dict:
    ds_rad, ds_cml, ds_gauge = load_event(source, key)

    # Concentrate on the wet part of the event: comparing dry frames measures
    # nothing and drags every correlation toward zero.
    cml_total = np.nansum(np.asarray(ds_cml.R), axis=1)
    order = np.argsort(cml_total)[::-1][:max_timesteps]
    steps = sorted(int(t) for t in order if cml_total[t] > 0)

    xg = np.asarray(ds_rad.x_grid)
    yg = np.asarray(ds_rad.y_grid)
    dist = distance_to_network(ds_cml, xg, yg)

    # Accumulate only where both sensors report. The radar is 22-59% NaN over
    # these domains - outside the beam, or below detection - and summing it
    # with NaN treated as zero while the interpolated CML field is dense makes
    # the accumulation ratio meaningless. Both are masked to the same cells.
    per_step, rad_acc, cml_acc, n_valid = [], None, None, None
    print(f"  {len(steps)} wet timesteps, grid {xg.shape}")
    for n, t in enumerate(steps):
        rad = np.asarray(ds_rad.R.isel(time=t))
        cml = cml_on_radar_grid(ds_rad, ds_cml, t)
        if not np.isfinite(cml).any():
            continue

        ok = np.isfinite(rad) & np.isfinite(cml)
        if rad_acc is None:
            rad_acc = np.zeros_like(rad, dtype=float)
            cml_acc = np.zeros_like(rad, dtype=float)
            n_valid = np.zeros_like(rad, dtype=float)
        rad_acc[ok] += rad[ok]
        cml_acc[ok] += cml[ok]
        n_valid[ok] += 1.0

        wet = ok & ((rad >= wet_threshold) | (cml >= wet_threshold))
        if wet.sum() < 20:
            continue
        a, b = rad[wet], cml[wet]
        denom = a.std() * b.std()
        per_step.append({
            "t": t,
            "corr": float(((a - a.mean()) * (b - b.mean())).mean() / denom)
                    if denom > 0 else np.nan,
            "bias": float((b - a).mean()),
            "radar_mean": float(a.mean()),
            "cml_mean": float(b.mean()),
            "wet_frac_radar": float((rad[ok] >= wet_threshold).mean()),
            "wet_frac_cml": float((cml[ok] >= wet_threshold).mean()),
        })
        if (n + 1) % 15 == 0:
            print(f"    {n + 1}/{len(steps)}", flush=True)

    # Cells never jointly observed carry no information either way.
    if n_valid is not None:
        never = n_valid == 0
        rad_acc = np.where(never, np.nan, rad_acc)
        cml_acc = np.where(never, np.nan, cml_acc)

    corr = np.array([p["corr"] for p in per_step])
    rad_means = np.array([p["radar_mean"] for p in per_step])
    cml_means = np.array([p["cml_mean"] for p in per_step])
    ratio_acc = (float(np.nansum(cml_acc) / np.nansum(rad_acc))
                 if rad_acc is not None and np.nansum(rad_acc) > 0 else np.nan)
    joint_frac = float(np.mean(n_valid > 0)) if n_valid is not None else np.nan

    # agreement as a function of distance from the link network
    bands, band_corr = [(0, 2), (2, 5), (5, 10), (10, np.inf)], []
    for lo, hi in bands:
        m = (dist >= lo) & (dist < hi) & np.isfinite(rad_acc) & np.isfinite(cml_acc)
        wet = m & ((rad_acc > 0) | (cml_acc > 0))
        if wet.sum() < 30:
            band_corr.append(np.nan)
            continue
        a, b = rad_acc[wet], cml_acc[wet]
        d = a.std() * b.std()
        band_corr.append(float(((a - a.mean()) * (b - b.mean())).mean() / d)
                         if d > 0 else np.nan)

    gauge_check = cml_vs_gauge(ds_cml, ds_gauge)

    return {"source": source, "event": key, "label": label, "regime": regime,
            "cml_vs_gauge": gauge_check,
            "n_steps": len(per_step),
            "corr_median": float(np.nanmedian(corr)) if corr.size else np.nan,
            "corr_p25": float(np.nanpercentile(corr, 25)) if corr.size else np.nan,
            "corr_p75": float(np.nanpercentile(corr, 75)) if corr.size else np.nan,
            "ratio_accumulation": ratio_acc,
            "ratio_wet_means": (float(np.nanmedian(cml_means / np.maximum(rad_means, 1e-9)))
                                if per_step else np.nan),
            "radar_acc_mean": float(np.nanmean(rad_acc)),
            "joint_coverage": joint_frac,
            "cml_acc_mean": float(np.nanmean(cml_acc)),
            "band_corr": dict(zip([f"{lo}-{hi} km" for lo, hi in bands],
                                  band_corr)),
            "_acc": (rad_acc, cml_acc, dist, xg, yg),
            "_per_step": per_step}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", choices=[e[1] for e in EVENTS])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--nyc", action="store_true",
                    help="only the NYC rain/snow phase comparison")
    ap.add_argument("--max-timesteps", type=int, default=40)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    todo = (NYC_EVENTS if args.nyc else
            EVENTS if args.all else
            [e for e in EVENTS if e[1] == args.event])
    if not todo:
        ap.error("pass --event or --all")

    out = []
    for source, key, label, regime in todo:
        print(f"\n{'=' * 72}\n{label}  ({regime})\n{'=' * 72}")
        r = compare(source, key, label, regime, args.max_timesteps)
        out.append(r)
        print(f"  spatial correlation   median {r['corr_median']:.3f}  "
              f"(IQR {r['corr_p25']:.3f}-{r['corr_p75']:.3f})")
        print(f"  accumulation          radar {r['radar_acc_mean']:.2f}   "
              f"cml {r['cml_acc_mean']:.2f}   ratio {r['ratio_accumulation']:.2f}"
              f"   (jointly observed {r['joint_coverage']*100:.0f}% of cells)")
        gc = r["cml_vs_gauge"]
        print(f"  CML vs point sensors  corr {gc['corr']:+.3f}   "
              f"ratio {gc['ratio']:.2f}   ({gc['n_pairs']} pairs)  "
              f"<- the control")
        print("  correlation by distance to the link network:")
        for band, c in r["band_corr"].items():
            print(f"      {band:>10}  {c:6.3f}" if np.isfinite(c)
                  else f"      {band:>10}     n/a")

    import json
    path = RESULTS / ("radar_vs_cml_nyc.json" if args.nyc
                      else "radar_vs_cml.json")
    path.write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in out],
        indent=1, default=float))
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")

    if len(out) > 1:
        import plots_radar_cml
        from core import viz_style as vs
        vs.use_style()
        fig_path = RESULTS / ("radar_vs_cml_nyc.png" if args.nyc
                              else "radar_vs_cml.png")
        plots_radar_cml.figure(out, fig_path)
        print(f"wrote {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
