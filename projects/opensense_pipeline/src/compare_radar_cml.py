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
sys.path.insert(0, str(REPO_ROOT / "projects/rainfall_field_sim/src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

RESULTS = HERE.parent / "results"

# (module, event key, human label, regime)
EVENTS = [
    ("openmrg", "torslanda", "Torslanda 2015-07-28", "convective cell"),
    ("openmrg", "aug25", "25 August 2015", "mixed"),
    ("openmrg", "jun17", "17 June 2015", "widespread frontal"),
    ("openrainer", "sep26", "26 September 2021", "widespread, Italy"),
]


def load_event(source: str, key: str):
    import ingest_openmrg as omrg
    import ingest_openrainer as orain

    module = omrg if source == "openmrg" else orain
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
    x0 = np.asarray(ds_cml.site_0_x)
    y0 = np.asarray(ds_cml.site_0_y)
    x1 = np.asarray(ds_cml.site_1_x)
    y1 = np.asarray(ds_cml.site_1_y)
    vx, vy = x1 - x0, y1 - y0
    len2 = vx * vx + vy * vy

    flat_x, flat_y = xg.ravel(), yg.ravel()
    best = np.full(flat_x.size, np.inf)
    for i in range(x0.size):
        if len2[i] <= 0:
            continue
        t = np.clip(((flat_x - x0[i]) * vx[i] + (flat_y - y0[i]) * vy[i])
                    / len2[i], 0.0, 1.0)
        d = np.hypot(flat_x - (x0[i] + t * vx[i]),
                     flat_y - (y0[i] + t * vy[i]))
        np.minimum(best, d, out=best)
    return (best / 1000.0).reshape(xg.shape)


# --------------------------------------------------------------------------
def compare(source: str, key: str, label: str, regime: str,
            max_timesteps: int = 40, wet_threshold: float = 0.1) -> dict:
    ds_rad, ds_cml, _ = load_event(source, key)
    n_t = ds_rad.sizes["time"]

    # Concentrate on the wet part of the event: comparing dry frames measures
    # nothing and drags every correlation toward zero.
    cml_total = np.nansum(np.asarray(ds_cml.R), axis=1)
    order = np.argsort(cml_total)[::-1][:max_timesteps]
    steps = sorted(int(t) for t in order if cml_total[t] > 0)

    xg = np.asarray(ds_rad.x_grid)
    yg = np.asarray(ds_rad.y_grid)
    dist = distance_to_network(ds_cml, xg, yg)

    per_step, rad_acc, cml_acc = [], None, None
    print(f"  {len(steps)} wet timesteps, grid {xg.shape}")
    for n, t in enumerate(steps):
        rad = np.asarray(ds_rad.R.isel(time=t))
        cml = cml_on_radar_grid(ds_rad, ds_cml, t)
        if not np.isfinite(cml).any():
            continue

        rad_acc = rad if rad_acc is None else rad_acc + np.nan_to_num(rad)
        cml_acc = cml if cml_acc is None else cml_acc + np.nan_to_num(cml)

        ok = np.isfinite(rad) & np.isfinite(cml)
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

    corr = np.array([p["corr"] for p in per_step])
    ratio_acc = (float(np.nansum(cml_acc) / np.nansum(rad_acc))
                 if rad_acc is not None and np.nansum(rad_acc) > 0 else np.nan)

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

    return {"source": source, "event": key, "label": label, "regime": regime,
            "n_steps": len(per_step),
            "corr_median": float(np.nanmedian(corr)) if corr.size else np.nan,
            "corr_p25": float(np.nanpercentile(corr, 25)) if corr.size else np.nan,
            "corr_p75": float(np.nanpercentile(corr, 75)) if corr.size else np.nan,
            "ratio_accumulation": ratio_acc,
            "radar_acc_mean": float(np.nanmean(rad_acc)),
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
    ap.add_argument("--max-timesteps", type=int, default=40)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    todo = EVENTS if args.all else [e for e in EVENTS if e[1] == args.event]
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
              f"cml {r['cml_acc_mean']:.2f}   ratio {r['ratio_accumulation']:.2f}")
        print("  correlation by distance to the link network:")
        for band, c in r["band_corr"].items():
            print(f"      {band:>10}  {c:6.3f}" if np.isfinite(c)
                  else f"      {band:>10}     n/a")

    import json
    path = RESULTS / "radar_vs_cml.json"
    path.write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in out],
        indent=1, default=float))
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")

    if len(out) > 1:
        import plots_radar_cml
        import viz_style as vs
        vs.use_style()
        fig_path = RESULTS / "radar_vs_cml.png"
        plots_radar_cml.figure(out, fig_path)
        print(f"wrote {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
