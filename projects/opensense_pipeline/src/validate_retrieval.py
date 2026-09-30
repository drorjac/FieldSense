"""
Check this project's CML retrieval against OpenSense's own.

Until now the retrieval chain had no independent check: it was scored against
rain gauges, which are sparse point sensors measuring something slightly
different from a path average, so a disagreement could not be attributed.

The OpenMRG ``8d`` example subset closes that gap. It ships raw ``tsl``/``rsl``
**and** a reference rain rate ``R`` retrieved by the OpenSense community from
exactly those signals, over 364 links at 10 s for 2015-07-22 to 07-29 - a
window that contains the Torslanda event this project already uses.

So both retrievals see identical input on identical links at identical times,
and any difference is the chain, not the reference.

    python -m validate_retrieval
    python -m validate_retrieval --event torslanda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from core.opensense import example_data  # noqa: E402
from core.opensense.retrieval import (RetrievalConfig, combine_sublinks,  # noqa: E402
                                      retrieve_dataset, sampling_interval_s)

RESULTS = HERE.parent / "results"

# Windows inside the 8d reference period worth looking at separately.
WINDOWS = {
    "full": (None, None),
    "torslanda": ("2015-07-28T12:00", "2015-07-28T22:00"),
}


def load_reference(offline: bool = False) -> xr.Dataset:
    """OpenMRG 8d subset: raw tsl/rsl plus the reference retrieval R."""
    cache = example_data.sample_dir("openmrg") / "openmrg_cml_8d.nc"
    if offline:
        if not cache.exists():
            raise SystemExit(
                f"--offline but {cache} is missing.\n"
                f"Run: python -m core.opensense.example_data "
                f"--dataset openmrg --subset 8d")
        return example_data.normalize_cml(xr.load_dataset(cache), "EPSG:32632")
    return example_data.load("openmrg", "8d")["cml"]


def run_our_retrieval(ds: xr.Dataset,
                      cfg: RetrievalConfig | None = None) -> np.ndarray:
    """Our chain on the reference subset's own tsl/rsl -> (time, cml_id).

    Before ``retrieve_dataset`` existed this function flattened the signals
    as (cml_id, sublink_id) and the metadata as the file stores it,
    (sublink_id, cml_id), so every link's second sublink was retrieved with
    another link's frequency. That inflated our totals by ~13% here and is
    why the README's first version of this comparison read 2.28x gauges.
    """
    out = combine_sublinks(retrieve_dataset(ds, cfg))
    return out.R.transpose("time", "cml_id").values


def reference_rain(ds: xr.Dataset) -> np.ndarray:
    """The published reference R, averaged over sublinks -> (time, cml_id)."""
    r = np.asarray(ds.R.transpose("time", "cml_id", "sublink_id"), dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(r, axis=2)


def compare(ours: np.ndarray, ref: np.ndarray, wet_threshold: float = 0.1) -> dict:
    """Agreement between the two retrievals over all link-timesteps."""
    a, b = ours.ravel(), ref.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]

    wet = (a >= wet_threshold) | (b >= wet_threshold)
    denom = a.std() * b.std()
    both_wet = (a >= wet_threshold) & (b >= wet_threshold)

    return {
        "n": int(ok.sum()),
        "n_wet": int(wet.sum()),
        "corr": float(((a - a.mean()) * (b - b.mean())).mean() / denom)
                if denom > 0 else float("nan"),
        "corr_wet": float(np.corrcoef(a[wet], b[wet])[0, 1]) if wet.sum() > 2 else float("nan"),
        "rmse": float(np.sqrt(((a - b) ** 2).mean())),
        "bias": float((a - b).mean()),
        "ratio_total": float(a.sum() / b.sum()) if b.sum() > 0 else float("nan"),
        "mean_ours": float(a.mean()),
        "mean_ref": float(b.mean()),
        # Detection agreement, which is a separate question from magnitude.
        "wet_agreement": float((( a >= wet_threshold) == (b >= wet_threshold)).mean()),
        "both_wet_frac": float(both_wet.mean()),
    }


def sweep_waa(ds: xr.Dataset, ref: np.ndarray,
              values=(0.0, 0.5, 1.0, 1.5, 2.0, 2.3)) -> None:
    """How the wet-antenna term trades magnitude against the reference.

    Worth running before trusting any single number out of this script: the
    ratio to the reference is strongly event-dependent, so a value calibrated
    on one event does not transfer to the next.
    """
    interval = sampling_interval_s(ds.time)

    m_ref = float(np.nanmean(ref))
    print(f"\nreference mean {m_ref:.4f} mm/h")
    print(f"{'waa_max_db':>11}{'our mean':>11}{'ratio/ref':>11}{'corr':>8}")
    for waa in values:
        ours = run_our_retrieval(
            ds, RetrievalConfig.for_interval(interval, waa_max_db=waa))
        a, b = ours.ravel(), ref.ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
        m = float(np.nanmean(ours))
        print(f"{waa:11.1f}{m:11.4f}{m / m_ref:11.2f}{corr:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", choices=sorted(WINDOWS), default="full")
    ap.add_argument("--offline", action="store_true",
                    help="use the cached subset; never download")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep wet-antenna attenuation against the reference")
    args = ap.parse_args()

    print("Loading the OpenMRG 8d reference subset "
          "(raw tsl/rsl + OpenSense's retrieved R)")
    ds = load_reference(args.offline)

    start, end = WINDOWS[args.window]
    if start:
        ds = ds.sel(time=slice(start, end))
    print(f"  window '{args.window}': {str(ds.time.values[0])[:16]} .. "
          f"{str(ds.time.values[-1])[:16]}  "
          f"({ds.sizes['cml_id']} links, {ds.sizes['time']} steps)")

    ref = reference_rain(ds)

    if args.sweep:
        sweep_waa(ds, ref)
        return

    print("Running our retrieval on the same signals...")
    ours = run_our_retrieval(ds)

    s = compare(ours, ref)
    print(f"\n{'=' * 66}\nOurs vs the OpenSense reference retrieval\n{'=' * 66}")
    print(f"  link-timesteps compared   {s['n']:>12,}")
    print(f"  correlation (all)         {s['corr']:>12.3f}")
    print(f"  correlation (wet)         {s['corr_wet']:>12.3f}")
    print(f"  wet/dry agreement         {s['wet_agreement']*100:>11.1f}%")
    print(f"  mean rain, ours           {s['mean_ours']:>12.3f} mm/h")
    print(f"  mean rain, reference      {s['mean_ref']:>12.3f} mm/h")
    print(f"  ratio of totals           {s['ratio_total']:>12.3f}")
    print(f"  bias                      {s['bias']:>+12.3f} mm/h")
    print(f"  RMSE                      {s['rmse']:>12.3f} mm/h")
    print("=" * 66)

    if not args.no_figure:
        import plots
        from core import viz_style as vs
        vs.use_style()
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"retrieval_vs_reference_{args.window}.png"
        plots.fig_retrieval_vs_reference(ours, ref, s, path,
                                         f"OpenMRG {args.window} window")
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
