"""
Rank CML retrieval variants against independent sensors, over 8 days.

``validate_retrieval.py`` established that this chain has the right skill and
the wrong magnitude: it over-reads the OpenSense reference by ~1.8x and the
municipal gauges by ~2x, and a single static wet-antenna constant cannot fix
both. This script asks which *algorithmic* change does, using the tools the
OpenSense ecosystem already provides:

wet-antenna model   (pycomlink)
    ``saturating``    the current default, 0.5 dB and the 2.3 dB literature
                      magnitude
    ``pastorek2021``  KR-alt, Pastorek et al. (2021)
    ``leijnse2008``   physical water-film model, Leijnse et al. (2008)
wet/dry mask        which samples the baseline is allowed to learn from
    ``rolling_std``   Schleiss & Berne (2010), the current default
    ``radar``         radar path-averaged along each link
                      (``poligrain.spatial.GridAtLines``)
    ``nearby``        the nearby-link approach of Overeem et al. (2016)
                      (``pycomlink``)
    ``cnn``           the CNN of Polz et al. (2020), trained on German links
                      against radar (``pycomlink`` loader, PyTorch)

Every variant is scored against three references that do not share its
errors, all matched with poligrain:

* **gauges** - the 10 municipal gauges, paired with every link whose *path*
  passes within 1 km (``get_closest_points_to_line``), at 15 minutes. The
  primary criterion: it is the only reference that is itself a direct
  measurement.
* **radar along the path** - 5 minutes, every link. Radar has its own bias
  here (see ``compare_radar_cml``), so this is read for correlation and
  wet/dry skill, not for magnitude.
* **the OpenSense reference retrieval** shipped with the subset - context,
  not truth.

And one robustness check the README asked for: the **per-day ratio to
gauges** over the wet days of the record. A calibration that is right on
average but swings 2x day to day is not transferable.

The same variants run on OpenRainER (``--dataset openrainer``) to test
whether any of this transfers from a dense 10 s urban network to a sparse
1-minute regional one. One combination - nearby-link wet/dry with the
Leijnse (2008) model, ``retrieval.retrieve_improved`` - beats the default on
every score on both.

    python projects/opensense_pipeline/src/retrieval_benchmark.py
    python projects/opensense_pipeline/src/retrieval_benchmark.py --dataset openrainer
    python projects/opensense_pipeline/src/retrieval_benchmark.py --quick   # one day
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from core.opensense import evaluation as ev  # noqa: E402
from core.opensense import example_data  # noqa: E402
from core.opensense import retrieval as rt  # noqa: E402
from core.opensense import wet_dry  # noqa: E402
from core.opensense.quality import censored_at_floor  # noqa: E402

RESULTS = HERE.parent / "results"

LINK_CHUNK = 48


@dataclass(frozen=True)
class Variant:
    """One retrieval configuration: a wet/dry mask and config overrides."""

    key: str
    label: str
    wet: str = "rolling_std"            # rolling_std | radar | nearby | cnn
    changes: dict = field(default_factory=dict)
    qc: bool = False                    # mask samples at the receiver floor


VARIANTS = (
    Variant("default", "default: rolling-std, saturating WAA 0.5 dB"),
    Variant("waa_none", "no wet-antenna correction",
            changes=dict(waa_model="none")),
    Variant("waa_sat_2.3", "saturating WAA 2.3 dB (Schleiss 2013 magnitude)",
            changes=dict(waa_max_db=2.3)),
    Variant("waa_pastorek", "WAA Pastorek 2021",
            changes=dict(waa_model="pastorek2021")),
    Variant("waa_leijnse", "WAA Leijnse 2008",
            changes=dict(waa_model="leijnse2008")),
    Variant("wet_radar", "radar wet/dry, saturating WAA 0.5 dB", wet="radar"),
    Variant("wet_nearby", "nearby-link wet/dry, saturating WAA 0.5 dB",
            wet="nearby"),
    Variant("wet_nearby_pastorek", "nearby-link wet/dry + WAA Pastorek 2021",
            wet="nearby", changes=dict(waa_model="pastorek2021")),
    Variant("wet_nearby_leijnse", "nearby-link wet/dry + WAA Leijnse 2008",
            wet="nearby", changes=dict(waa_model="leijnse2008")),
    Variant("wet_nearby_zero", "nearby-link wet/dry, zero when dry",
            wet="nearby", changes=dict(zero_when_dry=True)),
    Variant("wet_cnn", "CNN wet/dry (Polz 2020), saturating WAA 0.5 dB", wet="cnn"),
    Variant("wet_cnn_pastorek", "CNN wet/dry + WAA Pastorek 2021",
            wet="cnn", changes=dict(waa_model="pastorek2021")),
    Variant("wet_cnn_leijnse", "CNN wet/dry + WAA Leijnse 2008",
            wet="cnn", changes=dict(waa_model="leijnse2008")),
    Variant("wet_cnn_zero", "CNN wet/dry, zero when dry",
            wet="cnn", changes=dict(zero_when_dry=True)),
    Variant("default_qc", "default + receiver-floor QC", qc=True),
    Variant("improved", "retrieve_improved: nearby + Leijnse + QC",
            wet="nearby", changes=dict(waa_model="leijnse2008"), qc=True),
)
VARIANTS_BY_KEY = {v.key: v for v in VARIANTS}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    """How to benchmark one example dataset."""

    key: str
    name: str
    gauge: str                   # example_data component
    gauge_match_m: float         # link path to gauge
    gauge_step: str              # scoring step against gauges
    radar_step: str              # scoring step against radar
    quick_day: str
    has_reference: bool          # ships an OpenSense reference R

    @property
    def label(self) -> str:
        """Timestamp convention of the references (see evaluation.aggregate)."""
        return example_data.DATASETS[self.key].accumulation_label


SPECS = {
    # dense urban network, 10 s, 10 gauges, 5-min radar
    "openmrg": Spec("openmrg", "OpenMRG", "gauge_municipal", 1000.0,
                    "15min", "5min", "2015-07-28", has_reference=True),
    # sparse regional network, 1 min, 319 gauges, 15-min radar. Paths are
    # longer and gauges sparser, so 2 km matches 88 links where 1 km gets 62.
    "openrainer": Spec("openrainer", "OpenRainER", "gauge", 2000.0,
                       "15min", "15min", "2022-08-18", has_reference=False),
}


def load(spec: Spec, window: slice | None) -> dict:
    """CML signals, radar and gauges as rates, CML on a regular time axis.

    The rolling windows count samples, so a gap in the time axis silently
    shortens them. OpenRainER's 8-day subset has 11,412 of 11,520 minutes;
    reindexing to a regular axis turns the missing ones into NaN, which the
    chain masks as gaps.
    """
    data = example_data.load(spec.key, "8d", time=window, verbose=False)
    cml = data["cml"]
    step = pd.Timedelta(seconds=rt.sampling_interval_s(cml.time))
    regular = pd.date_range(cml.time.values[0], cml.time.values[-1], freq=step)
    if regular.size != cml.sizes["time"]:
        print(f"  CML time axis: {cml.sizes['time']:,} of {regular.size:,} "
              f"steps present; regularized")
        data["cml"] = cml.reindex(time=regular)

    g = data[spec.gauge]
    per_hour = 3600.0 / rt.sampling_interval_s(g.time)
    data["gauge_R"] = (g.rainfall_amount * per_hour).transpose("time", "id")
    data["gauge_R"].attrs["units"] = "mm h-1"
    return data


# --------------------------------------------------------------------------
# external wet/dry masks come from core.opensense.wet_dry
# --------------------------------------------------------------------------
def _flat_mask(mask: xr.DataArray | None, fallback: np.ndarray, n_sub: int):
    """Link mask (time, cml) -> sublink mask (time, cml*sub), NaN -> fallback."""
    if mask is None:
        return None
    m = np.repeat(np.asarray(mask, dtype=float)[:, :, None], n_sub, axis=2)
    m = m.reshape(fallback.shape)
    return np.where(np.isfinite(m), m > 0.5, fallback)


# --------------------------------------------------------------------------
# running the variants
# --------------------------------------------------------------------------
def run_variants(cml: xr.Dataset, variants, masks: dict,
                 aggregate=("5min", "15min"), label: str = "start") -> dict:
    """Every variant over every link, chunked by link to bound memory.

    Variants sharing a wet mask share its baseline, so the expensive rolling
    median is computed once per mask per chunk rather than once per variant.
    Returns {variant key: {freq: DataArray (time, cml_id)}}.
    """
    interval = rt.sampling_interval_s(cml.time)
    base_cfg = rt.RetrievalConfig.for_interval(interval)
    cml_ids = cml.cml_id.values
    out = {v.key: {f: [] for f in aggregate} for v in variants}

    for start in range(0, cml_ids.size, LINK_CHUNK):
        chunk = cml.isel(cml_id=slice(start, start + LINK_CHUNK))
        d = chunk.transpose("time", "cml_id", "sublink_id")
        raw_loss = rt.total_loss_from(d.rsl.values, d.tsl.values)
        n_t, n_c, n_s = raw_loss.shape
        raw_loss = raw_loss.reshape(n_t, n_c * n_s)
        censored = censored_at_floor(d.rsl.values.reshape(n_t, n_c * n_s), interval)
        prepared = {}
        for qc in dict.fromkeys(v.qc for v in variants):
            lo = np.where(censored, np.nan, raw_loss) if qc else raw_loss
            prepared[qc] = (~np.isfinite(lo), pd.DataFrame(lo).ffill().bfill().to_numpy())
        length, freq, pol = (a.ravel() for a in rt.link_metadata(
            chunk, d.rsl.isel(time=0, drop=True)))

        baselines = {}
        for v in variants:
            cfg = base_cfg.with_(**v.changes)
            gaps, loss = prepared[v.qc]
            if (v.wet, v.qc) not in baselines:
                rs_wet = rt.wet_dry_rolling_std(loss, base_cfg.wet_window,
                                                base_cfg.wet_threshold_db)
                m = masks.get(v.wet)
                wet = rs_wet if m is None else _flat_mask(
                    m.isel(cml_id=slice(start, start + LINK_CHUNK)), rs_wet, n_s)
                baselines[(v.wet, v.qc)] = (wet, rt.baseline_from_dry(
                    loss, wet, cfg.baseline_window))
            wet, baseline = baselines[(v.wet, v.qc)]

            a_obs = np.clip(loss - baseline, 0.0, None)
            a_obs[a_obs < cfg.min_attenuation_db] = 0.0
            if cfg.zero_when_dry:
                a_obs[~wet] = 0.0
            rain, _ = rt.rain_from_attenuation(a_obs, length, freq, pol, cfg)
            rain[gaps] = np.nan
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                per_link = np.nanmean(rain.reshape(n_t, n_c, n_s), axis=2)
            da = xr.DataArray(per_link, dims=("time", "cml_id"),
                              coords={"time": chunk.time,
                                      "cml_id": chunk.cml_id.values})
            for f in aggregate:
                out[v.key][f].append(ev.aggregate(da, f, label=label))
        print(f"    links {start + n_c:4d}/{cml_ids.size}", flush=True)

    return {k: {f: xr.concat(parts, "cml_id") for f, parts in by_f.items()}
            for k, by_f in out.items()}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def per_day_ratio(gauge_at_links: xr.DataArray, est: xr.DataArray, step: str,
                  min_gauge_mm: float = 1.0) -> dict:
    """Ratio of estimate to gauge totals, day by day, over matched pairs."""
    g, e = xr.align(gauge_at_links, est.transpose(*gauge_at_links.dims))
    both = np.isfinite(g) & np.isfinite(e)
    g, e = g.where(both), e.where(both)
    hours = pd.Timedelta(step).total_seconds() / 3600.0
    # mean rates per step -> mm per day, averaged over matched links
    g_day = (g.resample(time="1D").sum() * hours).mean("cml_id")
    e_day = (e.resample(time="1D").sum() * hours).mean("cml_id")
    wet = g_day >= min_gauge_mm
    ratios = (e_day / g_day).where(wet).dropna("time")
    return {str(t)[:10]: float(r) for t, r in
            zip(ratios.time.values, ratios.values)}


def common_support(series: dict, refs: dict, spec: Spec) -> tuple[dict, dict]:
    """Restrict every series and reference to the pairs all of them have.

    Without this the rows are not comparable: this chain masks links under
    0.5 km as unusable while the OpenSense reference fills them, so scoring
    each on its own valid pairs compares different samples.
    """
    keep = {}
    pairs = [("gauge", spec.gauge_step), ("radar", spec.radar_step)]
    if "pws" in refs:
        pairs.append(("pws", spec.gauge_step))
    for ref_key, step in pairs:
        m = np.isfinite(refs[ref_key].transpose("time", "cml_id"))
        for s in series.values():
            m = m & np.isfinite(s[step].transpose("time", "cml_id"))
        keep[ref_key] = m
    out_refs = {k: refs[k].transpose("time", "cml_id").where(keep[k]) for k, _ in pairs}
    out_series = {}
    for key, s in series.items():
        out_series[key] = {k: s[step].transpose("time", "cml_id").where(keep[k])
                           for k, step in pairs}
    if "opensense_reference" in out_series:
        out_refs["opensense"] = out_series["opensense_reference"]["radar"]
    return out_series, out_refs


def score_all(series: dict, refs: dict, spec: Spec) -> list[dict]:
    series, refs = common_support(series, refs, spec)
    rows = []
    for key, s in series.items():
        g = ev.rainfall_metrics(refs["gauge"], s["gauge"])
        r = ev.rainfall_metrics(refs["radar"], s["radar"])
        o = (ev.rainfall_metrics(refs["opensense"], s["radar"])
             if "opensense" in refs and key != "opensense_reference" else {})
        days = per_day_ratio(refs["gauge"], s["gauge"], spec.gauge_step)
        logs = np.log(np.array(list(days.values()))) if days else np.array([np.nan])
        label = (VARIANTS_BY_KEY[key].label if key in VARIANTS_BY_KEY
                 else "OpenSense reference retrieval (shipped R)")
        p = ev.rainfall_metrics(refs["pws"], s["pws"]) if "pws" in refs else {}
        rows.append({
            "key": key, "label": label,
            "gauge": g, "radar": r, "opensense_ref": o, "pws": p,
            "daily_ratio": days,
            # spread of the daily ratio, as a factor: exp(std(log ratio))
            "daily_ratio_spread": float(np.exp(np.nanstd(logs))),
        })
    return rows


def print_table(rows: list[dict], spec: Spec) -> None:
    gh = f"-- gauges, {spec.gauge_step} --"
    rh = f"- radar path, {spec.radar_step} -"
    print(f"\n{'variant':52s} {gh:>22s}   {rh:>22s}  {'daily':>6s}")
    print(f"{'':52s} {'ratio':>7s}{'r':>7s}{'MCC':>7s}   "
          f"{'ratio':>7s}{'r':>7s}{'MCC':>7s}   {'spread':>6s}")
    for row in rows:
        g, r = row["gauge"], row["radar"]
        p = row.get("pws") or {}
        extra = (f"   PWS ratio {p['ratio']:5.2f} r {p['r']:5.3f} MCC {p['mcc']:5.3f}"
                 if p else "")
        print(f"{row['label'][:52]:52s} {g['ratio']:7.2f}{g['r']:7.3f}{g['mcc']:7.3f}   "
              f"{r['ratio']:7.2f}{r['r']:7.3f}{r['mcc']:7.3f}   "
              f"{row['daily_ratio_spread']:6.2f}x{extra}")


# --------------------------------------------------------------------------
def pws_reference(cml: xr.Dataset, spec: Spec) -> tuple[xr.DataArray, dict]:
    """QC'd OpenMRG2 PWS, usable stations only, matched to link paths like the gauges.

    The PWS share the municipal gauges' start-stamped convention (the raw
    city-gauge file is identical to the curated one, and a lag scan puts
    PWS and gauges on the same labels).
    """
    from core.opensense import openmrg2, pws_qc

    d = openmrg2.load()
    window = slice(cml.time.values[0], cml.time.values[-1])
    qc = pws_qc.flag(d["pws"])
    use = pws_qc.usable(qc)
    per_hour = 3600.0 / pd.Timedelta(qc.attrs["step"]).total_seconds()
    rate = (qc.rainfall_qc.sel(id=use) * per_hour).transpose("time", "id").sel(time=window)
    pts = d["pws"].sel(id=use)
    rate = rate.assign_coords({c: pts[c] for c in ("x", "y", "lon", "lat")})
    closest = ev.closest_gauges(cml, rate.to_dataset(name="rainfall_amount"), spec.gauge_match_m)
    at_links = ev.gauge_series_at_links(ev.aggregate(rate, spec.gauge_step, label=spec.label),
                                        closest)
    n = int(np.isfinite(closest.distance.isel(n_closest=0)).sum())
    return at_links, {"stations": int(qc.sizes["id"]), "usable": len(use), "links_matched": n,
                      "qc": "pws_qc.flag defaults, usable() stations"}


def cnn_mask(spec: Spec, cml: xr.Dataset, quick: bool) -> xr.DataArray:
    """``wet_dry.cnn`` for the whole window, cached: it is the one slow mask.

    About 50 minutes for OpenMRG's 364 links over 8 days on a laptop CPU;
    the cache key is the dataset, window and model, so a changed model or
    threshold recomputes.
    """
    import hashlib

    key = hashlib.md5(f"{wet_dry.POLZ2020_MODEL}|0.82|{cml.time.values[0]!s}|"
                      f"{cml.time.values[-1]!s}|{cml.sizes['cml_id']}".encode()).hexdigest()[:10]
    path = example_data.CACHE / "_masks" / f"{spec.key}{'_quick' if quick else ''}_cnn_{key}.nc"
    if path.exists():
        return xr.open_dataarray(path).load()
    t = time.time()
    print("  running the CNN (cached afterwards) ...", flush=True)
    mask = wet_dry.cnn(cml)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.rename("wet").to_netcdf(path)
    print(f"  CNN done in {time.time() - t:.0f} s")
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(SPECS), default="openmrg")
    ap.add_argument("--quick", action="store_true",
                    help="one wet day instead of all 8")
    ap.add_argument("--variants", nargs="*", choices=sorted(VARIANTS_BY_KEY),
                    help="default: all")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--pws", action="store_true",
                    help="OpenMRG only: add QC'd OpenMRG2 PWS as a third reference")
    args = ap.parse_args()

    spec = SPECS[args.dataset]
    window = slice(spec.quick_day, spec.quick_day) if args.quick else None
    variants = ([VARIANTS_BY_KEY[k] for k in args.variants]
                if args.variants else list(VARIANTS))

    t0 = time.time()
    print(f"Loading {spec.name} 8d (CML signals, radar, gauges)")
    data = load(spec, window)
    cml, radar = data["cml"], data["radar"]
    print(f"  {cml.sizes['cml_id']} links x {cml.sizes['sublink_id']} sublinks, "
          f"{cml.sizes['time']:,} steps at {rt.sampling_interval_s(cml.time):g} s; "
          f"references stamped at interval {spec.label}")

    print("Matching sensors with poligrain")
    radar_path = ev.radar_along_links(radar.R, cml)
    closest = ev.closest_gauges(cml, data[spec.gauge], spec.gauge_match_m)
    gauge_at_links = ev.gauge_series_at_links(
        ev.aggregate(data["gauge_R"], spec.gauge_step, label=spec.label), closest)
    n_matched = int(np.isfinite(closest.distance.isel(n_closest=0)).sum())
    print(f"  radar along {radar_path.sizes['cml_id']} link paths; "
          f"{n_matched} links within {spec.gauge_match_m:g} m of a gauge")

    pws_info = None
    if args.pws:
        if spec.key != "openmrg":
            ap.error("--pws: PWS exist for OpenMRG (OpenMRG2) only")
        pws_at_links, pws_info = pws_reference(cml, spec)
        print(f"  PWS reference: {pws_info['usable']} of {pws_info['stations']} OpenMRG2 PWS "
              f"pass QC; {pws_info['links_matched']} links within "
              f"{spec.gauge_match_m:g} m of one")

    masks = {}
    if any(v.wet == "radar" for v in variants):
        masks["radar"] = wet_dry.from_radar(radar_path, cml.time)
    if any(v.wet == "nearby" for v in variants):
        masks["nearby"] = wet_dry.nearby_links(cml)
        decided = float(np.isfinite(masks["nearby"]).mean())
        print(f"  nearby-link wet/dry (Overeem 2016): decided {decided:.0%} "
              f"of link-samples, rolling-std elsewhere")
    if any(v.wet == "cnn" for v in variants):
        masks["cnn"] = cnn_mask(spec, cml, args.quick)
        decided = float(np.isfinite(masks["cnn"]).mean())
        print(f"  CNN wet/dry (Polz 2020): decided {decided:.0%} of link-samples, "
              f"wet {float(masks['cnn'].mean()):.1%}")

    steps = tuple(dict.fromkeys((spec.gauge_step, spec.radar_step)))
    print(f"Running {len(variants)} variants")
    series = run_variants(cml, variants, masks, aggregate=steps, label=spec.label)

    if spec.has_reference:
        ref = cml.R.mean("sublink_id").transpose("time", "cml_id")
        series["opensense_reference"] = {
            f: ev.aggregate(ref, f, label=spec.label) for f in steps}
    refs = {"gauge": gauge_at_links, "radar": radar_path}
    if pws_info:
        refs["pws"] = pws_at_links

    rows = score_all(series, refs, spec)
    print_table(rows, spec)

    RESULTS.mkdir(parents=True, exist_ok=True)
    # OpenMRG keeps the original file names
    tag = ("" if spec.key == "openmrg" else f"_{spec.key}") + \
          ("_quick" if args.quick else "")
    summary = {
        "dataset": f"{spec.name} example subset 8d",
        "window": spec.quick_day if args.quick else
        f"{str(cml.time.values[0])[:10]} .. {str(cml.time.values[-1])[:10]}",
        "links": int(cml.sizes["cml_id"]),
        "links_matched_to_gauges": n_matched,
        "gauge_match_m": spec.gauge_match_m,
        "gauge_step": spec.gauge_step, "radar_step": spec.radar_step,
        "reference_time_label": spec.label,
        "wet_threshold_mm_h": ev.WET_THRESHOLD_MM_H,
        "nearby_mask_decided": (float(np.isfinite(masks["nearby"]).mean())
                                if "nearby" in masks else None),
        "cnn_mask_decided": (float(np.isfinite(masks["cnn"]).mean())
                             if "cnn" in masks else None),
        "pws_reference": pws_info,
        "rows": rows,
    }
    out = RESULTS / f"retrieval_benchmark{tag}.json"
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}  ({time.time() - t0:.0f} s)")

    if not args.no_figure:
        import plots_retrieval
        from core import viz_style as vs
        vs.use_style()
        fig = RESULTS / f"retrieval_benchmark{tag}.png"
        plots_retrieval.figure(rows, fig, title=f"{spec.name} {summary['window']}",
                               gauge_step=spec.gauge_step, radar_step=spec.radar_step)
        print(f"wrote {fig.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
