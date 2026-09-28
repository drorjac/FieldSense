"""End-to-end study: every event x every method x every link set -> one report.

    from nyc_rain_maps.study import StudyConfig, run_study
    study = run_study(StudyConfig())            # catalog events with link data
    study.write_report("results/study")  # report.md + figures + CSV tables

or ``python src/run.py study -o results/study``.

What it does, in order:

1. **Events** - the catalogued events that fall inside the OpenMesh record (or any list
   of ``(label, type, start, end)`` given explicitly).
2. **Per event and link set** - :func:`nyc_rain_maps.pipeline.run_event`: link QC,
   every rain-retrieval method, IDW maps on the MRMS grid, hourly scores vs MRMS.
3. **Pooling** - besides per-event scores, the hourly (map cell, hour) pairs of all
   events are pooled per precipitation type, so the headline numbers weigh every hour
   equally instead of averaging per-event ratios (the README's scoring rules: report per
   event *and* pooled). Pooled scores use only cell-hours that every method has, so
   methods are compared on identical samples.
4. **Report** - ``report.md`` with provenance, QC, tables, automatically derived
   findings and limitations; figures; CSV tables for further work.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .settings import VERSION as __version__
from .scores import scores
from .pipeline import DEFAULT_METHODS, METHODS, run_event

log = logging.getLogger(__name__)

TYPE_ORDER = ["rain", "mix", "snow"]


@dataclass
class StudyConfig:
    """What to analyse. Defaults reproduce the repository's reference study."""

    name: str = "study"
    methods: list = field(default_factory=lambda: list(DEFAULT_METHODS))
    link_sets: list = field(default_factory=lambda: ["shared8", "selected", "qc"])
    sensors: bool = True                # multi-sensor comparison (CML / MRMS / PWS / ASOS)
    sensor_link_set: str = "selected"   # link set whose CML maps enter the sensor comparison
    events: list | None = None          # [(label, ptype, start, end)]; None = catalog
    only_selected: bool = True          # catalog: the top-5-per-type selection only
    near_km: float = 2.0
    spinup: str = "6h"
    wet_threshold_mm: float = 0.1
    min_coverage: float = 0.9           # methods below this define no common sample


@dataclass
class Study:
    config: StudyConfig
    events: pd.DataFrame
    event_scores: pd.DataFrame
    link_scores: pd.DataFrame
    pooled: pd.DataFrame
    pairs: pd.DataFrame
    qc: dict
    errors: dict
    started: str
    finished: str
    sensor_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    sensor_pooled: pd.DataFrame = field(default_factory=pd.DataFrame)
    asos_pooled: pd.DataFrame = field(default_factory=pd.DataFrame)
    sensor_figures: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- tables

    def summary_table(self, link_set: str = "shared8", near: bool = False) -> pd.DataFrame:
        """Pooled scores per type and method (rows ordered by type, then NRMSE)."""
        p = self.pooled[(self.pooled.link_set == link_set) & (self.pooled.near == near)]
        p = p.assign(_t=p.ptype.map({t: i for i, t in enumerate(TYPE_ORDER)}))
        return p.sort_values(["_t", "nrmse"]).drop(columns="_t").reset_index(drop=True)

    def findings(self) -> list[str]:
        """Plain-language findings derived from the numbers (no hand-written claims)."""
        out = []
        for ls in self.config.link_sets:
            tab = self.summary_table(ls)
            for t in TYPE_ORDER:
                sub = tab[tab.ptype == t]
                if sub.empty:
                    continue
                full = sub[sub.coverage >= self.config.min_coverage]
                full = full if not full.empty else sub
                best = full.iloc[0]
                closest = full.iloc[(full.rel_bias.abs()).argsort().iloc[0]]
                sign = "over" if sub.rel_bias.median() > 0 else "under"
                gaps = sub[sub.coverage < self.config.min_coverage]
                gap_txt = ("; incomplete output (not ranked): " + ", ".join(
                    f"`{g.method}` {g.coverage:.0%} coverage, NRMSE {g.nrmse:.2f} on it" for g in gaps.itertuples())
                    if len(gaps) else "")
                out.append(
                    f"**{t}, {ls} links** ({int(best.events)} events, {int(best.n):,} cell-hours): "
                    f"lowest hourly NRMSE {best.nrmse:.2f} with `{best.method}`; smallest total bias "
                    f"{closest.rel_bias:+.0%} with `{closest.method}`; the methods typically "
                    f"{sign}-estimate (median bias {sub.rel_bias.median():+.0%}){gap_txt}.")
        if not self.sensor_pooled.empty:
            for t in TYPE_ORDER:
                sp = self.sensor_pooled[self.sensor_pooled.ptype == t]
                pr = sp[(sp.estimate == "PWS") & (sp.reference == "MRMS")]
                cr = sp[sp.estimate.str.startswith("CML") & (sp.reference == "MRMS")
                        & (sp.coverage >= self.config.min_coverage)].sort_values("nrmse")
                if pr.empty or cr.empty:
                    continue
                out.append(f"**Sensors, {t}** (cells <= {self.config.near_km:g} km of links): the PWS gauge map "
                           f"differs from MRMS by NRMSE {pr.nrmse.iloc[0]:.2f} (bias {pr.rel_bias.iloc[0]:+.0%}); "
                           f"the best CML map (`{cr.estimate.iloc[0].replace('CML ', '')}`) by "
                           f"{cr.nrmse.iloc[0]:.2f} ({cr.rel_bias.iloc[0]:+.0%}) - the gauge-radar disagreement "
                           f"is the floor any CML score should be read against.")
        if "selected" in self.config.link_sets and "shared8" in self.config.link_sets:
            a, b = self.summary_table("shared8"), self.summary_table("selected")
            m = a.merge(b, on=["ptype", "method"], suffixes=("_8", "_sel"))
            if not m.empty:
                out.append(f"**Cherry-picked links** (gauge-calibrated selection, {int(self.events.shape[0])} "
                           f"held-out events): NRMSE within 0.05 of the hand-picked 8 in "
                           f"{int(((m.nrmse_sel - m.nrmse_8).abs() <= 0.05).sum())} of {len(m)} cases, better in "
                           f"{int((m.nrmse_sel < m.nrmse_8 - 0.05).sum())}, worse in "
                           f"{int((m.nrmse_sel > m.nrmse_8 + 0.05).sum())}.")
        if set(self.config.link_sets) >= {"shared8", "qc"}:
            a, b = self.summary_table("shared8"), self.summary_table("qc")
            m = a.merge(b, on=["ptype", "method"], suffixes=("_8", "_qc"))
            if not m.empty:
                better = int((m.nrmse_qc < m.nrmse_8).sum())
                out.append(f"**Link set:** the automatic-QC link set beats the 8 hand-picked links in "
                           f"{better} of {len(m)} (type, method) cases by pooled NRMSE.")
        return out

    # ---------------------------------------------------------------- report

    def write_report(self, out_dir: str | Path) -> Path:
        from . import plots as plotting

        out = Path(out_dir)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        self.events.to_csv(out / "events.csv", index=False)
        self.event_scores.round(4).to_csv(out / "event_scores.csv", index=False)
        self.link_scores.round(4).to_csv(out / "link_scores.csv", index=False)
        self.pooled.round(4).to_csv(out / "pooled_scores.csv", index=False)

        if not self.sensor_pooled.empty:
            self.sensor_events.round(4).to_csv(out / "sensor_event_scores.csv", index=False)
            self.sensor_pooled.round(4).to_csv(out / "sensor_pooled_scores.csv", index=False)
            self.asos_pooled.round(4).to_csv(out / "asos_point_scores.csv", index=False)
        for label, comp in self.sensor_figures.items():
            comp.save(out / "events" / _slug(label))
        figs = {
            "nrmse": plotting.study_scores_figure(self, "nrmse", out / "figures" / "nrmse_by_type.png"),
            "bias": plotting.study_event_bias_figure(self, out / "figures" / "bias_by_event.png"),
            "scatter": plotting.study_scatter_figure(self, out / "figures" / "hourly_scatter_rain.png"),
            "links": plotting.study_link_bias_figure(self, out / "figures" / "link_bias_rain.png"),
        }
        (out / "report.md").write_text(self._markdown(figs))
        (out / "study.json").write_text(json.dumps(
            {"config": asdict(self.config), "qc": self.qc, "errors": self.errors,
             "started": self.started, "finished": self.finished, "version": __version__},
            indent=2, default=str))
        return out

    def _markdown(self, figs) -> str:
        c = self.config
        L = [f"# CML rainfall maps vs MRMS radar - study `{c.name}`", "",
             f"_Generated {self.finished} by `nyc_rain_maps` {__version__} "
             f"(Python {platform.python_version()}). Regenerate: `python src/run.py study`._", "",
             "## Design", "",
             f"- **Events:** {len(self.events)} precipitation events inside the OpenMesh record "
             f"({', '.join(f'{(self.events.ptype == t).sum()} {t}' for t in TYPE_ORDER)}), "
             "from the MRMS/ASOS event catalog (`events/`).",
             f"- **Methods:** {', '.join(f'`{m}`' for m in c.methods)} "
             "(definitions: `docs/METHODS.md`).",
             f"- **Link sets:** {', '.join(f'`{s}`' for s in c.link_sets)} - `shared8` = the 8 sublinks "
             "both implementations picked by eye; `selected` = the cherry-picking pipeline "
             "(`nyc_rain_maps.link_selection`: implementation_1's contact-loss, rain-response and "
             "accumulation checks + one sublink per path, calibrated on PWS gauges over rain events "
             "*not* in this study - see `results/link_selection/`); `qc` = the README's automatic "
             "metadata/time-series QC over all 103 sublinks, re-run per event.",
             f"- **Common footing:** {c.spinup} spin-up; 1-min link rain -> hourly accumulations "
             "(hour-ending, >= 80 % coverage); IDW power 2, 10 km radius, on the MRMS 0.01 deg grid; "
             "reference MRMS `MultiSensor_QPE_01H_Pass2`; only cells/hours where both are valid.",
             f"- **Scores:** pooled over all cell-hours of a type (every hour weighs the same) and per "
             f"event. The sample is the cell-hours every full-coverage method has (>= "
             f"{c.min_coverage:.0%} of the best-covered method in that event); a method with gaps is "
             f"scored on its share of that sample and its **coverage** is reported, so it can neither "
             f"remove its failed hours from everyone's score nor look better by staying silent; NRMSE = RMSE / mean(radar); wet = >= {c.wet_threshold_mm} mm/h for POD/FAR/CSI. "
             f"'near' = cells within {c.near_km} km of a link.", ""]
        L += ["## Findings", ""] + [f"- {f}" for f in self.findings()] + [""]
        for ls in c.link_sets:
            for near in (False, True):
                tab = self.summary_table(ls, near)
                if tab.empty:
                    continue
                L += [f"## Pooled scores - `{ls}` links, {'cells near links' if near else 'all cells'}", "",
                      "| type | method | events | coverage | cell-hours | radar mean mm/h | CML mean mm/h | rel. bias "
                      "| NRMSE | corr | POD | FAR | CSI |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for r in tab.itertuples():
                    L.append(f"| {r.ptype} | `{r.method}` | {r.events} | {r.coverage:.0%} | {r.n:,} | {r.mean_ref:.2f} | "
                             f"{r.mean_est:.2f} | {r.rel_bias:+.0%} | {r.nrmse:.2f} | {r.corr:.2f} | "
                             f"{r.pod:.2f} | {r.far:.2f} | {r.csi:.2f} |")
                L.append("")
        L += ["![NRMSE by type](figures/nrmse_by_type.png)", "",
              "![Bias per event](figures/bias_by_event.png)", "",
              "![Hourly scatter, rain](figures/hourly_scatter_rain.png)", "",
              "![Per-link bias, rain](figures/link_bias_rain.png)", ""]
        if not self.sensor_pooled.empty:
            L += ["## Multi-sensor agreement", "",
                  f"CML maps (`{c.sensor_link_set}` links), the PWS gauge map (implementation_1's "
                  "5th-95th-percentile gauge filter, same IDW) and MRMS, compared pairwise on the same "
                  f"cell-hours within {c.near_km:g} km of a link; ASOS (never used in a map) is the "
                  "independent point check. Per-event snapshot grids, totals and time series: "
                  "`events/<event>/` next to this report (not tracked; `python src/run.py compare` redraws any one).", "",
                  "| type | estimate | reference | coverage | cell-hours | rel. bias | NRMSE | corr | CSI |",
                  "|---|---|---|---|---|---|---|---|---|"]
            sp = self.sensor_pooled.assign(_t=self.sensor_pooled.ptype.map({t: i for i, t in enumerate(TYPE_ORDER)}))
            for r in sp.sort_values(["_t", "reference", "nrmse"]).itertuples():
                L.append(f"| {r.ptype} | {r.estimate} | {r.reference} | {r.coverage:.0%} | {r.n:,} | {r.rel_bias:+.0%} | "
                         f"{r.nrmse:.2f} | {r.corr:.2f} | {r.csi:.2f} |")
            L += ["", "**At the ASOS stations** (hourly, map value in the station's cell vs the gauge):", "",
                  "| type | map | station-hours | rel. bias | NRMSE | corr |", "|---|---|---|---|---|---|"]
            ap = self.asos_pooled.assign(_t=self.asos_pooled.ptype.map({t: i for i, t in enumerate(TYPE_ORDER)}))
            for r in ap.sort_values(["_t", "nrmse"]).itertuples():
                L.append(f"| {r.ptype} | {r.map} | {r.n} | {r.rel_bias:+.0%} | {r.nrmse:.2f} | {r.corr:.2f} |")
            L.append("")
        L += ["## Per-event NRMSE (`shared8`)", "",
              "| event | type | " + " | ".join(f"`{m}`" for m in c.methods) + " |",
              "|---|---|" + "---|" * len(c.methods)]
        es = self.event_scores[self.event_scores.link_set == c.link_sets[0]]
        for ev, g in es.groupby("event", sort=False):
            vals = g.set_index("method").reindex(c.methods)["nrmse"]
            L.append(f"| {ev} | {g.ptype.iloc[0]} | " + " | ".join(
                "-" if pd.isna(v) else f"{v:.2f}" for v in vals) + " |")
        L.append("")
        if self.qc:
            L += ["## Link QC (automatic link set)", "",
                  "| event | links in | links out | rejected by reason |", "|---|---|---|---|"]
            for ev, q in self.qc.items():
                reasons = "; ".join(f"{k}: {v}" for k, v in q.get("by_reason", {}).items())
                L.append(f"| {ev} | {q.get('links_in')} | {q.get('links_out')} | {reasons} |")
            L.append("")
        if self.errors:
            L += ["## Errors", ""] + [f"- {k}: {v}" for k, v in self.errors.items()] + [""]
        L += ["## Limitations", "",
              "- MRMS is a radar estimate, not ground truth (beam height over NYC, urban clutter, "
              "empirical Z-R, uncertain snow QPE).",
              "- Snow and mixed-event sample sizes are small (the 2023-24 winter was snow-poor); "
              "treat those rows as indicative.",
              "- IDW from a few links cannot reproduce convective structure smaller than the link "
              "spacing; 'near' scores isolate cells the network actually constrains.",
              "- Methods run with the parameters of the original implementations; nothing is "
              "tuned against this radar data.",
              "", "## Files", "",
              "`events.csv`, `event_scores.csv` (per event x link set x method), `pooled_scores.csv`, "
              "`link_scores.csv` (per link vs radar along the path), `study.json` (config, QC, errors).",
              ""]
        return "\n".join(L)


# -------------------------------------------------------------------- running


def _catalog_events(only_selected: bool) -> pd.DataFrame:
    from .settings import repo_events_dir
    df = pd.read_csv(repo_events_dir() / "all_detected_events.csv")
    df = df[df["in_openmesh"].astype(bool)]
    if only_selected:
        df = df[df["selected"].astype(bool)]
    df = df.assign(label=[f"{p}#{int(r)} {pd.Timestamp(s):%Y-%m-%d}" for p, r, s in
                          zip(df.ptype, df["rank"], df.start)])
    return df[["label", "ptype", "start", "end", "total_mm"]].reset_index(drop=True)


def _pairs(res, method: str, near_km: float) -> pd.DataFrame:
    est, ref = res.maps_hourly[method], res.radar_hourly
    ref = ref.sel(time=est.time)
    near = np.broadcast_to((res.distance_to_link_km <= near_km).values, est.shape)
    e, r = est.values.ravel(), ref.values.ravel()
    ok = np.isfinite(e) & np.isfinite(r)
    return pd.DataFrame({"cell": np.flatnonzero(ok), "est": e[ok].astype("float32"),
                         "ref": r[ok].astype("float32"), "near": near.ravel()[ok]})


def run_study(config: StudyConfig | None = None, client=None) -> Study:
    """Run the full study (see module docstring)."""
    config = config or StudyConfig()
    unknown = [m for m in config.methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {sorted(METHODS)}")
    events = (pd.DataFrame(config.events, columns=["label", "ptype", "start", "end"])
              if config.events is not None else _catalog_events(config.only_selected))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ev_rows, link_rows, pair_frames, qc, errors = [], [], [], {}, {}
    sens_rows, sens_arrays, asos_arrays, sensor_figs = [], [], [], {}
    for ev in events.itertuples():
        for ls in config.link_sets:
            log.info("study: %s, %s links", ev.label, ls)
            try:
                res = run_event(ev.start, ev.end, methods=config.methods, link_set=ls,
                                spinup=config.spinup, near_km=config.near_km, client=client)
            except Exception as exc:
                errors[f"{ev.label} / {ls}"] = repr(exc)
                log.exception("event %s failed", ev.label)
                continue
            if ls == "qc" and res.qc_summary:
                qc[ev.label] = {k: v for k, v in res.qc_summary.items() if k != "config"}
            for k, v in res.errors.items():
                errors[f"{ev.label} / {ls} / {k}"] = v
            ms = res.map_scores.reset_index()
            ev_rows.append(ms.assign(event=ev.label, ptype=ev.ptype, link_set=ls))
            if not res.link_scores.empty:
                link_rows.append(res.link_scores.reset_index().assign(event=ev.label, ptype=ev.ptype,
                                                                     link_set=ls))
            for m in res.maps_hourly:
                pair_frames.append(_pairs(res, m, config.near_km).assign(
                    method=m, event=ev.label, ptype=ev.ptype, link_set=ls))
            if config.sensors and ls == config.sensor_link_set:
                try:
                    from .compare import compare_sensors
                    comp = compare_sensors(ev.start, ev.end, res=res)
                    sensor_figs[ev.label] = comp
                    pw = comp.pairwise(near_km=config.near_km)
                    sens_rows.append(pw.assign(event=ev.label, ptype=ev.ptype))
                    sens_arrays += _sensor_arrays(comp, config.near_km, ev)
                    asos_arrays += _asos_arrays(comp, ev)
                except Exception as exc:
                    errors[f"{ev.label} / sensors"] = repr(exc)
                    log.exception("sensor comparison failed for %s", ev.label)

    event_scores = pd.concat(ev_rows, ignore_index=True) if ev_rows else pd.DataFrame()
    link_scores = pd.concat(link_rows, ignore_index=True) if link_rows else pd.DataFrame()
    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()

    pooled = []
    coverage = pd.DataFrame()
    if not pairs.empty:
        # Common sample per (link set, event): the cell-hours every *full-coverage* method
        # has (>= min_coverage of the best-covered method). A method with gaps is scored on
        # its share of that sample and its coverage is reported - it must not remove the
        # hours it fails on from everybody's score, nor look better by staying silent.
        pairs, coverage = _common_sample(pairs, ["link_set", "event"], "method", "cell",
                                         config.min_coverage)
        for (t, ls, m), g in pairs.groupby(["ptype", "link_set", "method"]):
            cov = coverage[(coverage.link_set == ls) & (coverage.method == m) &
                           coverage.event.isin(g.event.unique())]
            for near in (False, True):
                gg = g[g.near] if near else g
                s = scores(gg.est.values, gg.ref.values, config.wet_threshold_mm)
                pooled.append({"ptype": t, "link_set": ls, "method": m, "near": near,
                               "events": int(gg.event.nunique()),
                               "coverage": float(cov.n.sum() / cov.base.sum()) if len(cov) else np.nan,
                               **s})
    sensor_pooled, asos_pooled = [], []
    if sens_arrays:
        sa = pd.concat(sens_arrays, ignore_index=True)
        sa = sa.assign(pair=sa.estimate + " | " + sa.reference)
        sa, scov = _common_sample(sa, ["event"], "pair", "cell", config.min_coverage)
        for (t, e_, r_), g in sa.groupby(["ptype", "estimate", "reference"]):
            cv = scov[(scov.pair == f"{e_} | {r_}") & scov.event.isin(g.event.unique())]
            sensor_pooled.append({"ptype": t, "estimate": e_, "reference": r_,
                                  "events": int(g.event.nunique()),
                                  "coverage": float(cv.n.sum() / cv.base.sum()) if len(cv) else np.nan,
                                  **scores(g.est.values, g.ref.values, config.wet_threshold_mm)})
    if asos_arrays:
        aa = pd.concat(asos_arrays, ignore_index=True)
        for (t, m_), g in aa.groupby(["ptype", "map"]):
            asos_pooled.append({"ptype": t, "map": m_, "events": int(g.event.nunique()),
                                **scores(g.est.values, g.ref.values, config.wet_threshold_mm)})
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return Study(config, events, event_scores, link_scores, pd.DataFrame(pooled), pairs, qc,
                 errors, started, finished,
                 sensor_events=pd.concat(sens_rows, ignore_index=True) if sens_rows else pd.DataFrame(),
                 sensor_pooled=pd.DataFrame(sensor_pooled), asos_pooled=pd.DataFrame(asos_pooled),
                 sensor_figures=sensor_figs)


def _common_sample(df: pd.DataFrame, by: list, who: str, key: str, min_coverage: float):
    """Restrict ``df`` to the keys shared by all full-coverage members within each ``by`` group.

    Returns ``(restricted df, coverage table)``; coverage = members' valid keys / the best
    member's. Members below ``min_coverage`` keep only their keys inside the common set.
    """
    out, cov_rows = [], []
    for grp, g in df.groupby(by):
        grp = grp if isinstance(grp, tuple) else (grp,)
        n = g.groupby(who)[key].nunique()
        base = n.max()
        full = n.index[n >= min_coverage * base]
        common = set.intersection(*(set(g.loc[g[who] == w, key]) for w in full))
        out.append(g[g[key].isin(common)])
        for w, nw in n.items():
            cov_rows.append({**dict(zip(by, grp)), who: w, "n": int(nw), "base": int(base),
                             "full": bool(w in full)})
    return pd.concat(out, ignore_index=True), pd.DataFrame(cov_rows)


def _slug(label: str) -> str:
    return label.replace("#", "").replace(" ", "_")


def _sensor_arrays(comp, near_km, ev) -> list:
    """Jointly valid near-link cell-hours for each sensor pair of one event."""
    times = comp._joint()
    near = (comp.distance_to_link_km <= near_km).values
    out = []
    cml = [n for n in comp.names if n.startswith("CML")]
    for est, ref in [(c, "MRMS") for c in cml] + [(c, "PWS") for c in cml] + [("PWS", "MRMS")]:
        e = comp.maps[est].sel(time=times).values[:, near].ravel()
        r = comp.maps[ref].sel(time=times).values[:, near].ravel()
        ok = np.isfinite(e) & np.isfinite(r)
        out.append(pd.DataFrame({"cell": np.flatnonzero(ok), "est": e[ok], "ref": r[ok], "estimate": est,
                                 "reference": ref, "event": ev.label, "ptype": ev.ptype}))
    return out


def _asos_arrays(comp, ev) -> list:
    a = comp.asos_points
    if a.sizes.get("station", 0) == 0:
        return []
    out = []
    for name, m in comp.maps.items():
        smp = m.sel(lat=xr.DataArray(a.lat.values, dims="station"),
                    lon=xr.DataArray(a.lon.values, dims="station"), method="nearest")
        times = np.intersect1d(smp.time.values, a.time.values)
        e = smp.sel(time=times).transpose("station", "time").values.ravel()
        r = a.sel(time=times).values.ravel()
        ok = np.isfinite(e) & np.isfinite(r)
        out.append(pd.DataFrame({"est": e[ok], "ref": r[ok], "map": name, "event": ev.label, "ptype": ev.ptype}))
    return out
