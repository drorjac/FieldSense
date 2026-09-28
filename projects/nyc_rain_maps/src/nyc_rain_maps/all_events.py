"""All-events comparison: every event of the OpenMesh record, every method, every sensor.

The study (:mod:`nyc_rain_maps.study`) pools the 10 catalog events. This module runs *all*
detected events of the OpenMesh period (52 events >= 1 mm) and asks event-level questions:

* per event: radar, PWS and official (ASOS) totals near the links, and every CML method's
  total, bias, NRMSE and correlation;
* which method wins how often, and how stable is the ranking across events;
* what drives the errors - event type, size, peak intensity, duration, temperature, and
  link outages (the share of missing link minutes);
* held-out vs calibration events: the gauge-calibrated link selection used the 31 rain
  events outside the catalog, so results on those are in-sample for the *selection*.

    python src/run.py all-events                    # -> results/all_events/README.md
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .study import StudyConfig, run_study
from .settings import repo_events_dir, repo_reports_dir
from .pipeline import DEFAULT_METHODS

log = logging.getLogger(__name__)

DRIVERS = ["total_mm", "peak_hourly_mm", "duration_h", "min_temp_c", "link_outage"]
MAX_OUTAGE = 0.5          # events with more missing link minutes are listed but not ranked


def all_events(min_total_mm: float = 1.0) -> pd.DataFrame:
    cat = pd.read_csv(repo_events_dir() / "all_detected_events.csv")
    cat = cat[cat.in_openmesh.astype(bool) & (cat.total_mm >= min_total_mm)].copy()
    cat["label"] = [f"{p} {pd.Timestamp(s):%Y-%m-%d %H}" for p, s in zip(cat.ptype, cat.start)]
    cat["held_out"] = cat.selected.astype(bool)          # catalog events: not used by the selection
    try:
        from .link_selection import load_or_select
        calib = set(map(str, load_or_select().calibration_events))
    except Exception:
        calib = set()
    cat["in_calibration"] = cat.event_id.astype(str).isin(calib)
    cat["sample"] = np.where(cat.held_out, "catalog (held out)",
                             np.where(cat.in_calibration, "selection calibration", "other (not used by selection)"))
    return cat.sort_values("start").reset_index(drop=True)


def run_all_events(min_total_mm: float = 1.0, methods=None, link_sets=("shared8", "selected")):
    cat = all_events(min_total_mm)
    cfg = StudyConfig(name="all_events", methods=list(methods or DEFAULT_METHODS), link_sets=list(link_sets),
                      events=[(r.label, r.ptype, r.start, r.end) for r in cat.itertuples()],
                      sensors=True, sensor_link_set="selected")
    study = run_study(cfg)
    return study, cat


def event_table(study, cat: pd.DataFrame, link_set: str = "selected", near_km: float = 2.0) -> pd.DataFrame:
    """One row per event: sensors' totals near the links and every method's scores."""
    rows = []
    es = study.event_scores[study.event_scores.link_set == link_set]
    for ev in cat.itertuples():
        row = {"label": ev.label, "ptype": ev.ptype, "start": ev.start, "end": ev.end, "held_out": ev.held_out,
               "sample": ev.sample,
               "total_mm": ev.total_mm, "peak_hourly_mm": ev.peak_hourly_mm, "duration_h": ev.duration_h,
               "min_temp_c": ev.min_temp_c}
        comp = study.sensor_figures.get(ev.label)
        if comp is not None:
            near = (comp.distance_to_link_km <= near_km).values
            t = comp._joint()
            for name in ("MRMS", "PWS"):
                if name in comp.maps:
                    row[f"{name}_mm"] = float(comp.maps[name].sel(time=t).values[:, near].mean(axis=1).sum())
            a = comp.asos_points
            if a.sizes.get("station", 0):
                row["ASOS_mm"] = float(a.sum("time", min_count=1).mean())
            rsl = comp.links["rsl"].sel(time=slice(pd.Timestamp(ev.start), pd.Timestamp(ev.end)))
            row["link_outage"] = float(rsl.isnull().mean())
        for r in es[es.event == ev.label].itertuples():
            row[f"{r.method}_nrmse"] = r.nrmse
            row[f"{r.method}_bias"] = r.rel_bias
            row[f"{r.method}_corr"] = r.corr
            row[f"{r.method}_mm"] = r.mean_est * r.hours if hasattr(r, "hours") else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    methods = [c[:-6] for c in df.columns if c.endswith("_nrmse")]
    nr = df[[f"{m}_nrmse" for m in methods]]
    df["best_method"] = nr.idxmin(axis=1).str[:-6].where(nr.notna().any(axis=1))
    return df


def win_counts(table: pd.DataFrame) -> pd.DataFrame:
    methods = [c[:-6] for c in table.columns if c.endswith("_nrmse")]
    ranks = table[[f"{m}_nrmse" for m in methods]].rank(axis=1)
    ranks.columns = methods
    out = pd.DataFrame({"wins": table.best_method.value_counts().reindex(methods).fillna(0).astype(int),
                        "median_rank": ranks.median(), "median_nrmse": [table[f"{m}_nrmse"].median() for m in methods],
                        "median_bias": [table[f"{m}_bias"].median() for m in methods],
                        "share_underestimating": [(table[f"{m}_bias"] < 0).mean() for m in methods]}, index=methods)
    return out.sort_values("median_rank")


def drivers(table: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each method's NRMSE and |bias| with event properties."""
    methods = [c[:-6] for c in table.columns if c.endswith("_nrmse")]
    rows = []
    for m in methods:
        for target in ("nrmse", "bias"):
            y = table[f"{m}_{target}"]
            y = y.abs() if target == "bias" else y
            rec = {"method": m, "metric": "NRMSE" if target == "nrmse" else "|bias|"}
            for d in DRIVERS:
                if d in table:
                    ok = y.notna() & table[d].notna()
                    varies = y[ok].nunique() > 1 and table.loc[ok, d].nunique() > 1
                    rec[d] = (float(y[ok].corr(table.loc[ok, d], method="spearman"))
                              if ok.sum() > 5 and varies else np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)


def write_report(study, cat: pd.DataFrame, out_dir: str | Path | None = None) -> Path:
    """Compute the event tables from ``study`` and render the report."""
    out = Path(out_dir or repo_reports_dir() / "all_events")
    out.mkdir(parents=True, exist_ok=True)
    event_table(study, cat).round(4).to_csv(out / "event_table_selected.csv", index=False)
    event_table(study, cat, link_set="shared8").round(4).to_csv(out / "event_table_shared8.csv", index=False)
    study.pooled.round(4).to_csv(out / "pooled_scores.csv", index=False)
    return render_report(out)


def render_report(out_dir: str | Path | None = None) -> Path:
    """(Re)render tables, figures and README from the saved event tables (no recomputation)."""
    from . import plots as plotting
    out = Path(out_dir or repo_reports_dir() / "all_events")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cat = all_events()
    tab = pd.read_csv(out / "event_table_selected.csv")
    tab8 = pd.read_csv(out / "event_table_shared8.csv")
    samp = cat.set_index("label")["sample"]
    for t in (tab, tab8):
        t["sample"] = t.label.map(samp)
    # events where the links were (almost) silent cannot be scored: listed, not ranked
    usable = tab.link_outage.fillna(0) <= MAX_OUTAGE
    rain = tab[(tab.ptype == "rain") & usable]
    wins = win_counts(rain)
    drv = drivers(rain)
    wins.round(4).to_csv(out / "win_counts_rain.csv")
    drv.round(3).to_csv(out / "error_drivers_rain.csv", index=False)
    plotting.all_events_heatmap(tab, out / "figures" / "nrmse_by_event.png")
    plotting.all_events_totals(tab, out / "figures" / "event_totals_by_sensor.png")
    plotting.all_events_drivers(tab, out / "figures" / "bias_vs_event_properties.png")
    (out / "README.md").write_text(_markdown(cat, tab, tab8, wins, drv, excluded=tab[~usable]))
    return out


def _fmt(v, f="{:.2f}"):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def _markdown(cat, tab, tab8, wins, drv, excluded=None) -> str:
    methods = [c[:-6] for c in tab.columns if c.endswith("_nrmse")]
    rain = tab[(tab.ptype == "rain") & (tab.link_outage.fillna(0) <= MAX_OUTAGE)]
    L = ["# All events of the OpenMesh record — methods, sensors and official gauges", "",
         f"Generated by `python src/run.py all-events` (`nyc_rain_maps.all_events`). {len(cat)} events with a "
         f"domain-mean radar total >= 1 mm ({', '.join(f'{(cat.ptype == t).sum()} {t}' for t in ('rain', 'mix', 'snow'))}), "
         "each run through every method on the gauge-selected links (`selected`) and the authors' links "
         "(`shared8`), scored hourly against MRMS on the 0.01 deg grid, with the PWS map and the official "
         "ASOS gauges alongside. Totals below are means over the cells within 2 km of a link (radar, PWS, "
         "CML) and over the ASOS stations in reach (Central Park, LaGuardia).", "",
         "**Samples.** The link selection was calibrated on PWS gauges over the rain events >= 5 mm "
         "that are not in the catalog (*selection calibration*); those are in-sample for the *selection* "
         "(never for the radar). The catalog events and the smaller rain events were not used by it.", "",
         "| sample | events |", "|---|---|"] + [f"| {k} | {v} |" for k, v in tab.groupby("sample").size().items()] + ["",
         *([f"**Excluded from rankings:** {len(excluded)} event(s) with more than {MAX_OUTAGE:.0%} of link "
            "minutes missing (network outage - nothing to score): "
            + ", ".join(f"{r.label} ({r.link_outage:.0%} missing)" for r in excluded.itertuples()) + ".", ""]
           if excluded is not None and len(excluded) else []),
         f"## Ranking across {len(rain)} rain events (`selected` links)", "",
         "| method | wins | median rank | median NRMSE | median bias | events under-estimated |",
         "|---|---|---|---|---|---|"]
    for m, r in wins.iterrows():
        L.append(f"| `{m}` | {int(r.wins)} | {r.median_rank:.1f} | {r.median_nrmse:.2f} | {r.median_bias:+.0%} | "
                 f"{r.share_underestimating:.0%} |")
    L += ["", "| rain sample | events | " + " | ".join(f"`{m}` median NRMSE / bias" for m in methods) + " |",
          "|---|---|" + "---|" * len(methods)]
    for name, sub in rain.groupby("sample"):
        L.append(f"| {name} | {len(sub)} | " + " | ".join(
            f"{sub[f'{m}_nrmse'].median():.2f} / {sub[f'{m}_bias'].median():+.0%}" for m in methods) + " |")
    L += ["", "| rain events | events | " + " | ".join(f"`{m}` median NRMSE / bias" for m in methods) + " |",
          "|---|---|" + "---|" * len(methods)]
    big = rain.MRMS_mm >= 10 if "MRMS_mm" in rain else rain.total_mm >= 10
    for name, sub in (("< 10 mm near links", rain[~big]), (">= 10 mm near links", rain[big])):
        if len(sub):
            L.append(f"| {name} | {len(sub)} | " + " | ".join(
                f"{sub[f'{m}_nrmse'].median():.2f} / {sub[f'{m}_bias'].median():+.0%}" for m in methods) + " |")
    L += ["", "## What drives the errors (rain events, Spearman correlation with the event property)", "",
          "| method | metric | " + " | ".join(DRIVERS) + " |", "|---|---|" + "---|" * len(DRIVERS)]
    for r in drv.itertuples():
        L.append(f"| `{r.method}` | {r.metric} | " + " | ".join(_fmt(getattr(r, d)) for d in DRIVERS) + " |")
    L += ["", *_driver_findings(drv), "",
          "![NRMSE by event](figures/nrmse_by_event.png)", "",
          "![totals by sensor](figures/event_totals_by_sensor.png)", "",
          "![bias vs properties](figures/bias_vs_event_properties.png)", "",
          "## Every event (`selected` links)", "",
          "| event | type | sample | radar mm | PWS mm | ASOS mm | outage | " +
          " | ".join(f"{m} NRMSE / bias" for m in methods) + " | best |",
          "|---|---|---|---|---|---|---|" + "---|" * len(methods) + "---|"]
    for r in tab.itertuples():
        d = r._asdict()
        L.append(f"| {r.label} | {r.ptype} | {'catalog' if r.held_out else ('calib.' if r.sample == 'selection calibration' else '')} | {_fmt(d.get('MRMS_mm'), '{:.1f}')} | "
                 f"{_fmt(d.get('PWS_mm'), '{:.1f}')} | {_fmt(d.get('ASOS_mm'), '{:.1f}')} | "
                 f"{_fmt(d.get('link_outage'), '{:.1%}')} | " +
                 " | ".join(f"{_fmt(d.get(f'{m}_nrmse'))} / {_fmt(d.get(f'{m}_bias'), '{:+.0%}')}" for m in methods) +
                 f" | {r.best_method if isinstance(r.best_method, str) else '-'} |")
    L += ["", "`event_table_selected.csv` / `event_table_shared8.csv` hold every number; "
          "`pooled_scores.csv` the pooled scores per type and link set.", ""]
    return "\n".join(L)


def _driver_findings(drv: pd.DataFrame) -> list:
    out = []
    for metric, word in (("NRMSE", "NRMSE"), ("|bias|", "|bias|")):
        for d in DRIVERS:
            if d not in drv:
                continue
            col = drv[drv.metric == metric][d]
            if col.notna().sum() == 0:
                continue
            med = col.median()
            if abs(med) >= 0.3:
                out.append(f"- {word} {'rises' if med > 0 else 'falls'} with **{d}** for most methods "
                           f"(median Spearman {med:+.2f}).")
    return out or ["- No event property is consistently (|rho| >= 0.3) related to the errors across methods."]
