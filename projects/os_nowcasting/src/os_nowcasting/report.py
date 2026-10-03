"""Tables, figures and ``results/report.md`` from the checkpointed network studies.

Every number in the report is computed here from ``CACHE_DIR/<network>/study_state.pkl``;
the score tables are also written as CSV next to the report.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .settings import EVENTS, FIGURES_DIR, NETWORKS, RESULTS_DIR
from .study import NetworkStudy

NAMES = {"openrainer": "OpenRainER (Emilia-Romagna, 15 min)", "openmrg": "OpenMRG (Gothenburg, 5 min)"}
METHOD_ORDER = ["persistence", "extrapolation", "sprog", "anvil", "linda"]
PRODUCT_ORDER = ["radar", "merged", "cml_idw40", "cml_idw20", "cml_idw10", "pws_idw"]


def load() -> dict:
    out = {}
    for n in NETWORKS:
        p = NetworkStudy(n).path
        if p.exists():
            out[n] = NetworkStudy.load_or_new(n).tables()
    return out


def _cat(tabs: dict, key: str) -> pd.DataFrame:
    parts = [t[key] for t in tabs.values() if len(t[key])]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _get(df, **sel):
    m = np.ones(len(df), bool)
    for k, v in sel.items():
        m &= (df[k] == v).values
    return df[m]


def _val(df, col, **sel):
    r = _get(df, **sel)
    return float(r[col].iloc[0]) if len(r) and col in r and pd.notna(r[col].iloc[0]) else np.nan


def _fmt(x, nd=2):
    return "-" if not np.isfinite(x) else f"{x:.{nd}f}"


def _md(rows, header) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------- tables
def table_radar_methods(det: pd.DataFrame, network: str, product="radar") -> str:
    d = _get(det, network=network, product=product)
    rows = []
    for m in METHOD_ORDER:
        if not len(_get(d, method=m)):
            continue
        r = [m]
        for lead in (30, 60, 90):
            r.append(_fmt(_val(d, "CSI_1", method=m, reference="radar", lead_min=lead)))
        r += [_fmt(_val(d, "FSS_10km", method=m, reference="radar", lead_min=60)),
              _fmt(_val(d, "MAE", method=m, reference="radar", lead_min=60)),
              _fmt(_val(d, "corr", method=m, reference="radar", lead_min=60)),
              _fmt(_val(d, "corr", method=m, reference="gauges", lead_min=60)),
              _fmt(_val(d, "RMSE", method=m, reference="gauges", lead_min=60)),
              int(_val(d, "n_forecasts", method=m, reference="radar", lead_min=60))]
        rows.append(r)
    return _md(rows, ["method", "CSI 1 mm/h, 30 min", "60 min", "90 min", "FSS 10 km, 60 min",
                      "MAE 60 min (mm/h)", "corr 60 min", "gauges: corr 60 min", "gauges: RMSE 60 min",
                      "forecasts"])


def table_products(det: pd.DataFrame, network: str, method="extrapolation", lead=60) -> str:
    d = _get(det, network=network, method=method)
    pers = _get(det, network=network, method="persistence")
    rows = []
    for p in PRODUCT_ORDER:
        if not len(_get(d, product=p)):
            continue
        csi_own = _val(d, "CSI_1", product=p, reference="own", lead_min=lead)
        csi_own_p = _val(pers, "CSI_1", product=p, reference="own", lead_min=lead)
        rows.append([p, _fmt(csi_own), _fmt(csi_own_p),
                     _fmt(_val(d, "FSS_10km", product=p, reference="own", lead_min=lead)),
                     _fmt(_val(d, "CSI_1", product=p, reference="radar", lead_min=lead)),
                     _fmt(_val(d, "FSS_10km", product=p, reference="radar", lead_min=lead)),
                     _fmt(_val(d, "MAE", product=p, reference="radar", lead_min=lead)),
                     _fmt(_val(d, "corr", product=p, reference="gauges", lead_min=lead)),
                     _fmt(_val(d, "RMSE", product=p, reference="gauges", lead_min=lead)),
                     _fmt(_val(d, "ME", product=p, reference="gauges", lead_min=lead))])
    return _md(rows, ["product", "own: CSI 1", "own: CSI 1, persistence", "own: FSS 10 km",
                      "radar: CSI 1", "radar: FSS 10 km", "radar: MAE", "gauges: corr",
                      "gauges: RMSE", "gauges: ME"])


def table_ensemble(ens: pd.DataFrame, network: str) -> str:
    d = _get(ens, network=network)
    rows = []
    for p in PRODUCT_ORDER + ["radar (LINDA-P subset)"]:
        for m in ("steps", "linda_p"):
            e = _get(d, product=p, method=m)
            if not len(e):
                continue
            rows.append([p, m,
                         _fmt(_val(e, "CRPS", reference="radar", lead_min=30)),
                         _fmt(_val(e, "CRPS", reference="radar", lead_min=60)),
                         _fmt(_val(e, "ROC_area", reference="radar", lead_min=60)),
                         _fmt(_val(e, "ensmean_CSI_1", reference="radar", lead_min=60)),
                         _fmt(_val(e, "CRPS", reference="gauges", lead_min=60)),
                         int(_val(e, "n_forecasts", reference="radar", lead_min=60))])
    return _md(rows, ["product", "ensemble", "CRPS 30 min (mm/h)", "CRPS 60 min", "ROC area 1 mm/h, 60 min",
                      "ens. mean CSI 1, 60 min", "gauges: CRPS 60 min", "forecasts"])


def table_motion(mot: pd.DataFrame, network: str) -> str:
    d = _get(mot, network=network)
    rows = []
    for m in ["LK", "LK (no dB)", "VET", "DARTS", "proesmans"]:
        if not len(_get(d, motion=m)):
            continue
        rows.append([m] + [_fmt(_val(d, "CSI_1", motion=m, lead_min=lead)) for lead in (30, 60, 90)]
                    + [_fmt(_val(d, "MAE", motion=m, lead_min=60)), _fmt(_val(d, "FSS_10km", motion=m, lead_min=60))])
    return _md(rows, ["motion", "CSI 1, 30 min", "60 min", "90 min", "MAE 60 min", "FSS 10 km, 60 min"])


def table_motion_vs_radar(mvr: pd.DataFrame, network: str) -> str:
    d = _get(mvr, network=network).dropna(subset=["mean_diff_kmh"])
    rows = []
    for p in PRODUCT_ORDER[1:]:
        e = _get(d, product=p)
        if len(e):
            rows.append([p, len(e), _fmt(e.mean_diff_kmh.median(), 0), _fmt(e.radar_speed_kmh.median(), 0)])
    return _md(rows, ["product", "issue times", "median vector difference to radar motion (km/h)",
                      "median radar speed (km/h)"])


def table_reach(reach: pd.DataFrame, network: str) -> str:
    d = _get(reach, network=network)
    cols = [c for c in d.columns if c.startswith("reach_")]
    rows = [[p] + [_fmt(100 * _get(d, product=p)[c].mean(), 0) for c in cols]
            for p in PRODUCT_ORDER if len(_get(d, product=p))]
    return _md(rows, ["product"] + [f"% of domain verifiable at {c.split('_')[1]} min" for c in cols])


def table_sal(det: pd.DataFrame, network: str) -> str:
    d = _get(det, network=network, reference="radar", lead_min=60)
    rows = []
    for p, m in [("radar", x) for x in METHOD_ORDER[:4]] + [("radar (LINDA subset)", "linda")] + \
                [(p, "extrapolation") for p in PRODUCT_ORDER[1:]]:
        e = _get(d, product=p, method=m)
        if len(e) and "SAL_S" in e and pd.notna(e.SAL_S.iloc[0]):
            rows.append([p, m, _fmt(e.SAL_S.iloc[0]), _fmt(e.SAL_A.iloc[0]), _fmt(e.SAL_L.iloc[0]),
                         int(e.SAL_n.iloc[0])])
    return _md(rows, ["product", "method", "S", "A", "L", "forecasts"])


def accumulation_summary(acc: pd.DataFrame) -> pd.DataFrame:
    if not len(acc):
        return pd.DataFrame()
    a = acc.dropna(subset=["estimate_mm", "gauge_mm"])
    rows = []
    for (net, m), g in a.groupby(["network", "method"]):
        for label, sub in [("all hours", g), ("gauge >= 1 mm", g[g.gauge_mm >= 1])]:
            e, o = sub.estimate_mm.values, sub.gauge_mm.values
            rows.append({"network": net, "method": m, "subset": label, "n": len(sub),
                         "rmse_mm": float(np.sqrt(np.mean((e - o) ** 2))),
                         "mae_mm": float(np.mean(np.abs(e - o))),
                         "corr": float(np.corrcoef(e, o)[0, 1]) if len(sub) > 2 else np.nan,
                         "bias_pct": float(100 * (e.sum() - o.sum()) / o.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- figures
def figures(det: pd.DataFrame, ens: pd.DataFrame) -> list:
    import matplotlib.pyplot as plt
    from core import viz_style as vs
    vs.use_style()
    slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    nets = [n for n in NETWORKS if n in det.network.unique()]
    out = []

    def lines(ax, d, key, order, col, ref, title):
        names = [x for x in order if x in d[key].unique()]
        for k, name in enumerate(names):
            e = _get(d, **{key: name}, reference=ref).sort_values("lead_min")
            ax.plot(e.lead_min, e[col], color=slots[k], marker="o", markersize=4, label=name)
            if len(names) > 4:              # the legend carries identity; labels would collide
                continue
            ax.annotate(name, (e.lead_min.iloc[-1], e[col].iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7.5, color=vs.INK_SECONDARY, va="center")
        ax.set_title(title)
        ax.set_xlabel("lead time (min)")

    fig, axes = plt.subplots(1, len(nets), figsize=(5.2 * len(nets), 3.8), squeeze=False,
                             layout="constrained")
    for ax, n in zip(axes[0], nets):
        lines(ax, _get(det, network=n, product="radar (LINDA subset)"), "method", METHOD_ORDER, "CSI_1",
              "radar", NAMES[n])
    axes[0, 0].set_ylabel("CSI at 1 mm/h against the radar")
    axes[0, -1].legend(loc="upper right")
    fig.suptitle("Radar nowcasts by method (issue times where LINDA ran)", fontsize=11)
    p = FIGURES_DIR / "csi_by_method.png"
    fig.savefig(p); plt.close(fig); out.append(p)

    fig, axes = plt.subplots(1, len(nets), figsize=(5.2 * len(nets), 3.8), squeeze=False,
                             layout="constrained")
    for ax, n in zip(axes[0], nets):
        lines(ax, _get(det, network=n, method="extrapolation"), "product", PRODUCT_ORDER, "FSS_10km",
              "radar", NAMES[n])
    axes[0, 0].set_ylabel("FSS (10 km, 1 mm/h) against the radar")
    axes[0, -1].legend(loc="upper right")
    fig.suptitle("Extrapolation nowcasts of each product, scored against the radar", fontsize=11)
    p = FIGURES_DIR / "fss_by_product.png"
    fig.savefig(p); plt.close(fig); out.append(p)

    fig, axes = plt.subplots(2, len(nets), figsize=(5.2 * len(nets), 7.0), squeeze=False,
                             layout="constrained")
    for j, n in enumerate(nets):
        e = _get(ens, network=n, product="radar", method="steps", reference="radar", lead_min=60)
        if not len(e):
            continue
        rp, rf = np.array(e.rel_prob.iloc[0]), np.array(e.rel_freq.iloc[0])
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color=vs.BASELINE, lw=1)
        ax.plot(rp, rf, color=slots[0], marker="o", markersize=4)
        ax.set_title(f"{NAMES[n]}\nreliability, 1 mm/h, 60 min")
        ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency")
        rh = np.array(e.rank_hist.iloc[0])
        ax = axes[1, j]
        ax.bar(np.arange(len(rh)), rh, color=slots[0], width=0.9)
        ax.axhline(1 / len(rh), color=vs.INK_MUTED, lw=1)
        ax.set_title("rank histogram, 60 min"); ax.set_xlabel("rank of the radar among the members")
        ax.set_ylabel("frequency")
    p = FIGURES_DIR / "steps_reliability_rank.png"
    fig.savefig(p); plt.close(fig); out.append(p)
    return out


# ---------------------------------------------------------------------- report
def write_report() -> str:
    tabs = load()
    det, ens, mot = _cat(tabs, "deterministic"), _cat(tabs, "ensemble"), _cat(tabs, "motion")
    mvr, acc, iss = _cat(tabs, "motion_vs_radar"), _cat(tabs, "accumulation"), _cat(tabs, "issues")
    reach = _cat(tabs, "reach")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ens_csv = ens.copy()
    for c in ("rel_prob", "rel_freq", "rank_hist"):
        if c in ens_csv:
            ens_csv[c] = ens_csv[c].map(json.dumps)
    det.to_csv(RESULTS_DIR / "scores_deterministic.csv", index=False, float_format="%.4f")
    ens_csv.to_csv(RESULTS_DIR / "scores_ensemble.csv", index=False, float_format="%.4f")
    mot.to_csv(RESULTS_DIR / "scores_motion_methods.csv", index=False, float_format="%.4f")
    accs = accumulation_summary(acc)
    accs.to_csv(RESULTS_DIR / "advection_interpolation.csv", index=False, float_format="%.4f")
    mvr.to_csv(RESULTS_DIR / "motion_vs_radar.csv", index=False, float_format="%.3f")
    iss.to_csv(RESULTS_DIR / "issue_times.csv", index=False)
    figs = figures(det, ens)

    L = ["# os_nowcasting - results", "",
         "Computed by `python projects/os_nowcasting/src/run.py report` from the checkpointed studies. "
         "Scores are pooled over all issue times of a network (one contingency table, one set of sums). "
         "`own`: against the product's own later field; `radar`: against the radar; `gauges`: at the "
         "independent gauges (OpenRainER: ARPAE, OpenMRG: the city gauges), at their grid cells. "
         "Rain rates in mm/h at the network's step.", ""]
    L += ["## Design", ""]
    rows = []
    for n, t in tabs.items():
        i = t["issues"]
        done = sorted(i.event.unique()) if len(i) else []
        rows.append([NAMES[n], f"{len(done)} of {len(EVENTS[n])}", ", ".join(done), len(i),
                     NETWORKS[n].max_lead_min, _fmt(i.seconds.sum() / 3600 if len(i) else np.nan, 1)])
    L += [_md(rows, ["network", "events", "event ids", "issue times", "max lead (min)",
                     "nowcasting time (h)"]), ""]
    for n in tabs:
        L += [f"## {NAMES[n]}", "",
              "Scores count only cells the product's own motion can reach from inside the domain; "
              "the rest is rain advected in from outside, unknown to every method. Share of the "
              "domain that is left (mean over issue times):", "", table_reach(reach, n), "",
              "### Radar: the nowcasting methods", "",
              "All issue times (LINDA, being slow, is in the next table):", "",
              table_radar_methods(det, n), "",
              "On the issue times where LINDA ran (every third), all methods on the same forecasts:", "",
              table_radar_methods(det, n, product="radar (LINDA subset)"), "",
              "### Every product, extrapolation nowcast at 60 min", "",
              table_products(det, n), "",
              "S-PROG at 60 min:", "", table_products(det, n, method="sprog"), "",
              "### STEPS ensembles (and LINDA-P on its subset)", "", table_ensemble(ens, n), "",
              "### Motion methods (radar, extrapolation nowcast against the radar)", "",
              table_motion(mot, n), "",
              "### How far each product's motion is from the radar's (LK, wet cells)", "",
              table_motion_vs_radar(mvr, n), "",
              "### SAL at 60 min against the radar (mean over forecasts with objects in both)", "",
              table_sal(det, n), ""]
    if len(accs):
        L += ["## Hourly totals from 5-min scans: plain vs advection interpolation (OpenMRG, city gauges)", "",
              _md([[r.method, r.subset, r.n, _fmt(r.rmse_mm), _fmt(r.mae_mm), _fmt(r.corr), _fmt(r.bias_pct, 1)]
                   for r in accs.itertuples()],
                  ["hourly total", "subset", "gauge-hours", "RMSE (mm)", "MAE (mm)", "corr", "bias (%)"]), ""]
    L += ["## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs] + [""]
    path = RESULTS_DIR / "report.md"
    path.write_text("\n".join(L))
    return str(path)
