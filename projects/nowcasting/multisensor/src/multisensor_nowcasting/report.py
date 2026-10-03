"""``results/report.md`` and the figures, from the CSV tables of :mod:`.evaluate` only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .settings import CACHE_DIR, FIGURES_DIR, NETWORKS, RESULTS_DIR

# forecasts shown in the short tables: the reference, the best physics per sensor set, the models
KEY = [
    "R|persistence", "R|extrapolation", "R|sprog", "R|steps_mean",
    "C|extrapolation", "C|extrapolation_Rmotion", "P|extrapolation_Rmotion",
    "CP|extrapolation_Rmotion", "CP|steps_mean",
    "RC|extrapolation", "RP|extrapolation", "RCP|extrapolation", "RCP|extrapolation_Rmotion", "RCP|steps_mean",
    "learned|in-R>tgt-R", "learned|in-C>tgt-R", "learned|in-P>tgt-R", "learned|in-CP>tgt-R",
    "learned|in-RC>tgt-R", "learned|in-RP>tgt-R", "learned|in-RCP>tgt-R",
    "learned|in-R>tgt-RCP", "learned|in-CP>tgt-RCP", "learned|in-RCP>tgt-RCP", "learned|hybrid",
    "learned_raw|in-R>tgt-R", "learned_raw|in-RCP>tgt-R", "learned_raw|hybrid",
]

LABELS = {
    "persistence": "persistence", "extrapolation": "extrapolation (own motion)", "sprog": "S-PROG (own motion)",
    "extrapolation_Rmotion": "extrapolation (radar motion)", "sprog_Rmotion": "S-PROG (radar motion)",
    "steps_mean": "STEPS mean",
}


def label(fid: str) -> str:
    kind, rest = fid.split("|", 1)
    if kind == "learned_raw":
        return label(f"learned|{rest}") + ", uncalibrated"
    if kind == "learned":
        if rest == "hybrid":
            return "U-Net: correction of the radar extrapolation (inputs R, C, P)"
        s, t = rest.removeprefix("in-").split(">tgt-")
        return f"U-Net: inputs {s} -> {'radar' if t == 'R' else 'merged RCP'}"
    return f"{kind}: {LABELS.get(rest, rest)}"


def _md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                       else ("-" if isinstance(v, (float, np.floating)) else str(v))
                                       for v in r.values) + " |")
    return "\n".join(out)


def _ci(row) -> str:
    return f"{row.estimate:+.3f} [{row.low:+.3f}, {row.high:+.3f}]"


def network_section(net: str) -> list:
    p = lambda name: RESULTS_DIR / f"{name}_{net}.csv"
    if not p("scores").exists():
        return []
    sc, ga, hl = pd.read_csv(p("scores")), pd.read_csv(p("gauges")), pd.read_csv(p("headline"))
    cube_meta = json.loads((CACHE_DIR / net / "cube" / "meta.json").read_text())
    issues = pd.read_csv(p("issues"))
    issues = issues[issues.scored]
    lines = [f"## {NETWORKS[net].label}", ""]
    lines += [f"Grid {cube_meta['grid_shape'][0]} x {cube_meta['grid_shape'][1]} pixels of "
              f"{cube_meta['pixel_km']:g} km, 5-minute steps ({cube_meta['n_steps']} in the cube); "
              f"{cube_meta['n_links']} links, {cube_meta['n_pws']} PWS after quality control (of "
              f"{cube_meta['n_pws_raw']}); {len(issues)} issue times on "
              f"{pd.to_datetime(issues.day).nunique()} storm days in the test weeks.", ""]

    have = [f for f in KEY if f in set(sc.forecast)]
    # 1. against the radar
    t = sc[(sc.region == "domain") & sc.forecast.isin(have)]
    piv = t.pivot_table(index="forecast", columns="lead_min", values="CSI_1").reindex(have)
    tab = pd.DataFrame({"forecast": [label(f) for f in have],
                        "CSI 1 mm/h, 15 min": piv[15].values, "30 min": piv[30].values, "60 min": piv[60].values})
    t60 = t[t.lead_min == 60].set_index("forecast").reindex(have)
    tab["FSS 10 km, 60 min"] = t60["FSS_10km"].values
    tab["MAE 60 min (mm/h)"] = t60["MAE"].values
    tab["bias 60 min"] = t60["bias"].values
    a60 = sc[(sc.region == "area") & (sc.lead_min == 60)].set_index("forecast").reindex(have)
    tab["CSI 60 min, sensor area"] = a60["CSI_1"].values
    lines += ["### Against the radar", "",
              "Pooled over all issue times, on the cells the radar's motion can reach; the last column "
              "only within 10 km of a link or PWS.", "", _md(tab), ""]

    # 2. at the gauges
    g = ga[ga.forecast.isin(have)]
    gp = g.pivot_table(index="forecast", columns="what", values=["RMSE", "corr", "bias"]).reindex(have)
    tab = pd.DataFrame({"forecast": [label(f) for f in have],
                        "rate at 30 min: RMSE": gp[("RMSE", "rate at 30 min")].values,
                        "corr": gp[("corr", "rate at 30 min")].values,
                        "next hour: RMSE (mm)": gp[("RMSE", "next-hour total")].values,
                        "corr ": gp[("corr", "next-hour total")].values,
                        "bias": gp[("bias", "next-hour total")].values})
    gname = "the 10 city gauges" if net == "openmrg" else "the ASOS stations with 1-minute data: EWR, JFK, LGA"
    lines += [f"### At the independent gauges ({gname})", "",
              "5-minute rates (mm/h) at 30 min lead, and the next-hour total (the mean of the 12 lead "
              "times against the gauge's total over the same hour).", "", _md(tab), ""]

    # 3. headline with intervals
    h = hl[hl.forecast.isin(have)]
    rows = []
    for f in have:
        r = {"forecast": label(f)}
        for metric, lm, region in (("CSI_1", 30, "domain"), ("CSI_1", 60, "domain"), ("CSI_1", 60, "area"),
                                   ("MAE", 60, "domain"), ("gauge next-hour RMSE", 60, "gauges")):
            q = h[(h.forecast == f) & (h.metric == metric) & (h.lead_min == lm) & (h.region == region)]
            r[f"{metric} {lm} min {region}"] = _ci(q.iloc[0]) if len(q) else "-"
        rows.append(r)
    lines += ["### Differences to the radar's extrapolation, with 95% intervals", "",
              "Paired bootstrap over storm days (1000 resamples). The radar extrapolation's own row gives "
              "its value; every other row its difference to it. Positive CSI and negative MAE / RMSE "
              "differences are better.", "",
              _md(pd.DataFrame(rows)), ""]

    if p("crps").exists():
        c = pd.read_csv(p("crps"))
        cp = c.pivot_table(index="product", columns=["where", "lead_min"], values="CRPS")
        cp.columns = [f"{w}, {lm} min" for w, lm in cp.columns]
        lines += ["### STEPS ensembles (10 members): CRPS (mm/h)", "", _md(cp.reset_index()), ""]
    if p("training").exists():
        tr = pd.read_csv(p("training"))
        tr["name"] = [label(f"learned|{n}") for n in tr.name]
        lines += ["### Training", "", _md(tr.rename(columns={"best_val": "best validation loss"}), "{:.4f}"), ""]
    if p("motion").exists():
        m = pd.read_csv(p("motion")).drop(columns="issue")
        lines += ["### Motion of each product against the radar's (km/h, wet cells, mean over issue times)", "",
                  _md(m.mean().to_frame("km/h").T.reset_index(drop=True), "{:.1f}"), ""]
    if p("sprog_fallback").exists():
        fb = pd.read_csv(p("sprog_fallback"))
        fb = fb[fb.issues > 0][["forecast", "fallback_to", "issues", "of"]]
        lines += ["### S-PROG runs that failed (nearly dry field) and fell back to extrapolation", "",
                  _md(fb, "{:.0f}"), ""]
    return lines


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.viz_style import use_style
    use_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    show = ["R|persistence", "R|extrapolation", "R|steps_mean", "CP|extrapolation_Rmotion",
            "RCP|extrapolation_Rmotion", "learned|in-R>tgt-R", "learned|in-CP>tgt-R",
            "learned|in-RCP>tgt-R", "learned|hybrid"]
    nets = [n for n in NETWORKS if (RESULTS_DIR / f"scores_{n}.csv").exists()]
    fig, axes = plt.subplots(1, len(nets), figsize=(5.5 * len(nets), 4), squeeze=False)
    for ax, net in zip(axes[0], nets):
        sc = pd.read_csv(RESULTS_DIR / f"scores_{net}.csv")
        for f in show:
            q = sc[(sc.forecast == f) & (sc.region == "domain")].sort_values("lead_min")
            if len(q):
                ax.plot(q.lead_min, q.CSI_1, marker="o", ms=3, label=label(f),
                        ls="--" if f.startswith("learned") else "-")
        ax.set_title(NETWORKS[net].label, fontsize=9)
        ax.set_xlabel("lead time (min)")
        ax.set_ylabel("CSI at 1 mm/h against the radar")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "csi_by_lead.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(nets), figsize=(6.5 * len(nets), 5), squeeze=False)
    for ax, net in zip(axes[0], nets):
        hl = pd.read_csv(RESULTS_DIR / f"headline_{net}.csv")
        q = hl[(hl.metric == "gauge next-hour RMSE") & hl.forecast.isin(KEY)].set_index("forecast")
        q = q.reindex([f for f in KEY if f in q.index])
        y = np.arange(len(q))
        cap = 2.0                                    # mm; points beyond are labelled at the edge
        est, lo, hi = q.estimate.clip(upper=cap), q.low.clip(upper=cap), q.high.clip(upper=cap)
        ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", ms=4, capsize=2)
        for yi, (e, h) in enumerate(zip(q.estimate, q.high)):
            if e > cap or h > cap:
                ax.annotate(f"{e:+.1f} [..{h:+.1f}]", (cap, yi), xytext=(-4, 3), textcoords="offset points",
                            ha="right", fontsize=6.5)
        ax.set_xlim(min(-0.45, float(lo.min()) - 0.05), cap + 0.05)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([label(f) for f in q.index], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("next-hour RMSE at the gauges minus the radar extrapolation's (mm)")
        ax.set_title(NETWORKS[net].label, fontsize=9)
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gauge_rmse_vs_extrapolation.png", dpi=130)
    plt.close(fig)

    for net in nets:
        d = CACHE_DIR / net / "learn"
        hs = [json.loads(p.read_text()) for p in sorted(d.glob("*/history.json"))]
        if not hs:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for h in hs:
            ax.plot(h["val"], label=label(f"learned|{h['name']}"), lw=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss (weighted MSE of log1p rate)")
        ax.set_title(f"{NETWORKS[net].label}: validation loss", fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"loss_{net}.png", dpi=130)
        plt.close(fig)


def write() -> str:
    figures()
    lines = ["# multisensor_nowcasting - results", "",
             "Computed by `python projects/nowcasting/multisensor/src/run.py evaluate --network <network>` and "
             "`report`. Every forecast is scored on the same issue times (the test weeks), the same cells and "
             "the same gauges. Rain rates in mm/h.", "",
             "![CSI by lead time](figures/csi_by_lead.png)", "",
             "![Gauge RMSE against the radar extrapolation](figures/gauge_rmse_vs_extrapolation.png)", ""]
    for net in NETWORKS:
        lines += network_section(net)
    out = RESULTS_DIR / "report.md"
    out.write_text("\n".join(lines))
    return str(out)
