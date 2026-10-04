"""Figures and ``results/report.md`` for trained models - every number computed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .settings import RESULTS_DIR

NAMES = {"openmrg": "Gothenburg", "openrainer": "Emilia-Romagna", "openmesh": "New York City"}
COLORS = {"rnn": "#e34948", "dynamic": "#1baf7a", "constant": "#4a3aa7", "pycomlink": "#e87ba4",
          "nearby": "#008300"}
TEXT = "#52514e"


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def scores_figure(table: pd.DataFrame, path, reference: str = "target"):
    """RMSE and correlation per network, the RNN against every power-law method (common sample)."""
    t = table[(table.reference == reference) & (table["sample"] == "common")]
    nets = list(dict.fromkeys(t.network))
    fig, axes = plt.subplots(2, len(nets), figsize=(4.6 * len(nets), 6.5), squeeze=False)
    for j, n in enumerate(nets):
        g = t[t.network == n]
        for i, metric in enumerate(("rmse", "corr")):
            ax = axes[i, j]
            ax.bar(g.method, g[metric], color=[COLORS.get(m, "#898781") for m in g.method])
            for k, (v, c) in enumerate(zip(g[metric], g.coverage)):
                ax.text(k, v, f"{v:.2f}" + ("" if c > 0.99 else f"\n({c:.0%})"), ha="center", va="bottom",
                        fontsize=7, color=TEXT)
            ax.set_title(f"{NAMES.get(n, n)}: {'RMSE (mm/h)' if metric == 'rmse' else 'correlation'}", fontsize=9)
            ax.tick_params(labelsize=7)
            if metric == "rmse":
                ax.set_ylim(0, min(g[metric].max(), 3 * g[metric].median()) * 1.25)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle(f"Test weeks, hourly link rain against the {reference}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def scatter_figure(d, est: dict, path, methods=("rnn", "nearby", "pycomlink")):
    """Hourly link rain against the target in test weeks (log-density)."""
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4), squeeze=False)
    test = np.broadcast_to(d.part == 2, d.y.shape)
    lim = float(np.nanpercentile(d.y[test], 99.9)) * 1.2
    for ax, m in zip(axes[0], methods):
        if m not in est:
            ax.axis("off")
            continue
        ok = test & np.isfinite(d.y) & np.isfinite(est[m]) & ((d.y > 0.1) | (est[m] > 0.1))
        ax.hexbin(d.y[ok], est[m][ok], gridsize=40, bins="log", extent=(0, lim, 0, lim), cmap="Blues")
        ax.plot([0, lim], [0, lim], color="#c3c2b7", lw=1)
        ax.set(xlabel="target (mm/h)", ylabel=f"{m} (mm/h)", title=f"{NAMES.get(d.network, d.network)}: {m}")
    fig.tight_layout()
    _save(fig, path)


def series_figure(d, est: dict, path, hours: int = 96):
    """The wettest test window of the link with the most target rain."""
    test = d.part == 2
    y = np.where(test[None, :], d.y, np.nan)
    li = int(np.nanargmax(np.nansum(y, axis=1)))
    center = int(np.nanargmax(np.where(test, np.nan_to_num(d.y[li]), -1)))
    sl = slice(max(0, center - hours // 2), center + hours // 2)
    t = pd.DatetimeIndex(d.ds.time.values[sl])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.step(t, d.ds.radar.values[li, sl], where="pre", color="#2a78d6", lw=1, label="radar along the path")
    for p, c in (("pws", "#eb6834"), ("city", "#eda100"), ("gauges", "#eb6834")):
        if p in d.ds:
            ax.step(t, d.ds[p].values[li, sl], where="pre", color=c, lw=1, label=f"{p} near the link")
    for m in ("rnn", "nearby"):
        if m in est:
            ax.step(t, est[m][li, sl], where="pre", color=COLORS[m], lw=1.6, label=m)
    ax.set(ylabel="mm/h", title=f"{NAMES.get(d.network, d.network)}, link {d.ds.link.values[li]} "
                                f"({float(d.ds.frequency[li]):.1f} GHz, {float(d.ds.length[li]):.1f} km)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    _save(fig, path)


def write(name: str, table: pd.DataFrame, verdicts: dict, out: Path = RESULTS_DIR) -> Path:
    from .settings import DATA_DIR
    meta = json.loads((DATA_DIR / "models" / name / "config.json").read_text())
    out = out / name
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "scores.csv", index=False)
    L = [f"# RNN `{name}` against the power law", "",
         f"_Trained on {', '.join(meta['networks'])} in {meta['minutes']:.0f} min "
         f"(best epoch {meta['history'].get('best_epoch')}); configuration in `config.json`._", ""]
    for ref, v in verdicts.items():
        L += [f"## Head to head against the {ref}", "",
              "Each power-law method against the RNN on that method's own valid test hours "
              "(hourly link rain, mm/h).", "",
              "| network | power law | link-hours | RMSE RNN / PL | corr RNN / PL | bias RNN / PL | CSI RNN / PL | RNN better |",
              "|---|---|---|---|---|---|---|---|"]
        for r in v.itertuples():
            L.append(f"| {r.network} | {r.power_law} | {r.n:,} | {r.rmse_rnn:.3f} / {r.rmse_pl:.3f} | "
                     f"{r.corr_rnn:.3f} / {r.corr_pl:.3f} | {r.bias_rnn:+.0%} / {r.bias_pl:+.0%} | "
                     f"{r.csi_rnn:.2f} / {r.csi_pl:.2f} | {'yes' if r.rnn_better else 'no'} |")
        L.append("")
    (out / "report.md").write_text("\n".join(L))
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    return out
