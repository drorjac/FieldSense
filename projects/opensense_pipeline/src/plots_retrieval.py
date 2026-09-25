"""Figure for ``retrieval_benchmark.py``: variants against gauges and radar."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from core import viz_style as vs

GAUGE = vs.SERIES["stratiform"]     # slot 1
RADAR = vs.SERIES["convective"]     # slot 2


def figure(rows: list[dict], path, highlight=("default", "wet_nearby_pastorek"),
           title: str = "OpenMRG 22-29 July 2015", gauge_step: str = "15min",
           radar_step: str = "5min"):
    """Three dot-plot panels sharing one variant axis.

    Ratio of totals is on a log axis, because 0.5x and 2x are equally wrong;
    the vertical line at 1 is the unbiased target.
    """
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))[::-1]

    panels = (
        ("ratio", "ratio of totals (log)", (0.4, 3.2), 1.0),
        ("r", "correlation r", None, None),
        ("mcc", "wet/dry MCC", None, None),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.42 * len(rows) + 1.8),
                             sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (key, title, xlim, target) in zip(axes, panels):
        g = [r["gauge"][key] for r in rows]
        rd = [r["radar"][key] for r in rows]
        if target is not None:
            ax.axvline(target, color=vs.INK_MUTED, lw=1.0, zorder=1)
        for yi, a, b in zip(y, g, rd):
            ax.plot([a, b], [yi, yi], color=vs.GRIDLINE, lw=2, zorder=1)
        ax.scatter(g, y, s=64, color=GAUGE, edgecolor=vs.SURFACE, lw=2,
                   zorder=3, label=f"vs gauges, {gauge_step}")
        ax.scatter(rd, y, s=64, color=RADAR, edgecolor=vs.SURFACE, lw=2,
                   zorder=3, label=f"vs radar along path, {radar_step}")
        if key == "ratio":
            ax.set_xscale("log")
            ax.set_xticks([0.5, 1, 2, 3])
            ax.set_xticklabels(["0.5x", "1x", "2x", "3x"])
            ax.minorticks_off()
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    for tick, row in zip(axes[0].get_yticklabels(), rows):
        if row["key"] in highlight:
            tick.set_fontweight("bold")
            tick.set_color(vs.INK_PRIMARY)
        else:
            tick.set_color(vs.INK_SECONDARY)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.55, -0.045),
                   ncol=2)
    fig.suptitle(f"CML retrieval variants, {title}: "
                 "same links, same pairs, scored against independent sensors",
                 x=0.12, ha="left", fontsize=11.5, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)
