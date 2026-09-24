"""Figure for the radar-versus-CML comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "rainfall_field_sim/src"))
import viz_style as vs  # noqa: E402

RADAR_C, CML_C = "#2a78d6", "#eb6834"


def figure(runs, path):
    n = len(runs)
    fig = plt.figure(figsize=(4.6 * n, 10.4))
    gs = fig.add_gridspec(3, n, height_ratios=[1.35, 1.0, 1.0], hspace=0.38,
                          wspace=0.24)

    for col, r in enumerate(runs):
        rad_acc, cml_acc, dist, xg, yg = r["_acc"]
        extent = (xg.min() / 1000, xg.max() / 1000,
                  yg.min() / 1000, yg.max() / 1000)
        vmax = float(np.nanpercentile(
            np.concatenate([rad_acc.ravel(), cml_acc.ravel()]), 99))

        # --- row 0: accumulation maps, radar over CML ---
        ax = fig.add_subplot(gs[0, col])
        half = rad_acc.shape[1] // 2
        combined = np.where(
            np.arange(rad_acc.shape[1])[None, :] < half, rad_acc, cml_acc)
        im = ax.imshow(combined, origin="lower", extent=extent, vmin=0,
                       vmax=vmax, cmap=vs.CMAP_RAIN, aspect="auto",
                       interpolation="nearest")
        ax.axvline(extent[0] + (extent[1] - extent[0]) * half
                   / rad_acc.shape[1], color=vs.INK_PRIMARY, lw=1.2)
        ax.text(0.02, 0.97, "RADAR", transform=ax.transAxes, va="top",
                fontsize=8.5, color=RADAR_C, fontweight="bold")
        ax.text(0.98, 0.97, "CML", transform=ax.transAxes, va="top",
                ha="right", fontsize=8.5, color=CML_C, fontweight="bold")
        ax.set_title(f"{r['label']}\n{r['regime']}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if col == n - 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("accumulated rain (mm h$^{-1}$ summed)", fontsize=8)

        # --- row 1: per-timestep correlation ---
        ax = fig.add_subplot(gs[1, col])
        corr = [p["corr"] for p in r["_per_step"]]
        ax.plot(range(len(corr)), corr, color=vs.INK_SECONDARY, lw=1.4)
        ax.axhline(r["corr_median"], color=RADAR_C, ls="--", lw=1.4,
                   label=f"median {r['corr_median']:.2f}")
        ax.set_ylim(-0.2, 1.0)
        ax.set_xlabel("wet timestep (wettest first)")
        if col == 0:
            ax.set_ylabel("spatial correlation")
        ax.set_title("agreement per timestep", fontsize=9.5)
        ax.legend(fontsize=7.5, loc="lower left")

        # --- row 2: correlation vs distance from the network ---
        ax = fig.add_subplot(gs[2, col])
        bands = list(r["band_corr"])
        vals = [r["band_corr"][b] for b in bands]
        colors = [CML_C if np.isfinite(v) else vs.GRIDLINE for v in vals]
        plotted = [v if np.isfinite(v) else 0.0 for v in vals]
        ax.bar(range(len(bands)), plotted, color=colors, width=0.62)
        ax.axhline(0, color=vs.BASELINE, lw=1.0)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.replace(" km", "").replace("-inf", "+")
                            for b in bands], fontsize=8.5)
        lo = min(0.0, float(np.nanmin(vals))) - 0.12
        ax.set_ylim(lo, 1.0)
        ax.set_xlabel("distance to nearest link (km)")
        if col == 0:
            ax.set_ylabel("correlation of accumulation")
        ax.set_title("agreement vs coverage", fontsize=9.5)
        ax.grid(axis="x", visible=False)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                off = 0.03 if v >= 0 else -0.09
                ax.text(i, v + off, f"{v:.2f}", ha="center", fontsize=8,
                        color=vs.STATUS_CRITICAL if v < 0 else vs.INK_SECONDARY)

    fig.suptitle("Radar against CML-derived rainfall, through precipitation "
                 "events", fontsize=13, color=vs.INK_PRIMARY, y=0.995)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
