"""Figures for the physics_ml notebooks; each returns the figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import viz_style as vs

# fixed order: the physics reference in ink, the rest in palette order
_STRATEGY_STYLE = {"physics": (vs.INK_PRIMARY, "--"), "mlp": (vs.INK_MUTED, "-"),
                   "scalar": (vs.SERIES["stratiform"], "-"),
                   "gate": (vs.SERIES["convective"], "-"),
                   "fc": (vs.SERIES["frontal"], "-"),
                   "residual": (vs.STATUS_CRITICAL, "-")}


def mixing_vs_noise(table: pd.DataFrame, strategies=("physics", "mlp", "scalar",
                                                     "gate", "fc", "residual")):
    """Test MSE (mean and 1 s.d. over repeats) against noise, one panel per frequency."""
    freqs = sorted(table.freq_ghz.unique())
    fig, axes = plt.subplots(1, len(freqs), figsize=(4.2 * len(freqs), 3.6),
                             squeeze=False, sharex=True)
    stats = table.groupby(["freq_ghz", "noise_db"])[list(strategies)].agg(["mean", "std"])
    for ax, f in zip(axes[0], freqs):
        s = stats.loc[f]
        for name in strategies:
            color, ls = _STRATEGY_STYLE[name]
            m, sd = s[(name, "mean")], s[(name, "std")]
            ax.plot(s.index, m, color=color, ls=ls, lw=1.8, label=name)
            ax.fill_between(s.index, m - sd, m + sd, color=color, alpha=0.12, lw=0)
        ax.set(title=f"{f:g} GHz", xlabel="noise σ (dB)")
    axes[0][0].set_ylabel("test MSE (mm/h)²")
    axes[0][-1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def weights_vs_noise(table: pd.DataFrame):
    """Fitted blend weight on the physics branch against noise, per frequency."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    colors = list(vs.SERIES.values()) + [vs.INK_MUTED]
    for ax, col, title in zip(axes, ("scalar_weight", "gate_weight_mean"),
                              ("scalar weight w*", "gate weight, mean over test")):
        g = table.groupby(["freq_ghz", "noise_db"])[col].agg(["mean", "std"])
        for f, c in zip(sorted(table.freq_ghz.unique()), colors):
            s = g.loc[f]
            ax.plot(s.index, s["mean"], color=c, label=f"{f:g} GHz")
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"],
                            color=c, alpha=0.12, lw=0)
        ax.set(title=title, xlabel="noise σ (dB)", ylim=(-0.05, 1.05))
    axes[0].set_ylabel("weight on physics")
    axes[1].legend()
    fig.tight_layout()
    return fig


def training_histories(results: dict, itu: tuple):
    """Loss, gate weight and learned (k, alpha) per epoch, staged vs joint."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.5))
    colors = {"staged": vs.SERIES["stratiform"], "joint": vs.SERIES["convective"]}
    for method, r in results.items():
        h, c = r["history"], colors[method]
        x = np.arange(1, len(h["val_loss"]) + 1)
        axes[0].plot(x, h["val_loss"], color=c, label=method)
        axes[1].plot(x, h["gate"], color=c)
        axes[2].plot(x, h["k"], color=c)
        axes[3].plot(x, h["alpha"], color=c)
    for ax, value in zip(axes[2:], itu):
        ax.axhline(value, color=vs.INK_MUTED, ls="--", lw=1, label="ITU-R")
    for ax, title in zip(axes, ("validation loss", "gate (1 = physics)", "learned k",
                                "learned α")):
        ax.set(title=title, xlabel="epoch")
    axes[1].set_ylim(0, 1)
    axes[0].legend()
    axes[2].legend()
    fig.tight_layout()
    return fig
