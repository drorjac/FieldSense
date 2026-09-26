"""Figures for the nowcasting notebook; each returns the figure.

The original notebook drew grouped bars for 15+ models in one palette,
which cannot be read at that count. Here the full comparison is a heatmap
(model x horizon, one metric), and colour lines are kept for a focus set of
at most five models.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import viz_style as vs

FOCUS_COLORS = [vs.INK_PRIMARY, vs.SERIES["stratiform"], vs.SERIES["convective"],
                vs.SERIES["frontal"], vs.INK_MUTED]


def skill_heatmap(table: pd.DataFrame, metric: str = "HSS", higher_is_better: bool = True,
                  title: str = ""):
    """Every model x horizon for one metric, models sorted by their mean."""
    wide = table.pivot(index="model", columns="horizon_min", values=metric)
    wide = wide.loc[wide.mean(axis=1).sort_values(ascending=not higher_is_better).index]
    fig, ax = plt.subplots(figsize=(1.3 * wide.shape[1] + 3.5, 0.34 * len(wide) + 1.2))
    cmap = vs.CMAP_RAIN if higher_is_better else vs.CMAP_RAIN.reversed()
    im = ax.imshow(wide.values, cmap=cmap, aspect="auto")
    for (i, j), v in np.ndenumerate(wide.values):
        if np.isfinite(v):
            dark = im.norm(v) > 0.6 if higher_is_better else im.norm(v) < 0.4
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=vs.SURFACE if dark else vs.INK_PRIMARY)
    ax.set_xticks(range(wide.shape[1]), [f"{h} min" for h in wide.columns])
    ax.set_yticks(range(len(wide)), wide.index, fontsize=8)
    ax.grid(False)
    ax.set_title(title or f"{metric} ({'higher' if higher_is_better else 'lower'} is better)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return fig


def skill_vs_horizon(table: pd.DataFrame, models, metrics=("HSS", "RMSE"), title: str = ""):
    """A handful of models' skill as the horizon grows."""
    models = [m for m in models if m in set(table.model)][:len(FOCUS_COLORS)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.6), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for m, c in zip(models, FOCUS_COLORS):
            t = table[table.model == m].sort_values("horizon_min")
            ax.plot(t.horizon_min, t[metric], "o-", color=c, label=m)
        ax.set(xlabel="horizon (min)", title=metric)
    axes[0][0].legend(fontsize=8)
    if title:
        fig.suptitle(title, x=0.05, ha="left")
    fig.tight_layout()
    return fig


def peak_maps(frames: dict, lon, lat, crop: tuple, gauges=None, vmax: float = 35.0,
              title: str = ""):
    """Truth and forecasts at one time, one shared power-law colour scale.

    ``gauges`` is (lon, lat, values), drawn in the same colours as the field.
    """
    y0, y1, x0, x1 = crop
    extent = [lon[y0:y1, x0:x1].min(), lon[y0:y1, x0:x1].max(),
              lat[y0:y1, x0:x1].min(), lat[y0:y1, x0:x1].max()]
    norm = vs.rain_norm(vmax)
    cols = min(4, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), squeeze=False)
    for ax, (name, field) in zip(axes.ravel(), frames.items()):
        im = ax.imshow(field[y0:y1, x0:x1], origin="lower", extent=extent, cmap=vs.CMAP_RAIN,
                       norm=norm, aspect="auto")
        if gauges is not None:
            ax.scatter(gauges[0], gauges[1], c=np.nan_to_num(gauges[2]), cmap=vs.CMAP_RAIN,
                       norm=norm, s=45, edgecolor=vs.INK_PRIMARY, lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.ravel()[len(frames):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01).set_label("mm/h")
    if title:
        fig.suptitle(title, x=0.05, ha="left")
    return fig


def training_curves(histories: dict, title: str = ""):
    """Validation loss per epoch for each trained forecaster."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (name, h), c in zip(histories.items(), FOCUS_COLORS * 4):
        ax.plot(h["val"], color=c, lw=1.2, label=name)
    ax.set(xlabel="epoch", ylabel="validation MSE", yscale="log", title=title)
    ax.legend(fontsize=7, ncol=2)
    return fig
