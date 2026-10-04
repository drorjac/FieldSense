"""Map panels for the notebooks: rain fields in km on the nowcasting grid, motion arrows."""

from __future__ import annotations

import numpy as np


def _extent(meta: dict):
    return [meta["x1"] / 1000, meta["x2"] / 1000, meta["y1"] / 1000, meta["y2"] / 1000]


def field(ax, rate: np.ndarray, meta: dict, title: str = "", vmax: float = 20.0, colorbar: bool = False,
          label: str = "mm/h"):
    """One rain-rate field (row 0 = south) on its km grid."""
    from core import viz_style as vs
    im = ax.imshow(np.where(np.isfinite(rate), rate, np.nan), origin="lower", extent=_extent(meta),
                   cmap=vs.CMAP_RAIN, norm=vs.rain_norm(vmax), interpolation="nearest")
    vs.style_map_axes(ax, _extent(meta))
    ax.set_title(title, fontsize=9.5)
    ax.tick_params(labelsize=7)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(label)
    return im


def quiver(ax, velocity: np.ndarray, meta: dict, step: int = 8, color: str = "#0b0b0b"):
    """Motion arrows (pixels per step) every ``step`` pixels."""
    ny, nx = velocity.shape[1:]
    px = meta["xpixelsize"] / 1000
    y, x = np.mgrid[0:ny:step, 0:nx:step]
    ax.quiver(meta["x1"] / 1000 + (x + 0.5) * px, meta["y1"] / 1000 + (y + 0.5) * px,
              velocity[0, ::step, ::step], velocity[1, ::step, ::step], color=color,
              angles="xy", scale_units="xy", scale=1 / px, width=0.004)


def row(fields: list, titles: list, meta: dict, vmax: float = 20.0, size: float = 3.0, label: str = "mm/h"):
    """A row of maps with one shared colour bar."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(fields), figsize=(size * len(fields) + 0.8, size * 0.9), squeeze=False)
    for ax, f, t in zip(axes[0], fields, titles):
        im = field(ax, f, meta, t, vmax)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label=label)
    return fig, axes[0]
