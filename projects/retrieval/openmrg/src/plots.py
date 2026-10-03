"""Figures for the cml_retrieval notebooks, so the notebooks stay narrative.

Every function draws onto new or given axes and returns the figure; none
calls ``plt.show()``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from core import viz_style as vs

C1, C2, C3 = (vs.SERIES[k] for k in vs.SERIES_ORDER)


# --------------------------------------------------------------------------
# data-driven (RNN)
# --------------------------------------------------------------------------
def rain_distribution(rain: np.ndarray, exp_fit: tuple):
    """Gauge rain-rate histogram with the fitted exponential the loss uses."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    import scipy.stats
    density, _, _ = ax.hist(rain, bins=100, density=True, color=C1, alpha=0.8)
    grid = np.linspace(0, np.max(rain), 200)
    ax.plot(grid, scipy.stats.expon.pdf(grid, *exp_fit), color=vs.INK_PRIMARY,
            lw=1.5, label=f"exponential, scale {exp_fit[1]:.3f}")
    ax.set_yscale("log")
    # the fitted pdf decays far below any observed density; keep the data in view
    ax.set_ylim(density[density > 0].min() / 3, density.max() * 3)
    ax.set(xlabel="gauge rain rate (mm/h)", ylabel="density")
    ax.legend()
    return fig


def training_curves(history: dict):
    """Total, estimation and detection loss per epoch."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for key, label, c in (("loss", "total", vs.INK_PRIMARY),
                          ("loss_est", f"rain rate x lambda ({history['lambda']:.3f})", C1),
                          ("loss_det", "wet/dry", C2)):
        values = np.asarray(history[key])
        if key == "loss_est":
            values = values * history["lambda"]
        ax.plot(values, color=c, label=label)
    ax.set(xlabel="epoch", ylabel="loss")
    ax.legend()
    return fig


def detection_timeline(ref: np.ndarray, detection: np.ndarray, link: int = 0,
                       n: int = 300, wet_threshold: float = 0.1):
    """One link's reference rain, shaded by detection outcome."""
    r, d = ref[link, :n], np.round(detection[link, :n])
    wet = r > wet_threshold
    x = np.arange(r.size)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    top = max(float(r.max()), 1.0)
    for mask, color, label in ((wet == d.astype(bool), vs.GRIDLINE, "correct"),
                               (wet & (d == 0), vs.STATUS_CRITICAL, "missed rain"),
                               (~wet & (d == 1), C1, "false alarm")):
        ax.fill_between(x, 0, top, where=mask, color=color, alpha=0.6,
                        step="mid", lw=0, label=label)
    ax.plot(x, r, color=vs.INK_PRIMARY, lw=1.2, label="gauge")
    ax.set(xlabel="sample (15 min)", ylabel="rain rate (mm/h)")
    ax.legend(loc="upper right", ncol=4)
    return fig


def accumulation(series: dict, dt_hours: float = 0.25):
    """Cumulative rain (mm) of several (time,) series, e.g. reference vs estimate."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    colors = [vs.INK_PRIMARY, C1, C2, C3]
    for (label, s), c in zip(series.items(), colors):
        ax.plot(np.cumsum(np.maximum(np.nan_to_num(s), 0)) * dt_hours, color=c,
                ls="--" if c == vs.INK_PRIMARY else "-", label=label)
    ax.set(xlabel="sample", ylabel="accumulated rain (mm)")
    ax.legend()
    return fig


def confusion(matrix: np.ndarray):
    """Detection confusion matrix, gauge (rows) against detector (columns)."""
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(matrix, cmap=vs.CMAP_RAIN)
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color=vs.SURFACE if v > matrix.max() / 2 else vs.INK_PRIMARY)
    ax.set_xticks([0, 1], ["dry", "wet"])
    ax.set_yticks([0, 1], ["dry", "wet"])
    ax.set(xlabel="detector", ylabel="gauge")
    ax.grid(False)
    return fig


# --------------------------------------------------------------------------
# model-driven
# --------------------------------------------------------------------------
def wet_dry_panels(detection: np.ndarray, sigma: np.ndarray, ref: np.ndarray):
    """Statistical wet/dry test: detection, its test statistic, the gauge."""
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax[0].step(np.arange(detection.size), detection, color=C1, where="mid")
    ax[0].set_ylabel("detection")
    ax[1].plot(sigma, color=C2)
    ax[1].set_ylabel(r"$\sigma_n$ (dB)")
    n = min(ref.size, detection.size)
    ax[2].fill_between(np.arange(n), 0, max(ref.max(), 1), where=detection[:n] > 0.5,
                       color=vs.GRIDLINE, step="mid", lw=0, label="classified wet")
    ax[2].plot(ref, color=vs.INK_PRIMARY, lw=1.2, label="gauge")
    ax[2].set(xlabel="sample (15 min)", ylabel="rain rate (mm/h)")
    ax[2].legend(loc="upper right")
    return fig


def baselines_panels(attenuation: np.ndarray, baselines: dict):
    """Baselines, raw max/min attenuation, and attenuation above each baseline."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for (label, b), c in zip(baselines.items(), (C1, C2)):
        ax[0].plot(b, color=c, label=label)
        ax[2].plot(np.maximum(attenuation[:, 0] - b, 0), color=c, label=label)
    ax[1].plot(attenuation[:, 0], color=C1, label=r"$A^{max}$")
    ax[1].plot(attenuation[:, 1], color=C2, label=r"$A^{min}$")
    for a, title in zip(ax, ("baseline (dB)", "attenuation (dB)",
                             "attenuation above baseline (dB)")):
        a.set(xlabel="sample", title=title)
        a.legend()
    return fig


# --------------------------------------------------------------------------
# rain maps
# --------------------------------------------------------------------------
def map_panels(maps: dict, title: str = "", vmax: float | None = None,
               show_links: bool = False):
    """One panel per reconstruction, each on its own grid, one colour scale.

    ``maps`` values carry ``grid``, ``x``, ``y`` and ``label`` (the shape
    ``rain_maps`` and ``model_driven`` return). With ``show_links``, each
    map's own ``links`` - endpoints in *that map's* coordinate frame, which
    differ between packages (lon/lat, UTM metres, PyNNcml's normalized grid)
    - are drawn on top.
    """
    if vmax is None:
        vmax = float(np.nanpercentile(
            np.concatenate([np.ravel(m["grid"]) for m in maps.values()]), 99.5))
    fig, axes = plt.subplots(1, len(maps), figsize=(4.1 * len(maps), 4.3),
                             squeeze=False)
    for ax, m in zip(axes[0], maps.values()):
        extent = [np.min(m["x"]), np.max(m["x"]), np.min(m["y"]), np.max(m["y"])]
        im = ax.imshow(m["grid"], origin="lower", extent=extent, vmin=0, vmax=vmax,
                       cmap=vs.CMAP_RAIN, norm=None, aspect="auto",
                       interpolation="nearest")
        if show_links and "links" in m:
            for x0, y0, x1, y1 in zip(*m["links"]):
                ax.plot([x0, x1], [y0, y1], color=vs.INK_PRIMARY, lw=0.5, alpha=0.6)
        ax.set_title(m["label"], fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.012).set_label("rain rate (mm/h)")
    if title:
        fig.suptitle(title, y=1.02)
    return fig
