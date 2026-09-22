"""
Shared matplotlib styling for this project.

Colors come from the repository's data-visualization palette. The categorical
slots are used in their documented order (blue, orange, aqua) - that ordering is
the colorblind-safety mechanism, so it is not reshuffled. Three slots is also
the documented cap for all-pairs chart forms such as scatter, which is exactly
what this project needs.
"""

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- categorical slots (documented order) ---------------------------------
SERIES = {
    "stratiform": "#2a78d6",   # slot 1, blue
    "convective": "#eb6834",   # slot 2, orange
    "frontal": "#1baf7a",      # slot 3, aqua
}
SERIES_ORDER = ("stratiform", "convective", "frontal")

# --- chart chrome & ink (light surface) -----------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- status (used only for pass/fail style callouts, never as a series) ---
STATUS_CRITICAL = "#d03b3b"
STATUS_GOOD = "#0ca30c"

# Sequential blue ramp, steps 100-700 of the palette. Step 0 is the chart
# surface so that "no rain" recedes into the background rather than reading as
# a value.
_BLUE_RAMP = [
    "#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
    "#104281", "#0d366b",
]
CMAP_RAIN = LinearSegmentedColormap.from_list("rain_blue", _BLUE_RAMP)

# Diverging blue <-> gray <-> red for signed error. The blue arm uses the
# palette's sequential steps; the red arm is generated to mirror it in
# lightness, anchored on the palette's categorical red (#e34948) and the
# darker critical red (#d03b3b). Midpoint is the palette's neutral gray.
_DIVERGING = [
    "#104281", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb",
    "#f0efec",
    "#f9cfcf", "#f0a3a3", "#e87070", "#e34948", "#d03b3b", "#a62e2e",
]
CMAP_ERROR = LinearSegmentedColormap.from_list("error_div", _DIVERGING)


def use_style() -> None:
    """Apply the project's matplotlib defaults."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titleweight": "semibold",
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": INK_PRIMARY,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 8.0 ** 0.5 * 3,  # >= 8 px markers
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def style_map_axes(ax, extent_km) -> None:
    """Common treatment for the 2-D field maps: equal aspect, no grid."""
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlim(extent_km[0], extent_km[1])
    ax.set_ylim(extent_km[2], extent_km[3])
    for spine in ax.spines.values():
        spine.set_color(BASELINE)


def rain_norm(vmax: float):
    """Shared color normalization for rain-rate maps.

    Rain-rate distributions are heavy-tailed, so a linear scale hides almost
    everything below the convective peaks. A power-law norm with gamma < 1
    keeps the light rain legible without the perceptual dishonesty of a
    log scale that has to fake a zero.
    """
    return mpl.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)


def label_axis_km(ax) -> None:
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")


def annotate_corner(ax, text: str, loc: str = "upper left") -> None:
    """Small recessive annotation inside a map panel."""
    xy = {"upper left": (0.03, 0.97), "upper right": (0.97, 0.97),
          "lower left": (0.03, 0.03), "lower right": (0.97, 0.03)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=8, color=INK_PRIMARY, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                      edgecolor=BASELINE, linewidth=0.6, alpha=0.92))
