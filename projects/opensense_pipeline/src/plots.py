"""Figures for the OpenSense merge pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "rainfall_field_sim/src"))
import viz_style as vs                                  # noqa: E402

# One colour per method family, so the eye groups by what the method *uses*
# rather than by its name. Slots follow the documented categorical order.
FAMILY_COLOR = {
    "radar only": "#2a78d6",   # slot 1
    "CML only": "#eb6834",     # slot 2
    "merged": "#1baf7a",       # slot 3
}


def _rain_map(ax, field, extent, vmax, cmap=None):
    im = ax.imshow(field, origin="lower", extent=extent,
                   cmap=cmap or vs.CMAP_RAIN, norm=vs.rain_norm(vmax),
                   interpolation="nearest", aspect="equal")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(vs.BASELINE)
    return im


def _extent_km(ds_rad):
    xg = np.asarray(ds_rad.x_grid) / 1000.0
    yg = np.asarray(ds_rad.y_grid) / 1000.0
    return (xg.min(), xg.max(), yg.min(), yg.max())


# --------------------------------------------------------------------------
def fig_sensors(ds_rad, ds_cml, ds_gauge, t_index, title, path):
    """What each sensor type sees at one timestep: grid, lines, points."""
    extent = _extent_km(ds_rad)
    rad = np.asarray(ds_rad.R.isel(time=t_index))
    cml = np.asarray(ds_cml.R.isel(time=t_index))
    gau = np.asarray(ds_gauge.R.isel(time=t_index))

    vmax = float(max(np.nanmax(rad), np.nanmax(cml), np.nanmax(gau), 1.0))

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8))

    im = _rain_map(axes[0], rad, extent, vmax)
    axes[0].set_title("Radar", color=FAMILY_COLOR["radar only"])

    # CML: draw each link as a line coloured by its retrieved path rain
    _rain_map(axes[1], np.full_like(rad, np.nan), extent, vmax)
    x0 = np.asarray(ds_cml.site_0_x) / 1000.0
    y0 = np.asarray(ds_cml.site_0_y) / 1000.0
    x1 = np.asarray(ds_cml.site_1_x) / 1000.0
    y1 = np.asarray(ds_cml.site_1_y) / 1000.0
    norm = vs.rain_norm(vmax)
    order = np.argsort(np.nan_to_num(cml))
    for i in order:
        v = cml[i]
        color = vs.CMAP_RAIN(norm(v)) if np.isfinite(v) else vs.GRIDLINE
        axes[1].plot([x0[i], x1[i]], [y0[i], y1[i]], color=color,
                     lw=2.0 if np.isfinite(v) and v > 0.1 else 0.7,
                     solid_capstyle="round", zorder=3)
    axes[1].set_title(f"CML ({ds_cml.sizes['cml_id']} links)",
                      color=FAMILY_COLOR["CML only"])

    _rain_map(axes[2], np.full_like(rad, np.nan), extent, vmax)
    gx = np.asarray(ds_gauge.x) / 1000.0
    gy = np.asarray(ds_gauge.y) / 1000.0
    finite = np.isfinite(gau)
    axes[2].scatter(gx[finite], gy[finite], c=gau[finite], cmap=vs.CMAP_RAIN,
                    norm=norm, s=46, edgecolor=vs.INK_SECONDARY, linewidth=0.5,
                    zorder=3)
    axes[2].set_title(f"Rain gauges ({int(finite.sum())} reporting)",
                      color=vs.INK_PRIMARY)

    for ax in axes:
        ax.set_xlabel("easting (km)")
    axes[0].set_ylabel("northing (km)")

    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label("rain rate (mm h$^{-1}$)", color=vs.INK_SECONDARY)
    cb.ax.tick_params(colors=vs.INK_MUTED)
    cb.outline.set_edgecolor(vs.BASELINE)

    fig.suptitle(title, fontsize=12.5, color=vs.INK_PRIMARY, y=1.0)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_merged_maps(ds_rad, fields: dict, labels: dict, title, path,
                    ncols: int = 4):
    """A rainfall map per reconstruction method."""
    extent = _extent_km(ds_rad)
    keys = list(fields)
    vmax = float(max(np.nanpercentile(np.asarray(f), 99.9)
                     for f in fields.values()))
    vmax = max(vmax, 1.0)

    # Size the panels from the domain's real aspect ratio. OpenMRG is roughly
    # square, OpenRainER is nearly 3:1 wide; a fixed square cell leaves one of
    # them swimming in whitespace.
    span_x = extent[1] - extent[0]
    span_y = extent[3] - extent[2]
    aspect = float(np.clip(span_y / span_x, 0.32, 2.2))

    nrows = int(np.ceil(len(keys) / ncols))
    panel_w = 3.5
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols,
                                      (panel_w * aspect + 0.55) * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        im = _rain_map(ax, np.asarray(fields[key]), extent, vmax)
        ax.set_title(labels[key], fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(keys):]:
        ax.axis("off")

    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.024, pad=0.012)
    cb.set_label("rain rate (mm h$^{-1}$)", color=vs.INK_SECONDARY)
    cb.ax.tick_params(colors=vs.INK_MUTED)
    cb.outline.set_edgecolor(vs.BASELINE)

    fig.suptitle(title, fontsize=12.5, color=vs.INK_PRIMARY, y=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_benchmark(rows, path):
    """Synthetic-truth ranking: one facet per rainfall regime.

    Drawn as a dot plot, not bars. The RMSE range spans four orders of
    magnitude once multiplicative merging blows up, which forces a log axis -
    and a bar on a log axis encodes nothing, because its length depends on
    where the axis happens to start. A dot encodes position, which stays
    honest under any scale.
    """
    regimes = list(dict.fromkeys(r["regime"] for r in rows))
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.3 * len(regimes), 5.0))
    axes = np.atleast_1d(axes)

    for ax, regime in zip(axes, regimes):
        sel = [r for r in rows if r["regime"] == regime]
        sel.sort(key=lambda r: r["rmse"])

        vals = np.array([r["rmse"] for r in sel])
        colors = [FAMILY_COLOR[r["family"]] for r in sel]
        blew = [r["frac_implausible"] > 0.001 for r in sel]
        y = np.arange(len(sel))

        floor = float(np.nanmin(vals)) * 0.75
        for yi, (v, c) in enumerate(zip(vals, colors)):
            ax.plot([floor, v], [yi, yi], color=c, lw=1.2, alpha=0.45,
                    zorder=2, solid_capstyle="round")
        ax.scatter(vals, y, s=90, c=colors, zorder=3,
                   edgecolor=vs.SURFACE, linewidth=1.2)

        ax.set_yticks(y)
        ax.set_yticklabels([r["method"] for r in sel], fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("field RMSE (mm h$^{-1}$), log scale")
        ax.set_title(regime, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(floor, float(np.nanmax(vals)) * 14)
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(
                lambda v, _pos: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        for yi, (v, bad) in enumerate(zip(vals, blew)):
            tag = f"{v:,.1f}" + ("   blow-up" if bad else "")
            ax.text(v * 1.35, yi, tag, va="center", fontsize=7.5,
                    color=vs.STATUS_CRITICAL if bad else vs.INK_SECONDARY)

    handles = [Line2D([], [], marker="o", ls="", ms=8, color=c, label=k)
               for k, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Ranked against synthetic truth, sampled by the real "
                 "OpenMRG network geometry",
                 fontsize=12.5, color=vs.INK_PRIMARY, y=1.01)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_gauge_validation(results: dict, labels: dict, gauge_obs, path, title):
    """Real-data check: each method sampled at the gauges, against the gauges."""
    keys = [k for k in results if np.isfinite(results[k]).any()]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))

    lim = 1.0
    for i, key in enumerate(keys):
        est = np.asarray(results[key]).ravel()
        obs = np.asarray(gauge_obs).ravel()
        ok = np.isfinite(est) & np.isfinite(obs)
        if ok.sum() == 0:
            continue
        color = FAMILY_COLOR.get(labels[key]["family"], vs.INK_MUTED)
        ax.scatter(obs[ok], est[ok], s=24, alpha=0.72, color=color,
                   edgecolor=vs.SURFACE, linewidth=0.5,
                   label=labels[key]["label"], zorder=3)
        lim = max(lim, np.nanpercentile(obs[ok], 99.5),
                  np.nanpercentile(est[ok], 99.5))

    ax.plot([0, lim], [0, lim], ls=":", lw=1.2, color=vs.INK_MUTED, zorder=2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("gauge rain rate (mm h$^{-1}$)")
    ax.set_ylabel("reconstructed at gauge (mm h$^{-1}$)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_coverage(rows, path, title):
    """RMSE by method, split by each gauge's distance to the link network.

    The y-axis is capped at a readable range and over-range bars are labelled,
    because the unstable multiplicative merge reaches three figures and would
    otherwise flatten every other method into the baseline.
    """
    bands = list(dict.fromkeys(r["band"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    family = {r["method"]: r["family"] for r in rows}

    stable = [r["rmse"] for r in rows if r["frac_implausible"] <= 0.001
              and np.isfinite(r["rmse"])]
    cap = float(np.nanmax(stable)) * 1.45 if stable else float("nan")

    fig, ax = plt.subplots(figsize=(1.9 * len(bands) + 5.4, 5.0))
    width = 0.8 / len(methods)
    x = np.arange(len(bands))

    for i, m in enumerate(methods):
        vals, over = [], []
        for b in bands:
            hit = [r for r in rows if r["method"] == m and r["band"] == b]
            v = hit[0]["rmse"] if hit else np.nan
            vals.append(min(v, cap) if np.isfinite(v) else np.nan)
            over.append(np.isfinite(v) and v > cap)
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.92, color=FAMILY_COLOR[family[m]],
               label=m, edgecolor=vs.SURFACE, linewidth=0.6)
        for xi, (v, o) in enumerate(zip(vals, over)):
            if o:
                true_v = [r for r in rows if r["method"] == m
                          and r["band"] == bands[xi]][0]["rmse"]
                ax.text(xi + offset, cap * 1.01, f"{true_v:,.0f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90,
                        color=vs.STATUS_CRITICAL)

    ax.set_ylim(0, cap * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n(n={next(r['n_gauges'] for r in rows if r['band'] == b)})"
         for b in bands], fontsize=9)
    ax.set_xlabel("gauge distance to the nearest CML path")
    ax.set_ylabel("RMSE against gauge (mm h$^{-1}$)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", visible=False)

    handles = [Line2D([], [], marker="s", ls="", ms=8, color=c, label=k)
               for k, c in FAMILY_COLOR.items()]
    handles.append(Line2D([], [], ls="", marker="", label="red = over axis cap"))
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", ncol=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_retrieval_vs_reference(ours, ref, stats, path, subtitle=""):
    """Our retrieval against the OpenSense reference, on identical signals."""
    a = np.asarray(ours).ravel()
    b = np.asarray(ref).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6))
    ax_sc, ax_ts, ax_cdf = axes

    # (a) paired scatter. Wet-only, on log axes: the great majority of
    # link-timesteps are dry zeros, which on a linear axis pile onto the
    # origin and hide the part of the range anyone cares about.
    wet = (a > 0.1) & (b > 0.1)
    ax_sc.scatter(b[wet], a[wet], s=5, alpha=0.18,
                  color=FAMILY_COLOR["CML only"], edgecolor="none", zorder=3)
    lim = float(max(np.percentile(a[wet], 99.9), np.percentile(b[wet], 99.9))) \
        if wet.any() else 10.0
    ax_sc.plot([0.1, lim], [0.1, lim], ls=":", lw=1.2, color=vs.INK_MUTED, zorder=4)
    ax_sc.set_xscale("log")
    ax_sc.set_yscale("log")
    ax_sc.set_xlim(0.1, lim)
    ax_sc.set_ylim(0.1, lim)
    ax_sc.set_xlabel("OpenSense reference R (mm h$^{-1}$)")
    ax_sc.set_ylabel("our retrieval (mm h$^{-1}$)")
    ax_sc.set_title("Paired link-timesteps, wet only")
    vs.annotate_corner(
        ax_sc,
        f"r = {stats['corr_wet']:.3f}\nratio = {stats['ratio_total']:.2f}\n"
        f"n = {int(wet.sum()):,}", loc="upper left")

    # (b) network-mean time series
    ours2d = np.asarray(ours)
    ref2d = np.asarray(ref)
    with np.errstate(invalid="ignore"):
        m_ours = np.nanmean(ours2d, axis=1)
        m_ref = np.nanmean(ref2d, axis=1)
    t = np.arange(m_ours.size)
    ax_ts.plot(t, m_ref, color=vs.INK_SECONDARY, lw=1.6, label="reference")
    ax_ts.plot(t, m_ours, color=FAMILY_COLOR["CML only"], lw=1.6, label="ours")
    ax_ts.set_xlabel("timestep")
    ax_ts.set_ylabel("network-mean rain (mm h$^{-1}$)")
    ax_ts.set_title("Network mean over time")
    ax_ts.legend(loc="upper right")

    # (c) exceedance, which shows where in the distribution they part company
    levels = np.logspace(-1, np.log10(max(lim, 1.1)), 60)
    ax_cdf.plot(levels, [(b > lv).mean() * 100 for lv in levels],
                color=vs.INK_SECONDARY, lw=1.8, label="reference")
    ax_cdf.plot(levels, [(a > lv).mean() * 100 for lv in levels],
                color=FAMILY_COLOR["CML only"], lw=1.8, label="ours")
    ax_cdf.set_xscale("log")
    ax_cdf.set_yscale("log")
    ax_cdf.set_xlabel("rain rate r (mm h$^{-1}$)")
    ax_cdf.set_ylabel("link-timesteps with R > r  (%)")
    ax_cdf.set_title("Intensity exceedance")
    ax_cdf.legend(loc="lower left")

    head = "Our retrieval vs the OpenSense reference, identical input signals"
    fig.suptitle(f"{head}  —  {subtitle}" if subtitle else head,
                 fontsize=12.5, color=vs.INK_PRIMARY, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
