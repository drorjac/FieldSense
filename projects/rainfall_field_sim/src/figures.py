"""Figure builders for the rainfall-field / CML demonstration."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core import viz_style as vs
from core.simulation.rain_fields import (WET_THRESHOLD_MM_H, exceedance_curve,
                                         radial_autocorrelation)


def _rain_map(ax, rain, grid, vmax, cmap=None):
    im = ax.imshow(rain, origin="lower", extent=grid.extent_km,
                   cmap=cmap or vs.CMAP_RAIN, norm=vs.rain_norm(vmax),
                   interpolation="nearest")
    vs.style_map_axes(ax, grid.extent_km)
    return im


# --------------------------------------------------------------------------
def fig_fields(models, fields, stats, grid, path):
    """The three rainfall fields side by side."""
    vmax = max(f.max() for f in fields.values())
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))

    for ax, m in zip(axes, models):
        r = fields[m.key]
        s = stats[m.key]
        im = _rain_map(ax, r, grid, vmax)
        ax.set_title(m.name, color=vs.SERIES[m.key])
        ax.set_xlabel("x (km)")
        if ax is axes[0]:
            ax.set_ylabel("y (km)")
        vs.annotate_corner(
            ax,
            f"wet area   {s['war']*100:.0f}%\n"
            f"mean       {s['imf']:.2f} mm/h\n"
            f"wet mean   {s['cmf']:.2f} mm/h\n"
            f"peak       {s['max']:.0f} mm/h\n"
            f"decorr.    {s['decorrelation_km']:.1f} km",
        )

    cbar = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.015,
                        ticks=[0, 1, 5, 15, 30, 60, int(vmax)])
    cbar.set_label("rain rate (mm h$^{-1}$)", color=vs.INK_SECONDARY)
    cbar.ax.tick_params(colors=vs.INK_MUTED)
    cbar.outline.set_edgecolor(vs.BASELINE)

    fig.suptitle("Three rainfall regimes on a 32 x 32 km domain, "
                 "comparable mean rain, very different structure",
                 fontsize=12.5, color=vs.INK_PRIMARY, y=1.00)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_proportions(models, fields, stats, grid, path):
    """How the water is distributed - the 'proportion' view."""
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9))
    ax_war, ax_exc, ax_acf, ax_lor = axes

    # (a) wet-area ratio
    names = [m.name for m in models]
    wars = [stats[m.key]["war"] * 100 for m in models]
    colors = [vs.SERIES[m.key] for m in models]
    bars = ax_war.bar(range(len(models)), wars, color=colors, width=0.62)
    for b, v in zip(bars, wars):
        ax_war.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=9.5,
                    color=vs.INK_PRIMARY, fontweight="semibold")
    ax_war.set_xticks(range(len(models)))
    ax_war.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8.5)
    ax_war.set_ylabel("wet area (% of domain)")
    ax_war.set_ylim(0, 100)
    ax_war.set_title("Wet-area ratio")
    ax_war.grid(axis="x", visible=False)

    # (b) exceedance
    levels = np.logspace(-1, np.log10(90), 60)
    for m in models:
        ax_exc.plot(levels, exceedance_curve(fields[m.key], levels) * 100,
                    color=vs.SERIES[m.key], label=m.name)
    ax_exc.set_xscale("log")
    ax_exc.set_yscale("log")
    ax_exc.set_xlabel("rain rate r (mm h$^{-1}$)")
    ax_exc.set_ylabel("area with R > r  (%)")
    ax_exc.set_title("Intensity exceedance")
    ax_exc.set_ylim(0.005, 120)
    ax_exc.legend(loc="lower left")

    # (c) spatial autocorrelation
    for m in models:
        lags, acf = radial_autocorrelation(fields[m.key], grid)
        ax_acf.plot(lags, acf, color=vs.SERIES[m.key], label=m.name)
        d = stats[m.key]["decorrelation_km"]
        ax_acf.plot([d], [np.exp(-1)], marker="o", ms=6,
                    color=vs.SERIES[m.key], mec=vs.SURFACE, mew=1.5)
    ax_acf.axhline(np.exp(-1), color=vs.INK_MUTED, lw=1.0, ls=":")
    ax_acf.text(11.6, np.exp(-1) + 0.03, "1/e", fontsize=8,
                color=vs.INK_MUTED, ha="right")
    ax_acf.set_xlabel("lag (km)")
    ax_acf.set_ylabel("spatial autocorrelation")
    ax_acf.set_title("Decorrelation")
    ax_acf.set_ylim(-0.25, 1.02)
    ax_acf.legend(loc="upper right")

    # (d) concentration of water volume
    fracs = np.linspace(0.001, 1.0, 200)
    for m in models:
        flat = np.sort(fields[m.key].ravel())[::-1]
        cum = np.cumsum(flat) / flat.sum()
        idx = np.clip((fracs * flat.size).astype(int) - 1, 0, flat.size - 1)
        ax_lor.plot(fracs * 100, cum[idx] * 100, color=vs.SERIES[m.key],
                    label=m.name)
    ax_lor.plot([0, 100], [0, 100], color=vs.INK_MUTED, lw=1.0, ls=":")
    ax_lor.text(62, 52, "uniform", fontsize=8, color=vs.INK_MUTED, rotation=33)
    ax_lor.set_xlabel("wettest share of area (%)")
    ax_lor.set_ylabel("share of total water (%)")
    ax_lor.set_title("Water concentration")
    ax_lor.set_xlim(0, 100)
    ax_lor.set_ylim(0, 101)
    ax_lor.legend(loc="lower right")

    fig.suptitle("Same order of mean rainfall, entirely different proportions",
                 fontsize=12.5, color=vs.INK_PRIMARY, y=1.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_network(models, fields, results, net, grid, path):
    """What the link network actually sees."""
    vmax = max(f.max() for f in fields.values())
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))

    for ax, m in zip(axes, models):
        _rain_map(ax, fields[m.key], grid, vmax)
        r_hat = results[m.key]["R_retrieved"]
        order = np.argsort(r_hat)          # draw wet links last, on top
        for i in order:
            wet = r_hat[i] >= WET_THRESHOLD_MM_H
            ax.plot([net.xa[i], net.xb[i]], [net.ya[i], net.yb[i]],
                    color=vs.STATUS_CRITICAL if wet else vs.INK_MUTED,
                    lw=2.2 if wet else 0.8,
                    alpha=0.95 if wet else 0.45,
                    solid_capstyle="round", zorder=3 if wet else 2)
        n_wet = int((r_hat >= WET_THRESHOLD_MM_H).sum())
        ax.set_title(m.name, color=vs.SERIES[m.key])
        ax.set_xlabel("x (km)")
        if ax is axes[0]:
            ax.set_ylabel("y (km)")
        vs.annotate_corner(ax, f"links reporting rain\n"
                               f"{n_wet} of {net.n_links}", loc="upper left")

    handles = [Line2D([], [], color=vs.STATUS_CRITICAL, lw=2.2,
                      label="link reporting rain (>= 0.1 mm/h retrieved)"),
               Line2D([], [], color=vs.INK_MUTED, lw=1.0, alpha=0.6,
                      label="link reporting dry")]
    fig.suptitle(f"A {net.n_links}-link network over each field: "
                 f"the same sensors, wildly different coverage of the rain",
                 fontsize=12.5, color=vs.INK_PRIMARY, y=0.99)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.005))
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_sensor_physics(models, results, net, cfg, path):
    """The two errors that survive a perfect receiver, plus the ones that do not."""
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
    ax_bias, ax_budget, ax_scatter = axes

    # (a) path-averaging (Jensen) bias vs along-path heterogeneity
    for m in models:
        res = results[m.key]
        Rt = res["R_path_true"]
        Rc = res["R_retrieved_clean"]
        wet = Rt > 0.5
        if not wet.any():
            continue
        cv = res["samples"].std(axis=1) / np.maximum(Rt, 1e-9)
        rel = (Rc - Rt) / np.maximum(Rt, 1e-9) * 100
        hi = wet & (net.alpha > 1.0)
        lo = wet & (net.alpha <= 1.0)
        ax_bias.scatter(cv[hi], rel[hi], s=34, marker="o",
                        color=vs.SERIES[m.key], edgecolor=vs.SURFACE,
                        linewidth=0.8, label=m.name, zorder=3)
        ax_bias.scatter(cv[lo], rel[lo], s=34, marker="v",
                        color=vs.SERIES[m.key], edgecolor=vs.SURFACE,
                        linewidth=0.8, zorder=3)

    ax_bias.axhline(0.0, color=vs.BASELINE, lw=1.2, zorder=1)
    ax_bias.set_xlabel("along-path variability of rain  (CV)")
    ax_bias.set_ylabel("retrieval bias vs true path mean (%)")
    ax_bias.set_title("Path-averaging bias")

    shape_handles = [
        Line2D([], [], marker="o", ls="", color=vs.INK_SECONDARY,
               mec=vs.SURFACE, label=r"$\alpha>1$  (below ~23 GHz)"),
        Line2D([], [], marker="v", ls="", color=vs.INK_SECONDARY,
               mec=vs.SURFACE, label=r"$\alpha<1$  (above ~23 GHz)"),
    ]
    leg1 = ax_bias.legend(loc="upper left", fontsize=8)
    ax_bias.add_artist(leg1)
    ax_bias.legend(handles=shape_handles, loc="lower right", fontsize=8)

    # (b) median magnitude of each term in the attenuation budget
    labels = ["rain\n(signal)", "wet\nantenna", "baseline\nerror",
              "receiver\nnoise", "quantiz-\nation"]
    key = models[1].key          # convective: the most demanding case
    res = results[key]
    wet = res["R_path_true"] > 0.5
    medians = [
        float(np.median(res["A_rain"][wet])),
        float(np.median(res["A_waa"][wet])),
        cfg.baseline_sigma_db,
        cfg.noise_sigma_db,
        cfg.quantization_db / np.sqrt(12.0),
    ]
    bar_colors = [vs.SERIES["convective"]] + [vs.INK_MUTED] * 4
    bars = ax_budget.bar(range(5), medians, color=bar_colors, width=0.62)
    for b, v in zip(bars, medians):
        ax_budget.text(b.get_x() + b.get_width() / 2,
                       v + max(medians) * 0.03, f"{v:.2f}",
                       ha="center", va="bottom", fontsize=8.5,
                       color=vs.INK_PRIMARY)
    ax_budget.set_xticks(range(5))
    ax_budget.set_xticklabels(labels, fontsize=8)
    ax_budget.set_ylabel("attenuation (dB)")
    ax_budget.set_ylim(0, max(medians) * 1.22)
    ax_budget.set_title("Attenuation budget, wet links\n(convective field)",
                        fontsize=10.5)
    ax_budget.grid(axis="x", visible=False)

    # (c) retrieved vs true path average, full chain
    lim = 0.0
    for m in models:
        res = results[m.key]
        ax_scatter.scatter(res["R_path_true"], res["R_retrieved"], s=30,
                           color=vs.SERIES[m.key], edgecolor=vs.SURFACE,
                           linewidth=0.8, label=m.name, zorder=3)
        lim = max(lim, res["R_path_true"].max(), res["R_retrieved"].max())
    lim *= 1.06
    ax_scatter.plot([0, lim], [0, lim], color=vs.INK_MUTED, lw=1.2, ls=":",
                    zorder=2)
    ax_scatter.text(lim * 0.72, lim * 0.79, "1:1", fontsize=8.5,
                    color=vs.INK_MUTED, rotation=38)
    ax_scatter.set_xlim(-0.4, lim)
    ax_scatter.set_ylim(-0.4, lim)
    ax_scatter.set_xlabel("true path-mean rain (mm h$^{-1}$)")
    ax_scatter.set_ylabel("retrieved (mm h$^{-1}$)")
    ax_scatter.set_title("Full sensor chain, per link")
    ax_scatter.legend(loc="upper left")

    fig.suptitle("Modelling the CML sensor: what each stage costs",
                 fontsize=12.5, color=vs.INK_PRIMARY, y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_reconstruction(models, fields, recon, grid, path):
    """Truth, ideal-sampling reconstruction, full-sensor reconstruction, error."""
    vmax = max(f.max() for f in fields.values())
    fig, axes = plt.subplots(3, 4, figsize=(14.0, 10.2))

    err_lim = float(np.percentile(
        np.concatenate([np.abs(recon[m.key]["full"] - fields[m.key]).ravel()
                        for m in models]), 99.5))

    col_titles = ["truth", "reconstructed from\ntrue path means",
                  "reconstructed from\nretrieved path means", "error\n(full chain)"]

    for row, m in enumerate(models):
        truth = fields[m.key]
        ideal = recon[m.key]["ideal"]
        full = recon[m.key]["full"]
        err = full - truth

        im_rain = _rain_map(axes[row, 0], truth, grid, vmax)
        _rain_map(axes[row, 1], ideal, grid, vmax)
        _rain_map(axes[row, 2], full, grid, vmax)
        im_err = axes[row, 3].imshow(err, origin="lower", extent=grid.extent_km,
                                     cmap=vs.CMAP_ERROR, vmin=-err_lim,
                                     vmax=err_lim, interpolation="nearest")
        vs.style_map_axes(axes[row, 3], grid.extent_km)

        axes[row, 0].set_ylabel(f"{m.name}\ny (km)", color=vs.SERIES[m.key],
                                fontsize=10, fontweight="semibold")
        s = recon[m.key]["score_full"]
        vs.annotate_corner(axes[row, 2],
                           f"r = {s['corr']:.2f}\n"
                           f"peak kept {s['peak_ratio']*100:.0f}%",
                           loc="upper left")
        for col in range(4):
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10)
            if row == 2:
                axes[row, col].set_xlabel("x (km)")

    # Two horizontal colorbars along the bottom, each under the columns it
    # actually describes. A shared bar on the right would sit beside the error
    # column and read as if it applied to it.
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.115,
                        wspace=0.22, hspace=0.16)

    left_first = axes[2, 0].get_position().x0
    left_last = axes[2, 2].get_position().x1
    err_box = axes[2, 3].get_position()

    cax1 = fig.add_axes([left_first, 0.055, left_last - left_first, 0.013])
    cb1 = fig.colorbar(im_rain, cax=cax1, orientation="horizontal",
                       ticks=[0, 1, 5, 15, 30, 60, int(vmax)])
    cb1.set_label("rain rate (mm h$^{-1}$)", color=vs.INK_SECONDARY, fontsize=9)

    cax2 = fig.add_axes([err_box.x0, 0.055, err_box.width, 0.013])
    cb2 = fig.colorbar(im_err, cax=cax2, orientation="horizontal")
    cb2.set_label("reconstruction error (mm h$^{-1}$)",
                  color=vs.INK_SECONDARY, fontsize=9)

    for cb in (cb1, cb2):
        cb.ax.tick_params(colors=vs.INK_MUTED, labelsize=8)
        cb.outline.set_edgecolor(vs.BASELINE)

    fig.suptitle("Rainfall fields rebuilt from link measurements: "
                 "structure, not sensor noise, sets the limit",
                 fontsize=12.5, color=vs.INK_PRIMARY, y=0.965)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_error_budget(models, recon, sweep, path, n_links=90):
    """Where the reconstruction error comes from, and how density changes it."""
    fig = plt.figure(figsize=(14.6, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.34)
    ax_bar = fig.add_subplot(gs[0, 0])
    sweep_axes = [fig.add_subplot(gs[0, i + 1]) for i in range(3)]

    # (a) sampling vs sensor, at the operating density
    width = 0.36
    xs = np.arange(len(models))
    samp = [recon[m.key]["decomp"]["rmse_sampling"] for m in models]
    tot = [recon[m.key]["decomp"]["rmse_total"] for m in models]

    b1 = ax_bar.bar(xs - width / 2, samp, width, color=vs.SERIES["stratiform"],
                    label="network geometry alone")
    b2 = ax_bar.bar(xs + width / 2, tot, width, color=vs.SERIES["convective"],
                    label="geometry + sensor chain")
    head = max(tot) * 1.30
    for bars in (b1, b2):
        for b in bars:
            ax_bar.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + head * 0.015,
                        f"{b.get_height():.1f}", ha="center", va="bottom",
                        fontsize=8, color=vs.INK_PRIMARY)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([m.name.replace(" ", "\n") for m in models],
                           fontsize=8.5)
    ax_bar.set_ylabel("field RMSE (mm h$^{-1}$)")
    ax_bar.set_ylim(0, head)
    ax_bar.set_title(f"Error budget at {n_links} links")
    ax_bar.legend(loc="upper left", fontsize=8)
    ax_bar.grid(axis="x", visible=False)

    # (b-d) density sweep, one facet per model. Six lines on a single axis
    # would exceed the readable series count, so they are faceted instead.
    for ax, m in zip(sweep_axes, models):
        d = sweep[m.key]
        ax.plot(d["n_links"], d["rmse_sampling"], color=vs.SERIES["stratiform"],
                marker="o", ms=5, label="geometry alone")
        ax.plot(d["n_links"], d["rmse_total"], color=vs.SERIES["convective"],
                marker="o", ms=5, label="+ sensor chain")

        share = (d["rmse_total"][-1] - d["rmse_sampling"][-1]) / d["rmse_total"][-1]
        ax.annotate(f"sensor chain is\n{max(share, 0)*100:.0f}% of the error\n"
                    f"at {d['n_links'][-1]} links",
                    xy=(d["n_links"][-1], d["rmse_total"][-1]),
                    xytext=(0.96, 0.60 if share > 0.05 else 0.08),
                    textcoords="axes fraction", ha="right",
                    fontsize=7.8, color=vs.INK_SECONDARY, linespacing=1.4)

        ax.set_title(m.name, color=vs.SERIES[m.key], fontsize=10)
        ax.set_xlabel("links in network")
        ax.set_xscale("log")
        ax.set_xticks(d["n_links"])
        ax.set_xticklabels([str(n) for n in d["n_links"]], fontsize=8)
        ax.minorticks_off()
        # Each regime has its own error scale; a shared axis would flatten the
        # stratiform panel into a line and hide the effect this figure is about.
        lo = min(min(d["rmse_sampling"]), min(d["rmse_total"]))
        hi = max(max(d["rmse_sampling"]), max(d["rmse_total"]))
        pad = (hi - lo) * 0.45 + 0.05
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
        ax.set_ylabel("field RMSE (mm h$^{-1}$)")
        if ax is sweep_axes[0]:
            ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Sampling geometry sets the error - except on smooth fields, "
                 "where a dense network hits the sensor floor",
                 fontsize=12.5, color=vs.INK_PRIMARY, y=1.03)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_moving(models, moving, grid, path):
    """Each regime over an hour, evolving; and how fast motion stops predicting it.

    ``moving`` is ``{key: {tau_min or None: MovingSequence}}`` from
    ``core.simulation.moving_fields``. The maps use the fastest evolution;
    the right column is the correlation between the truth and the field
    moved at the true velocity, for every evolution time.
    """
    taus = list(next(iter(moving.values())).keys())
    shown = taus[-1]
    fig, axes = plt.subplots(len(models), 4, figsize=(13.4, 3.3 * len(models)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    tau_color = dict(zip(taus, [vs.INK_PRIMARY, vs.INK_SECONDARY, vs.INK_MUTED]))
    for row, m in zip(axes, models):
        seqs = moving[m.key]
        seq = seqs[shown]
        vmax = float(seq.frames.max())
        last = len(seq.frames) - 1
        for ax, t in zip(row[:3], (0, last // 2, last)):
            im = _rain_map(ax, seq.frames[t], grid, vmax)
            ax.set_title(f"{m.name}, +{seq.times_min[t]:.0f} min", color=vs.SERIES[m.key],
                         fontsize=10)
        u, v = seq.velocity_kmh
        vs.annotate_corner(row[0], f"moves {np.hypot(u, v):.0f} km/h\n"
                                   f"evolves, tau {shown} min")
        fig.colorbar(im, ax=row[2], fraction=0.046, pad=0.03).set_label("mm/h")

        ax = row[3]
        for tau, s in seqs.items():
            h = np.arange(1, len(s.frames))
            ax.plot(h * s.dt_min, [s.predictability(k) for k in h], "o-", ms=4, lw=2,
                    color=tau_color[tau],
                    label="frozen" if tau is None else f"evolving, tau {tau} min")
        ax.set(ylim=(0, 1.05), xlabel="horizon (min)", ylabel="corr. with moved field")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Moving rain: the true motion is a perfect forecast only while the "
                 "pattern holds", fontsize=12.5, color=vs.INK_PRIMARY, x=0.03, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
