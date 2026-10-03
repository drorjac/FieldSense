"""Figures: the three networks, event totals by sensor, and the pooled scores."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.opensense.networks import NETWORKS

TEXT = "#52514e"
# one hue per kind of sensor; the link methods share one family
COLORS = {"radar": "#2a78d6", "pws": "#eb6834", "city": "#eda100", "gauges": "#eb6834",
          "dynamic": "#1baf7a", "constant": "#4a3aa7", "pycomlink": "#e87ba4", "nearby": "#008300",
          "rnn": "#e34948"}
NAMES = {"openmrg": "Gothenburg (OpenMRG)", "openrainer": "Emilia-Romagna (OpenRainER)",
         "openmesh": "New York City (OpenMesh)"}


def color(name: str) -> str:
    return COLORS.get(name.split(" ")[0], "#898781")


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return fig


def _links(ax, table, color_="#0b0b0b", lw=0.7, alpha=0.9):
    for r in table.itertuples():
        ax.plot([r.site_0_lon, r.site_1_lon], [r.site_0_lat, r.site_1_lat], color=color_, lw=lw,
                alpha=alpha, solid_capstyle="round")


def networks_figure(points: dict, path=None):
    """The three networks: links, point gauges and the radar grid's extent."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, key in zip(axes, ("openmrg", "openrainer", "openmesh")):
        net = NETWORKS[key]
        table = net.links_table()
        d = net.domain
        ax.add_patch(plt.Rectangle((d.lon_min, d.lat_min), d.lon_max - d.lon_min, d.lat_max - d.lat_min,
                                   fill=False, ls="--", lw=1, ec=COLORS["radar"], label="radar grid"))
        _links(ax, table)
        for name, p in points.get(key, {}).items():
            ax.scatter(p.lon, p.lat, s=12, color=color(name), zorder=5, label=f"{name} ({p.sizes['station']})")
        ax.plot([], [], color="#0b0b0b", label=f"links ({len(table)} sublinks)")
        ax.set(title=NAMES[key], xlabel="lon", ylabel="lat")
        ax.set_aspect(1 / np.cos(np.radians((d.lat_min + d.lat_max) / 2)))
        ax.legend(fontsize=7, loc="lower left")
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    return _save(fig, path)


def event_totals_figure(res, path=None):
    """Event totals of every map on one colour scale, links on top."""
    names = list(res.maps)
    tot = {n: res.maps[n].sum("time", min_count=1) for n in names}
    vmax = float(np.nanpercentile(np.concatenate([t.values.ravel() for t in tot.values()]), 99))
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6), squeeze=False,
                             layout="constrained")
    table = pd.DataFrame({c: res.links[c].values for c in ["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"]})
    lat = float(res.maps["radar"].lat.mean())
    for ax, n in zip(axes[0], names):
        m = ax.pcolormesh(tot[n].lon, tot[n].lat, tot[n], cmap="Blues", vmin=0, vmax=vmax, shading="nearest")
        _links(ax, table, lw=0.5, alpha=0.6)
        if n in res.points:
            p = res.points[n]
            ax.scatter(p.lon, p.lat, s=6, color="#eb6834", zorder=5)
        ax.set_title(n, fontsize=9)
        ax.set_aspect(1 / np.cos(np.radians(lat)))
        ax.tick_params(labelsize=6)
    fig.colorbar(m, ax=axes[0].tolist(), shrink=0.8, label="event total (mm)")
    fig.suptitle(f"{NAMES[res.network]}: {res.start:%Y-%m-%d %H:%M} - {res.end:%m-%d %H:%M} UTC",
                 fontsize=10, x=0.01, ha="left")
    return _save(fig, path)


def pooled_figure(pooled_maps: pd.DataFrame, pooled_points: pd.DataFrame, path=None):
    """Per network: every map against the radar (cells near links), and at the check gauges."""
    nets = [n for n in ("openmrg", "openrainer", "openmesh") if n in set(pooled_maps.network)]
    fig, axes = plt.subplots(2, len(nets), figsize=(5.2 * len(nets), 7.5), squeeze=False)
    for j, n in enumerate(nets):
        pm = pooled_maps[(pooled_maps.network == n) & (pooled_maps.reference == "radar")
                         & (pooled_maps.cells != "all")].sort_values("nrmse")
        ax = axes[0, j]
        ax.barh(pm.estimate, pm.nrmse, color=[color(e) for e in pm.estimate])
        for i, (v, b) in enumerate(zip(pm.nrmse, pm.rel_bias)):
            ax.text(v, i, f" {v:.2f} ({b:+.0%})", va="center", fontsize=7, color=TEXT)
        ax.invert_yaxis()
        ax.set(title=f"{NAMES[n]}\nmaps vs radar, cells <= 2 km of a link", xlabel="hourly NRMSE (bias)")
        ax.set_xlim(0, pm.nrmse.max() * 1.45)
        pp = pooled_points[(pooled_points.network == n) & (pooled_points.subset == "all gauges")].sort_values("nrmse")
        ax = axes[1, j]
        ax.barh(pp.estimate, pp.nrmse, color=[color(e) for e in pp.estimate])
        for i, (v, b) in enumerate(zip(pp.nrmse, pp.rel_bias)):
            ax.text(v, i, f" {v:.2f} ({b:+.0%})", va="center", fontsize=7, color=TEXT)
        ax.invert_yaxis()
        check = pp.check.iloc[0] if len(pp) else ""
        ax.set(title=f"at the {check} gauges (each left out of its own map)", xlabel="hourly NRMSE (bias)")
        ax.set_xlim(0, pp.nrmse.max() * 1.45 if len(pp) else 1)
        for a in axes[:, j]:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(labelsize=8)
    fig.tight_layout()
    return _save(fig, path)
