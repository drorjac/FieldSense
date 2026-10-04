"""Figures: rain maps, event overviews and method comparisons (matplotlib).

Colour rules: rain amounts use one sequential hue (light -> dark blue, zero = white);
differences use a diverging pair with a neutral midpoint; methods get categorical
colours in a fixed order that follows the method, never its rank.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import xarray as xr

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_ORDER = ["impl1_dynamic", "impl1_dynamic_maxfill", "impl2_classical_dynamic",
                "impl2_classical_constant", "impl2_pycomlink", "impl2_pycomlink_linear",
                "impl2_nearby", "impl2_gru"]
PTYPE_COLORS = {"rain": "#2a78d6", "snow": "#4a3aa7", "mix": "#eb6834"}
RAIN_CMAP = "Blues"
DIFF_CMAP = "RdBu"
TEXT = "#52514e"


def method_color(name: str) -> str:
    i = METHOD_ORDER.index(name) if name in METHOD_ORDER else len(METHOD_ORDER) - 1
    return SERIES[i % len(SERIES)]


def _aspect(ax, lat):
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(lat)))))


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=7)


def draw_links(ax, links, color="#0b0b0b", lw=1.2, alpha=0.9):
    """Draw link paths from a link set or a DataFrame with site coordinates."""
    if isinstance(links, xr.Dataset):
        rows = zip(links.site_0_lon.values, links.site_0_lat.values,
                   links.site_1_lon.values, links.site_1_lat.values)
    else:
        rows = zip(links.site_0_lon, links.site_0_lat, links.site_1_lon, links.site_1_lat)
    for lo0, la0, lo1, la1 in rows:
        ax.plot([lo0, lo1], [la0, la1], color=color, lw=lw, alpha=alpha, solid_capstyle="round")


def plot_field(field: xr.DataArray, ax=None, title: str = "", vmax: float | None = None,
               cmap: str = RAIN_CMAP, label: str = "mm", links=None, colorbar: bool = True,
               vmin: float = 0.0, basemap: bool = False):
    """Map a (lat, lon) field with optional link overlay and OSM basemap (contextily)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    vmax = vmax if vmax is not None else float(np.nanmax(field.values)) if np.isfinite(field).any() else 1.0
    m = ax.pcolormesh(field.lon, field.lat, field.values, cmap=cmap, vmin=vmin, vmax=max(vmax, 1e-6),
                      shading="nearest", alpha=0.85 if basemap else 1.0)
    if basemap:
        try:
            import contextily as cx
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.PositronNoLabels,
                           attribution_size=5)
        except Exception:  # offline or contextily missing: plain map
            pass
    if links is not None:
        draw_links(ax, links)
    _aspect(ax, field.lat)
    _style(ax)
    ax.set_title(title, fontsize=9, color="#0b0b0b", loc="left")
    if colorbar:
        cb = plt.colorbar(m, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(label, fontsize=8, color=TEXT)
        cb.ax.tick_params(labelsize=7, colors=TEXT)
    return m


def event_comparison_figure(res, path: str | Path | None = None, near_only: bool = False):
    """Event totals: radar + every method on one colour scale, plus NRMSE bars."""
    tot = res.event_totals()
    names = [m for m in METHOD_ORDER if m in tot] + [m for m in tot if m not in METHOD_ORDER and m != "radar"]
    panels = ["radar"] + names
    vmax = float(np.nanpercentile(np.concatenate([tot[p].values.ravel() for p in panels]), 99))
    n = len(panels) + 1
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.4 * rows), squeeze=False,
                             layout="constrained")
    axes = axes.ravel()
    mappable = None
    for ax, p in zip(axes, panels):
        f = tot[p]
        if near_only and res.distance_to_link_km is not None:
            f = f.where(res.distance_to_link_km <= res.config.get("near_km", 2.0))
        mappable = plot_field(f, ax, title=("MRMS radar" if p == "radar" else p), vmax=vmax,
                              links=res.links, colorbar=False)
    cb = fig.colorbar(mappable, ax=axes[:len(panels)].tolist(), shrink=0.6, pad=0.01)
    cb.set_label("event total (mm)", fontsize=8, color=TEXT)
    cb.ax.tick_params(labelsize=7, colors=TEXT)
    # the bar chart spans the rest of the last row: its long labels would otherwise
    # widen one grid column and push that column's maps off centre
    row, col = divmod(len(panels), cols)
    for a in axes[len(panels):]:
        a.remove()
    ax = fig.add_subplot(axes[0].get_subplotspec().get_gridspec()[row, col:])
    if not res.map_scores.empty:
        sc = res.map_scores.reindex(names)["nrmse"]
        ax.barh(range(len(sc)), sc.values, color=[method_color(m) for m in sc.index], height=0.6)
        # names at the bar ends, not as tick labels (see above)
        ax.set_yticks([])
        for i, (m, v) in enumerate(sc.items()):
            ax.text(v if np.isfinite(v) else 0, i, f" {v:.2f}  {m}", va="center", fontsize=7,
                    color=TEXT)
        ax.set_xlim(0, np.nanmax(sc.values) * 1.8 if np.isfinite(sc.values).any() else 1)
        ax.invert_yaxis()
        ax.set_xlabel("hourly NRMSE vs MRMS (lower is better)", fontsize=8, color=TEXT)
        _style(ax)
    fig.suptitle(f"Event accumulation {res.start:%Y-%m-%d %H:%M} - {res.end:%Y-%m-%d %H:%M} UTC (mm)",
                 fontsize=10, x=0.01, ha="left")
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def event_overview_figure(event, hourly: xr.DataArray, path: str | Path | None = None, links=None,
                          asos_hourly: pd.DataFrame | None = None):
    """Catalog figure: event-total radar map + domain-mean hyetograph."""
    t0, t1 = pd.Timestamp(event.start), pd.Timestamp(event.end)
    sub = hourly.sel(time=slice(t0 + pd.Timedelta("1h"), t1))
    total = sub.sum("time", min_count=1)
    fig = plt.figure(figsize=(10, 4.2))
    ax1 = fig.add_axes([0.05, 0.1, 0.38, 0.8])
    plot_field(total, ax1, title="MRMS total, liquid equivalent (mm)", links=links, label="mm")
    ax2 = fig.add_axes([0.52, 0.18, 0.45, 0.62])
    s = sub.mean(("lat", "lon")).to_series()
    color = PTYPE_COLORS.get(event.ptype, SERIES[0])
    ax2.bar(s.index - pd.Timedelta("30min"), s.values, width=1 / 24 * 0.85, color=color, align="center")
    if asos_hourly is not None and not asos_hourly.empty:
        a = asos_hourly.loc[t0:t1].mean(axis=1)
        ax2.plot(a.index - pd.Timedelta("30min"), a.values, color="#0b0b0b", lw=1.5,
                 marker="o", ms=3, label="ASOS mean (mm/h)")
        ax2.legend(fontsize=7, frameon=False)
    ax2.set_ylabel("domain-mean mm/h", fontsize=8, color=TEXT)
    _style(ax2)
    fig.autofmt_xdate()
    sf = event.snow_fraction
    text = (f"{event.ptype.upper()}  |  total {event.total_mm:.1f} mm  |  peak {event.peak_hourly_mm:.1f} mm/h"
            f"  |  snow fraction {sf:.2f}" if np.isfinite(sf) else f"{event.ptype.upper()}")
    fig.text(0.52, 0.9, f"{event.event_id}  {t0:%Y-%m-%d %H:%M} - {t1:%Y-%m-%d %H:%M} UTC",
             fontsize=10, color="#0b0b0b")
    fig.text(0.52, 0.85, text, fontsize=8, color=TEXT)
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def link_series_figure(res, link: str, path: str | Path | None = None):
    """RSL and every method's rain rate for one link, with radar along its path."""
    from core.radar.mrms import path_average
    from core.opensense.openmesh import links_frame

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    rsl = res.links["rsl"].sel(link=link).sel(time=slice(res.start, res.end))
    a1.plot(rsl.time, rsl, color="#0b0b0b", lw=0.8)
    a1.set_ylabel("RSL (dBm)", fontsize=8, color=TEXT)
    for m, ds in res.link_rain.items():
        r = ds["rain"].sel(link=link)
        a2.plot(r.time, r, lw=1.0, color=method_color(m), label=m)
    if res.radar_hourly is not None:
        rp = path_average(res.radar_hourly, links_frame(res.links).loc[[link]]).isel(link=0)
        a2.step(rp.time, rp, where="pre", color="#0b0b0b", lw=1.5, label="MRMS along path (mm/h)")
    a2.set_ylabel("rain rate (mm/h)", fontsize=8, color=TEXT)
    a2.legend(fontsize=7, frameon=False, ncol=3)
    for a in (a1, a2):
        _style(a)
    fig.suptitle(link, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


# ------------------------------------------------------------ study figures

LINKSET_COLORS = {"shared8": SERIES[0], "selected": SERIES[2], "qc": SERIES[1]}


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def study_scores_figure(study, metric: str = "nrmse", path=None):
    """Pooled ``metric`` per method, one panel per precipitation type, one bar per link set."""
    types = [t for t in ["rain", "mix", "snow"] if t in set(study.pooled.ptype)]
    methods = [m for m in METHOD_ORDER if m in set(study.pooled.method)]
    sets = list(study.config.link_sets)
    fig, axes = plt.subplots(1, len(types), figsize=(4.2 * len(types), 0.45 * len(methods) + 1.6),
                             sharey=True, squeeze=False, layout="constrained")
    h = 0.8 / len(sets)
    for ax, t in zip(axes[0], types):
        p = study.pooled[(study.pooled.ptype == t) & (~study.pooled.near)]
        for j, ls in enumerate(sets):
            v = p[p.link_set == ls].set_index("method").reindex(methods)[metric]
            y = np.arange(len(methods)) + (j - (len(sets) - 1) / 2) * h
            ax.barh(y, v.values, height=h * 0.9, color=LINKSET_COLORS.get(ls, SERIES[2 + j]),
                    label=f"{ls} links")
            for yy, vv in zip(y, v.values):
                if np.isfinite(vv):
                    ax.text(vv, yy, f" {vv:.2f}", va="center", fontsize=6.5, color=TEXT)
        n_ev = int(p.events.max()) if len(p) else 0
        ax.set_title(f"{t} ({n_ev} events)", loc="left", fontsize=9)
        ax.set_yticks(range(len(methods)), methods, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"pooled hourly {metric.upper()} vs MRMS", fontsize=8, color=TEXT)
        _style(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=len(sets), loc="outside upper right")
    return _save(fig, path)


def study_event_bias_figure(study, path=None, link_set: str = "shared8"):
    """Relative bias of each event (dots, coloured by type) per method."""
    es = study.event_scores[study.event_scores.link_set == link_set]
    methods = [m for m in METHOD_ORDER if m in set(es.method)]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(methods) + 1.4), layout="constrained")
    for t, color in PTYPE_COLORS.items():
        g = es[es.ptype == t]
        y = g.method.map({m: i for i, m in enumerate(methods)})
        jitter = {"rain": -0.18, "mix": 0.0, "snow": 0.18}[t]
        ax.scatter(g.rel_bias.clip(-1, 3), y + jitter, s=34, color=color, edgecolor="white",
                   linewidth=0.8, label=t, zorder=3)
    ax.axvline(0, color="#8a8984", lw=1, zorder=1)
    ax.set_yticks(range(len(methods)), methods, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"event relative bias vs MRMS ({link_set} links; clipped at -100 % / +300 %)",
                  fontsize=8, color=TEXT)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.legend(*ax.get_legend_handles_labels(), fontsize=8, frameon=False, ncol=3,
               loc="outside upper right")
    _style(ax)
    return _save(fig, path)


def study_scatter_figure(study, path=None, ptype: str = "rain", link_set: str = "shared8"):
    """Hourly CML vs MRMS for cells near links, one hexbin panel per method."""
    p = study.pairs[(study.pairs.ptype == ptype) & (study.pairs.link_set == link_set) & study.pairs.near]
    methods = [m for m in METHOD_ORDER if m in set(p.method)]
    fig, axes = plt.subplots(1, len(methods), figsize=(2.9 * len(methods), 3.2), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    lim = float(np.nanpercentile(p[["est", "ref"]].values, 99.5)) if len(p) else 1.0
    for ax, m in zip(axes[0], methods):
        g = p[p.method == m]
        ax.hexbin(g.ref, g.est, gridsize=28, extent=(0, lim, 0, lim), bins="log", cmap=RAIN_CMAP,
                  mincnt=1, linewidths=0)
        ax.plot([0, lim], [0, lim], color="#0b0b0b", lw=0.8)
        ax.set_title(m, fontsize=8, loc="left")
        ax.set_xlabel("MRMS (mm/h)", fontsize=8, color=TEXT)
        ax.set_aspect("equal")
        _style(ax)
    axes[0][0].set_ylabel(f"CML map (mm/h), {ptype}, cells <= {study.config.near_km:g} km",
                          fontsize=8, color=TEXT)
    return _save(fig, path)


def study_link_bias_figure(study, path=None, ptype: str = "rain", link_set: str = "shared8"):
    """Median relative bias per link (rows) and method (columns) against radar along the path."""
    ls_ = study.link_scores
    if ls_.empty:
        fig, ax = plt.subplots()
        ax.axis("off")
        return _save(fig, path)
    d = ls_[(ls_.ptype == ptype) & (ls_.link_set == link_set)]
    tab = d.groupby(["link", "method"]).rel_bias.median().unstack("method")
    tab = tab[[m for m in METHOD_ORDER if m in tab.columns]]
    fig, ax = plt.subplots(figsize=(1.3 * tab.shape[1] + 2.2, 0.38 * tab.shape[0] + 1.6),
                           layout="constrained")
    m = ax.imshow(tab.values, cmap=DIFF_CMAP, vmin=-1, vmax=1, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.0%}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "#0b0b0b")
    ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(tab.shape[0]), tab.index, fontsize=7)
    cb = fig.colorbar(m, ax=ax, shrink=0.8)
    cb.set_label("median relative bias vs MRMS along the path", fontsize=7, color=TEXT)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"per-link bias, {ptype} events, {link_set} links", fontsize=9, loc="left")
    return _save(fig, path)


# ----------------------------------------------------------- sensor figures

SENSOR_COLORS = {"MRMS": "#0b0b0b", "PWS": SERIES[2], "ASOS": "#52514e"}


def _clip_to(ax, field):
    """Keep the axes on the map extent (points outside it must not stretch the panel)."""
    dlon = float(abs(np.diff(field.lon.values[:2])[0])) / 2 if field.lon.size > 1 else 0.005
    dlat = float(abs(np.diff(field.lat.values[:2])[0])) / 2 if field.lat.size > 1 else 0.005
    ax.set_xlim(float(field.lon.min()) - dlon, float(field.lon.max()) + dlon)
    ax.set_ylim(float(field.lat.min()) - dlat, float(field.lat.max()) + dlat)


def _sensor_color(name):
    if name in SENSOR_COLORS:
        return SENSOR_COLORS[name]
    return method_color(name.replace("CML ", ""))


def sensor_snapshot_figure(comp, n_times: int = 6, path=None, diff: bool = True):
    """implementation_2-style snapshot grid: rows = sensors, columns = hours around the peak.

    With ``diff`` a last row shows the first CML map minus MRMS (implementation_1 5-row
    layout). Links are drawn on CML rows, stations on the PWS row.
    """
    peak = comp.peak_time()
    times = pd.DatetimeIndex(comp.maps["MRMS"].time.values)
    i0 = max(0, min(times.get_indexer([peak], method="nearest")[0] - n_times // 2, len(times) - n_times))
    sel = times[i0:i0 + n_times]
    rows = list(comp.maps)
    cml = [r for r in rows if r.startswith("CML")]
    n_rows = len(rows) + (1 if diff and cml else 0)
    vmax = max(1e-3, float(np.nanpercentile(np.concatenate(
        [comp.maps[r].sel(time=sel).values.ravel() for r in rows]), 99)))
    fig, axes = plt.subplots(n_rows, len(sel), figsize=(1.9 * len(sel) + 1.2, 2.2 * n_rows),
                             squeeze=False, layout="constrained")
    m_rain = m_diff = None
    for i, r in enumerate(rows):
        for j, t in enumerate(sel):
            ax = axes[i][j]
            f = comp.maps[r].sel(time=t)
            m_rain = ax.pcolormesh(f.lon, f.lat, f.values, cmap=RAIN_CMAP, vmin=0, vmax=vmax, shading="nearest")
            if r.startswith("CML"):
                draw_links(ax, comp.links, lw=0.8)
            if r == "PWS":
                ax.scatter(comp.pws_points.lon, comp.pws_points.lat, s=6, marker="^", color="#0b0b0b")
            _clip_to(ax, f)
            ax.set_xticks([]), ax.set_yticks([])
            _aspect(ax, f.lat)
            if i == 0:
                ax.set_title(f"{pd.Timestamp(t):%m-%d %H:%M}", fontsize=7)
            if j == 0:
                ax.set_ylabel(r.replace("CML ", "CML\n"), fontsize=7)
    if diff and cml:
        for j, t in enumerate(sel):
            ax = axes[-1][j]
            d = comp.maps[cml[0]].sel(time=t) - comp.maps["MRMS"].sel(time=t)
            m_diff = ax.pcolormesh(d.lon, d.lat, d.values, cmap=DIFF_CMAP + "_r", vmin=-vmax / 2,
                                   vmax=vmax / 2, shading="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            _aspect(ax, d.lat)
            if j == 0:
                ax.set_ylabel(f"{cml[0].replace('CML ', '')}\nminus MRMS", fontsize=7)
    fig.colorbar(m_rain, ax=axes[:len(rows)].ravel().tolist(), shrink=0.6, label="hourly rain (mm)")
    if m_diff is not None:
        fig.colorbar(m_diff, ax=axes[-1].tolist(), shrink=0.9, label="difference (mm)")
    fig.suptitle(f"Hourly maps by sensor around the peak ({comp.start:%Y-%m-%d})", fontsize=10,
                 x=0.01, ha="left")
    return _save(fig, path)


def sensor_totals_figure(comp, path=None):
    """Event totals per sensor on one scale, ASOS totals as annotated dots."""
    tot = comp.totals()
    names = list(tot)
    vmax = float(np.nanpercentile(np.concatenate([tot[n].values.ravel() for n in names]), 99))
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.6), squeeze=False,
                             layout="constrained")
    m = None
    a = comp.asos_points
    for ax, n in zip(axes[0], names):
        m = plot_field(tot[n], ax, title=n, vmax=vmax, colorbar=False,
                       links=comp.links if n.startswith("CML") else None)
        if a.sizes.get("station", 0):
            at = a.sum("time", min_count=1)
            ax.scatter(a.lon, a.lat, c=at.values, cmap=RAIN_CMAP, vmin=0, vmax=vmax, s=60,
                       edgecolor="#0b0b0b", linewidth=1.0, zorder=5)
            for st, lo, la, v in zip(a.station.values, a.lon.values, a.lat.values, at.values):
                ax.annotate(f"{st} {v:.0f}", (lo, la), xytext=(3, 3), textcoords="offset points",
                            fontsize=6, color="#0b0b0b")
        if n == "PWS":
            ax.scatter(comp.pws_points.lon, comp.pws_points.lat, s=8, marker="^", color="#0b0b0b")
        _clip_to(ax, tot[n])
    fig.colorbar(m, ax=axes[0].tolist(), shrink=0.8, label="event total (mm); circles = ASOS")
    return _save(fig, path)


def sensor_series_figure(comp, near_km: float = 2.0, path=None):
    """Hourly area-mean rain near the links for every sensor, with the ASOS mean."""
    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
    mask = None
    if comp.distance_to_link_km is not None:
        mask = comp.distance_to_link_km <= near_km
    for n, mp in comp.maps.items():
        s = (mp.where(mask) if mask is not None else mp).mean(("lat", "lon")).to_series()
        ax.step(s.index, s.values, where="pre", lw=2.2 if n == "MRMS" else 1.4,
                color=_sensor_color(n), label=n)
    a = comp.asos_points
    if a.sizes.get("station", 0):
        s = a.mean("station").to_series()
        ax.plot(s.index - pd.Timedelta("30min"), s.values, "o", ms=5, mfc="none", mew=1.4,
                color=SENSOR_COLORS["ASOS"],
                label="ASOS mean (" + ", ".join(map(str, a.station.values)) + ")")
    ax.set_ylabel(f"hourly rain (mm), cells <= {near_km:g} km of links", fontsize=8, color=TEXT)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=7, frameon=False, ncol=4, loc="outside upper left")
    _style(ax)
    return _save(fig, path)


# -------------------------------------------------------- validation figures

VALIDATION_COLORS = {"official": "#0b0b0b", "MRMS Pass 2": SERIES[0], "PWS nearest (<3 km)": SERIES[2],
                     "PWS IDW": SERIES[3]}


def validation_cumulative_figure(v, path=None, min_coverage: float = 0.5):
    """Cumulative precipitation at each official gauge: ASOS vs radar vs PWS.

    Per station, curves accumulate over the hours where the official gauge and every drawn
    source have a value (so a gap never looks like a dry spell); sources covering less than
    ``min_coverage`` of the official hours at that station are left out (no PWS within reach).
    """
    off = v.series["official"]
    names = ["MRMS Pass 2", "PWS nearest (<3 km)", "PWS IDW"]
    stations = list(off.columns)
    fig, axes = plt.subplots(1, len(stations), figsize=(4.2 * len(stations), 3.6), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, st in zip(axes[0], stations):
        o = off[st]
        ok = o.notna()
        drawn = [n for n in names if v.series[n][st].reindex(off.index)[ok].notna().mean() >= min_coverage]
        for n in drawn:
            ok &= v.series[n][st].reindex(off.index).notna()
        ax.plot(o.index[ok], o[ok].cumsum(), lw=2.4, color=VALIDATION_COLORS["official"], label="ASOS (official)")
        for n in drawn:
            e = v.series[n][st].reindex(off.index)[ok]
            ax.plot(e.index, e.cumsum(), lw=1.4, color=VALIDATION_COLORS[n], label=n)
        missing = [n for n in names if n not in drawn]
        ax.set_title(st + (f"  (no {', '.join(m.split(' (')[0] for m in missing)} in reach)" if missing else ""),
                     loc="left", fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        _style(ax)
    axes[0][0].set_ylabel("cumulative precipitation (mm), common hours", fontsize=8, color=TEXT)
    handles = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), fontsize=8, frameon=False, ncol=4, loc="outside upper left")
    return _save(fig, path)


def validation_scatter_figure(v, path=None, hours: str = "rain"):
    """Hourly source vs official gauge (all stations), rain hours, log-count hexbins."""
    off, pt = v.series["official"], v.series["ptype"]
    names = ["MRMS Pass 2", "PWS nearest (<3 km)", "PWS IDW"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.5), sharex=True, sharey=True,
                             layout="constrained")
    lim = 15.0
    for ax, name in zip(axes, names):
        xs, ys = [], []
        for st in off.columns:
            m = pt[st].reindex(off.index).isin(["rain"]) if hours == "rain" else slice(None)
            o = off[st][m]
            e = v.series[name][st].reindex(o.index)
            ok = o.notna() & e.notna()
            xs.append(o[ok].values)
            ys.append(e[ok].values)
        x, y = np.concatenate(xs), np.concatenate(ys)
        ax.hexbin(x, y, gridsize=30, extent=(0, lim, 0, lim), bins="log", cmap=RAIN_CMAP, mincnt=1, linewidths=0)
        ax.plot([0, lim], [0, lim], color="#0b0b0b", lw=0.8)
        ax.set_title(name, fontsize=8, loc="left")
        ax.set_xlabel("ASOS official (mm/h)", fontsize=8, color=TEXT)
        ax.set_aspect("equal")
        _style(ax)
    axes[0].set_ylabel(f"source (mm/h), {hours} hours", fontsize=8, color=TEXT)
    return _save(fig, path)


def validation_accumulation_figure(v, path=None):
    """Domain-mean daily total: sum of 24 hourly Pass-2 fields vs the 24-h Pass-2 product."""
    a = v.accumulation_check
    fig, ax = plt.subplots(figsize=(4.2, 4), layout="constrained")
    if not a.empty:
        ax.scatter(a.mean_24h_product, a.mean_sum_of_hourly, s=30, color=SERIES[0], edgecolor="white")
        m = float(max(a.mean_24h_product.max(), a.mean_sum_of_hourly.max())) * 1.05
        ax.plot([0, m], [0, m], color="#0b0b0b", lw=0.8)
    ax.set_xlabel("MRMS 24-h Pass-2 product (mm, domain mean)", fontsize=8, color=TEXT)
    ax.set_ylabel("sum of 24 hourly Pass-2 fields (mm)", fontsize=8, color=TEXT)
    ax.set_title("A4: accumulation consistency", loc="left", fontsize=9)
    _style(ax)
    return _save(fig, path)


# -------------------------------------------------------- all-events figures


def _methods_in(table):
    return [m for m in METHOD_ORDER if f"{m}_nrmse" in table.columns]


def all_events_heatmap(table, path=None):
    """NRMSE per event (rows, chronological) and method (columns); event type in the label."""
    ms = _methods_in(table)
    v = table[[f"{m}_nrmse" for m in ms]].values
    fig, ax = plt.subplots(figsize=(1.3 * len(ms) + 3.5, 0.24 * len(table) + 1.6), layout="constrained")
    im = ax.imshow(np.clip(v, 0, 3), cmap="Blues", vmin=0, vmax=3, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v[i, j] > 1.6 else "#0b0b0b")
    labels = [f"{r.label}{' *' if r.held_out else (' c' if getattr(r, 'sample', '') == 'selection calibration' else '')}"
              for r in table.itertuples()]
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xticks(range(len(ms)), ms, rotation=30, ha="right", fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.5)
    cb.set_label("hourly NRMSE vs MRMS (clipped at 3)", fontsize=7, color=TEXT)
    ax.set_title("NRMSE per event (* catalog event, c selection-calibration event)", loc="left", fontsize=9)
    return _save(fig, path)


def all_events_totals(table, path=None):
    """Event totals near the links: PWS, ASOS and the best CML methods against MRMS."""
    ms = _methods_in(table)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    ax = axes[0]
    lim = float(np.nanmax(table[["MRMS_mm", "PWS_mm", "ASOS_mm"]].values)) * 1.05
    for col, color, lab in (("PWS_mm", SERIES[2], "PWS map"), ("ASOS_mm", "#52514e", "ASOS official")):
        if col in table:
            ax.scatter(table.MRMS_mm, table[col], s=26, color=color, edgecolor="white", label=lab)
    ax.plot([0, lim], [0, lim], color="#0b0b0b", lw=0.8)
    ax.set_xlabel("MRMS event total near links (mm)", fontsize=8, color=TEXT)
    ax.set_ylabel("gauge event total (mm)", fontsize=8, color=TEXT)
    ax.set_title("references vs radar", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    for m in ms:
        b = table[f"{m}_bias"]
        ax.scatter(table.MRMS_mm, b.clip(-1, 3), s=20, color=method_color(m), edgecolor="white", label=m)
    ax.axhline(0, color="#8a8984", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("MRMS event total near links (mm, log)", fontsize=8, color=TEXT)
    ax.set_ylabel("CML relative bias vs MRMS (clipped)", fontsize=8, color=TEXT)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("CML bias by event size", loc="left", fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    _style(ax)
    return _save(fig, path)


def all_events_drivers(table, path=None):
    """Relative bias of each method against peak intensity, duration, temperature and link outages."""
    ms = _methods_in(table)
    props = [("peak_hourly_mm", "peak hourly rain (mm/h)"), ("duration_h", "duration (h)"),
             ("min_temp_c", "min ASOS temperature (C)"), ("link_outage", "missing link minutes")]
    props = [p for p in props if p[0] in table]
    fig, axes = plt.subplots(1, len(props), figsize=(3.4 * len(props), 3.4), sharey=True, layout="constrained")
    for ax, (col, lab) in zip(np.atleast_1d(axes), props):
        for m in ms:
            ax.scatter(table[col], table[f"{m}_bias"].clip(-1, 3), s=14, color=method_color(m),
                       edgecolor="white", linewidth=0.5, label=m)
        ax.axhline(0, color="#8a8984", lw=1)
        ax.set_xlabel(lab, fontsize=8, color=TEXT)
        if col == "link_outage":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("relative bias vs MRMS (clipped)", fontsize=8, color=TEXT)
    first.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.legend(*first.get_legend_handles_labels(), fontsize=7, frameon=False, ncol=len(ms), loc="outside upper left")
    return _save(fig, path)
