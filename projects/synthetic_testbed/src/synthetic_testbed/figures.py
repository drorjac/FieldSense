"""Figures of the synthetic testbed (static PNGs in ``results/figures``).

``all_from_tables`` redraws every figure that is made from the study tables; the gallery and
the example maps need their fields and are drawn by ``run.py gallery`` / the notebooks.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import viz_style as vs  # noqa: E402

from .settings import FIGURES, TABLES  # noqa: E402

# categorical slots in the palette's fixed order (blue, orange, aqua, yellow, magenta, green, violet)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
FAMILY_COLOR = {"radar only": SLOTS[0], "no radar": SLOTS[1], "radar adjusted": SLOTS[2]}

TITLES = {"convective_cells": "Gaussian cells", "clustered_storms": "Clustered HyCell storms",
          "squall_line": "Squall line (cells)", "stratiform_matern": "Stratiform (Matern)",
          "banded_anisotropic": "Anisotropic bands", "scale_free": "Scale-free GRF",
          "rainfarm": "RainFARM", "cascade": "Beta-lognormal cascade",
          "multifractal": "Universal multifractal", "frontal": "Frontal band",
          "cloud_model": "Warm-rain cloud model"}


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    return FIGURES / name


def _map(ax, f, extent, vmax, title=None):
    im = ax.imshow(f, origin="lower", extent=extent, cmap=vs.CMAP_RAIN, norm=vs.rain_norm(vmax),
                   interpolation="nearest")
    vs.style_map_axes(ax, extent)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9.5)
    return im


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
def fig_gallery(fields: dict, stats: pd.DataFrame, extent=(0, 64, 0, 64), name="fig01_generators.png"):
    vs.use_style()
    keys = list(fields)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow), layout="constrained")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        im = _map(ax, fields[k], extent, 40, TITLES.get(k, k))
        s = stats.set_index("model").loc[k]
        vs.annotate_corner(ax, f"wet {s.wet_fraction:.0%}\nmax {s.max_mm_h:.0f} mm/h\n"
                               f"L {s.decorrelation_km:.1f} km", "lower right")
    fig.colorbar(im, ax=axes, shrink=0.6).set_label("rain rate (mm/h)")
    fig.suptitle("Eleven 2-D rain generators on a 64 km domain (0.5 km pixels)", x=0.02, ha="left",
                 fontsize=12, fontweight="semibold")
    return _save(fig, name)


def fig_evolution(seqs: dict, pred: pd.DataFrame, extent=(0, 64, 0, 64), show=(0, 6, 12),
                  name="fig02_evolution.png"):
    vs.use_style()
    names = list(seqs)
    fig = plt.figure(figsize=(3.0 * len(show) + 4.6, 2.6 * len(names)))
    gs = fig.add_gridspec(len(names), len(show) + 1, width_ratios=[1] * len(show) + [1.6])
    for i, k in enumerate(names):
        s = seqs[k]
        for j, t in enumerate(show):
            ax = fig.add_subplot(gs[i, j])
            _map(ax, s.frames[t], extent, 40, f"t = {t * s.dt_min:.0f} min" if i == 0 else None)
            if j == 0:
                ax.set_ylabel(k, fontsize=10, color=vs.INK_PRIMARY)
    ax = fig.add_subplot(gs[:, -1])
    for c, k in zip(SLOTS, names):
        d = pred[pred.evolution == k]
        ax.plot(d.lead_min, d.correlation, color=c, label=k)
    ax.set_xlabel("lead time (min)")
    ax.set_ylabel("correlation with the field carried by the true flow")
    ax.set_title("What the true motion alone can predict", fontsize=10)
    ax.set_ylim(-0.05, 1.02)
    ax.legend(loc="lower left")
    fig.suptitle("Clustered storms in a rotating flow, under each kind of evolution",
                 x=0.02, ha="left", fontsize=12, fontweight="semibold")
    return _save(fig, name)


def fig_other_fields(name="fig03_other_fields.png"):
    from core.simulation import fields_1d as f1
    from core.simulation import generators as gen
    from core.simulation import met_fields as mf
    from core.simulation.rain_fields import Grid
    vs.use_style()
    grid = Grid(n=128, dx_km=0.5)
    extent = (0, 64, 0, 64)
    rain = gen.make("clustered_storms", seed=3).build(grid)
    fields = {"temperature": (mf.Temperature(rain=rain, seed=1).build(grid), "deg C"),
              "humidity": (mf.Humidity(rain=rain, seed=2).build(grid), "%"),
              "pressure": (mf.Pressure(seed=3).build(grid), "hPa"),
              "wind": (mf.Wind(seed=4).build(grid), "m/s")}
    rng = np.random.default_rng(0)
    t = np.arange(48 * 12) * 5 / 60
    series = {"Bartlett-Lewis": (t, f1.bartlett_lewis(48, 5, rng, storm_rate_per_h=0.05)),
              "Neyman-Scott": (t, f1.neyman_scott(48, 5, rng, storm_rate_per_h=0.05))}
    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.8])
    cmaps = {"temperature": "RdBu_r", "humidity": "BrBG", "pressure": "PuOr_r", "wind": "viridis"}
    for j, (k, (f, unit)) in enumerate(fields.items()):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(f, origin="lower", extent=extent, cmap=cmaps[k])
        vs.style_map_axes(ax, extent)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(k, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75).set_label(unit)
    ax = fig.add_subplot(gs[1, :2])
    for c, (k, (tt, r)) in zip(SLOTS, series.items()):
        ax.plot(tt, r, color=c, lw=1.2, label=k)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("rain rate (mm/h)")
    ax.set_title("1-D in time: point rain series (5 min)", fontsize=10)
    ax.legend(loc="upper right")
    ax = fig.add_subplot(gs[1, 2:])
    d, v = f1.transect(rain, grid.dx_km, 20, 22, 28, 28)
    ax.plot(d, v, color=SLOTS[0], lw=1.5)
    ax.fill_between(d, v, color=SLOTS[0], alpha=0.15, lw=0)
    ax.axhline(v.mean(), color=vs.INK_MUTED, lw=1, ls="--")
    ax.text(d[-1], v.mean(), "  path mean", color=vs.INK_SECONDARY, va="bottom", ha="right", fontsize=8.5)
    ax.set_xlabel("distance along the path (km)")
    ax.set_ylabel("rain rate (mm/h)")
    ax.set_title("1-D in space: what a 10 km link averages", fontsize=10)
    fig.suptitle("Other meteorological fields, and 1-D fields", x=0.02, ha="left", fontsize=12,
                 fontweight="semibold")
    return _save(fig, name)


# --------------------------------------------------------------------------
# one scenario
# --------------------------------------------------------------------------
def fig_sensors(case, products: dict, show=None, name="fig04_sensors.png", title=None):
    """First-hour totals: the truth with the sensors on it, and five products."""
    from core.simulation.benchmark import hourly
    vs.use_style()
    n = case.truth.sizes["lat"]
    ext = (0, n * case.pixel_km, 0, n * case.pixel_km)
    show = show or ["radar", "idw links+gauges", "gmz links", "mfb [links+gauges]", "ked [links+gauges]"]
    panels = [("truth", case.truth)] + [(k, products[k]) for k in show if products.get(k) is not None]
    truth_h = hourly(case.truth, case.interval_min)[0]
    vmax = float(np.percentile(truth_h, 99.5))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), layout="constrained")
    lat0, lon0 = case.truth.lat.values[0], case.truth.lon.values[0]
    dlat = np.diff(case.truth.lat.values[:2])[0]
    dlon = np.diff(case.truth.lon.values[:2])[0]

    def km(lat, lon):
        return (np.asarray(lon) - lon0) / dlon * case.pixel_km + 0.5 * case.pixel_km, \
            (np.asarray(lat) - lat0) / dlat * case.pixel_km + 0.5 * case.pixel_km

    for ax, (k, da) in zip(axes.ravel(), panels):
        f = hourly(da, case.interval_min)[0]
        im = _map(ax, f, ext, vmax, k)
        if k == "truth":
            L = case.links
            if L is not None:
                x0, y0 = km(L.site_0_lat.values, L.site_0_lon.values)
                x1, y1 = km(L.site_1_lat.values, L.site_1_lon.values)
                for a, b, c, d in zip(x0, x1, y0, y1):
                    ax.plot([a, b], [c, d], color=vs.INK_PRIMARY, lw=0.6)
            if case.gauges is not None:
                gx, gy = km(case.gauges.lat.values, case.gauges.lon.values)
                ax.scatter(gx, gy, s=22, color=SLOTS[1], edgecolor=vs.SURFACE, lw=1, zorder=3, label="gauges")
            if case.pws is not None:
                px, py = km(case.pws.lat.values, case.pws.lon.values)
                ax.scatter(px, py, s=12, marker="s", color=SLOTS[4], edgecolor=vs.SURFACE, lw=0.8,
                           zorder=3, label="PWS")
            ax.legend(loc="upper right", fontsize=7.5, frameon=True)
        else:
            r = np.corrcoef(np.nan_to_num(f).ravel(), truth_h.ravel())[0, 1]
            vs.annotate_corner(ax, f"r = {r:.2f} vs truth", "lower right")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6).set_label("first-hour total (mm)")
    m = case.meta
    fig.suptitle(title or f"One scenario: {m['model']}, {m['flow']}, {m['radar_band']}-band radar, "
                          f"{m['n_links']} links, {m['n_gauges']} gauges, {m['n_pws']} PWS",
                 x=0.02, ha="left", fontsize=12, fontweight="semibold")
    return _save(fig, name)


# --------------------------------------------------------------------------
# regional study
# --------------------------------------------------------------------------
def fig_maps(maps: pd.DataFrame, metric: str = "nrmse", top: int = 24, name="fig05_maps.png"):
    """Median and interquartile range of a score per method over all scenarios."""
    vs.use_style()
    g = maps.groupby(["method", "family"])[metric]
    d = pd.DataFrame({"med": g.median(), "q1": g.quantile(0.25), "q3": g.quantile(0.75)}).reset_index()
    d = d.sort_values("med").head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 0.32 * len(d) + 1.2))
    y = np.arange(len(d))
    for fam, c in FAMILY_COLOR.items():
        s = d.family == fam
        ax.hlines(y[s], d.q1[s], d.q3[s], color=c, lw=2)
        ax.scatter(d.med[s], y[s], color=c, s=40, zorder=3, edgecolor=vs.SURFACE, lw=1.5, label=fam)
    ax.set_yticks(y, d.method)
    ax.set_xlabel(f"hourly {metric.upper()} against the truth (median and IQR over scenarios)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    return _save(fig, name)


def fig_motion(motion: pd.DataFrame, name="fig06_motion.png"):
    vs.use_style()
    d = motion[motion.region == "wet"]
    products = [p for p in ("truth", "radar", "ked [links+gauges]", "idw links") if p in set(d["product"])]
    methods = [m for m in ("LK", "VET", "DARTS", "proesmans", "learned", "learned+LK") if m in set(d.method)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    w = 0.8 / len(methods)
    for ax, (groups, col, sub, title) in zip(axes, [
            (products, "product", d, "Motion estimated from each product"),
            (sorted(set(d.flow)), "flow", d[d["product"] == "radar"], "From the radar, by kind of flow")]):
        for i, (m, c) in enumerate(zip(methods, SLOTS)):
            med = [sub[(sub[col] == g) & (sub.method == m)].epe_kmh.median() for g in groups]
            ax.bar(np.arange(len(groups)) + (i - (len(methods) - 1) / 2) * w, med, w * 0.92, color=c, label=m)
        ax.set_xticks(np.arange(len(groups)), groups)
        ax.set_ylabel("median endpoint error where it rains (km/h)")
        ax.set_title(title, fontsize=10)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=len(methods), loc="lower center", bbox_to_anchor=(0.5, -0.04))
    return _save(fig, name)


NOWCAST_METHODS = ("persistence", "extrapolation", "extrapolation (learned+LK)", "sprog", "steps (mean)",
                   "extrapolation (true motion)")


def fig_nowcast(now: pd.DataFrame, score: str = "CSI_1", name="fig07_nowcast.png"):
    vs.use_style()
    products = [p for p in ("truth", "radar", "ked [links+gauges]") if p in set(now["product"])]
    methods = [m for m in NOWCAST_METHODS if m in set(now.method)]
    fig, axes = plt.subplots(1, len(products), figsize=(4.6 * len(products), 4.0), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), products):
        for m, c in zip(methods, SLOTS):
            d = now[(now["product"] == p) & (now.method == m)].sort_values("lead_min")
            ax.plot(d.lead_min, d[score], color=c, label=m, ls="--" if "true" in m else "-")
        ax.set_title(f"nowcast of {p}", fontsize=10)
        ax.set_xlabel("lead time (min)")
    np.atleast_1d(axes)[0].set_ylabel(f"{score.replace('_', ' at ')} mm/h against the truth")
    np.atleast_1d(axes)[-1].legend(loc="upper right", fontsize=7.5)
    return _save(fig, name)


# --------------------------------------------------------------------------
# city study: accuracy by scale and by distance to the links
# --------------------------------------------------------------------------
CITY_METHODS = {
    "radar": "C-band radar (800 m)",
    "X radar": "X-band radar (200 m)",
    "add [links+gauges]": "C-band + links + gauges",
    "X add [links+gauges+pws]": "X-band + links + gauges + PWS",
    "idw links+gauges+pws": "links + gauges + PWS, no radar",
    "gmz links": "links only (GMZ)",
    "idw pws": "PWS only",
    "mul [links+gauges]": "C-band x links + gauges (multiplicative)",
    "ked [links+gauges]": "C-band + links + gauges (KED)",
}


def fig_scales(city: pd.DataFrame, metric: str = "nrmse", name="fig08_scales.png"):
    """Median score against spatial scale, one panel per accumulation time."""
    vs.use_style()
    d = city[city.region == "all"]
    times = sorted(set(d.time_min))
    methods = [m for m in list(CITY_METHODS)[:7] if m in set(d.method)]
    fig, axes = plt.subplots(1, len(times), figsize=(4.0 * len(times), 4.2), sharey=True, layout="constrained")
    for ax, t in zip(np.atleast_1d(axes), times):
        for m, c in zip(methods, SLOTS):
            s = d[(d.time_min == t) & (d.method == m)].groupby("space_km")[metric].median()
            ax.plot(s.index, s.values, color=c, marker="o", ms=4, label=CITY_METHODS[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4], ["0.1", "0.2", "0.4", "0.8", "1.6", "3.2", "6.4"])
        ax.set_title(f"{t:g}-min totals", fontsize=10)
        ax.set_xlabel("pixel size (km)")
    np.atleast_1d(axes)[0].set_ylabel(f"{metric.upper()} against the truth (median of scenarios)")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("How accuracy grows with pixel size and accumulation time (city, 100 m truth)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return _save(fig, name)


def fig_link_distance(city: pd.DataFrame, min_share: float = 0.02, name="fig09_link_distance.png"):
    """NRMSE at 200 m by distance from the nearest link, 5-min and hourly totals; distance bands
    holding less than ``min_share`` of the city are left out (too few cells to score)."""
    vs.use_style()
    d = city[city.region != "all"]
    share = d.groupby("region").cell_share.median()
    d = d[d.region.map(share) >= min_share]
    methods = [m for m in ("radar", "X radar", "add [links+gauges]", "mul [links+gauges]",
                           "ked [links+gauges]", "gmz links") if m in set(d.method)]
    labels = d.groupby("band_lo").region.first()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for ax, t in zip(axes, (5, 60)):
        for m, c in zip(methods, SLOTS):
            s_ = d[(d.method == m) & (d.time_min == t)].groupby("band_lo").nrmse.median()
            ax.plot(np.arange(len(s_)), s_.values, color=c, marker="o", ms=5, label=CITY_METHODS.get(m, m))
        ax.set_xticks(np.arange(len(labels)), labels.values)
        ax.set_xlabel("distance from the nearest link")
        ax.set_title(f"{t}-min totals at 200 m", fontsize=10)
    axes[0].set_ylabel("NRMSE against the truth (median of scenarios)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.15))
    return _save(fig, name)


def fig_real_radar(tables: dict, score: str = "CSI_1", name="fig10_real_radar.png"):
    vs.use_style()
    methods = ("persistence", "LK", "VET", "DARTS", "proesmans", "learned", "learned+LK")
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 4.0), layout="constrained")
    for ax, (net, d) in zip(np.atleast_1d(axes), tables.items()):
        for m, c in zip(methods, SLOTS):
            s = d[d.method == m].sort_values("lead_min")
            if len(s):
                ax.plot(s.lead_min, s[score], color=c, label=m, ls="--" if m.startswith("learned") else "-")
        ax.set_title(f"{net}: extrapolation along each motion ({int(d.n_issues.iloc[0])} issue times)",
                     fontsize=10)
        ax.set_xlabel("lead time (min)")
        ax.set_ylabel(f"{score.replace('_', ' at ')} mm/h against the radar")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=len(lab), loc="lower center", bbox_to_anchor=(0.5, -0.1))
    return _save(fig, name)


def all_from_tables():
    """Every figure that is drawn from the study tables, from whatever tables exist."""
    out = []
    reg = TABLES / "regional"
    if (reg / "maps.csv").exists():
        out += [fig_maps(pd.read_csv(reg / "maps.csv")), fig_motion(pd.read_csv(reg / "motion.csv")),
                fig_nowcast(pd.read_csv(reg / "nowcast.csv"))]
    from .city import load
    city = load()
    if len(city):
        out += [fig_scales(city), fig_link_distance(city)]
    real = {n: pd.read_csv(TABLES / f"real_radar_{n}.csv") for n in ("openmrg", "openrainer")
            if (TABLES / f"real_radar_{n}.csv").exists()}
    if real:
        out.append(fig_real_radar(real))
    return out
