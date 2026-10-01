"""``results/merging/``: figures and ``report.md`` of a :class:`~multisensor_maps.merging.MergingStudy`."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.viz_style import BASELINE, CMAP_RAIN, INK_MUTED, INK_SECONDARY, SERIES

from .merging import ADJUSTMENTS, DEFAULT_TAG, FAMILIES, NEAR_KM, OUT_DIR, PACKAGE, best_per_family
from .plots import NAMES, _links, _save
from .settings import CHECK_POINTS

FAMILY_NAMES = {"R": "radar", "L": "links", "G": "gauges", "L+G": "links + gauges",
                "R+G": "radar + gauges", "R+L": "radar + links", "R+L+G": "radar + links + gauges"}
CHECK_NAMES = {"city": "municipal gauges", "pws": "PWS", "gauges": "ARPAE gauges"}
METHOD_NAMES = {"mfb": "mean-field bias", "add": "additive IDW", "mul": "multiplicative IDW",
                "idw_add": "difference IDW, additive", "idw_mul": "difference IDW, multiplicative",
                "okrig_add": "difference block kriging", "ked": "KED block kriging"}
BLUE, ORANGE = SERIES["stratiform"], SERIES["convective"]
MAIN = "held-out gauges"
NEAR = f"held-out gauges <= {NEAR_KM:g} km of a link"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(labelsize=8, colors=INK_SECONDARY)


def families_figure(study, subset: str = MAIN, path=None):
    """Per network: the best product of each input combination, hourly NRMSE at held-out gauges."""
    nets = [n for n in NAMES if (n, subset) in study.rankings]
    fig, axes = plt.subplots(1, len(nets), figsize=(5.4 * len(nets), 3.8), squeeze=False)
    for ax, n in zip(axes[0], nets):
        b = best_per_family(study.rankings[(n, subset)])
        best = b.nrmse.idxmin()
        y = np.arange(len(b))
        ax.barh(y, b.nrmse, color=[ORANGE if p == best else BLUE for p in b.index], height=0.62)
        for i, (p, r) in enumerate(b.iterrows()):
            ax.text(r.nrmse, i, f" {r.nrmse:.2f} ({r.rel_bias:+.0%})", va="center", fontsize=7.5, color=INK_SECONDARY)
        ax.set_yticks(y, [FAMILY_NAMES[f] if p == FAMILY_NAMES[f] else f"{FAMILY_NAMES[f]}\n{p}"
                          for p, f in zip(b.index, b.family)], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, b.nrmse.max() * 1.35)
        ax.set_title(NAMES[n], fontsize=10, loc="left")
        ax.set_xlabel("hourly NRMSE (bias) at held-out gauges", fontsize=8, color=INK_SECONDARY)
        _style(ax)
    fig.tight_layout()
    return _save(fig, path)


def method_matrix(ranking: pd.DataFrame, radar: str) -> pd.DataFrame:
    """NRMSE of each adjustment method (rows) with each source (columns): the gauges, the
    best retrieval's links, and both."""
    r = ranking[(ranking.radar == radar) & ranking.method.notna()]
    if "variogram" in r:
        r = r[r.variogram != "mergeplg default"]
    cols = {"gauges": r[r.source == "gauges"]}
    rl = r[r.family == "R+L"]
    rlg = r[r.family == "R+L+G"]
    if len(rl):
        m = rl.groupby("retrieval").nrmse.min().idxmin()
        cols[f"links ({m})"] = rl[rl.retrieval == m]
    if len(rlg):
        m = rlg.groupby("retrieval").nrmse.min().idxmin()
        cols[f"links ({m}) + gauges"] = rlg[rlg.retrieval == m]
    out = pd.DataFrame({c: v.set_index("method").nrmse for c, v in cols.items()})
    return out.reindex(list(ADJUSTMENTS))


def methods_figure(study, subset: str = MAIN, path=None):
    """Heat maps: adjustment method x source, per network (main radar)."""
    nets = [n for n in NAMES if (n, subset) in study.rankings]
    fig, axes = plt.subplots(1, len(nets), figsize=(6.2 * len(nets), 4.2), squeeze=False)
    for ax, n in zip(axes[0], nets):
        ranking = study.rankings[(n, subset)]
        M = method_matrix(ranking, "radar")
        radar_alone = ranking.loc["radar", "nrmse"]
        # colour clipped at 1.5 x the radar alone, so one blow-up does not flatten the rest
        lo, hi = np.nanmin(M.values), max(min(np.nanmax(M.values), 1.5 * radar_alone), radar_alone)
        im = ax.imshow(np.clip(M.values, lo, hi), cmap=CMAP_RAIN.reversed(), aspect="auto", vmin=lo, vmax=hi)
        for (i, j), v in np.ndenumerate(M.values):
            if np.isfinite(v):
                dark = v < lo + 0.45 * (hi - lo)
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if dark else "#0b0b0b")
        ax.set_xticks(range(M.shape[1]), [c.replace(" (", "\n(").replace(" + ", "\n+ ") for c in M.columns],
                      fontsize=7.5)
        ax.set_yticks(range(M.shape[0]), [f"{m} ({PACKAGE[m]})" for m in M.index], fontsize=7.5)
        ax.set_title(f"{NAMES[n]}\nradar alone: {radar_alone:.2f}", fontsize=9, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.8).set_label("hourly NRMSE", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def example_figure(network: str, em, fields: dict, path=None):
    """Event totals: the radar and the best map of each input combination, one scale."""
    names = list(fields)
    tot = {k: v.sum("time", min_count=1) for k, v in fields.items()}
    vmax = float(np.nanpercentile(np.concatenate([t.values.ravel() for t in tot.values()]), 99))
    # panel shape follows the domain: wide domains get fewer, wider panels per row
    ratio = (np.ptp(em.grid.lon) * np.cos(np.radians(float(np.mean(em.grid.lat))))) / max(np.ptp(em.grid.lat), 1e-6)
    ncol = min(len(names), 4 if ratio < 1.5 else 2)
    nrow = int(np.ceil(len(names) / ncol))
    w = 3.3 if ratio < 1.5 else 5.2
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol + 1, w / max(ratio, 0.5) * nrow + 0.8), squeeze=False,
                             layout="constrained")
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    lh = next(iter(em.links.values()))
    table = pd.DataFrame({c: lh[c].values for c in ["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"]})
    g = em.gauges
    for ax, k in zip(axes.ravel(), names):
        m = ax.pcolormesh(tot[k].lon, tot[k].lat, tot[k], cmap=CMAP_RAIN, vmin=0, vmax=vmax, shading="nearest")
        _links(ax, table, lw=0.5, alpha=0.55)
        if g is not None:
            ax.scatter(g.lon, g.lat, s=4, color=ORANGE, zorder=5, linewidths=0)
        ax.set_title(k, fontsize=7.5)
        ax.set_aspect(1 / np.cos(np.radians(float(np.mean(em.grid.lat)))))
        ax.set(xlim=(em.grid.lon[0], em.grid.lon[-1]), ylim=(em.grid.lat[0], em.grid.lat[-1]))
        ax.tick_params(labelsize=6, colors=INK_MUTED)
    fig.colorbar(m, ax=axes.ravel().tolist(), shrink=0.6, label="event total (mm)")
    fig.suptitle(f"{NAMES[network]}: {em.inp.start:%Y-%m-%d %H:%M} - {em.inp.end:%m-%d %H:%M} UTC; "
                 "lines: links, dots: gauges", fontsize=9, x=0.01, ha="left")
    return _save(fig, path)


# ------------------------------------------------------------------------ markdown
def _table(df: pd.DataFrame, cols: list, names: dict | None = None) -> list:
    names = names or {}
    lines = ["| " + " | ".join(names.get(c, c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        cells = []
        for c in cols:
            v = idx if c == "product" else r[c]
            if c == "product":
                cells.append(f"`{v}`")
            elif c in ("rel_bias",):
                cells.append(f"{v:+.0%}")
            elif c == "coverage":
                cells.append(f"{v:.0%}")
            elif c == "family":
                cells.append(FAMILY_NAMES.get(v, v))
            elif isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


COLS = ["product", "family", "n", "rel_bias", "nrmse", "corr", "csi"]
COLNAMES = {"n": "station-hours", "rel_bias": "bias", "nrmse": "NRMSE", "corr": "r", "csi": "CSI",
            "median_event_nrmse": "median per event", "event_wins": "events best"}


def findings(study) -> list:
    out = []
    for n in NAMES:
        if (n, MAIN) not in study.rankings:
            continue
        r = study.rankings[(n, MAIN)]
        b = best_per_family(r)
        get = lambda f: b[b.family == f].iloc[0] if (b.family == f).any() else None  # noqa: E731
        best = b.nrmse.idxmin()
        R, G, RG, RL, RLG = (get(f) for f in ("R", "G", "R+G", "R+L", "R+L+G"))
        txt = (f"**{NAMES[n]}** (at held-out {CHECK_NAMES[CHECK_POINTS[n]]}): best map `{best}`, "
               f"NRMSE {b.loc[best, 'nrmse']:.2f} ({b.loc[best, 'rel_bias']:+.0%}). ")
        if R is not None and G is not None:
            txt += f"Radar alone {R.nrmse:.2f}, gauges alone {G.nrmse:.2f}"
        if RG is not None:
            txt += f"; radar + gauges {RG.nrmse:.2f} (`{RG.name}`)"
        if RL is not None:
            txt += f"; radar + links {RL.nrmse:.2f} (`{RL.name}`)"
        if RLG is not None:
            txt += f"; all three {RLG.nrmse:.2f} (`{RLG.name}`)"
        txt += "."
        if (n, NEAR) in study.rankings:
            bn = best_per_family(study.rankings[(n, NEAR)])
            gn = bn[bn.family.isin(["R+G"])]
            ln = bn[bn.family.isin(["R+L+G", "R+L"])].sort_values("nrmse")
            if len(gn) and len(ln):
                txt += (f" Near the links (<= {NEAR_KM:g} km): radar + gauges {gn.nrmse.iloc[0]:.2f}, best with "
                        f"links `{ln.index[0]}` {ln.nrmse.iloc[0]:.2f}.")
        out.append(txt)
    return out


def write_report(study, out: Path = OUT_DIR) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    families_figure(study, MAIN, figs / "families.png")
    families_figure(study, NEAR, figs / "families_near_links.png")
    methods_figure(study, MAIN, figs / "methods.png")
    for n, (em, fields) in study.examples.items():
        example_figure(n, em, fields, figs / f"largest_event_{n}.png")

    L = ["# Merging links, gauges and radar on three networks", "",
         "_Generated by `python src/run.py merge`; every number below is computed._", "",
         "The best hourly rain map from any combination of the three sensors, and what each sensor adds. "
         "Per network, the same 10 largest events as `results/report.md`. Products, on the network's grid:", "",
         "- **radar** as it is (New York: MRMS Pass 2, which NOAA already corrects with gauges, and MRMS "
         "radar-only, which it does not);",
         "- **links** alone - every retrieval (four power laws and the RNN of `projects/cml_rnn`), "
         "IDW from midpoints and from virtual gauges along each path (`line`);",
         "- **gauges** alone (IDW), and **links + gauges** (one IDW over both);",
         "- **radar adjusted** with the gauges, the links, or both, by seven methods: mean-field bias, "
         "additive and multiplicative IDW of the residuals from `pcpn_maps` (`core.maps.merge`), and "
         "difference IDW (additive, multiplicative), difference block kriging and block kriging with external "
         "drift from the OpenSense package `mergeplg` 0.1.0 (`core.maps.mergeplg_methods`, identical to "
         "`mergeplg`'s own `adjust()` - `tests/test_merging.py`). The kriging variogram is fitted to each "
         "network's radar fields, never to a gauge.", "",
         f"**Scoring.** The check gauges are split into 5 groups by station; each group is held out in turn, "
         "every product that uses gauges is rebuilt without it and scored at it, hour by hour (links and radar "
         "do not change). All products are scored on the station-hours every full-coverage product has "
         "(a link-only map has no value beyond 10 km of a link, so away from the links it is shown with its "
         "coverage). NRMSE is the hourly RMSE over the mean gauge value. Independent gauges that no product "
         "uses give a second check where they exist.", "",
         "## Findings", ""]
    L += [f"- {t}" for t in findings(study)]
    L += ["", "![families](figures/families.png)", "",
          "Near the links, where they can add something:", "", "![near](figures/families_near_links.png)", "",
          "## Which merging method", "",
          "Hourly NRMSE at held-out gauges of the radar adjusted by each method, with the gauges, the links of "
          "the best retrieval for that column, and both (method keys in the table below):", "",
          "![methods](figures/methods.png)", "",
          "`idw_mul` is `mergeplg`'s multiplicative difference IDW: the ratio gauge / radar is interpolated with "
          "no bound, so one wet observation over a nearly dry radar cell gives a factor of hundreds; its "
          "blow-ups are the method's, reproduced exactly (the `pcpn_maps` `mul` clips the factor to [0.2, 5] "
          "and adds 0.5 mm to both sides). In Gothenburg every municipal gauge is within 5 km of a link, so the "
          "near-link scores equal the overall ones.", ""]
    for n in NAMES:
        if (n, MAIN) not in study.rankings:
            continue
        r = study.rankings[(n, MAIN)]
        for radar in [x for x in ("radar", "radar only") if (r.radar == x).any()]:
            M = method_matrix(r, radar)
            L += [f"**{NAMES[n]}, `{radar}` (alone: {r.loc[radar, 'nrmse']:.2f})**", "",
                  "| method | package | " + " | ".join(M.columns) + " |", "|---|---|" + "---|" * M.shape[1]]
            for m, row in M.iterrows():
                L.append(f"| {METHOD_NAMES[m]} (`{m}`) | {PACKAGE[m]} | "
                         + " | ".join("" if np.isnan(v) else f"{v:.2f}" for v in row) + " |")
            L.append("")
    L += ["## Which link retrieval, once merged", "",
          "Best NRMSE at held-out gauges over methods, per retrieval (radar + links; radar + links + gauges; "
          "links + gauges):", ""]
    for n in NAMES:
        if (n, MAIN) not in study.rankings:
            continue
        r = study.rankings[(n, MAIN)]
        if "variogram" in r:
            r = r[r.variogram != "mergeplg default"]
        t = pd.DataFrame({f: r[r.family == f].groupby("retrieval").nrmse.min() for f in ("R+L", "R+L+G", "L+G")})
        t = t.sort_values("R+L+G")
        L += [f"**{NAMES[n]}**", "", "| retrieval | radar + links | radar + links + gauges | links + gauges |",
              "|---|---|---|---|"]
        L += [f"| {m} | {row['R+L']:.2f} | {row['R+L+G']:.2f} | {row['L+G']:.2f} |" for m, row in t.iterrows()]
        L.append("")

    L += ["## The kriging variogram", "",
          "The kriging products use a spherical variogram fitted to each network's radar (range and nugget "
          "below). Rebuilt with `mergeplg`'s default instead (sill 0.9, range 5 km, nugget 0.1) - NRMSE at "
          "held-out gauges, radar adjusted with the gauges and with the best retrieval's links + gauges:", "",
          "| network | radar | product | radar fit | 5 km default |", "|---|---|---|---|---|"]
    for n in NAMES:
        if (n, MAIN) not in study.rankings or "variogram" not in study.rankings[(n, MAIN)]:
            continue
        r = study.rankings[(n, MAIN)]
        fit = r[r.variogram == "radar fit"]
        for radar in [x for x in ("radar", "radar only") if (fit.radar == x).any()]:
            f = fit[fit.radar == radar]
            rlg = f[f.family == "R+L+G"]
            best_m = rlg.groupby("retrieval").nrmse.min().idxmin() if len(rlg) else None
            for p in f[(f.source == "gauges") | (f.source == f"{best_m} + gauges")].index:
                d = p + DEFAULT_TAG
                if d in r.index:
                    L.append(f"| {NAMES[n]} | {radar} | `{p}` | {r.loc[p, 'nrmse']:.2f} | {r.loc[d, 'nrmse']:.2f} |")
    L.append("")

    for n in NAMES:
        keys = [k for k in study.rankings if k[0] == n]
        if not keys:
            continue
        vg = study.variograms.get(n, {})
        evs = study.events[study.events.network == n]
        scored = study.tables[n].event.nunique()
        L += [f"## {NAMES[n]}", "",
              f"{scored} of {len(evs)} events scored, {pd.Timestamp(evs.start.min()):%Y-%m-%d} to {pd.Timestamp(evs.end.max()):%Y-%m-%d}; "
              f"check gauges: {CHECK_NAMES[CHECK_POINTS[n]]}. Kriging variogram from the radar: spherical, range "
              f"{vg.get('range', np.nan) / 1000:.0f} km, nugget {vg.get('nugget', np.nan):.0%} of the sill "
              f"({vg.get('hours', '?')} wet hours).", ""]
        if n in study.examples:
            L += [f"![largest event](figures/largest_event_{n}.png)", ""]
        for _, subset in keys:
            r = study.rankings[(n, subset)]
            L += [f"**Best of each input combination, {subset}:**", ""]
            L += _table(best_per_family(r), COLS, COLNAMES) + [""]
            if subset == MAIN:
                top = r[(r.coverage >= 0.9) & (r.get("variogram") != "mergeplg default")].head(15)
                L += ["**Top 15 products** (median per-event NRMSE and number of events on which each is the best "
                      "product):", ""]
                L += _table(top, COLS + ["median_event_nrmse", "event_wins"], COLNAMES) + [""]
    if study.errors:
        L += ["## Not scored", ""] + [f"- {k}: {v}" for k, v in study.errors.items()] + [""]
    L += ["## Files", "",
          "`ranking_<network>_<subset>.csv` (every product, pooled), `scores_per_event.csv`, `events.csv`, "
          "`variograms.json`. Per-event inputs and station-hour tables are cached under "
          "`dataset/open_datasets/_multisensor_maps/merging/`.", ""]
    (out / "report.md").write_text("\n".join(L))
    return out / "report.md"
