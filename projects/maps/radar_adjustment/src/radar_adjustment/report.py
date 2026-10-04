"""``results/report.md``, the score tables and figures - every number from the saved runs."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from radar_adjustment.prepare import PREPARED, load
from radar_adjustment.score import distance_to_links_km, load_product, radar_at_gauges, score_products
from radar_adjustment.settings import CHECKS, FIELDS, LABELS, RESULTS_DIR, VARIANTS

NETS = {"openmrg": "OpenMRG (Gothenburg, JJA 2015)", "openrainer": "OpenRainER (Emilia-Romagna, JJA 2022)"}
PRODUCTS = ["radar", *VARIANTS]

# OpenSense's own result: the table printed in 4_analysis.ipynb of
# OpenSenseAction/radar_adjustment_intercomparison (OpenMRG, range checks of
# 3_adjust_radar; RMSE and MAE in mm/h, PBIAS in %). OpenRainER's is not in the notebooks.
PUBLISHED_OPENMRG = pd.DataFrame({
    "product": ["radar", "add_p_idw", "add_p_ok", "add_b_ok", "ked_p", "ked_b", "mul_p_idw", "mul_p_ok", "mul_b_ok"],
    "pcc": [0.552351, 0.682600, 0.696003, 0.696606, 0.686943, 0.687618, 0.677492, 0.691845, 0.692403],
    "rmse": [1.440088, 1.305720, 1.266565, 1.266235, 1.290163, 1.289670, 1.317976, 1.282964, 1.282181],
    "mae": [0.835733, 0.701064, 0.675495, 0.674293, 0.683296, 0.682588, 0.701177, 0.681952, 0.680866],
    "pbias": [-18.814609, -10.177743, -10.312691, -10.388008, -11.278192, -11.336716, -6.682407, -6.202346, -6.321689],
})


def _gather(version, network, checks, tags, rad_at, g):
    products = {"radar": rad_at} if rad_at is not None else {}
    for tag in tags:
        p = load_product(version, network, checks, tag)
        if p is not None and p.sizes["time"] == g.sizes["time"]:
            products[tag] = p
    return products


def score_all() -> dict:
    """Score every saved run; write the CSV tables; return them."""
    tables = {k: [] for k in ("overall", "classes", "per_gauge", "bands")}
    for net in NETS:
        if not (PREPARED / net / "cml.nc").exists():
            continue
        rad, cml, g = load(net)
        rad_at = radar_at_gauges(rad, g)
        dist = distance_to_links_km(g, cml)
        runs = [(v, c) for v in ("pinned", "main") for c in CHECKS]
        runs += [("main", c) for c in ("mapping", "stations", "vg5km", "vgfit", "c0within")]
        for version, checks in runs:
            folder = FIELDS / version / net / checks
            if not folder.exists():
                continue
            tags = sorted({f.name.rsplit("_", 1)[0] for f in folder.glob("*.nc")})
            products = _gather(version, net, checks, tags, rad_at if checks in CHECKS else None, g)
            if not products:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = score_products(products, g, dist)
            for k, df in res.items():
                if len(df):
                    tables[k].append(df.assign(network=net, version=version, checks=checks))
    out = {k: pd.concat(v, ignore_index=True) for k, v in tables.items() if v}
    names = {"overall": "scores", "classes": "scores_by_intensity", "per_gauge": "scores_per_gauge",
             "bands": "scores_by_distance"}
    for k, df in out.items():
        cols = ["network", "version", "checks", "product"] + [c for c in df.columns
                                                            if c not in ("network", "version", "checks", "product")]
        df[cols].to_csv(RESULTS_DIR / f"{names[k]}.csv", index=False, float_format="%.4f")
    return out


# ---------------------------------------------------------------------------
def _table(df: pd.DataFrame, cols: list, fmt: dict | None = None) -> list:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(fmt.get(c, "{:.3f}").format(v) if np.isfinite(v) else "-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _label(p: str) -> str:
    base, _, adj = p.partition("@")
    name = {**LABELS, "idw_map": "IDW map (no radar)", "okp_map": "point OK map (no radar)",
            "okb_map": "block OK map (no radar)", "radolan": "RADOLAN"}.get(base, base)
    return f"{name} [{adj}]" if adj else name


def figures(t: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.viz_style import GRIDLINE, INK_SECONDARY, SURFACE

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    s = t["overall"]
    colors = {"pinned": "#2a78d6", "main": "#eb6834"}
    # 1. RMSE and PCC per product, pinned vs main, the published range checks
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row", facecolor=SURFACE)
    for j, net in enumerate(NETS):
        for i, metric in enumerate(("rmse", "pcc")):
            ax = axes[i, j]
            x = np.arange(len(PRODUCTS))
            for k, v in enumerate(("pinned", "main")):
                d = s[(s.network == net) & (s.version == v) & (s.checks == "default")].set_index("product")
                vals = [d[metric].get(p, np.nan) for p in PRODUCTS]
                ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=colors[v], label=f"mergeplg {v}")
            if net == "openmrg":
                pub = PUBLISHED_OPENMRG.set_index("product")[metric]
                ax.scatter(x, [pub.get(p) for p in PRODUCTS], marker="_", s=300, color="k", zorder=3,
                           label="OpenSense, published")
            ax.set_xticks(x, [LABELS[p] for p in PRODUCTS], rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{NETS[net]}", fontsize=9, color=INK_SECONDARY)
            ax.set_ylabel({"rmse": "RMSE (mm/h)", "pcc": "Pearson r"}[metric])
            ax.grid(axis="y", color=GRIDLINE)
            ax.set_axisbelow(True)
            if metric == "pcc":
                lo = np.nanmin(s[s.network.isin(list(NETS))].pcc) - 0.05
                ax.set_ylim(max(0, lo), 1)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Radar adjusted with links, scored at independent gauges (hourly, 0.2 mm threshold)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "intercomparison.png", dpi=130)
    plt.close(fig)

    # 2. effect of the range checks (pinned)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    ccol = {"default": "#2a78d6", "cc": "#eb6834", "nc": "#1baf7a"}
    for j, net in enumerate(NETS):
        ax = axes[j]
        x = np.arange(len(VARIANTS))
        for k, c in enumerate(CHECKS):
            d = s[(s.network == net) & (s.version == "pinned") & (s.checks == c)].set_index("product")
            ax.bar(x + (k - 1) * 0.27, [d.rmse.get(p, np.nan) for p in VARIANTS], 0.27, color=ccol[c],
                   label={"default": "published checks", "cc": "conservative", "nc": "none"}[c])
        r = s[(s.network == net) & (s.version == "pinned") & (s["product"] == "radar")].rmse
        if len(r):
            ax.axhline(r.iloc[0], color="k", lw=1, ls="--", label="radar")
        ax.set_xticks(x, [LABELS[p] for p in VARIANTS], rotation=45, ha="right", fontsize=8)
        ax.set_title(NETS[net], fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("RMSE (mm/h)")
        ax.grid(axis="y", color=GRIDLINE)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "range_checks.png", dpi=130)
    plt.close(fig)

    # 3. RMSE by distance of the gauge to the nearest link
    b = t.get("bands")
    if b is not None and len(b):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
        for j, net in enumerate(NETS):
            ax = axes[j]
            d = b[(b.network == net) & (b.version == "main") & (b.checks == "default")]
            bands = list(dict.fromkeys(d.band))
            for p, col in (("radar", "k"), ("add_b_ok", "#2a78d6"), ("ked_b", "#eb6834"), ("mul_b_ok", "#1baf7a")):
                dd = d[d["product"] == p].set_index("band")
                ax.plot(bands, [dd.rmse.get(x, np.nan) for x in bands], marker="o", color=col, label=LABELS[p])
            n = d[d["product"] == "radar"].set_index("band").n_gauges
            ax.set_xticks(range(len(bands)), [f"{x}\n({n.get(x, 0)} gauges)" for x in bands], fontsize=8)
            ax.set_title(NETS[net], fontsize=9, color=INK_SECONDARY)
            ax.set_ylabel("RMSE (mm/h)")
            ax.grid(color=GRIDLINE)
        axes[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "distance_to_links.png", dpi=130)
        plt.close(fig)

    # 4. three-month totals: radar and the best adjustment
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor=SURFACE)
    for j, net in enumerate(NETS):
        best = s[(s.network == net) & (s.version == "main") & (s.checks == "default") & (s["product"] != "radar")]
        if best.empty:
            continue
        bp = best.sort_values("rmse")["product"].iloc[0]
        rad, cml, g = load(net)
        files = sorted((FIELDS / "main" / net / "default").glob(f"{bp}_????-??.nc"))
        tot = sum(xr.open_dataset(f).total.load() for f in files)
        rad_tot = rad.sum("time")
        from radar_adjustment.adjust import crop
        rad_tot = crop(rad.isel(time=slice(0, 1)), cml, g).isel(time=0).copy(
            data=crop(rad, cml, g).sum("time").values)
        vmax = float(np.nanpercentile(rad_tot.values, 99))
        for i, (title, f) in enumerate((("radar", rad_tot.values), (LABELS[bp], tot.values))):
            ax = axes[i, j]
            im = ax.pcolormesh(rad_tot.lon, rad_tot.lat, f, vmin=0, vmax=vmax, cmap="Blues", shading="auto")
            ax.plot(np.c_[cml.site_0_lon, cml.site_1_lon].T, np.c_[cml.site_0_lat, cml.site_1_lat].T,
                    color="#eb6834", lw=0.6)
            ax.scatter(g.lon, g.lat, s=8, c="k")
            ax.set_title(f"{NETS[net]}: {title}", fontsize=9)
            fig.colorbar(im, ax=ax, label="3-month total (mm)")
    fig.tight_layout()
    fig.savefig(fig_dir / "totals.png", dpi=130)
    plt.close(fig)


def write_report() -> str:
    t = score_all()
    figures(t)
    s, cls, bands = t["overall"], t.get("classes"), t.get("bands")
    cols = ["product", "pcc", "rmse", "mae", "pbias", "n_pairs", "n_nan"]
    L = ["# radar_adjustment - results", "",
         "Every number below is computed by `run.py report` from the runs saved by `run.py adjust` "
         "and `run.py extend`. Hourly, at the gauges, which no adjustment uses; poligrain's "
         "`calculate_rainfall_metrics`, 0.2 mm threshold on both sides (`n_pairs`: gauge-hours left).", ""]

    # replication
    L += ["## 1. Replication of the OpenSense result (OpenMRG)", "",
          "`published` = the table printed in the OpenSense repository's `4_analysis`; `pinned` = "
          "mergeplg 9894b9c (the submodule it records), `main` = mergeplg dd380b1; range checks of "
          "`3_adjust_radar` (difference 10 mm, ratio 0.1-15).", ""]
    pin = s[(s.network == "openmrg") & (s.version == "pinned") & (s.checks == "default")].set_index("product")
    mai = s[(s.network == "openmrg") & (s.version == "main") & (s.checks == "default")].set_index("product")
    if len(pin) and len(mai):
        rows = []
        for _, r in PUBLISHED_OPENMRG.iterrows():
            p = r["product"]
            if p in pin.index and p in mai.index:
                rows.append({"product": _label(p),
                             "rmse published": r.rmse, "rmse pinned": pin.loc[p, "rmse"], "rmse main": mai.loc[p, "rmse"],
                             "pcc published": r.pcc, "pcc pinned": pin.loc[p, "pcc"], "pcc main": mai.loc[p, "pcc"],
                             "pbias published": r.pbias, "pbias pinned": pin.loc[p, "pbias"],
                             "pbias main": mai.loc[p, "pbias"]})
        f = {c: "{:+.2f}" for c in rows[0] if c.startswith("pbias")}
        f.update({c: "{:.4f}" for c in rows[0] if c.startswith(("rmse", "pcc"))})
        L += _table(pd.DataFrame(rows), list(rows[0]), f) + [""]

    # full tables
    L += ["## 2. The intercomparison, both networks", ""]
    for net, title in NETS.items():
        for version in ("pinned", "main"):
            for c, desc in (("default", "published range checks"), ("cc", "conservative checks (diff 5, ratio 0.2-8)"),
                            ("nc", "no range checks")):
                d = s[(s.network == net) & (s.version == version) & (s.checks == c)].copy()
                if d.empty:
                    continue
                d["product"] = d["product"].map(_label)
                L += [f"**{title} - mergeplg {version}, {desc}**", ""]
                L += _table(d[cols], cols, {"pbias": "{:+.1f}", "n_pairs": "{:.0f}", "n_nan": "{:.0f}"}) + [""]

    if cls is not None and len(cls):
        L += ["## 3. By intensity of the gauge value (main, published checks): RMSE (mm/h)", ""]
        for net, title in NETS.items():
            d = cls[(cls.network == net) & (cls.version == "main") & (cls.checks == "default")]
            if d.empty:
                continue
            piv = d.pivot_table(index="product", columns="class", values="rmse", sort=False)
            n = d.groupby("class", sort=False).n.first()
            piv = piv.reindex([p for p in PRODUCTS if p in piv.index])
            piv.index = piv.index.map(_label)
            piv = piv.reset_index()
            piv.columns = ["product"] + [f"{c} (n={n[c]})" for c in piv.columns[1:]]
            L += [f"**{title}**", ""] + _table(piv, list(piv.columns)) + [""]

    if bands is not None and len(bands):
        L += ["## 4. By distance of the gauge to the nearest link (main, published checks): RMSE (mm/h)", ""]
        for net, title in NETS.items():
            d = bands[(bands.network == net) & (bands.version == "main") & (bands.checks == "default")]
            if d.empty:
                continue
            piv = d.pivot_table(index="product", columns="band", values="rmse", sort=False)
            n = d.groupby("band", sort=False).n_gauges.first()
            piv = piv.reindex([p for p in PRODUCTS if p in piv.index])
            piv.index = piv.index.map(_label)
            piv = piv.reset_index()
            piv.columns = ["product"] + [f"{c} ({n[c]} gauges)" for c in piv.columns[1:]]
            L += [f"**{title}**", ""] + _table(piv, list(piv.columns)) + [""]

    # extensions
    ext = s[s.checks.isin(["mapping", "stations", "vg5km", "vgfit", "c0within"])]
    if len(ext):
        L += ["## 5. Extensions (mergeplg main)", "",
              "`run`: `links (default)` the intercomparison's main run; `mapping` maps with no radar and RADOLAN; "
              "`stations` the radar adjusted with PWS or links + PWS (`[pws]`, `[cml+pws]`); `vg5km`, `vgfit`, "
              "`c0within` block kriging with mergeplg's default variogram, one fitted to the radar, and the nugget "
              "from the link geometry.", ""]
        base = s[(s.version == "main") & (s.checks == "default")]
        for net, title in NETS.items():
            d = pd.concat([base[(base.network == net) & base["product"].isin(["radar", "add_p_idw", "add_b_ok",
                                                                             "ked_b", "mul_b_ok"])].assign(run="links (default)"),
                           ext[ext.network == net].assign(run=lambda x: x.checks)])
            if d.empty:
                continue
            d["product"] = d["product"].map(_label)
            L += [f"**{title}**", ""] + _table(d.sort_values("rmse")[["run"] + cols], ["run"] + cols,
                                               {"pbias": "{:+.1f}", "n_pairs": "{:.0f}", "n_nan": "{:.0f}"}) + [""]
        vg = RESULTS_DIR / "variograms.json"
        if vg.exists():
            L += ["Fitted variograms (spherical, sill 1, from standardised wet radar hours):", "", "```",
                  json.dumps(json.loads(vg.read_text()), indent=1), "```", ""]
    ny = RESULTS_DIR / "newyork_asos.csv"
    if ny.exists():
        d = pd.read_csv(ny)
        ev = json.loads((RESULTS_DIR / "newyork_asos_events.json").read_text())
        L += ["## 6. New York City: MRMS radar-only adjusted, at the 4 ASOS gauges", "",
              f"The storms of `projects/maps/multisensor` with every product ({len(ev['events'])}; left out for "
              f"missing inputs: {', '.join(ev['excluded_incomplete']) or 'none'}), {ev['station_hours']} ASOS "
              "station-hours, the same for every product. `links rnn` / `links nearby`: link rain from the RNN of "
              "`projects/retrieval/rnn_three_networks` / the nearby-link power law; `pws`: the WU PWS; `MRMS Pass 2`: the radar "
              "corrected by NOAA with gauges, as a reference product.", ""] + _table(d[cols], cols, {"pbias": "{:+.1f}", "n_pairs": "{:.0f}", "n_nan": "{:.0f}"}) + [""]
    L += ["## Figures", "", "![intercomparison](figures/intercomparison.png)", "",
          "![range checks](figures/range_checks.png)", "", "![distance](figures/distance_to_links.png)", "",
          "![totals](figures/totals.png)", ""]
    path = RESULTS_DIR / "report.md"
    path.write_text("\n".join(L))
    return str(path)
