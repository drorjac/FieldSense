"""results/report.md from the result tables."""

from __future__ import annotations

import json

import pandas as pd

from .settings import GAUGE_KM, GRID_RES, IDW, MIN_COVERAGE, P_CUT, RESULTS_DIR, WET_MM

# Overeem, Leijnse and Uijlenhoet (2016), AMT 9, 2425-2444, Sect. 4.4: 12 days (June,
# August, September 2011), Nokia links, RAINLINK defaults, IDW power 2 on a 0.9 km^2
# grid, against gauge-adjusted radar (not gauges), every land pixel.
SRC = "Overeem et al. 2016, AMT"
PUBLISHED = pd.DataFrame([
    {"source": SRC, "what": "path-averaged links", "scale": "15 min", "rel_bias": 0.105, "cv": 3.84, "r2": 0.54},
    {"source": SRC, "what": "map, IDW", "scale": "15 min", "rel_bias": 0.083, "cv": 3.30, "r2": 0.41},
    {"source": SRC, "what": "map, IDW", "scale": "daily", "rel_bias": 0.082, "cv": 0.51, "r2": 0.70},
    {"source": SRC, "what": "map, kriging", "scale": "daily", "rel_bias": 0.014, "cv": 0.54, "r2": 0.73},
])


def _table(df: pd.DataFrame, digits: int = 2) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.{digits}f}")
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def write() -> None:
    net = json.loads((RESULTS_DIR / "network.json").read_text())
    path = pd.read_csv(RESULTS_DIR / "path_level.csv")
    maps = pd.read_csv(RESULTS_DIR / "map_level.csv")
    pairs = pd.read_csv(RESULTS_DIR / "path_pairs.csv")
    stations = pd.read_csv(RESULTS_DIR / "map_stations.csv")
    PUBLISHED.to_csv(RESULTS_DIR / "published.csv", index=False)
    ratio = (pairs.total_link / pairs.total_gauge).where(pairs.total_gauge > 0)
    cols = ["n", "mean_ref", "mean_est", "rel_bias", "nrmse", "cv", "corr", "r2", "pod", "far", "csi"]
    L = ["# maps/netherlands: results", "",
         "Computed by `python projects/maps/netherlands/src/run.py report` from the CSVs here. "
         "RAINLINK (pycomlink's port, default parameters, ITU-R P.838-3 vertical) on the Dutch "
         "network, summer (JJA) 2012, against KNMI's hourly automatic gauges. Hourly totals in mm; "
         f"wet = {WET_MM} mm. A link hour needs {MIN_COVERAGE:.0%} of its 15-min steps; a day "
         "needs 20 hours valid in both series, summed over the same hours.", "",
         "`rel_bias` = sum(est)/sum(ref) - 1; `nrmse` = RMSE / mean(ref); `cv` = std(residual) / "
         "mean(ref) and `r2` = corr^2, the two numbers RAINLINK's papers report.", "",
         "## Network", "", _table(pd.DataFrame(net.items(), columns=["what", "value"])), "",
         f"## Path level: each path against the nearest KNMI gauge within {GAUGE_KM:g} km of its midpoint",
         "", _table(path[["pairs_used", "scale", "pairs", "gauges"] + cols]), "",
         f"Pairs: {len(pairs)} paths at {pairs.station.nunique()} gauges, median distance "
         f"{pairs.dist_km.median():.1f} km, median path length {pairs.length_km.median():.1f} km. "
         f"Per pair (`path_pairs.csv`): median hourly correlation {pairs['corr'].median():.2f}; "
         f"the ratio of link to gauge summer totals has median {ratio.median():.2f}, "
         f"quartiles {ratio.quantile(0.25):.2f} to {ratio.quantile(0.75):.2f}, and is above 3 "
         f"for {(ratio > 3).sum()} paths, of which {((ratio > 3) & (pairs.length_km < 1)).sum()} "
         "are shorter than 1 km.", "",
         f"## Map level: IDW (power {IDW['power']:g}, {IDW['radius_m'] / 1000:g} km) at every KNMI gauge",
         "", f"Maps on a {GRID_RES} deg grid, read at the cell holding each gauge; `masked` is "
         f"`core.maps.wet_area.masked_idw` (p = {P_CUT}). Gauges with no link within the radius "
         "have no map value and are not scored (`gauges_covered`).", "",
         _table(maps[["map", "scale", "gauges_covered"] + cols]), "",
         "Per gauge: `map_stations.csv`.", "",
         "## Published RAINLINK numbers", "",
         "Overeem, Leijnse and Uijlenhoet (2016, AMT 9, 2425-2444, Sect. 4.3-4.4), 12 days of "
         "2011 from Nokia links, against gauge-adjusted radar on every land pixel (here: point "
         "gauges, a different reference), with RAINLINK's own DSD-based power law:", "",
         _table(PUBLISHED, 3), ""]
    idw = stations[stations["map"] == "idw"].dropna(subset=["corr"])
    if len(idw):
        ratio = (idw.total_map / idw.total_gauge).describe()
        L += ["## Per gauge (IDW, hourly)", "",
              f"Ratio of the map's summer total to the gauge's, over the {len(idw)} covered "
              f"gauges: median {ratio['50%']:.2f}, quartiles {ratio['25%']:.2f} to {ratio['75%']:.2f}.", ""]
    (RESULTS_DIR / "report.md").write_text("\n".join(L))
