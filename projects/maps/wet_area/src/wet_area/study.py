"""The three maps (plain IDW, IDW with a wet mask, conditional IDW) scored on known truth
and on the storms of projects/maps/multisensor.

Every map is of hourly totals (mm) from links alone. Synthetic cases are scored against
the true field; real events against the radar on the same grid and at held-out gauges.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid
from core.maps import wet_area as wa
from core.maps.idw import idw_map
from core.maps.scores import scores

from .settings import EVENT_INPUTS, GAUGES, IDW, N_SYNTHETIC, P_CUTS, RETRIEVALS, WET_MM

log = logging.getLogger(__name__)


def link_maps(links: xr.DataArray, grid: Grid, idw: dict, p_cuts=P_CUTS) -> dict:
    """``{name: (time, lat, lon)}``: plain IDW, and the masked and conditional maps per cut."""
    out = {"idw": idw_map(links, grid, **idw)}
    for p in p_cuts:
        out[f"masked p{p:.1f}"] = wa.masked_idw(links, grid, WET_MM, p, **idw)
        out[f"conditional p{p:.1f}"] = wa.conditional_idw(links, grid, WET_MM, p, **idw)
    return out


def score_row(est: xr.DataArray, ref: xr.DataArray) -> dict:
    s = scores(est.values, ref.values, wet_threshold=WET_MM)
    return {**{k: s.get(k, np.nan) for k in ("n", "nrmse", "corr", "rel_bias", "pod", "far", "csi")},
            **wa.wet_area_scores(est.values, ref.values, WET_MM)}


# -------------------------------------------------------------------------- synthetic
def _hourly(da: xr.DataArray, interval_min: float) -> xr.DataArray:
    k = int(round(60 / interval_min))
    n = da.sizes["time"] // k
    da = da.isel(time=slice(0, n * k))
    return da.coarsen(time=k).sum().assign_coords(time=da.time.values[k - 1::k])


def synthetic(n_cases: int = N_SYNTHETIC) -> pd.DataFrame:
    """One row per case and map: links alone against the true hourly field."""
    from core.simulation.scenario import random_scenario
    rows = []
    for seed in range(n_cases):
        case = random_scenario(seed, n_pws=0).run()
        truth = _hourly(case.truth, case.interval_min)
        links = _hourly(case.links, case.interval_min)
        for name, m in link_maps(links, case.geo_grid, {"power": 2.0, "radius_m": None}).items():
            rows.append({"case": seed, "model": case.meta.get("model"), "n_links": links.sizes["link"],
                         "map": name, **score_row(m, truth)})
        log.info("synthetic case %d done", seed)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- real events
def events() -> list:
    return sorted(p for p in EVENT_INPUTS.glob("*_*") if (p / "meta.json").exists())


def _at(field: xr.DataArray, points: xr.DataArray) -> xr.DataArray:
    lat = xr.DataArray(points.lat.values, dims="station")
    lon = xr.DataArray(points.lon.values, dims="station")
    v = field.sel(lat=lat, lon=lon, method="nearest").transpose("station", "time")
    return v.drop_vars(["lat", "lon"]).assign_coords(station=points.station.values)


def real() -> pd.DataFrame:
    """One row per event, retrieval, map and reference (radar on the grid, held-out gauges)."""
    rows = []
    for folder in events():
        network = folder.name.split("_")[0]
        meta = json.loads((folder / "meta.json").read_text())
        radar = xr.open_dataarray(folder / "radar_0.nc").load()
        grid = Grid(radar.lat.values, radar.lon.values)
        g = GAUGES.get(network)
        gauges = xr.open_dataarray(folder / f"points_{g}.nc").load() if g in meta["points"] else None
        for ret in RETRIEVALS:
            if ret not in meta["links"]:
                continue
            links = xr.open_dataarray(folder / f"links_{ret}.nc").load().transpose("link", "time")
            for name, m in link_maps(links, grid, IDW).items():
                t = np.intersect1d(m.time.values, radar.time.values)
                base = {"network": network, "event": folder.name, "retrieval": ret, "map": name}
                rows.append({**base, "reference": "radar", **score_row(m.sel(time=t), radar.sel(time=t))})
                if gauges is not None:
                    tg = np.intersect1d(m.time.values, gauges.time.values)
                    rows.append({**base, "reference": "gauges",
                                 **score_row(_at(m.sel(time=tg), gauges), gauges.sel(time=tg).transpose("station", "time"))})
        log.info("%s done", folder.name)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- summary
COLS = ["nrmse", "corr", "rel_bias", "csi", "far", "war_ratio", "peak_ratio"]


def summary(df: pd.DataFrame, by: list) -> pd.DataFrame:
    """Median over cases/events of each score."""
    return df.groupby(by + ["map"])[COLS].median().round(3).reset_index()
