"""ARPAE's own gauge-adjusted radar (OpenRainER ``RADadj``) as a baseline for Emilia-Romagna.

The intercomparison adjusts the unadjusted composite (``RADrain``), which reads about double
the gauges in summer 2022. ARPAE also publishes the composite corrected by kriging the
gauge/radar ratio of its own gauge network. Here it is prepared exactly as ``RADrain`` is
(hourly totals from 15-min depths, the radar zero threshold, first and last hour dropped)
and scored at the same 319 gauges with the same metric.

It is not an independent baseline: ARPAE's correction uses its gauges, and the gauges
scored here are ARPAE's. Its scores are therefore an upper reference for what gauge
adjustment can reach, not a fair competitor to link adjustment.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from core.opensense import intercomparison_chain as ic
from radar_adjustment.prepare import PREPARED, _openrainer_file, load
from radar_adjustment.score import distance_to_links_km, radar_at_gauges, score_products
from radar_adjustment.settings import MONTHS, RESULTS_DIR

log = logging.getLogger(__name__)
PAD_DEG = 0.1          # cells kept around the gauges; the score reads the nearest cell only


def prepare(force: bool = False) -> xr.Dataset:
    """Hourly ``RADadj`` on the gauges' part of the grid, saved next to the prepared radar."""
    path = PREPARED / "openrainer" / "radar_adj.nc"
    if path.exists() and not force:
        return xr.open_dataset(path)
    _, _, g = load("openrainer")
    la0, la1 = float(g.lat.min()) - PAD_DEG, float(g.lat.max()) + PAD_DEG
    lo0, lo1 = float(g.lon.min()) - PAD_DEG, float(g.lon.max()) + PAD_DEG
    parts = []
    for m in MONTHS["openrainer"]:
        with xr.open_dataset(_openrainer_file("RADadj", m)) as r:
            rr = r.rainfall_amount.sel(lat=slice(la0, la1), lon=slice(lo0, lo1)).load()
        parts.append(rr.rename({"lon": "x", "lat": "y"}))
        log.info("RADadj: read %s", m)
    radar = xr.concat(parts, "time").to_dataset(name="rainfall_amount")
    full = pd.date_range(radar.time.min().values, radar.time.max().values, freq="15min")
    radar = ic.hourly_from_mean(radar.reindex(time=full), 4)
    radar["rainfall_amount"] = ic.threshold_radar(radar.rainfall_amount)
    radar = radar.drop_isel(time=-1).drop_isel(time=0)
    lon2, lat2 = np.meshgrid(radar.x.values, radar.y.values)
    radar.coords["lon"] = (("y", "x"), lon2.astype(float))
    radar.coords["lat"] = (("y", "x"), lat2.astype(float))
    radar.coords["x_grid"], radar.coords["y_grid"] = ic.project(radar.lon, radar.lat)
    radar.astype("float32").to_netcdf(path)
    return radar


def score() -> dict:
    """The unadjusted radar, ARPAE's adjusted radar and every saved adjustment, at the gauges."""
    rad, cml, g = load("openrainer")
    adj = prepare().rainfall_amount.load()
    adj = adj.reindex(time=rad.time.values)
    products = {"radar": radar_at_gauges(rad, g), "arpae_adjusted": radar_at_gauges(adj, g)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = score_products(products, g, distance_to_links_km(g, cml))
    names = {"overall": "arpae_adjusted_scores", "bands": "arpae_adjusted_by_distance"}
    for k, name in names.items():
        res[k].to_csv(RESULTS_DIR / f"{name}.csv", index=False, float_format="%.4f")
    return res
