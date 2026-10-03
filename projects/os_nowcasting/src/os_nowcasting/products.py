"""The rain fields that get nowcast, all at the radar's native step on one grid.

``radar``        the radar itself (:mod:`os_nowcasting.data`)
``cml_idw<d>``   link rain by IDW within ``d`` km - the session's
                 ``rainfall_interpolateIDW_10/20/40``. Links: metadata and time-series QC
                 (``core.cml.link_qc``), the nearby-link retrieval (Overeem et al. 2016,
                 ``core.cml.estimators.NearbyLinks``, 15-min min/max RSL; the best power law
                 in ``projects/multisensor_maps``), sublinks averaged per link, accumulated
                 to the radar's step; IDW power 2 from path midpoints, 12 nearest links.
                 Cells with no link in range are 0 (the session filled them with the
                 field's minimum).
``merged``       the radar adjusted with the links: ``mergeplg`` difference block kriging,
                 additive (``core.maps.mergeplg_methods``), with a variogram fitted to the
                 event's radar fields; beyond 60 km from every link the radar is kept
                 (mergeplg main's ``max_distance`` / ``fill_radar``).
``pws_idw``      quality-controlled PWS (``core.opensense.pws_qc``) by IDW within 20 km.

The nearby-link retrieval gives 15-minute totals. On OpenMRG (5-min radar) each 15-min
link total is spread evenly over its three 5-min steps, so the link maps change every
15 minutes - which limits what a 5-min nowcast of them can show (see the README).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from core.cml.estimators import study_estimator
from core.cml.link_qc import QCConfig, metadata_qc, timeseries_qc
from core.geo import Grid, to_local_xy
from core.maps.idw import idw_map
from core.opensense.networks import NETWORKS as SOURCES

from .data import grid_for, points
from .settings import CML_IDW_KM, NETWORKS

log = logging.getLogger(__name__)

IDW_NNEAR = 12
PWS_RADIUS_KM = 20
MIN_AVAILABILITY = 0.6
MERGE_MAX_DISTANCE_M = 60_000.0


def link_rain(network: str, start, end, spinup: str = "24h") -> xr.DataArray:
    """Per-link rain (mm per radar step, interval-ending), ``(link, time)``, sublinks averaged."""
    s = NETWORKS[network]
    net = SOURCES[network]
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    # OpenRainER's 1-min record misses whole minutes (~12% of a month, a quarter of some
    # hours), so the default 80% availability over a few hours rejects every link
    cfg = QCConfig(domain=net.domain.pad(0.05), min_availability=MIN_AVAILABILITY)
    keep, _ = metadata_qc(net.links_table(), cfg)
    links = net.links(t0 - pd.Timedelta(spinup), t1, keep)
    keep2, _ = timeseries_qc(links.sel(time=slice(t0, t1)), cfg)
    links = links.sel(link=keep2)
    rate = study_estimator("nearby", t0 - pd.Timedelta(spinup), t1).estimate(links)["rain"]   # mm/h, 15 min
    amount = rate * 0.25                                                  # mm per 15 min
    # one value per physical link: mean of its sublinks
    amount = amount.groupby("cml_id").mean("link", skipna=True)
    first = pd.DataFrame({c: links[c].values for c in
                          ["cml_id", "site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon",
                           "mid_lat", "mid_lon"]}).drop_duplicates("cml_id").set_index("cml_id")
    first = first.loc[amount.cml_id.values]
    amount = amount.rename(cml_id="link").assign_coords(
        {c: ("link", first[c].to_numpy()) for c in first.columns})
    if s.step_min != 15:                                  # spread each 15-min total evenly
        n = 15 // s.step_min
        times = pd.DatetimeIndex(amount.time.values)
        fine = pd.DatetimeIndex(np.concatenate([times - pd.Timedelta(minutes=s.step_min * k)
                                                for k in range(n)])).sort_values()
        amount = amount.reindex(time=fine, method="bfill", tolerance=pd.Timedelta("15min")) / n
    out = amount.sel(time=slice(t0, t1)).astype("float32")
    out.attrs = {"units": "mm", "retrieval": "nearby links (core.cml.estimators.NearbyLinks)",
                 "n_links": int(out.sizes["link"])}
    return out


def cml_map(link_amount: xr.DataArray, grid: Grid, radius_km: float) -> xr.DataArray:
    m = idw_map(link_amount, grid, power=2.0, radius_m=radius_km * 1000.0, nnear=IDW_NNEAR)
    return m.fillna(0.0).rename(f"cml_idw{radius_km:g}")


def merged_map(radar: xr.DataArray, link_amount: xr.DataArray) -> xr.DataArray:
    from core.maps.mergeplg_methods import Merger, fit_radar_variogram
    times = np.intersect1d(radar.time.values, link_amount.time.values)
    r = radar.sel(time=times)
    filled = r.fillna(0.0)
    vario = fit_radar_variogram(filled)
    m = Merger(filled, links=link_amount.sel(time=times), variogram=vario)
    out = m.field("okrig_add", name="merged")
    # mergeplg main's rule (max_distance 60 km, fill_radar): beyond 60 km from every link
    # the kriged difference is the links' mean difference, not information - keep the radar
    g = Grid(r.lat.values, r.lon.values)
    glat, glon = g.mesh()
    lat0, lon0 = float(np.mean(g.lat)), float(np.mean(g.lon))
    gx, gy = to_local_xy(glat.ravel(), glon.ravel(), lat0, lon0)
    lx, ly = to_local_xy(link_amount.mid_lat.values, link_amount.mid_lon.values, lat0, lon0)
    from scipy.spatial import cKDTree
    far = (cKDTree(np.c_[lx, ly]).query(np.c_[gx, gy])[0] > MERGE_MAX_DISTANCE_M).reshape(g.shape)
    out = out.where(~far, r).where(np.isfinite(r))
    out.attrs.update(variogram=str({k: v for k, v in vario.items()}), max_distance_m=MERGE_MAX_DISTANCE_M)
    return out


def pws_map(network: str, start, end, grid: Grid) -> xr.DataArray | None:
    s = NETWORKS[network]
    if s.pws is None:
        return None
    p = points(network, start, end, s.pws)
    try:
        from core.opensense import pws_qc
        lat0, lon0 = float(p.lat.mean()), float(p.lon.mean())
        x, y = to_local_xy(p.lat.values, p.lon.values, lat0, lon0)
        ds = xr.Dataset({"rainfall": (("id", "time"), p.values)},
                        coords={"id": p.station.values, "time": p.time.values,
                                "lat": ("id", p.lat.values), "lon": ("id", p.lon.values),
                                "x": ("id", x), "y": ("id", y)})
        qc = pws_qc.flag(ds, step=f"{s.step_min}min")
        p = p.copy(data=qc.rainfall_qc.transpose("id", "time").values)
    except Exception as exc:                      # QC needs enough neighbours; keep raw
        log.warning("PWS QC skipped on %s: %r", network, exc)
    da = p.rename(station="link").assign_coords(mid_lat=("link", p.lat.values),
                                                mid_lon=("link", p.lon.values))
    m = idw_map(da, grid, power=2.0, radius_m=PWS_RADIUS_KM * 1000.0, nnear=IDW_NNEAR)
    return m.fillna(0.0).rename("pws_idw")


def build(network: str, start, end, radar: xr.DataArray) -> dict:
    """All products of a network for one window, ``{name: (time, lat, lon) mm per step}``,
    on the radar's times; the radar's no-data cells are NaN in every product."""
    grid = grid_for(network)
    valid = np.isfinite(radar)
    out = {"radar": radar}
    try:
        links = link_rain(network, start, end)
        for d in CML_IDW_KM:
            out[f"cml_idw{d}"] = cml_map(links, grid, d).reindex(time=radar.time.values)
        out["merged"] = merged_map(radar, links).reindex(time=radar.time.values)
        out["_links"] = links
    except Exception as exc:
        log.exception("link products failed on %s %s: %r", network, start, exc)
    pws = pws_map(network, start, end, grid)
    if pws is not None:
        out["pws_idw"] = pws.reindex(time=radar.time.values)
    for k, v in out.items():
        if k not in ("radar", "_links"):
            out[k] = v.where(valid).astype("float32")
    return out
