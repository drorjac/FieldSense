"""RAINLINK over the Dutch network for summer 2012, scored at the KNMI automatic gauges.

Steps (each cached under ``dataset/open_datasets/_netherlands/``):

1. ``retrieve``: monthly OpenSense files -> usable sub-links -> RAINLINK's chain
   (``core.opensense.netherlands.rainlink_retrieval``) -> 15-min rain per sub-link.
2. ``hourly``: hourly totals per sub-link (3 of 4 steps needed), averaged over the
   sub-links of a path (as RAINLINK does before mapping); KNMI hourly gauges.
3. ``score``: (a) path level - each path against the nearest gauge within 5 km of its
   midpoint; (b) map level - IDW (power 2, 10 km) and the wet-masked IDW of the hourly
   path totals, at the centre of the 0.02 deg cell holding each gauge. Hourly and daily.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid, haversine_m
from core.maps import wet_area as wa
from core.maps.idw import accumulate, idw_map
from core.maps.scores import scores
from core.opensense import netherlands as nl

from .settings import (CACHE, END, GAUGE_KM, GRID_RES, IDW, MIN_COVERAGE, MIN_KM, NL, P_CUT,
                       POL, RESULTS_DIR, SPINUP, START, WET_MM)

log = logging.getLogger(__name__)

RAIN_15MIN = CACHE / "rain_15min.nc"
LINKS_HOURLY = CACHE / "links_hourly.nc"
GAUGES_HOURLY = CACHE / "gauges_hourly.nc"
MAPS_AT_GAUGES = CACHE / "maps_at_gauges.nc"
COLS = ["n", "mean_ref", "mean_est", "rel_bias", "nrmse", "cv", "corr", "r2", "pod", "far", "csi"]


def score_row(est, ref) -> dict:
    """``core.maps.scores.scores`` plus Overeem et al.'s (2016) CV and r^2.

    CV is the standard deviation of the residuals over the mean of the reference, r^2
    the squared Pearson correlation: the two numbers RAINLINK's papers report.
    """
    e = np.asarray(est, float).ravel()
    r = np.asarray(ref, float).ravel()
    ok = np.isfinite(e) & np.isfinite(r)
    s = scores(e[ok], r[ok], WET_MM)
    if s["n"] == 0:
        return {k: np.nan for k in COLS} | {"n": 0}
    s["cv"] = float(np.std(e[ok] - r[ok]) / s["mean_ref"]) if s["mean_ref"] > 0 else np.nan
    s["r2"] = s["corr"] ** 2
    return {k: s.get(k, np.nan) for k in COLS}


# ------------------------------------------------------------------ 0. availability

def availability() -> pd.DataFrame:
    """Rows per month of the whole 2011-2015 file and sub-links per 15 min (median day).

    The basis for choosing the study summer; one pass over the zip (~6 min).
    """
    s = nl.rows_per_day()
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="D"), fill_value=0)
    m = s.resample("MS").agg(["sum", "median", lambda x: int((x == 0).sum())])
    m.columns = ["rows", "median_rows_per_day", "days_without_data"]
    m["sublinks_per_step"] = (m.median_rows_per_day / 96).round().astype(int)
    m.index = m.index.strftime("%Y-%m")
    m.index.name = "month"
    return m.reset_index()


# ------------------------------------------------------------------ 1. retrieval

def retrieve(force: bool = False) -> xr.Dataset:
    """15-min RAINLINK rain per sub-link, spin-up day included; cached."""
    if RAIN_15MIN.exists() and not force:
        return xr.open_dataset(RAIN_15MIN)
    ds = nl.open_months(START - SPINUP, END)
    sub_all = nl.sublinks(ds)
    sub = nl.usable(sub_all)
    log.info("%d paths, %d sub-links, %d usable; %d steps", ds.sizes["cml_id"],
             sub_all.sizes["cml_id"], sub.sizes["cml_id"], sub.sizes["time"])
    out = nl.rainlink_retrieval(sub, pol=POL)
    keep = ["path", "rainlink_id", "frequency", "length", "vendor", "mid_lat", "mid_lon",
            "site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"]
    res = xr.Dataset({"rain": out.rain.astype("float32"), "wet": out.wet.astype("float32")},
                     coords={c: ("cml_id", sub[c].values) for c in keep})
    res.attrs = {"n_paths_in_data": ds.sizes["cml_id"],
                 "n_sublinks_in_data": sub_all.sizes["cml_id"],
                 "n_sublinks_usable": sub.sizes["cml_id"]}
    RAIN_15MIN.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ("rain", "wet")}
    res.to_netcdf(RAIN_15MIN, encoding=enc)
    return res


# ------------------------------------------------------------------ 2. hourly series

def hourly(force: bool = False) -> tuple[xr.DataArray, xr.DataArray]:
    """``(links, gauges)``: hourly path totals (link, time) and KNMI hourly (station, time), mm."""
    if LINKS_HOURLY.exists() and GAUGES_HOURLY.exists() and not force:
        return xr.open_dataarray(LINKS_HOURLY).load(), xr.open_dataarray(GAUGES_HOURLY).load()
    rain = retrieve().rain.load()
    h = accumulate(rain, "1h", min_coverage=MIN_COVERAGE)
    h = h.sel(time=slice(START + pd.Timedelta("1h"), END))
    links = nl.path_rain(h).astype("float32")
    links.attrs = {"units": "mm", "long_name": "hourly path rain (mean of sub-links)",
                   "time_label": "end of the hour (UTC)"}
    gauges = nl.knmi_hourly(START, END)
    links.to_netcdf(LINKS_HOURLY)
    gauges.to_netcdf(GAUGES_HOURLY)
    return links, gauges


def daily(h: xr.DataArray, other: xr.DataArray, min_hours: int = 20):
    """Daily totals of two hourly series over their joint valid hours (>= ``min_hours``).

    A day is the 24 hours ending 01:00 .. 24:00 UTC. Both totals are over the same
    hours, so a gap in either series does not bias the comparison.
    """
    joint = h.notnull() & other.notnull()
    def day_sum(x):
        return x.where(joint).resample(time="1D", closed="right", label="left").sum(min_count=1)
    n = joint.resample(time="1D", closed="right", label="left").sum()
    ok = n >= min_hours
    return day_sum(h).where(ok), day_sum(other).where(ok)


# ------------------------------------------------------------------ 3a. path level

def gauge_pairs(links: xr.DataArray, gauges: xr.DataArray, max_km: float = GAUGE_KM) -> pd.DataFrame:
    """Each path's nearest KNMI gauge, kept when within ``max_km`` of the path midpoint."""
    d = haversine_m(links.mid_lat.values[:, None], links.mid_lon.values[:, None],
                    gauges.lat.values[None, :], gauges.lon.values[None, :]) / 1000
    j = d.argmin(axis=1)
    df = pd.DataFrame({"link": links.link.values, "station": gauges.station.values[j],
                       "name": gauges.station_name.values[j], "dist_km": d[np.arange(d.shape[0]), j],
                       "length_km": links.length.values})
    return df[df.dist_km <= max_km].reset_index(drop=True)


def path_level() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(pooled, per_pair)``: hourly and daily scores of paths vs their nearest gauge."""
    links, gauges = hourly()
    pairs = gauge_pairs(links, gauges)
    est = links.sel(link=xr.DataArray(pairs.link.values, dims="pair"))
    ref = gauges.sel(station=xr.DataArray(pairs.station.values, dims="pair"))
    est = est.drop_vars([c for c in est.coords if c not in ("time",)])
    ref = ref.drop_vars([c for c in ref.coords if c not in ("time",)])
    est_d, ref_d = daily(est, ref)
    # all pairs (RAINLINK as published), and the paths of at least MIN_KM: a sensitivity
    # check, since below a few hundred metres a 1 dB step is tens of mm/h
    rows = []
    for subset, keep in (("all paths", np.ones(len(pairs), bool)),
                         (f"paths >= {MIN_KM:g} km", pairs.length_km.values >= MIN_KM)):
        k = xr.DataArray(keep, dims="pair")
        for scale, e, r in (("hourly", est, ref), ("daily", est_d, ref_d)):
            rows.append({"pairs_used": subset, "scale": scale, "pairs": int(keep.sum()),
                         "gauges": pairs.station[keep].nunique(),
                         **score_row(e.where(k), r.where(k))})
    pooled = pd.DataFrame(rows)
    per = []
    for i, p in pairs.iterrows():
        e, r = est.isel(pair=i), ref.isel(pair=i)
        ok = e.notnull() & r.notnull()
        s = score_row(e, r)
        per.append({**p.to_dict(), "n_hours": s["n"], "total_link": float(e.where(ok).sum()),
                    "total_gauge": float(r.where(ok).sum()), "corr": s["corr"]})
    return pooled, pd.DataFrame(per)


# ------------------------------------------------------------------ 3b. map level

def grid() -> Grid:
    return Grid.from_domain(NL, GRID_RES)


def gauge_cells(gauges: xr.DataArray, g: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Index (lat, lon) of the grid cell holding each gauge."""
    ilat = np.abs(g.lat[None, :] - gauges.lat.values[:, None]).argmin(axis=1)
    ilon = np.abs(g.lon[None, :] - gauges.lon.values[:, None]).argmin(axis=1)
    return ilat, ilon


def maps_at_gauges(force: bool = False) -> xr.Dataset:
    """IDW and wet-masked IDW maps of the hourly path totals, at the gauges' cells.

    Each map is computed on the full 0.02 deg grid over the Netherlands, one day at a
    time, and read at the cell holding each gauge (the gauges are never mapped).
    """
    if MAPS_AT_GAUGES.exists() and not force:
        return xr.open_dataset(MAPS_AT_GAUGES).load()
    links, gauges = hourly()
    g = grid()
    ilat, ilon = gauge_cells(gauges, g)
    ia, io = xr.DataArray(ilat, dims="station"), xr.DataArray(ilon, dims="station")
    out = {"idw": [], "masked": []}
    days = pd.date_range(START, END - pd.Timedelta("1D"), freq="1D")
    for d in days:
        chunk = links.sel(time=slice(d + pd.Timedelta("1h"), d + pd.Timedelta("1D")))
        m = idw_map(chunk, g, **IDW)
        mm = wa.masked_idw(chunk, g, WET_MM, P_CUT, **IDW)
        out["idw"].append(m.isel(lat=ia, lon=io).drop_vars(["lat", "lon"]))
        out["masked"].append(mm.isel(lat=ia, lon=io).drop_vars(["lat", "lon"]))
        log.info("maps %s", d.date())
    ds = xr.Dataset({k: xr.concat(v, "time").transpose("station", "time") for k, v in out.items()})
    ds = ds.assign_coords(station=gauges.station.values, lat=("station", gauges.lat.values),
                          lon=("station", gauges.lon.values),
                          station_name=("station", gauges.station_name.values),
                          gauge=(("station", "time"), gauges.transpose("station", "time").values))
    ds.to_netcdf(MAPS_AT_GAUGES)
    return ds


def map_level() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(pooled, per_station)``: hourly and daily scores of each map at all KNMI gauges."""
    ds = maps_at_gauges()
    rows, per = [], []
    for name in ("idw", "masked"):
        est, ref = ds[name], ds.gauge
        est_d, ref_d = daily(est, ref)
        for scale, e, r in (("hourly", est, ref), ("daily", est_d, ref_d)):
            covered = int((e.notnull() & r.notnull()).any("time").sum())
            rows.append({"map": name, "scale": scale, "gauges_covered": covered,
                         **score_row(e, r)})
        for s in ds.station.values:
            e, r = est.sel(station=s), ref.sel(station=s)
            ok = e.notnull() & r.notnull()
            per.append({"map": name, "station": int(s), "name": str(ds.station_name.sel(station=s).item()),
                        "n_hours": int(ok.sum()), "total_map": float(e.where(ok).sum()),
                        "total_gauge": float(r.where(ok).sum()),
                        "corr": score_row(e, r)["corr"]})
    return pd.DataFrame(rows), pd.DataFrame(per)


# ------------------------------------------------------------------ network summary

def network() -> dict:
    """Counts of the network over the study period, for the report."""
    r = retrieve()
    links, gauges = hourly()
    rain = r.rain.sel(time=slice(START + pd.Timedelta("15min"), END))
    per_step = rain.notnull().sum("cml_id")
    return {
        "period": f"{START.date()} .. {(END - pd.Timedelta('1D')).date()} (interval ends in "
                  f"({START}, {END}])",
        "paths_in_data": int(r.attrs["n_paths_in_data"]),
        "sublinks_in_data": int(r.attrs["n_sublinks_in_data"]),
        "sublinks_usable": int(r.attrs["n_sublinks_usable"]),
        "paths_usable": int(np.unique(r.path.values).size),
        "sublinks_with_rain_estimate_median_per_step": int(per_step.median()),
        "paths_with_hourly_value_median_per_hour": int(links.notnull().sum("link").median()),
        "wet_fraction_of_estimates": round(float(r.wet.sel(time=rain.time).mean()), 3),
        "knmi_gauges_with_data": int(gauges.sizes["station"]),
        "grid": f"{GRID_RES} deg, {grid().shape[0]} x {grid().shape[1]} cells",
    }


def run_all() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pooled, per_pair = path_level()
    pooled.to_csv(RESULTS_DIR / "path_level.csv", index=False)
    per_pair.to_csv(RESULTS_DIR / "path_pairs.csv", index=False)
    mpooled, per_station = map_level()
    mpooled.to_csv(RESULTS_DIR / "map_level.csv", index=False)
    per_station.to_csv(RESULTS_DIR / "map_stations.csv", index=False)
    (RESULTS_DIR / "network.json").write_text(json.dumps(network(), indent=2) + "\n")
