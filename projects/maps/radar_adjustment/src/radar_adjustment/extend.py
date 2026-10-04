"""
Beyond the intercomparison, on its inputs and scoring (all with mergeplg main):

``stations``   weather stations as adjusters. Gothenburg: the OpenMRG2 Netatmo PWS,
               quality-controlled with pypwsqc (``core.opensense.pws_qc``), hourly by the
               same mean-times-steps rule as the gauges; the radar adjusted with the
               links, the PWS, or both, scored at the 11 official gauges, which are used
               by none of them.
``mapping``    maps with no radar: links (and in Gothenburg PWS) interpolated by IDW and
               by ordinary kriging of points or of whole lines (``mergeplg.interpolate``);
               and the DWD's RADOLAN adjustment (``MergeRADOLAN``).
``variogram``  the best kriging adjustment with mergeplg's default variogram (range 5 km),
               the study's (30 km), one fitted to each network's radar, and the nugget
               estimated from the link geometry (``c0_within``).
``newyork``    New York City, on the 10 storms of ``projects/maps/multisensor``: MRMS
               radar-only adjusted with NYC Mesh links (the RNN of ``projects/retrieval/rnn_three_networks`` and
               the nearby-link power law), the WU PWS, or both; scored at the 4 ASOS
               gauges, against MRMS Pass 2 (radar corrected by NOAA with gauges) as a
               reference product.

All runs are :class:`~radar_adjustment.adjust.Task` s with their own ``checks`` label,
saved and scored like the intercomparison.
"""

from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from radar_adjustment.adjust import (Task, adjust_series, build_merger, merger_kwargs,
                                     run_tasks)
from radar_adjustment.settings import (DATA_DIR, MONTHS, NNEAR, PREPARED, RESULTS_DIR,
                                       VARIOGRAM)

log = logging.getLogger(__name__)

MAPS = ("idw_map", "okp_map", "okb_map")
STATION_PRODUCTS = ("add_p_idw", "add_b_ok", "ked_b", "mul_b_ok", "radolan")
NYC_PRODUCTS = ("add_p_idw", "add_b_ok", "ked_b", "radolan", "idw_map", "okb_map")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def prepare_pws_openmrg(force: bool = False):
    """QC'd Netatmo PWS over JJA 2015, hour-ending mm (id, time), on the prepared hours."""
    from core.opensense import intercomparison_chain as ic
    from core.opensense import pws_qc
    from core.opensense.networks import NETWORKS

    path = PREPARED / "openmrg" / "pws.nc"
    if path.exists() and not force:
        return path
    raw = NETWORKS["openmrg"].points("2015-06-01", "2015-09-01 00:05", sets=["pws"])["pws"]
    ds = xr.Dataset({"rainfall": raw.rain.rename(station="id")},
                    coords={"lon": ("id", raw.lon.values), "lat": ("id", raw.lat.values)})
    ds.coords["x"], ds.coords["y"] = ic.project(ds.lon, ds.lat)
    qc = pws_qc.flag(ds, step="5min")
    keep = pws_qc.usable(qc, max_missing=0.5, max_flagged=0.05)
    summary = pws_qc.summary(qc)
    rain = qc.rainfall_qc.sel(id=keep)
    hourly = ic.hourly_from_mean(rain, 12)
    # an hour is kept only when at least 80% of its 12 steps passed QC
    n_ok = rain.notnull().resample(time="1h", label="right", closed="right").sum()
    hourly = hourly.where(n_ok >= 10)
    with xr.open_dataset(PREPARED / "openmrg" / "radar.nc") as r:
        hourly = hourly.reindex(time=r.time.values)
    out = hourly.astype("float32").to_dataset(name="rainfall_amount")
    out = out.assign_coords(id=out.id.values.astype(str))
    out.attrs.update(qc="pypwsqc FZ/HI/SO + rate check (core.opensense.pws_qc), usable() stations",
                     n_raw=int(ds.sizes["id"]), n_kept=len(keep))
    out.to_netcdf(path)
    summary.to_csv(RESULTS_DIR / "pws_qc_openmrg.csv")
    return path


def load_pws(network: str, month: str) -> xr.DataArray:
    with xr.open_dataset(PREPARED / network / "pws.nc") as ds:
        da = ds.rainfall_amount
        t = pd.DatetimeIndex(da.time.values)
        return da.isel(time=np.flatnonzero(t.strftime("%Y-%m") == month)).load()


def fit_variogram_xy(rad: xr.DataArray, min_wet_fraction: float = 0.2, wet_mm: float = 0.1,
                     max_lag_m: float = 60_000.0, n_pairs: int = 200_000, seed: int = 0) -> dict:
    """Spherical variogram shape from standardised wet radar hours, on UTM ``x_grid``/``y_grid``.

    The method of ``core.maps.mergeplg_methods.fit_radar_variogram`` for grids given by
    2-D projected coordinates. Sill 1; only the range and the nugget share matter.
    """
    from scipy.optimize import curve_fit

    from core.maps.mergeplg_methods import standardised_semivariogram
    x, y = rad.x_grid.values.ravel(), rad.y_grid.values.ravel()
    R = rad.values.reshape(rad.shape[0], -1)
    lag, gamma, n_wet = standardised_semivariogram(R, x, y, min_wet_fraction, wet_mm, max_lag_m,
                                                   n_pairs, seed)

    def sph(h, nug, rng_):
        s = np.where(h < rng_, 1.5 * h / rng_ - 0.5 * (h / rng_) ** 3, 1.0)
        return nug + (1 - nug) * s

    (nug, r), _ = curve_fit(sph, lag, gamma / gamma[-3:].mean(), p0=[0.1, 20000],
                            bounds=([0, 1000], [0.99, 500_000]))
    return {"sill": 1.0, "range": float(r), "nugget": float(nug), "n_hours": n_wet}


def variograms(networks) -> dict:
    """Fitted per network, cached in results/variograms.json."""
    from radar_adjustment.prepare import load
    path = RESULTS_DIR / "variograms.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    for net in networks:
        if net not in known:
            rad, _, _ = load(net)
            known[net] = fit_variogram_xy(rad)
    path.write_text(json.dumps(known, indent=2))
    return known


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------
def _with_name(da, name):
    return None if da is None else da.rename(name)


OPENMRG_STERE = "+proj=stere +lat_ts=60 +ellps=bessel +lon_0=14 +lat_0=90"


def regular_plane(rad, cmls=None, pts=None, network: str = ""):
    """Grid, links and points in a plane where the radar grid is regular, for RADOLAN.

    mergeplg's RADOLAN code takes the grid from the 1-D ``x``/``y`` coordinates of the
    radar, in the stations' projection (the DWD grid is regular). OpenMRG's composite is
    regular in its polar stereographic projection; the lat/lon grids of OpenRainER and
    New York are regular in an equirectangular projection centred on the domain.
    """
    import pyproj
    if network == "openmrg":
        proj = OPENMRG_STERE
        x1, y1 = rad.x.values.astype(float), rad.y.values.astype(float)
    else:
        lat0, lon0 = float(rad.lat.mean()), float(rad.lon.mean())
        proj = f"+proj=eqc +lat_ts={lat0} +lon_0={lon0} +ellps=WGS84"
        tr0 = pyproj.Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        x1, _ = tr0.transform(rad.lon.values[0, :], np.full(rad.sizes["x"], lat0))
        _, y1 = tr0.transform(np.full(rad.sizes["y"], lon0), rad.lat.values[:, 0])
    tr = pyproj.Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    X, Y = np.meshgrid(x1, y1)
    rad = rad.assign_coords(x=x1, y=y1, x_grid=(("y", "x"), X), y_grid=(("y", "x"), Y))
    if cmls is not None:
        cmls = cmls.copy()
        for s_ in ("0", "1"):
            xx, yy = tr.transform(cmls[f"site_{s_}_lon"].values, cmls[f"site_{s_}_lat"].values)
            cmls = cmls.assign_coords({f"site_{s_}_x": ("cml_id", xx), f"site_{s_}_y": ("cml_id", yy)})
        cmls = cmls.assign_coords(x=(cmls.site_0_x + cmls.site_1_x) / 2, y=(cmls.site_0_y + cmls.site_1_y) / 2)
    if pts is not None:
        xx, yy = tr.transform(pts.lon.values, pts.lat.values)
        pts = pts.assign_coords(x=("id", xx), y=("id", yy))
    return rad, cmls, pts


def make_fields(task: Task, rad, cml, gauges, pws=None) -> np.ndarray:
    """Hourly fields (time, y, x) of an extension product."""
    from mergeplg import interpolate, merge
    cmls = cml if "cml" in task.adjusters else None
    pts = pws if "pws" in task.adjusters else None
    kw = dict(task.kwargs)
    if task.variant in MAPS:
        if task.variant == "idw_map":
            interp = interpolate.InterpolateIDW(ds_grid=rad, ds_cmls=cmls, ds_gauges=pts, nnear=NNEAR,
                                                idw_method="radolan", **kw)
        else:
            interp = interpolate.InterpolateOrdinaryKriging(
                ds_grid=rad, ds_cmls=cmls, ds_gauges=pts, nnear=NNEAR,
                variogram_parameters=dict(VARIOGRAM), full_line=task.variant == "okb_map", **kw)
        out = []
        for t in rad.time.values:
            f = interp(da_cmls=None if cmls is None else cmls.sel(time=t),
                       da_gauges=None if pts is None else pts.sel(time=t))
            out.append(np.asarray(f, "float32").reshape(rad.shape[1:]))
        return np.stack(out)
    if task.variant == "radolan":
        r, cmls, pts = regular_plane(rad.astype("float64"), cmls, pts, task.network)
        merger = merge.MergeRADOLAN(ds_rad=r, ds_cmls=_with_name(cmls, "R"), ds_gauges=_with_name(pts, "R"),
                                    nnear=NNEAR, **kw)
        return np.stack(adjust_series(merger, r, _with_name(cmls, "R"), _with_name(pts, "R"),
                                      call_kwargs={"start_index_in_relevant_stations": 0}, radar_if_empty=True))
    cls, mkw = merger_kwargs(task.variant, "default")
    mkw.update(kw)
    merger = build_merger(cls, mkw, rad, cmls, pts)
    return np.stack(adjust_series(merger, rad, cmls, pts))


def _product(task, rad, cml, g):
    pws = load_pws(task.network, task.month) if "pws" in task.adjusters else None
    if pws is not None:
        pws = pws.sel(time=rad.time)
    return make_fields(task, rad, cml, g, pws)


def _run_ext(task):
    from radar_adjustment.adjust import run_task
    try:
        return run_task(task, product=_product)
    except Exception as e:                       # noqa: BLE001
        log.exception("task failed: %s", task)
        return f"FAILED {task}: {type(e).__name__}: {e}"


def extension_tasks(networks, best_kriging: str = "add_b_ok") -> list[Task]:
    vg = variograms(networks)
    tasks = []
    for net in networks:
        for m in MONTHS[net]:
            # mapping and RADOLAN with the links
            for p in (*MAPS, "radolan"):
                tasks.append(Task("main", net, "mapping", p, m))
            # variogram and nugget
            for label, params in (("vg5km", {"sill": 0.9, "range": 5000.0, "nugget": 0.1}),
                                  ("vgfit", {k: vg[net][k] for k in ("sill", "range", "nugget")})):
                tasks.append(Task("main", net, label, best_kriging, m,
                                  kwargs=(("variogram_parameters", params),)))
            tasks.append(Task("main", net, "c0within", best_kriging, m, kwargs=(("c0_within", True),)))
            if net == "openmrg":
                for adj in ("pws", "cml+pws"):
                    for p in STATION_PRODUCTS:
                        tasks.append(Task("main", net, "stations", p, m, adjusters=adj))
                    for p in MAPS:
                        tasks.append(Task("main", net, "mapping", p, m, adjusters=adj))
    return tasks


# ---------------------------------------------------------------------------
# New York City
# ---------------------------------------------------------------------------
NYC_DIR = DATA_DIR / "newyork"
NYC_EPSG = "EPSG:32618"


def _nyc_inputs():
    """The cached multisensor_maps inputs of the 10 OpenMesh storms, in mergeplg's layout."""
    import poligrain as plg
    from core.data_paths import OUTPUTS
    folders = sorted((OUTPUTS / "_multisensor_maps" / "merging" / "inputs").glob("openmesh_*"))
    for f in folders:
        meta = json.loads((f / "meta.json").read_text())
        radars = {r: xr.open_dataarray(f / f"radar_{i}.nc").load() for i, r in enumerate(meta["radars"])}
        if "radar only" not in radars:
            continue
        points = {p: xr.open_dataarray(f / f"points_{p}.nc").load() for p in meta["points"]}
        links = {m: xr.open_dataarray(f / f"links_{m}.nc").load() for m in meta["links"] if m in ("nearby", "rnn")}

        def grid(da):
            lon2, lat2 = np.meshgrid(da.lon.values, da.lat.values)
            da = da.rename(lat="y", lon="x").assign_coords(lon=(("y", "x"), lon2), lat=(("y", "x"), lat2))
            da.coords["x_grid"], da.coords["y_grid"] = plg.spatial.project_point_coordinates(da.lon, da.lat, NYC_EPSG)
            return da.transpose("time", "y", "x")

        def pts(da):
            da = da.rename(station="id").transpose("id", "time")
            da.coords["x"], da.coords["y"] = plg.spatial.project_point_coordinates(da.lon, da.lat, NYC_EPSG)
            return da

        def lnk(da):
            da = da.rename(link="cml_id").transpose("cml_id", "time")
            for s in ("0", "1"):
                da.coords[f"site_{s}_x"], da.coords[f"site_{s}_y"] = plg.spatial.project_point_coordinates(
                    da[f"site_{s}_lon"], da[f"site_{s}_lat"], NYC_EPSG)
            da.coords["x"] = (da.site_0_x + da.site_1_x) / 2
            da.coords["y"] = (da.site_0_y + da.site_1_y) / 2
            return da.drop_vars([c for c in da.coords if c in ("frequency", "length", "polarization",
                                                               "mid_lat", "mid_lon")])
        yield f.name, {"radar only": grid(radars["radar only"]), "pass2": grid(radars["radar"])}, \
            {k: pts(v) for k, v in points.items()}, {k: lnk(v) for k, v in links.items()}


def run_newyork(retrievals=("rnn", "nearby")) -> pd.DataFrame:
    """Radar-only MRMS adjusted with links / PWS / both, per storm; values at ASOS gauges."""
    import poligrain as plg
    from radar_adjustment.score import metrics
    NYC_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, radars, points, links in _nyc_inputs():
        if "asos" not in points:
            continue
        rad = radars["radar only"].fillna(0)
        asos = points["asos"]
        times = rad.time.values
        gap = plg.spatial.GridAtPoints(da_gridded_data=rad.isel(time=0), da_point_data=asos.isel(time=0),
                                       nnear=1, stat="best")
        out = {"radar only": gap(da_gridded_data=rad, da_point_data=asos),
               "MRMS Pass 2": gap(da_gridded_data=radars["pass2"].reindex(time=times), da_point_data=asos)}
        pws = points["pws"].reindex(time=times)
        for ret in retrievals:
            if ret not in links:
                continue
            cml = links[ret].reindex(time=times)
            for adj in ("cml", "pws", "cml+pws"):
                if ret != retrievals[0] and adj == "pws":
                    continue
                for p in NYC_PRODUCTS:
                    t = Task("main", "openmesh", "newyork", p, name, adjusters=adj)
                    cache = NYC_DIR / f"{name}_{p}_{adj}_{ret}.nc"
                    if cache.exists():
                        da = xr.open_dataarray(cache).load()
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                f = make_fields(t, rad, cml, None, pws)
                            except Exception as e:      # noqa: BLE001
                                log.warning("%s %s %s %s: %s", name, p, adj, ret, e)
                                continue
                        da = gap(da_gridded_data=rad.copy(data=f), da_point_data=asos)
                        da.to_netcdf(cache)
                    label = f"{p} [{adj if adj == 'pws' else adj.replace('cml', 'links ' + ret)}]"
                    out[label] = da
        ref = asos.transpose("id", "time")
        for label, da in out.items():
            rows.append({"event": name, "product": label,
                         "ref": ref.values.ravel(), "est": da.transpose("id", "time").values.ravel()})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # the same station-hours for every product: events that have every product, and
    # within them the station-hours where every product and the gauge have a value
    n_products = df["product"].nunique()
    complete = df.groupby("event")["product"].nunique() == n_products
    excluded = sorted(complete.index[~complete])
    df = df[df.event.isin(complete.index[complete])]
    est, ref = {}, []
    for ev, g in df.groupby("event", sort=False):
        r = g.ref.iloc[0]
        ok = np.isfinite(r)
        for e in g.est:
            ok &= np.isfinite(e)
        ref.append(r[ok])
        for p, e in zip(g["product"], g.est):
            est.setdefault(p, []).append(e[ok])
    ref = np.concatenate(ref)
    table = pd.DataFrame([{"product": p, **metrics(ref, np.concatenate(v))} for p, v in est.items()])
    table = table.sort_values("rmse")
    table.attrs["excluded"] = excluded
    table.to_csv(RESULTS_DIR / "newyork_asos.csv", index=False)
    (RESULTS_DIR / "newyork_asos_events.json").write_text(json.dumps(
        {"events": sorted(set(df.event)), "excluded_incomplete": excluded,
         "station_hours": int(ref.size)}, indent=2))
    return table


def run_extensions(networks=None, workers: int = 6):
    networks = networks or ["openmrg", "openrainer"]
    if "openmrg" in networks:
        prepare_pws_openmrg()
    nets = [n for n in networks if n in MONTHS]
    run_tasks(extension_tasks(nets), workers, func=_run_ext)
    print(run_newyork())
