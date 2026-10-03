"""
Inputs of the intercomparison, as ``1_data_preparation`` and ``2_cml_processing`` build them.

Per network, three files under ``PREPARED/<network>/``:

``radar.nc``     hour-ending radar totals (mm) on the radar's own grid, ``rainfall_amount``
                 (time, y, x) with ``lon``/``lat`` and UTM ``x_grid``/``y_grid``
``cml.nc``       hour-ending link totals (mm), ``R_acc`` (cml_id, time) with UTM site and
                 midpoint coordinates, from :mod:`core.opensense.intercomparison_chain`
``gauges.nc``    hour-ending gauge totals (mm), ``rainfall_amount`` (id, time), ``x``/``y``

OpenMRG: the 10-s CML record taken to 1 min with the first valid sample of each minute
(the OpenSense transformer's ``resample("1min").first()``), the 5-min radar, 10 city
gauges (1 min, summed to 15 min) and the SMHI gauge (15 min). OpenRainER: 1-min CML,
15-min radar and gauges, June-August 2022, links and gauges present in all three months.
The CML chain runs in batches of links; every step of it is per link, so the batches
change nothing.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from core import data_paths as dp
from core.opensense import intercomparison_chain as ic
from radar_adjustment.settings import MONTHS, PREPARED, RAD_FREQ

log = logging.getLogger(__name__)
BATCH = 40          # links per chain batch


def _save(ds: xr.Dataset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)


def _run_chain(ds_cml: xr.Dataset, radar_along: xr.DataArray, rad_freq: int):
    parts, removed = [], []
    ids = ds_cml.cml_id.values
    for k in range(0, ids.size, BATCH):
        sel = ids[k:k + BATCH]
        h, r = ic.chain(ds_cml.sel(cml_id=sel), radar_along.sel(cml_id=sel), rad_freq)
        parts.append(h)
        removed += r
        log.info("chain: links %d-%d of %d", k, k + sel.size, ids.size)
    return xr.concat(parts, "cml_id"), removed


def _cml_out(hourly: xr.DataArray, ds_cml: xr.Dataset) -> xr.Dataset:
    coords = ds_cml[["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon", "length"]]
    coords = coords.sel(cml_id=hourly.cml_id).reset_coords()
    out = xr.Dataset({"R_acc": hourly.transpose("cml_id", "time").astype("float32")})
    out = out.assign_coords(cml_id=out.cml_id.values.astype(str))
    for c in coords.data_vars:
        out.coords[c] = ("cml_id", coords[c].values)
    return ic.add_link_xy(out)


# ---------------------------------------------------------------------------
# OpenMRG
# ---------------------------------------------------------------------------
def openmrg_cml_1min() -> xr.Dataset:
    """(cml_id, sublink_id, time) TSL/RSL at 1 min, frequency in MHz and length in m."""
    meta = pd.read_csv(dp.data_path(dp.OPENMRG_CML_META))
    edges = pd.date_range("2015-06-01", "2015-09-02", freq="10D").append(
        pd.DatetimeIndex(["2015-09-02"]))
    parts = []
    with xr.open_dataset(dp.data_path(dp.OPENMRG_CML)) as raw:
        raw = raw.sel(time=slice("2015-06-01", None))
        for t0, t1 in zip(edges[:-1], edges[1:]):
            chunk = raw.sel(time=slice(t0, t1 - pd.Timedelta("1ns"))).load()
            if chunk.sizes["time"]:
                parts.append(chunk.resample(time="1min").first())
    ds = xr.concat(parts, "time").drop_duplicates("time")
    ds = ds.drop_isel(time=0)               # as the notebook: the first minute is dropped
    m = meta.set_index("Sublink").loc[ds.sublink.values]
    ds = ds.assign_coords(cml_id=("sublink", m.Link.values),
                          sublink_id=("sublink", np.where(m.Direction.values == "A", *ic.SUBLINKS)))
    ds = ds.set_index(sublink=["cml_id", "sublink_id"]).unstack("sublink")
    first = meta.sort_values("Sublink").drop_duplicates("Link").set_index("Link").loc[ds.cml_id.values]
    for col, name in [("NearLatitude_DecDeg", "site_0_lat"), ("NearLongitude_DecDeg", "site_0_lon"),
                      ("FarLatitude_DecDeg", "site_1_lat"), ("FarLongitude_DecDeg", "site_1_lon")]:
        ds.coords[name] = ("cml_id", first[col].values)
    ds.coords["length"] = ("cml_id", first.Length_km.values * 1000)
    per = meta.set_index(["Link", "Direction"])
    fr, pol = [], []
    for c in ds.cml_id.values:
        fr.append([per.loc[(c, d), "Frequency_GHz"] * 1000 for d in ("A", "B")])
        pol.append(["v" if str(per.loc[(c, d), "Polarization"]).lower().startswith("v") else "h"
                    for d in ("A", "B")])
    ds.coords["frequency"] = (("cml_id", "sublink_id"), np.asarray(fr, float))
    ds.coords["polarization"] = (("cml_id", "sublink_id"), np.asarray(pol))
    return ds.transpose("cml_id", "sublink_id", "time")


def openmrg_radar_5min() -> xr.Dataset:
    with xr.open_dataset(dp.data_path(dp.OPENMRG_RADAR)) as ds:
        ds = ds.load()
    out = xr.Dataset({"rainfall_amount": ic.openmrg_radar_rate(ds.data) * (5 / 60)})
    out.coords["lon"] = (("y", "x"), ds.lon.values.astype(float))
    out.coords["lat"] = (("y", "x"), ds.lat.values.astype(float))
    out.coords["x_grid"], out.coords["y_grid"] = ic.project(out.lon, out.lat)
    return out


def openmrg_gauges(t_first, t_last) -> xr.Dataset:
    folder = dp.data_path(dp.OPENMRG_GAUGES)
    city = pd.read_csv(folder / "city" / "CityGauges-2015JJA.csv", index_col=0, parse_dates=True)
    cmeta = pd.read_csv(folder / "city" / "CityGauges-metadata.csv", index_col=0, encoding="utf-8-sig")
    city.index = city.index.tz_localize(None)
    ds_city = xr.Dataset({"rainfall_amount": (("id", "time"), city.T.values.astype(float))},
                         coords={"id": cmeta.index.values, "time": city.index.values,
                                 "lon": ("id", cmeta.Longitude_DecDeg.values),
                                 "lat": ("id", cmeta.Latitude_DecDeg.values)})
    ds_city = ds_city.resample(time="15min", label="right").sum()
    smhi = pd.read_csv(folder / "smhi" / "GbgA-71420-2015JJA.csv", index_col=0, parse_dates=True)
    smhi.index = smhi.index.tz_localize(None)
    ds_smhi = xr.Dataset({"rainfall_amount": (("id", "time"), [smhi.Pvol_mm.values.astype(float)])},
                         coords={"id": ["SMHI"], "time": smhi.index.values,
                                 "lon": ("id", [11.9924]), "lat": ("id", [57.7156])})
    g = xr.concat([ds_city, ds_smhi], dim="id").sel(time=slice(t_first, t_last))
    g = ic.hourly_from_mean(g, 4).drop_isel(time=-1)
    g = g.assign_coords(id=g.id.values.astype(str))
    g.coords["x"], g.coords["y"] = ic.project(g.lon, g.lat)
    return g


def prepare_openmrg(force: bool = False):
    out = PREPARED / "openmrg"
    if (out / "cml.nc").exists() and not force:
        return out
    rad5 = openmrg_radar_5min()
    log.info("openmrg: reading CML")
    cml = openmrg_cml_1min()
    along = ic.radar_along_links(rad5.rainfall_amount, cml, rad5.lon.values, rad5.lat.values)
    hourly, removed = _run_chain(cml, along, RAD_FREQ["openmrg"])
    hourly = hourly.drop_isel(time=-1)
    radar = ic.hourly_from_mean(rad5[["rainfall_amount"]], 12)
    radar["rainfall_amount"] = ic.threshold_radar(radar.rainfall_amount)
    radar = radar.drop_isel(time=-1)
    gauges = openmrg_gauges(cml.time.values[0], cml.time.values[-1])
    _save(radar.astype("float32"), out / "radar.nc")
    _save(gauges, out / "gauges.nc")
    c = _cml_out(hourly, cml)
    c.attrs["removed_by_qc"] = ",".join(map(str, removed))
    _save(c, out / "cml.nc")
    return out


# ---------------------------------------------------------------------------
# OpenRainER
# ---------------------------------------------------------------------------
def _openrainer_file(prefix, month):
    from core.opensense.networks import NETWORKS
    archive = {"CML": "CML.tar", "AWS": "AWS.tar", "RADrain": "RADrain.tar"}[prefix]
    return NETWORKS["openrainer"]._file(prefix, pd.Timestamp(month), archive)


def prepare_openrainer(force: bool = False):
    out = PREPARED / "openrainer"
    if (out / "cml.nc").exists() and not force:
        return out
    months = MONTHS["openrainer"]
    # gauges: stations in every month, hourly mean x 4, first and last hour dropped
    ids, parts = [], []
    for m in months:
        with xr.open_dataset(_openrainer_file("AWS", m)) as t:
            ids.append(set(t.id.values))
    common = sorted(set.intersection(*ids))
    for m in months:
        with xr.open_dataset(_openrainer_file("AWS", m)) as t:
            parts.append(t.rainfall_amount.sel(id=common).load())
    g = xr.concat(parts, "time").rename({"longitude": "lon", "latitude": "lat"})
    g = ic.hourly_from_mean(g, 4).drop_isel(time=-1).drop_isel(time=0)
    g = g.to_dataset(name="rainfall_amount")
    g = g.assign_coords(id=g.id.values.astype(str))
    g.coords["x"], g.coords["y"] = ic.project(g.lon, g.lat)
    # links in every month
    ids = []
    for m in months:
        with xr.open_dataset(_openrainer_file("CML", m)) as t:
            ids.append(set(t.cml_id.values))
    common = sorted(set.intersection(*ids))
    cml_parts, along_parts, radar_parts = [], [], []
    for m in months:
        with xr.open_dataset(_openrainer_file("CML", m)) as t:
            c = t.sel(cml_id=common).load()
        with xr.open_dataset(_openrainer_file("RADrain", m)) as r:
            rr = r.rainfall_amount.load()
        lon2, lat2 = np.meshgrid(rr.lon.values, rr.lat.values)
        along_parts.append(ic.radar_along_links(rr, c, lon2, lat2))
        radar_parts.append(rr.rename({"lon": "x", "lat": "y"}))
        cml_parts.append(c)
        log.info("openrainer: read %s", m)
    cml = xr.concat(cml_parts, "time")
    cml["polarization"] = cml.polarization.astype(str)
    along = xr.concat(along_parts, "time")
    radar = xr.concat(radar_parts, "time").to_dataset(name="rainfall_amount")
    full = pd.date_range(radar.time.min().values, radar.time.max().values, freq="15min")
    radar = radar.reindex(time=full)
    along = along.reindex(time=pd.date_range(along.time.min().values, along.time.max().values,
                                             freq="15min"))
    hourly, removed = _run_chain(cml.transpose("cml_id", "sublink_id", "time"), along,
                                 RAD_FREQ["openrainer"])
    hourly = hourly.drop_isel(time=-1).drop_isel(time=0)
    radar = ic.hourly_from_mean(radar, 4)
    radar["rainfall_amount"] = ic.threshold_radar(radar.rainfall_amount)
    radar = radar.drop_isel(time=-1).drop_isel(time=0)
    lon2, lat2 = np.meshgrid(radar.x.values, radar.y.values)
    radar.coords["lon"] = (("y", "x"), lon2.astype(float))
    radar.coords["lat"] = (("y", "x"), lat2.astype(float))
    radar.coords["x_grid"], radar.coords["y_grid"] = ic.project(radar.lon, radar.lat)
    _save(radar.astype("float32"), out / "radar.nc")
    _save(g, out / "gauges.nc")
    c = _cml_out(hourly, cml)
    c.attrs["removed_by_qc"] = ",".join(map(str, removed))
    _save(c, out / "cml.nc")
    return out


def load(network: str, month: str | None = None):
    """``(radar, cml, gauges)`` prepared for ``network``: DataArrays of hourly mm.

    ``month`` ("YYYY-MM") reads only that month's hours."""
    d = PREPARED / network
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, var in (("radar", "rainfall_amount"), ("cml", "R_acc"), ("gauges", "rainfall_amount")):
            with xr.open_dataset(d / f"{name}.nc") as ds:
                da = ds[var]
                if month is not None:
                    t = pd.DatetimeIndex(da.time.values)
                    da = da.isel(time=np.flatnonzero(t.strftime("%Y-%m") == month))
                da = da.load()
            out.append(da.transpose("id", "time") if name == "gauges" else da)
    return tuple(out)


PREPARE = {"openmrg": prepare_openmrg, "openrainer": prepare_openrainer}
