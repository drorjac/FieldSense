"""Radar and gauges at their native time step, on the nowcasting grid.

``core.opensense.networks`` gives the radar as hour-ending totals; nowcasting needs the
native step. The readers here use the same files and conventions (paths from
``core.data_paths``, time labels from ``networks.LABELS``, OpenMRG's Z-R from the file's
attributes) and only change the time resolution and the grid:

=============  ==================================================  =========
network        radar                                               step
=============  ==================================================  =========
OpenRainER     ARPAE-SIMC composite, 15-min rain depth (mm)          15 min
OpenMRG        SMHI composite, 5-min reflectivity -> Z = 200 R^1.5    5 min
=============  ==================================================  =========

Each field is in mm per step, interval-ending, on the square grid of
:func:`~os_nowcasting.grid.square_grid`. Event windows are cached under
``CACHE_DIR/<network>/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from core.geo import Grid
from core.opensense.networks import LABELS, NETWORKS as SOURCES, regrid

from .grid import square_grid, pixel_km
from .settings import CACHE_DIR, NETWORKS


def grid_for(network: str) -> Grid:
    s = NETWORKS[network]
    return square_grid(s.domain, s.pixel_km)


def _coverage_mask(src_lat, src_lon, grid: Grid, max_km: float) -> np.ndarray:
    """True where the nearest source pixel centre is within ``max_km`` of the cell centre."""
    c = np.cos(np.radians(np.mean(grid.lat)))
    tree = cKDTree(np.column_stack([np.ravel(src_lat), np.ravel(src_lon) * c]))
    glat, glon = grid.mesh()
    d, _ = tree.query(np.column_stack([glat.ravel(), glon.ravel() * c]))
    return (d * 111.195 <= max_km).reshape(grid.shape)


def _field(data, times, grid: Grid, name: str, **attrs) -> xr.DataArray:
    return xr.DataArray(np.asarray(data, dtype="float32"), dims=("time", "lat", "lon"),
                        coords={"time": pd.DatetimeIndex(times), "lat": grid.lat, "lon": grid.lon},
                        name=name, attrs=attrs)


def _openrainer_radar(t0, t1, grid: Grid) -> xr.DataArray:
    net = SOURCES["openrainer"]
    parts = []
    for month in pd.period_range(t0.to_period("M"), t1.to_period("M"), freq="M"):
        path = net._file("RADrain", month.to_timestamp(), "RADrain.tar")
        with xr.open_dataset(path) as ds:
            lat_desc = ds.lat.values[0] > ds.lat.values[-1]
            lat_sl = slice(grid.lat.max() + 0.05, grid.lat.min() - 0.05) if lat_desc else \
                slice(grid.lat.min() - 0.05, grid.lat.max() + 0.05)
            sub = ds.rainfall_amount.sel(time=slice(t0, t1), lat=lat_sl,
                                         lon=slice(grid.lon.min() - 0.05, grid.lon.max() + 0.05)).load()
        parts.append(sub)
    amount = xr.concat(parts, "time") if len(parts) > 1 else parts[0]
    amount = amount.where(amount >= 0)
    lon2, lat2 = np.meshgrid(amount.lon.values, amount.lat.values)
    data = regrid(amount.values.reshape(amount.sizes["time"], -1), lat2, lon2, grid)
    data[:, ~_coverage_mask(lat2, lon2, grid, 1.5)] = np.nan
    return _field(data, amount.time.values, grid, "radar", units="mm", step="15min",
                  time_label=LABELS[("openrainer", "radar")],
                  source="ARPAE-SIMC radar 15-min rain depth (OpenRainER RADrain)")


def _openmrg_radar(t0, t1, grid: Grid) -> xr.DataArray:
    net = SOURCES["openmrg"]
    with xr.open_dataset(net.RADAR) as ds:
        ds = ds.sel(time=slice(t0, t1)).load()
        za, zb = float(ds.data.attrs.get("zr_a", 200)), float(ds.data.attrs.get("zr_b", 1.5))
        rate = (10.0 ** (ds.data / 10.0) / za) ** (1.0 / zb)               # mm/h
        lat, lon = ds.lat.values, ds.lon.values
    amount = rate.values / 12.0                                              # mm per 5 min
    data = regrid(amount.reshape(amount.shape[0], -1), lat, lon, grid)
    data[:, ~_coverage_mask(lat, lon, grid, 1.5 * pixel_km(grid))] = np.nan
    return _field(data, ds.time.values, grid, "radar", units="mm", step="5min",
                  time_label=LABELS[("openmrg", "radar")],
                  source="SMHI radar composite, Z = 200 R^1.5 (OpenMRG)")


def radar(network: str, start, end, cache: bool = True) -> xr.DataArray:
    """Native-step radar (mm per step) for steps ending in ``[start, end]``."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    grid = grid_for(network)
    path = CACHE_DIR / network / f"radar_{t0:%Y%m%dT%H%M}_{t1:%Y%m%dT%H%M}.nc"
    if cache and path.exists():
        with xr.open_dataarray(path) as da:
            return da.load()
    reader = {"openrainer": _openrainer_radar, "openmrg": _openmrg_radar}[network]
    da = reader(t0, t1, grid)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        da.to_netcdf(path, encoding={"radar": {"zlib": True, "complevel": 4}})
    return da


def to_step(values: xr.DataArray, step: str, label: str = "end") -> xr.DataArray:
    """Point accumulations ``(station, time)`` at a finer step -> sums per ``step`` (mm),
    interval-ending; a step with any missing sample is NaN."""
    t = pd.DatetimeIndex(values.time.values)
    native = pd.Series(t).diff().median()
    n = int(pd.Timedelta(step) / native)
    if n <= 1:
        return values
    closed = "right" if label == "end" else "left"
    r = values.resample(time=step, closed=closed, label="right")
    return r.sum(min_count=1).where(r.count() == n)


def points(network: str, start, end, which: str) -> xr.DataArray:
    """Point set ``which`` as ``(station, time)`` mm per native radar step, interval-ending."""
    s = NETWORKS[network]
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    ds = SOURCES[network].points(t0 - pd.Timedelta(minutes=s.step_min), t1, sets=[which])[which]
    out = to_step(ds["rain"], f"{s.step_min}min", ds.attrs.get("label", "end"))
    out = out.sel(time=slice(t0, t1))
    g = grid_for(network)
    inside = (out.lat.values >= g.lat.min()) & (out.lat.values <= g.lat.max()) & \
             (out.lon.values >= g.lon.min()) & (out.lon.values <= g.lon.max())
    return out.isel(station=np.flatnonzero(inside)).assign_attrs(units="mm", source=which)
