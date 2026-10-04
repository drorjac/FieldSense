"""From MRMS fields to rainfall-map products.

The functions here turn cached MRMS fields into the objects the rest of the package
compares against:

* :func:`hourly_rainfall` - hourly accumulations (mm), hour-ENDING labels;
* :func:`event_accumulation` - total over a window, with per-cell coverage;
* :func:`rain_rate` - instantaneous rate (mm/h) from ``PrecipRate``, optionally averaged;
* :func:`to_grid` - put any radar field on a target :class:`~core.geo.Grid`;
* :func:`sample_points` / :func:`path_average` - radar at gauges, or along CML paths.

Quality control applied throughout: no-coverage / missing codes are NaN (never 0);
accumulations over several hours are NaN wherever any hour is missing, unless
``min_coverage`` < 1 is requested; optional masking by ``RadarQualityIndex``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Domain, Grid

from .client import MRMSClient

QPE_1H = "MultiSensor_QPE_01H_Pass2"


def _client(client: MRMSClient | None) -> MRMSClient:
    return client or MRMSClient()


def hourly_rainfall(start, end, domain: Domain, product: str = QPE_1H,
                    client: MRMSClient | None = None) -> xr.DataArray:
    """Hourly accumulations (mm) whose hour-ending labels fall in ``(start, end]``.

    An hour labelled 12:00 covers 11:00-12:00 UTC, so ``hourly_rainfall("2024-01-09",
    "2024-01-10", ...)`` returns exactly the 24 hours of 9 January.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    first = start.floor("h") + pd.Timedelta("1h")
    da = _client(client).load(product, first, end.floor("h"), domain)
    da.attrs["units"] = "mm"
    return da


def event_accumulation(start, end, domain: Domain, product: str = QPE_1H,
                       min_coverage: float = 1.0, client: MRMSClient | None = None
                       ) -> xr.DataArray:
    """Total rainfall (mm) over ``(start, end]`` from hourly accumulations.

    ``attrs['coverage']`` holds the fraction of hours available (archive gaps), and a
    ``coverage`` coordinate gives the per-cell fraction of valid hours. Cells below
    ``min_coverage`` are NaN; with ``min_coverage < 1`` the total is scaled up by the
    missing fraction (mean-rate infilling), which is documented in the attributes.
    """
    hourly = hourly_rainfall(start, end, domain, product, client)
    n_expected = int((pd.Timestamp(end).floor("h") - pd.Timestamp(start).floor("h")) / pd.Timedelta("1h"))
    valid = hourly.notnull().sum("time")
    frac = valid / max(n_expected, 1)
    total = hourly.sum("time", min_count=1)
    if min_coverage < 1:
        total = total / frac.where(frac > 0)
    total = total.where(frac >= min_coverage - 1e-9)
    total = total.assign_coords(coverage=frac)
    total.name = "accumulation"
    total.attrs = {"units": "mm", "product": hourly.attrs.get("product"),
                   "start": str(pd.Timestamp(start)), "end": str(pd.Timestamp(end)),
                   "hours_expected": n_expected, "hours_available": int(hourly.sizes["time"]),
                   "coverage": hourly.sizes["time"] / max(n_expected, 1),
                   "min_coverage": min_coverage,
                   "note": "partial-coverage cells rescaled by 1/coverage" if min_coverage < 1 else ""}
    return total


def rain_rate(start, end, domain: Domain, average: str | None = None, freq: str | None = None,
              client: MRMSClient | None = None) -> xr.DataArray:
    """Instantaneous surface rain rate (mm/h) from MRMS ``PrecipRate`` (2-min).

    ``freq`` subsamples the 2-min stream (e.g. ``"10min"``) to limit downloads;
    ``average`` then block-averages to a coarser, interval-ENDING step (e.g. ``"15min"``).
    """
    da = _client(client).load("PrecipRate", start, end, domain, freq=freq)
    if average:
        da = da.resample(time=average, label="right", closed="right").mean()
        da.attrs["time_label"] = f"end of {average} averaging interval (UTC)"
    return da


def mask_low_quality(field: xr.DataArray, rqi: xr.DataArray, threshold: float = 0.5) -> xr.DataArray:
    """NaN-out cells whose RadarQualityIndex is below ``threshold``.

    ``rqi`` may be a single field or a time series (it is averaged over time first).
    """
    q = rqi.mean("time") if "time" in rqi.dims else rqi
    q = q.reindex_like(field, method="nearest", tolerance=0.006)
    return field.where(q >= threshold)


def to_grid(field: xr.DataArray, grid: Grid, method: str = "auto") -> xr.DataArray:
    """Resample a radar field to ``grid``.

    ``method``: ``"nearest"`` (cell containing the target centre), ``"mean"`` (average of
    radar cells whose centres fall in each target cell - use when the target is coarser),
    ``"linear"``, or ``"auto"`` (mean if target is >= 1.5x coarser, else nearest).
    """
    res_src = float(abs(np.diff(field.lat.values[:2])[0]))
    res_dst = float(abs(np.diff(grid.lat[:2])[0])) if grid.lat.size > 1 else res_src
    if method == "auto":
        method = "mean" if res_dst >= 1.5 * res_src else "nearest"
    if method == "nearest":
        return field.sel(lat=grid.lat, lon=grid.lon, method="nearest",
                         tolerance=max(res_src, res_dst)).assign_coords(lat=grid.lat, lon=grid.lon)
    if method == "linear":
        return field.interp(lat=grid.lat, lon=grid.lon)
    if method == "mean":
        def edges(c):
            d = np.diff(c)
            return np.concatenate([[c[0] - d[0] / 2], c[:-1] + d / 2, [c[-1] + d[-1] / 2]])
        out = field.groupby_bins("lat", edges(grid.lat), labels=grid.lat).mean()
        out = out.groupby_bins("lon", edges(grid.lon), labels=grid.lon).mean()
        return out.rename(lat_bins="lat", lon_bins="lon").transpose(*field.dims)
    raise ValueError(f"unknown method {method!r}")


def sample_points(field: xr.DataArray, lat, lon, names=None) -> xr.DataArray:
    """Radar value in the cell containing each point, as dimension ``point``."""
    lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
    names = list(names) if names is not None else list(range(lat.size))
    out = field.sel(lat=xr.DataArray(lat, dims="point"), lon=xr.DataArray(lon, dims="point"),
                    method="nearest")
    return out.assign_coords(point=names, point_lat=("point", lat), point_lon=("point", lon))


def path_average(field: xr.DataArray, links: pd.DataFrame, n_samples: int | None = None
                 ) -> xr.DataArray:
    """Radar averaged along each CML path (what a link actually "sees").

    See ``core.maps.geometry.path_average_points``: points ~250 m apart (at least 3 per
    link), NaN-aware mean of the cells they fall in.
    """
    from core.maps.geometry import path_average_points
    return path_average_points(field, links, n_samples=n_samples)


def domain_mean_series(field: xr.DataArray, min_valid_fraction: float = 0.5) -> pd.Series:
    """Spatial mean per time step, NaN where fewer than ``min_valid_fraction`` cells are valid."""
    valid = field.notnull().mean(("lat", "lon"))
    return field.mean(("lat", "lon")).where(valid >= min_valid_fraction).to_series()
