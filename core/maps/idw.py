"""Inverse-distance-weighted (IDW) mapping of link rain rates to a grid.

Each link is represented by its path midpoint. For a grid cell at distance d_i from
link i, ``R = sum(w_i r_i) / sum(w_i)`` with ``w_i = 1 / d_i**power``, restricted to
links within ``radius_m`` and (optionally) to the ``nnear`` closest links.

Both implementations use IDW with power 2 and a 10 km radius; they differ in details
that this module exposes as parameters:

=====================  ==============================  =================================
                       implementation_1 (PyNNcml IDW)   implementation_2 (pycomlink KDTree)
=====================  ==============================  =================================
distance space         UTM 18N, normalised by extent    UTM 18N metres
neighbours             all within radius                8 nearest (``nnear=8``) in radius
NaN link values        set to 0 before IDW              excluded per time step
weight regulariser     ``1/(d^2 + 1e-6)``               exact value at d = 0
grid                   padded link bbox, ~1 km          12 x 12 km box, 1 km cells
=====================  ==============================  =================================

:func:`idw_map` is the general tool used for every sensor here.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid, to_local_xy


def idw_weights(src_x, src_y, dst_x, dst_y, power: float = 2.0, radius_m: float | None = 10_000.0,
                nnear: int | None = None, eps: float = 0.0) -> np.ndarray:
    """Weight matrix ``(n_dst, n_src)``; rows are NOT normalised.

    A destination that coincides with a source (d == 0, ``eps == 0``) gets weight 1 on
    that source only (exact interpolation).
    """
    d = np.hypot(dst_x[:, None] - src_x[None, :], dst_y[:, None] - src_y[None, :])
    with np.errstate(divide="ignore"):
        w = 1.0 / (d ** power + eps)
    exact = d == 0
    if eps == 0 and exact.any():
        rows = exact.any(1)
        w[rows] = exact[rows].astype(float)
    if radius_m is not None:
        w[d > radius_m] = 0.0
    if nnear is not None and nnear < d.shape[1]:
        order = np.argsort(d, axis=1)
        far = order[:, nnear:]
        np.put_along_axis(w, far, 0.0, axis=1)
    return w


def idw_map(link_rain: xr.DataArray, grid: Grid, power: float = 2.0,
            radius_m: float | None = 10_000.0, nnear: int | None = None,
            nan_policy: str = "exclude", eps: float = 0.0) -> xr.DataArray:
    """Interpolate ``link_rain(link, time)`` (needs ``mid_lat``/``mid_lon`` coords) to ``grid``.

    ``nan_policy``: ``"exclude"`` (a NaN link drops out of that time step's average) or
    ``"zero"`` (NaN treated as 0 mm/h, implementation_1). Cells with no link in range
    are NaN. Returns ``(time, lat, lon)`` in the input units.
    """
    lat0, lon0 = float(np.mean(grid.lat)), float(np.mean(grid.lon))
    sx, sy = to_local_xy(link_rain.mid_lat.values, link_rain.mid_lon.values, lat0, lon0)
    glat, glon = grid.mesh()
    dx, dy = to_local_xy(glat.ravel(), glon.ravel(), lat0, lon0)
    W = idw_weights(sx, sy, dx, dy, power, radius_m, nnear, eps)          # (cells, links)

    V = link_rain.transpose("link", ...).values.reshape(link_rain.sizes["link"], -1)
    if nan_policy == "zero":
        valid = np.ones_like(V)
        V = np.nan_to_num(V, nan=0.0)
    elif nan_policy == "exclude":
        valid = np.isfinite(V).astype(float)
        V = np.where(valid > 0, V, 0.0)
    else:
        raise ValueError("nan_policy must be 'exclude' or 'zero'")
    num, den = W @ V, W @ valid
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)

    other = [d for d in link_rain.dims if d != "link"]
    shape = [link_rain.sizes[d] for d in other]
    out = out.reshape(grid.shape + tuple(shape))
    out = np.moveaxis(out, [0, 1], [-2, -1])
    coords = {d: link_rain[d].values for d in other}
    coords.update(lat=grid.lat, lon=grid.lon)
    da = xr.DataArray(out.astype("float32"), dims=other + ["lat", "lon"], coords=coords,
                      name=link_rain.name or "rain")
    da.attrs = dict(link_rain.attrs, interpolation="IDW", power=power,
                    radius_m=radius_m if radius_m is not None else "none",
                    nnear=nnear if nnear is not None else "all", nan_policy=nan_policy,
                    n_links=int(link_rain.sizes["link"]))
    return da


def accumulate(rate: xr.DataArray, freq: str = "1h", min_coverage: float = 0.8) -> xr.DataArray:
    """Rate series (mm/h at a regular step) -> accumulation per ``freq`` (mm), interval-ENDING.

    The interval (t - freq, t] is labelled t, matching MRMS 1-h QPE. The accumulation is the
    mean rate over the valid samples times the interval length, so gaps are infilled with the
    interval's mean rate rather than counted as zero; intervals with fewer than
    ``min_coverage`` valid samples are NaN.
    """
    t = pd.DatetimeIndex(rate.time.values)
    step = pd.Series(t).diff().median()
    hours = pd.Timedelta(freq) / pd.Timedelta("1h")
    expected = pd.Timedelta(freq) / step
    r = rate.resample(time=freq, label="right", closed="right")
    acc = r.mean() * hours
    acc = acc.where(r.count() >= min_coverage * expected)
    acc.attrs = dict(rate.attrs, units="mm", accumulation=freq, sample_step=str(step),
                     min_coverage=min_coverage, time_label="end of accumulation window (UTC)")
    return acc
