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


class IDW:
    """IDW from fixed sources to fixed destinations, applied to values with gaps.

    ``idw_weights`` builds one matrix for all time steps, which is right only while every
    source has a value: with ``nnear`` it picks the k nearest sources before the missing
    ones drop out, so a cell can average fewer than k (or none), and a cell on a missing
    source gets no value at all. Here each pattern of missing sources gets its own
    weights: the k nearest *valid* sources (as pycomlink's KDTree IDW does), and an exact
    hit only when the source on the cell has a value. Without gaps, or without ``nnear``
    and exact hits, it is ``apply_weights`` with ``idw_weights`` - the same numbers.

    ``src_weights`` multiplies each source's weight (e.g. per-kind or per-link weights).
    """

    def __init__(self, src_x, src_y, dst_x, dst_y, power: float = 2.0, radius_m: float | None = 10_000.0,
                 nnear: int | None = None, eps: float = 0.0, src_weights=None, dtype=float):
        src_x, src_y = np.asarray(src_x, float), np.asarray(src_y, float)
        dst_x, dst_y = np.asarray(dst_x, float), np.asarray(dst_y, float)
        self.nnear, self.eps, self.dtype = nnear, eps, dtype
        self.sw = np.ones(src_x.size) if src_weights is None else np.asarray(src_weights, float)
        self.W0 = (idw_weights(src_x, src_y, dst_x, dst_y, power, radius_m, nnear, eps)
                   * self.sw[None, :]).astype(dtype)
        self.d = np.hypot(dst_x[:, None] - src_x[None, :], dst_y[:, None] - src_y[None, :])
        with np.errstate(divide="ignore"):
            self.w = 1.0 / (self.d ** power + eps)
        if radius_m is not None:
            self.w[self.d > radius_m] = 0.0
        self.exact = (self.d == 0) if eps == 0 else np.zeros(self.d.shape, bool)
        self.w[self.exact] = 0.0
        self.order = np.argsort(self.d, axis=1, kind="stable") if nnear is not None else None
        self._needs_masks = nnear is not None or self.exact.any()

    def weights_for(self, valid: np.ndarray) -> np.ndarray:
        """``(dst, src)`` weights when only the sources where ``valid`` is True have a value."""
        valid = np.asarray(valid, bool)
        w = self.w * valid[None, :]
        if self.nnear is not None and self.nnear < valid.sum():
            vs = valid[self.order]                                     # valid, nearest first
            keep_sorted = vs & (np.cumsum(vs, axis=1) <= self.nnear)
            keep = np.zeros_like(keep_sorted)
            np.put_along_axis(keep, self.order, keep_sorted, axis=1)
            w = w * keep
        hit = self.exact & valid[None, :]
        rows = hit.any(1)
        if rows.any():
            w[rows] = hit[rows].astype(float)
        return (w * self.sw[None, :]).astype(self.dtype)

    def __call__(self, V: np.ndarray, valid_dtype=float) -> np.ndarray:
        """``V (src, ...)`` -> ``(dst, ...)``, NaN sources excluded per step."""
        V = np.asarray(V)
        valid = np.isfinite(V)
        if not self._needs_masks or valid.all():
            return apply_weights(self.W0, V, valid_dtype)
        flat_v = V.reshape(V.shape[0], -1)
        flat_ok = valid.reshape(valid.shape[0], -1)
        out = np.full((self.d.shape[0], flat_v.shape[1]), np.nan)
        patterns, inverse = np.unique(flat_ok.T, axis=0, return_inverse=True)
        for p, mask in enumerate(patterns):
            cols = np.flatnonzero(inverse.ravel() == p)
            if mask.all():
                out[:, cols] = apply_weights(self.W0, flat_v[:, cols], valid_dtype)
            else:
                out[:, cols] = apply_weights(self.weights_for(mask), flat_v[:, cols], valid_dtype)
        return out.reshape((self.d.shape[0],) + V.shape[1:])


def apply_weights(W: np.ndarray, V: np.ndarray, valid_dtype=float) -> np.ndarray:
    """``W (dst, src) @ V (src, ...)`` normalised per column, NaN sources excluded per step.

    Destinations with no valid source in range are NaN. ``valid_dtype`` sets the precision
    of the weight sums (float32 keeps a float32 pipeline in float32).
    """
    valid = np.isfinite(V)
    num, den = W @ np.where(valid, V, 0.0), W @ valid.astype(valid_dtype)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def idw_map(link_rain: xr.DataArray, grid: Grid, power: float = 2.0,
            radius_m: float | None = 10_000.0, nnear: int | None = None,
            nan_policy: str = "exclude", eps: float = 0.0, weights=None) -> xr.DataArray:
    """Interpolate ``link_rain(link, time)`` (needs ``mid_lat``/``mid_lon`` coords) to ``grid``.

    ``nan_policy``: ``"exclude"`` (a NaN link drops out of that time step's average) or
    ``"zero"`` (NaN treated as 0 mm/h, implementation_1). Cells with no link in range
    are NaN. ``weights`` (one per link) multiplies each link's IDW weight, e.g. the
    inverse of its expected error variance. Returns ``(time, lat, lon)`` in the input units.
    """
    lat0, lon0 = float(np.mean(grid.lat)), float(np.mean(grid.lon))
    sx, sy = to_local_xy(link_rain.mid_lat.values, link_rain.mid_lon.values, lat0, lon0)
    glat, glon = grid.mesh()
    dx, dy = to_local_xy(glat.ravel(), glon.ravel(), lat0, lon0)
    op = IDW(sx, sy, dx, dy, power, radius_m, nnear, eps, src_weights=weights)   # (cells, links)

    V = link_rain.transpose("link", ...).values.reshape(link_rain.sizes["link"], -1)
    if nan_policy == "zero":
        out = op(np.nan_to_num(V, nan=0.0))
    elif nan_policy == "exclude":
        out = op(V)
    else:
        raise ValueError("nan_policy must be 'exclude' or 'zero'")

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


def points_idw_map(values: xr.DataArray, grid: Grid, **idw) -> xr.DataArray:
    """IDW map from point sensors ``values(station, time)`` with ``lat``/``lon`` coords.

    Gauges and weather stations through :func:`idw_map`, each station as its own midpoint;
    ``idw`` is passed on (power, radius_m, nnear, nan_policy, eps).
    """
    da = values.rename(station="link").assign_coords(mid_lat=("link", values.lat.values),
                                                    mid_lon=("link", values.lon.values))
    return idw_map(da, grid, **idw)


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
