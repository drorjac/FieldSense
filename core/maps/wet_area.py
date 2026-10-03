"""Rain maps that can be dry: a wet/dry decision before the interpolation.

IDW and kriging of rain amounts spread every wet observation over its whole radius, so a
map of a storm that covers a tenth of the domain is wet almost everywhere and its peaks are
diluted by the dry observations around them. Two corrections, both from the same sensors:

``wet_probability``   IDW of the wet indicator (value >= ``wet_threshold``): the fraction
                      of nearby sensors, weighted by distance, that saw rain
``masked_idw``        an IDW map set to zero where the wet probability is below ``p_cut``
``conditional_idw``   IDW of the wet observations only (the rain where it rains), set to
                      zero where the wet probability is below ``p_cut`` - the indicator
                      approach of geostatistics, with IDW weights instead of kriging

All three take link rain ``(link, time)`` with ``mid_lat``/``mid_lon`` (as ``idw_map``) or
point sensors ``(station, time)`` with ``lat``/``lon``, and return ``(time, lat, lon)``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.geo import Grid
from core.maps.idw import idw_map

WET_THRESHOLD = 0.1          # mm per interval: wet
P_CUT = 0.5                  # wet probability above which a cell is wet


def _as_links(values: xr.DataArray) -> xr.DataArray:
    if "link" in values.dims:
        return values
    return values.rename(station="link").assign_coords(mid_lat=("link", values.lat.values),
                                                      mid_lon=("link", values.lon.values))


def wet_probability(values: xr.DataArray, grid: Grid, wet_threshold: float = WET_THRESHOLD,
                    **idw) -> xr.DataArray:
    """Distance-weighted share of sensors that are wet, per cell and time (0..1)."""
    v = _as_links(values)
    ind = xr.where(v >= wet_threshold, 1.0, 0.0).where(v.notnull())
    p = idw_map(ind, grid, **idw)
    p.attrs = {"long_name": "wet probability", "wet_threshold": wet_threshold}
    return p.rename("wet_probability")


def masked_idw(values: xr.DataArray, grid: Grid, wet_threshold: float = WET_THRESHOLD,
               p_cut: float = P_CUT, **idw) -> xr.DataArray:
    """IDW of all observations, zero where the wet probability is below ``p_cut``."""
    m = idw_map(_as_links(values), grid, **idw)
    p = wet_probability(values, grid, wet_threshold, **idw)
    out = m.where((p >= p_cut) | m.isnull(), 0.0)
    out.attrs = dict(m.attrs, interpolation="IDW, wet mask", p_cut=p_cut, wet_threshold=wet_threshold)
    return out


def conditional_idw(values: xr.DataArray, grid: Grid, wet_threshold: float = WET_THRESHOLD,
                    p_cut: float = P_CUT, **idw) -> xr.DataArray:
    """IDW of the wet observations only, zero where the wet probability is below ``p_cut``.

    Cells judged wet but with no wet observation in range keep the plain IDW value.
    """
    v = _as_links(values)
    plain = idw_map(v, grid, **idw)
    wet_only = idw_map(v.where(v >= wet_threshold), grid, **idw)
    p = wet_probability(values, grid, wet_threshold, **idw)
    rain = wet_only.fillna(plain)
    out = rain.where((p >= p_cut) | plain.isnull(), 0.0).where(plain.notnull())
    out.attrs = dict(plain.attrs, interpolation="IDW of wet observations, wet mask", p_cut=p_cut,
                     wet_threshold=wet_threshold)
    return out


def wet_area_scores(est, ref, wet_threshold: float = WET_THRESHOLD, peak_q: float = 0.99) -> dict:
    """How well a map gets where it rains: wet-area ratio and peak recovery, on joint valid cells.

    ``war_ratio`` is the estimate's wet fraction over the reference's (1 is right, above 1
    too wet); ``peak_ratio`` the ``peak_q`` quantile of the estimate over the reference's.
    """
    e = np.asarray(est, dtype=float).ravel()
    r = np.asarray(ref, dtype=float).ravel()
    ok = np.isfinite(e) & np.isfinite(r)
    e, r = e[ok], r[ok]
    if not ok.any():
        return {"war_est": np.nan, "war_ref": np.nan, "war_ratio": np.nan, "peak_ratio": np.nan}
    war_e, war_r = float((e >= wet_threshold).mean()), float((r >= wet_threshold).mean())
    qe, qr = float(np.quantile(e, peak_q)), float(np.quantile(r, peak_q))
    return {"war_est": war_e, "war_ref": war_r,
            "war_ratio": war_e / war_r if war_r > 0 else np.nan,
            "peak_ratio": qe / qr if qr > 0 else np.nan}
