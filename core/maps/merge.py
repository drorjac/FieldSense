"""Combine sensors: links + gauges in one map, and radar adjusted with links and/or gauges.

Ported from ``pcpn_maps.mapping.merge`` (``drorjac/pcpn_maps``, private), where it was built
for the NYC extreme-event study; here it works on any :class:`~core.geo.Grid`.

Everything is hourly accumulations (mm, hour-ending) and *observations* - an
``obs(point, time)`` array with ``lat``/``lon``/``kind`` coordinates. A gauge is a point at
its station; a link is a point at its path midpoint whose value is a path average, so the
radar it is compared with is the radar averaged along the same path (:func:`radar_at`).
:func:`observations` builds that pair for any mix of sources.

Merging without radar: :func:`merge_idw` - one IDW over links and gauges together.

Radar adjustment, the standard gauge-adjustment family (e.g. Goudenhoofdt & Delobbe 2009;
wradlib ``adjust``), each hour on its own:

* ``"mfb"`` - mean-field bias: ``R * sum(obs) / sum(radar at obs)``; the factor is clipped
  to ``[1/max_factor, max_factor]`` and needs ``min_radar_mm`` of radar at the observations
  and ``min_points`` of them, else the event-cumulative factor is used;
* ``"add"`` - additive: ``R + IDW(obs - radar at obs)``, clipped at 0;
* ``"mul"`` - multiplicative: ``R * IDW((obs + eps) / (radar at obs + eps))``, the factor
  clipped to the same range.

Cells farther than ``radius_m`` from every observation keep the mean-field factor
(``add``/``mul``), so the field stays complete. The geostatistical methods of the OpenSense
package ``mergeplg`` are in :mod:`core.maps.mergeplg_methods`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.geo import Grid, to_local_xy
from core.maps.idw import IDW as IDWOperator

IDW = {"power": 2.0, "radius_m": 10_000.0, "nnear": None}
ADJUSTMENTS = ("mfb", "add", "mul")
PATH_COLS = ["site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon"]


def as_points(values: xr.DataArray, kind: str) -> xr.DataArray:
    """``(point, time)`` with ``lat``/``lon``/``kind`` from a ``(link, time)`` array (at its
    midpoints, path ends kept) or a ``(station, time)`` array."""
    if "link" in values.dims:
        v = values.transpose("link", "time")
        lat, lon, ids = values.mid_lat.values, values.mid_lon.values, values.link.values
    else:
        v = values.transpose("station", "time")
        lat, lon, ids = values.lat.values, values.lon.values, values.station.values
    out = xr.DataArray(v.values.astype(float), dims=("point", "time"),
                       coords={"point": np.asarray(ids).astype(str), "time": v.time.values},
                       name=values.name, attrs=values.attrs)
    out = out.assign_coords(lat=("point", np.asarray(lat, float)), lon=("point", np.asarray(lon, float)),
                            kind=("point", np.array([kind] * out.sizes["point"], dtype=object)))
    # a gauge is a zero-length path, so links and gauges stack with one set of coordinates
    ends = {c: values[c].values if "link" in values.dims else (lat if c.endswith("lat") else lon)
            for c in PATH_COLS}
    return out.assign_coords({c: ("point", np.asarray(v, float)) for c, v in ends.items()})


def radar_at(radar: xr.DataArray, values: xr.DataArray) -> xr.DataArray:
    """Radar at each observation, same shape as ``values``: the path average for links
    (``(link, time)`` input), the containing cell for gauges (``(station, time)``)."""
    from core.opensense.networks import radar_along_links
    t = np.intersect1d(values.time.values, radar.time.values)
    if "link" in values.dims:
        r = radar_along_links(radar.sel(time=t), values.to_dataset(name="v")).transpose("link", "time")
        return r.reindex(time=values.time.values)
    lat = xr.DataArray(values.lat.values, dims="station")
    lon = xr.DataArray(values.lon.values, dims="station")
    r = radar.sel(time=t).sel(lat=lat, lon=lon, method="nearest").transpose("station", "time")
    r = r.drop_vars(["lat", "lon"]).assign_coords(station=values.station.values)
    return r.reindex(time=values.time.values)


def observations(radar: xr.DataArray | None = None, **sources) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Stack sources (e.g. ``cml=link_hourly``, ``gauges=station_hourly``) into ``obs(point, time)``.

    Returns ``(obs, radar_at_obs)`` on the times of the first source (others are reindexed,
    missing hours NaN); the second is ``None`` without ``radar``.
    """
    obs, rad, times = [], [], None
    for kind, v in sources.items():
        if v is None or v.size == 0:
            continue
        times = v.time.values if times is None else times
        v = v.reindex(time=times)
        p = as_points(v, kind)
        obs.append(p)
        if radar is not None:
            r = radar_at(radar, v)
            other = next(d for d in r.dims if d != "time")
            rad.append(p.copy(data=r.transpose(other, "time").values.astype(float)))
    if not obs:
        raise ValueError("no observations")
    O = xr.concat(obs, dim="point", join="outer", coords="different", compat="equals") if len(obs) > 1 else obs[0]
    R = (xr.concat(rad, dim="point", coords="different", compat="equals") if len(rad) > 1 else rad[0]) if rad else None
    return O, R


def _point_weights(obs: xr.DataArray, kind_weights: dict | None) -> np.ndarray:
    """Per-point multiplier from ``{kind: weight}`` (missing kinds weigh 1)."""
    return np.array([(kind_weights or {}).get(str(k), 1.0) for k in obs.kind.values], dtype=float)


def _weights(obs: xr.DataArray, grid: Grid, idw: dict, kind_weights: dict | None = None) -> IDWOperator:
    lat0, lon0 = float(np.mean(grid.lat)), float(np.mean(grid.lon))
    sx, sy = to_local_xy(obs.lat.values, obs.lon.values, lat0, lon0)
    glat, glon = grid.mesh()
    dx, dy = to_local_xy(glat.ravel(), glon.ravel(), lat0, lon0)
    return IDWOperator(sx, sy, dx, dy, idw.get("power", 2.0), idw.get("radius_m"), idw.get("nnear"),
               src_weights=_point_weights(obs, kind_weights))


def _interp(W: IDWOperator, V: np.ndarray) -> np.ndarray:
    """IDW of ``V(point, time)`` with NaN excluded per time step -> ``(cells, time)``."""
    return W(V)


def _field(values: np.ndarray, grid: Grid, times, name: str) -> xr.DataArray:
    arr = values.reshape(grid.shape + (len(times),)).transpose(2, 0, 1)
    return xr.DataArray(arr.astype("float32"), dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": grid.lat, "lon": grid.lon}, name=name,
                        attrs={"units": "mm"})


def merge_idw(obs: xr.DataArray, grid: Grid, idw: dict | None = None, name: str = "merged",
              kind_weights: dict | None = None) -> xr.DataArray:
    """One IDW map from all observations (links at midpoints, gauges at stations).

    ``kind_weights`` scales the IDW weight of each source, e.g. ``{"cml": 0.3}`` makes a link
    count 0.3 of a gauge at the same distance."""
    W = _weights(obs, grid, {**IDW, **(idw or {})}, kind_weights)
    return _field(_interp(W, obs.transpose("point", "time").values), grid, obs.time.values, name)


def adjust(radar: xr.DataArray, obs: xr.DataArray, rad_obs: xr.DataArray, method: str = "mfb",
           idw: dict | None = None, max_factor: float = 5.0, min_radar_mm: float = 0.5,
           min_points: int = 3, eps: float = 0.5, name: str | None = None,
           kind_weights: dict | None = None) -> xr.DataArray:
    """Adjust hourly ``radar(time, lat, lon)`` to ``obs`` (see the module docstring).

    ``kind_weights`` weights each source in the mean-field sums and the residual IDW."""
    if method not in ADJUSTMENTS:
        raise ValueError(f"method must be one of {ADJUSTMENTS}")
    times = np.intersect1d(radar.time.values, obs.time.values)
    R = radar.sel(time=times)
    O = obs.sel(time=times).transpose("point", "time").values.astype(float)
    P = rad_obs.sel(time=times).transpose("point", "time").values.astype(float)
    both = np.isfinite(O) & np.isfinite(P)
    pw = _point_weights(obs, kind_weights)[:, None]
    O0, P0 = np.where(both, O, 0.0) * pw, np.where(both, P, 0.0) * pw
    lo, hi = 1.0 / max_factor, max_factor
    # event-cumulative factor: the fallback for hours with too little radar at the points
    f_event = np.clip(O0.sum() / P0.sum(), lo, hi) if P0.sum() >= min_radar_mm else 1.0
    so, sp, n = O0.sum(0), P0.sum(0), both.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_hour = np.where((sp >= min_radar_mm) & (n >= min_points), np.clip(so / sp, lo, hi), f_event)
    grid = Grid(radar.lat.values, radar.lon.values)
    Rv = R.transpose("time", "lat", "lon").values.reshape(len(times), -1).T      # (cells, time)

    if method == "mfb":
        out = Rv * f_hour[None, :]
    else:
        W = _weights(obs, grid, {**IDW, **(idw or {})}, kind_weights)
        covered = (W.W0.sum(1) > 0)[:, None]
        if method == "add":
            res = _interp(W, np.where(both, O - P, np.nan))
            out = np.where(covered & np.isfinite(res), np.clip(Rv + res, 0, None), Rv * f_hour[None, :])
        else:
            ratio = np.where(both, (O + eps) / (P + eps), np.nan)
            fac = np.clip(_interp(W, ratio), lo, hi)
            out = np.where(covered & np.isfinite(fac), Rv * fac, Rv * f_hour[None, :])
    da = _field(out, grid, times, name or f"radar_{method}")
    da.attrs.update({"adjustment": method, "event_factor": float(f_event)})
    return da
