"""Links as lines: a field sampled along each path, and the distance to the nearest path.

A link reports one number for its whole path, so comparing it with a gridded field means
reducing the field to the same path, and how well a cell is covered depends on its distance
to the nearest path, not to a link's midpoint. Every project does both through here.

``path_average_points``     the field at points spaced along each path (nearest cell), averaged;
                            lat/lon fields, links as a table
``path_average_intersect``  the field weighted by each path's intersection length with the
                            cells (poligrain), in the projected plane or on a lon/lat grid
``segment_distance_km``     exact point-to-segment distance in a projected plane (m in, km out)
``distance_to_links_km``    great-circle distance from lat/lon cells to points spaced along
                            each path
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import haversine_m


# --------------------------------------------------------------------------
# a field along each path
# --------------------------------------------------------------------------
def path_average_points(field: xr.DataArray, links: pd.DataFrame, n_samples: int | None = None,
                        spacing_m: float = 250.0) -> xr.DataArray:
    """``field`` (``lat``, ``lon``, any other dims) averaged along each link path.

    ``links`` needs ``site_0_lat, site_0_lon, site_1_lat, site_1_lon`` and is indexed by
    the link identifier. Points are spaced ``spacing_m`` apart (at least 3 per link); the
    mean is over the cells those points fall in, NaN-aware. Returns (link, ...).
    """
    res = []
    for _, r in links.iterrows():
        length = haversine_m(r.site_0_lat, r.site_0_lon, r.site_1_lat, r.site_1_lon)
        n = n_samples or max(3, int(np.ceil(length / spacing_m)) + 1)
        s = np.linspace(0, 1, n)
        lat = r.site_0_lat + s * (r.site_1_lat - r.site_0_lat)
        lon = r.site_0_lon + s * (r.site_1_lon - r.site_0_lon)
        pts = field.sel(lat=xr.DataArray(lat, dims="s"), lon=xr.DataArray(lon, dims="s"),
                        method="nearest")
        res.append(pts.mean("s", skipna=True))
    # object labels: xarray < 2024.9 cannot index by pandas 3's string dtype
    out = xr.concat(res, dim=pd.Index(list(links.index), dtype=object, name="link"))
    out.attrs = dict(field.attrs, note="mean along link path")
    return out


def line_geometry(ds_cml: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """Just the per-link geometry poligrain needs, with no time axis."""
    names = ("site_0_x", "site_0_y", "site_1_x", "site_1_y",
             "site_0_lon", "site_0_lat", "site_1_lon", "site_1_lat")
    missing = [n for n in names[:4] if n not in ds_cml.coords]
    if missing:
        raise ValueError(f"CML data is missing projected endpoints {missing}; "
                         f"run conventions.project_cml first")
    geo = xr.Dataset(coords={"cml_id": ds_cml.cml_id})
    for n in names:
        if n in ds_cml.coords:
            c = ds_cml[n]
            if "sublink_id" in c.dims:
                c = c.isel(sublink_id=0, drop=True)
            geo.coords[n] = ("cml_id", np.asarray(c))
    # GridAtLines copies site lon/lat onto its output for plotting even when
    # it computed in projected metres, and fails without them. They are not
    # read numerically, so NaN placeholders are honest when a caller has
    # projected coordinates only.
    for n in names[4:]:
        if n not in geo.coords:
            geo.coords[n] = ("cml_id", np.full(geo.sizes["cml_id"], np.nan))
    return geo


def path_average_intersect(field: xr.DataArray, ds_cml: xr.Dataset | xr.DataArray,
                           plane: str = "xy", lon2d=None, lat2d=None,
                           grid_point_location: str = "center") -> xr.DataArray:
    """``field`` weighted by each path's intersection length with the grid cells (poligrain).

    ``plane="xy"``: ``field`` is (time, y, x) or (y, x) with 2-D ``x_grid``/``y_grid`` in the
    links' projected CRS (``site_*_x/y``); links that leave the grid get the average of the
    part inside it. ``plane="lonlat"``: weights on the ``lon2d``/``lat2d`` grid from the
    links' ``site_*_lon/lat``. Returns (time, cml_id).
    """
    import poligrain as plg

    if plane == "xy":
        if "time" in field.dims:
            field = field.transpose("time", "y", "x")
        gal = plg.spatial.GridAtLines(field, line_geometry(ds_cml),
                                      grid_point_location=grid_point_location, use_lon_lat=False)
        return gal(field)
    if plane == "lonlat":
        w = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
            x1_line=ds_cml.site_0_lon.values, y1_line=ds_cml.site_0_lat.values,
            x2_line=ds_cml.site_1_lon.values, y2_line=ds_cml.site_1_lat.values,
            cml_id=ds_cml.cml_id.values, x_grid=np.asarray(lon2d), y_grid=np.asarray(lat2d),
            grid_point_location=grid_point_location)
        return plg.spatial.get_grid_time_series_at_intersections(grid_data=field, intersect_weights=w)
    raise ValueError(f"plane must be 'xy' or 'lonlat', not {plane!r}")


# --------------------------------------------------------------------------
# distance to the nearest path
# --------------------------------------------------------------------------
def segment_distance_km(x0, y0, x1, y1, x, y, chunk: int = 20000) -> np.ndarray:
    """Shortest distance (km) from each point ``(x, y)`` to any segment ``(x0, y0)-(x1, y1)``.

    Projected coordinates in metres. Point-to-segment, not point-to-midpoint: a point beside
    the middle of a long link is well covered even though both endpoints are far away.
    ``x``/``y`` can have any shape (a gauge list, a 2-D grid); the result has the same shape.
    """
    x0, y0, x1, y1 = (np.asarray(v, dtype=float).ravel() for v in (x0, y0, x1, y1))
    vx, vy = x1 - x0, y1 - y0
    len2 = np.maximum(vx * vx + vy * vy, 1e-9)

    px = np.asarray(x, dtype=float)
    py = np.asarray(y, dtype=float)
    flat_x, flat_y = px.ravel(), py.ravel()
    out = np.empty(flat_x.size)
    for s in range(0, flat_x.size, chunk):
        qx = flat_x[s:s + chunk, None]
        qy = flat_y[s:s + chunk, None]
        t = np.clip(((qx - x0) * vx + (qy - y0) * vy) / len2, 0.0, 1.0)
        out[s:s + chunk] = np.hypot(qx - (x0 + t * vx), qy - (y0 + t * vy)).min(axis=1)
    return (out / 1000.0).reshape(px.shape)


def distance_to_links_km(lat, lon, links: xr.Dataset | pd.DataFrame, step_m: float = 200.0) -> np.ndarray:
    """Great-circle distance (km) from each ``(lat, lon)`` to the nearest link path.

    Each path is sampled every ``step_m`` (at least its two ends); ``lat``/``lon`` can have
    any shape and the result has the same shape.
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    best = np.full(lat.shape, np.inf)
    get = (lambda c: links[c].values) if isinstance(links, xr.Dataset) else (lambda c: links[c].to_numpy())
    for la0, lo0, la1, lo1 in zip(get("site_0_lat"), get("site_0_lon"), get("site_1_lat"), get("site_1_lon")):
        n = max(2, int(haversine_m(la0, lo0, la1, lo1) / step_m) + 1)
        for s in np.linspace(0, 1, n):
            best = np.minimum(best, haversine_m(lat, lon, la0 + s * (la1 - la0), lo0 + s * (lo1 - lo0)))
    return best / 1000.0
