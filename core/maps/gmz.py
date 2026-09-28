"""GMZ rain fields from link path averages (Goldshtein, Messer & Zinevich 2009).

A link measures the rain *averaged along its path*; midpoint IDW puts that average at one
point. GMZ keeps the path: each link becomes ``k`` virtual gauges along it, all starting at
the link's value; the field is interpolated from them (IDW), sampled back at the virtual
gauges, and each link's gauges are shifted so that their mean in the attenuation domain
(``R**b``, ``b`` from ITU-R P.838-3) equals the link's own. After ``n_iter`` passes the rain
along each link is no longer uniform: it leans towards its neighbours while the link's
average is kept.

This follows PyNNcml's ``generate_link_set_gmz`` / ``GMZInterpolation`` (10 passes, the
same update), but on a lat/lon :class:`~core.geo.Grid` shared with the radar, and samples
the field at the virtual gauges bilinearly with the corrections FieldSense applies to
PyNNcml (``core/scientific_packages/pynncml_compat.patch_pynncml_gmz``).

``n_iter=0`` is *line IDW*: the same virtual gauges, no correction.

    from core.maps.gmz import gmz_map
    field = gmz_map(link_hourly, grid)         # (time, lat, lon), same units as the input
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.cml.power_law import itu_ab
from core.geo import Grid, haversine_m, to_local_xy


def virtual_gauges(links: xr.DataArray | xr.Dataset, k: int | None = None, per_km: float = 1.0,
                   k_min: int = 3) -> tuple:
    """Points along each path: ``(lat, lon, link index)``. ``k`` fixed, else ``per_km`` x length (>= ``k_min``)."""
    la0, lo0 = links.site_0_lat.values, links.site_0_lon.values
    la1, lo1 = links.site_1_lat.values, links.site_1_lon.values
    lat, lon, owner = [], [], []
    for i in range(la0.size):
        n = k or max(k_min, int(np.ceil(per_km * haversine_m(la0[i], lo0[i], la1[i], lo1[i]) / 1000)))
        s = np.linspace(0, 1, n)
        lat.append(la0[i] + s * (la1[i] - la0[i]))
        lon.append(lo0[i] + s * (lo1[i] - lo0[i]))
        owner.append(np.full(n, i))
    return np.concatenate(lat), np.concatenate(lon), np.concatenate(owner)


def _bilinear(grid: Grid, lat, lon):
    """Indices and weights of the four grid cells around each point (edges clamped)."""
    fi = np.clip((lat - grid.lat[0]) / (grid.lat[1] - grid.lat[0]), 0, grid.lat.size - 1.000001)
    fj = np.clip((lon - grid.lon[0]) / (grid.lon[1] - grid.lon[0]), 0, grid.lon.size - 1.000001)
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    di, dj = fi - i0, fj - j0
    idx = [(i0, j0), (i0 + 1, j0), (i0, j0 + 1), (i0 + 1, j0 + 1)]
    w = [(1 - di) * (1 - dj), di * (1 - dj), (1 - di) * dj, di * dj]
    return idx, w


def gmz_map(link_rain: xr.DataArray, grid: Grid, n_iter: int = 10, k: int | None = None,
            per_km: float = 1.0, power: float = 2.0, radius_m: float = 10_000.0) -> xr.DataArray:
    """GMZ field ``(time, lat, lon)`` from ``link_rain(link, time)``.

    ``link_rain`` needs site coordinates, ``frequency`` (GHz) and ``polarization``. NaN
    links drop out of that time step. Cells with no virtual gauge within ``radius_m`` are NaN.
    """
    from core.maps.idw import idw_weights

    lat, lon, owner = virtual_gauges(link_rain, k, per_km)
    lat0, lon0 = float(np.mean(grid.lat)), float(np.mean(grid.lon))
    px, py = to_local_xy(lat, lon, lat0, lon0)
    glat, glon = grid.mesh()
    gx, gy = to_local_xy(glat.ravel(), glon.ravel(), lat0, lon0)
    W = idw_weights(px, py, gx, gy, power, radius_m)                       # (cells, points)
    idx, bw = _bilinear(grid, lat, lon)
    flat = [i * grid.lon.size + j for i, j in idx]

    f = np.clip(link_rain.frequency.values.astype(float), 1, 100)
    _, b = itu_ab(f, np.asarray(link_rain.polarization.values, dtype=str), "ITU_2005")
    b_pt = b[owner][:, None]

    R = link_rain.transpose("link", ...).values.reshape(link_rain.sizes["link"], -1).astype(float)
    valid = np.isfinite(R)[owner]                                          # (points, time)
    target = np.where(np.isfinite(R), np.clip(R, 0, None), 0.0) ** b[:, None]   # R_link^b
    pts = np.where(valid, np.nan_to_num(R)[owner], 0.0)                    # start: the link value
    n_link = R.shape[0]
    counts = np.bincount(owner, minlength=n_link)[:, None]

    def field(values):
        num, den = W @ (values * valid), W @ valid.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)                     # (cells, time)

    F = field(pts)
    for _ in range(n_iter):
        at_pts = sum(w[:, None] * np.nan_to_num(F[c]) for c, w in zip(flat, bw))
        rb = np.clip(at_pts, 0, None) ** b_pt                              # (points, time)
        mean_rb = np.zeros((n_link, rb.shape[1]))
        np.add.at(mean_rb, owner, rb)
        mean_rb /= counts
        u = target[owner] - mean_rb[owner] + rb
        pts = np.where(valid, np.clip(u, 0, None) ** (1.0 / b_pt), 0.0)
        F = field(pts)

    other = [d for d in link_rain.dims if d != "link"]
    out = F.reshape(grid.shape + tuple(link_rain.sizes[d] for d in other))
    out = np.moveaxis(out, [0, 1], [-2, -1])
    coords = {d: link_rain[d].values for d in other}
    coords.update(lat=grid.lat, lon=grid.lon)
    return xr.DataArray(out.astype("float32"), dims=other + ["lat", "lon"], coords=coords,
                        name=link_rain.name or "rain",
                        attrs=dict(link_rain.attrs, interpolation="GMZ" if n_iter else "line IDW",
                                   n_iter=n_iter, radius_m=radius_m, power=power))
