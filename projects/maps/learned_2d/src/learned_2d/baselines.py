"""The proposal's three model-driven baselines on a ``core.maps.learning`` dataset.

``idw``  inverse-distance weighting from link midpoints (``core.maps.idw``)
``ok``   ordinary kriging from link midpoints (``pykrige``), one hour at a time, with a
         spherical variogram whose shape is fitted to the *training* radar fields
         (``core.maps.mergeplg_methods.fit_radar_variogram``) and whose sill is each
         hour's link variance
``gmz``  Goldshtein-Messer-Zinevich virtual gauges along each path (``core.maps.gmz``)

All three use the links only, power 2 / radius 10 km where it applies, and leave cells
farther than ``radius_m`` from every link empty rather than extrapolate into them.

    from learned_2d.baselines import baseline_maps
    maps = baseline_maps(ds.sel(time=ds.split == "test"), variogram)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.geo import Grid, to_local_xy
from core.maps.gmz import gmz_map
from core.maps.idw import idw_map
from core.maps.learning import link_rain_of

RADIUS_M = 10_000.0
MIN_LINKS = 3


def fit_variogram(ds: xr.Dataset, split: str = "train") -> dict:
    """Spherical variogram shape from the radar fields of ``split`` (no test hour is used)."""
    from core.maps.mergeplg_methods import fit_radar_variogram
    return fit_radar_variogram(ds.target.sel(time=ds.split == split))


def ordinary_kriging_map(link_rain: xr.DataArray, grid: Grid, variogram: dict,
                         radius_m: float = RADIUS_M, min_links: int = MIN_LINKS) -> xr.DataArray:
    """Ordinary kriging of link midpoints, ``(time, lat, lon)``; negative values set to 0.

    ``variogram``: ``{"range" (m), "nugget" (share of the sill)}``. Hours with fewer than
    ``min_links`` valid links are NaN; an hour with no spread gets the common value.
    """
    from pykrige.ok import OrdinaryKriging

    lat0, lon0 = float(np.mean(grid.lat)), float(np.mean(grid.lon))
    sx, sy = to_local_xy(link_rain.mid_lat.values, link_rain.mid_lon.values, lat0, lon0)
    glat, glon = grid.mesh()
    gx, gy = to_local_xy(glat.ravel(), glon.ravel(), lat0, lon0)
    near = np.hypot(gx[:, None] - sx[None, :], gy[:, None] - sy[None, :]).min(axis=1) <= radius_m
    V = link_rain.transpose("link", "time").values.astype(float)
    out = np.full((V.shape[1], gx.size), np.nan)
    for t in range(V.shape[1]):
        ok = np.isfinite(V[:, t])
        if ok.sum() < min_links:
            continue
        z = V[ok, t]
        if z.std() == 0:
            out[t] = z[0]
            continue
        var = float(z.var())
        model = OrdinaryKriging(sx[ok], sy[ok], z, variogram_model="spherical",
                                variogram_parameters={"sill": var, "range": float(variogram["range"]),
                                                      "nugget": float(variogram["nugget"]) * var},
                                verbose=False, enable_plotting=False)
        est, _ = model.execute("points", gx, gy, backend="vectorized")
        out[t] = np.clip(np.asarray(est), 0, None)
    out[:, ~near] = np.nan
    return xr.DataArray(out.reshape((V.shape[1],) + grid.shape).astype("float32"), dims=("time", "lat", "lon"),
                        coords={"time": link_rain.time.values, "lat": grid.lat, "lon": grid.lon},
                        name="ok", attrs={"interpolation": "ordinary kriging", "variogram": str(variogram)})


def baseline_maps(ds: xr.Dataset, variogram: dict, which=("idw", "ok", "gmz"),
                  radius_m: float = RADIUS_M) -> dict:
    """``{name: (time, lat, lon)}`` on the dataset's grid and times, from its link table."""
    grid = Grid(ds.lat.values, ds.lon.values)
    links = link_rain_of(ds)
    out = {}
    if "idw" in which:
        out["idw"] = idw_map(links, grid, power=2.0, radius_m=radius_m)
    if "ok" in which:
        out["ok"] = ordinary_kriging_map(links, grid, variogram, radius_m)
    if "gmz" in which:
        out["gmz"] = gmz_map(links, grid, n_iter=10, power=2.0, radius_m=radius_m)
    return {k: v.transpose("time", "lat", "lon") for k, v in out.items()}
