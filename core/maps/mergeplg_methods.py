"""The merging methods of the OpenSense package ``mergeplg`` on hourly lat/lon maps.

``mergeplg`` 0.1.0 merges radar with links and gauges one time step at a time, over the
whole grid, with a Python loop over grid cells - minutes per hour of data on a regional
grid, and a study that scores at held-out gauges needs the field at a handful of cells,
rebuilt for every fold. This module runs the same methods:

* the geometry is ``mergeplg``'s own (``calculate_cml_line`` / ``calculate_cml_midpoint``,
  lines cut into ``discretization`` intervals, gauges as zero-length lines);
* the radar along each link comes from ``poligrain``'s intersect weights, at a gauge from
  the cell containing it, as in ``mergeplg.base``;
* each hour follows the ``adjust`` of the corresponding class line by line - which
  observations are kept, the minimum number of observations, neighbourhoods, the kriging
  systems (``pinv`` of the same matrices), clipping - but is evaluated at any set of
  target cells, vectorised over targets, with the block-to-block variogram of all
  observations computed once;

and ``tests/test_mergeplg_methods.py`` checks the result against ``mergeplg``'s own
``adjust()`` on the full grid.

====================  =============================================  ==========================
key                   mergeplg 0.1.0                                 settings (class defaults)
====================  =============================================  ==========================
``idw_add``           ``MergeDifferenceIDW(method="additive")``      RADOLAN IDW, 8 nearest, 60 km
``idw_mul``           ``MergeDifferenceIDW(method="multiplicative")``    as above
``okrig_add``         ``MergeDifferenceOrdinaryKriging`` additive    block kriging, 8 nearest
``ked``               ``MergeKrigingExternalDrift``                  block KED, 8 nearest
====================  =============================================  ==========================

All need more than ``min_observations`` (5) usable observations in an hour; otherwise the
radar is returned unchanged. The kriging methods take a spherical variogram; mergeplg's
default (sill 0.9, range 5 km, nugget 0.1) or one fitted to the radar fields
(:func:`fit_radar_variogram`, gauge-free). Ordinary kriging and KED weights do not change
when the whole variogram is scaled, so only its shape (range, nugget share) matters.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.geo import Grid, to_local_xy

METHODS = ("idw_add", "idw_mul", "okrig_add", "ked")
DEFAULT_VARIOGRAM = {"sill": 0.9, "range": 5000.0, "nugget": 0.1}
IDW = {"nnear": 8, "max_distance": 60_000.0, "idw_method": "radolan", "p": 2}
N_CLOSEST = 8
MIN_OBSERVATIONS = 5
DISCRETIZATION = 8


def spherical(params: dict, h: np.ndarray) -> np.ndarray:
    """pykrige's spherical model with ``{"sill", "range", "nugget"}`` (sill includes the nugget)."""
    from pykrige.variogram_models import spherical_variogram_model
    p = [params["sill"] - params["nugget"], params["range"], params["nugget"]]
    return spherical_variogram_model(p, h)


class Merger:
    """``mergeplg`` merging of hourly ``radar(time, lat, lon)`` with links and/or gauges.

    ``links``: ``(link, time)`` hour-ending mm with site coordinates; ``gauges``:
    ``(station, time)`` with ``lat``/``lon``. Geometry and the radar at every observation
    are computed once; :meth:`adjust` then gives any method at any target cells.
    """

    def __init__(self, radar: xr.DataArray, links: xr.DataArray | None = None,
                 gauges: xr.DataArray | None = None, variogram: dict | None = None,
                 discretization: int = DISCRETIZATION):
        import poligrain as plg
        from mergeplg.base import calculate_cml_line

        if links is None and gauges is None:
            raise ValueError("need links or gauges")
        self.grid = Grid(radar.lat.values, radar.lon.values)
        self.times = radar.time.values
        self.radar = radar.transpose("time", "lat", "lon").values.reshape(len(self.times), -1).astype(float)
        self.lat0, self.lon0 = float(np.mean(self.grid.lat)), float(np.mean(self.grid.lon))
        glat, glon = self.grid.mesh()
        gx, gy = to_local_xy(glat, glon, self.lat0, self.lon0)
        self.x_grid, self.y_grid = gx, gy                          # (lat, lon) metres
        self.variogram = dict(variogram or DEFAULT_VARIOGRAM)

        obs, rad, blocks = [], [], []
        if links is not None and links.sizes.get("link", 0):
            L = links.transpose("link", "time").reindex(time=self.times)
            x0, y0 = to_local_xy(L.site_0_lat.values, L.site_0_lon.values, self.lat0, self.lon0)
            x1, y1 = to_local_xy(L.site_1_lat.values, L.site_1_lon.values, self.lat0, self.lon0)
            ids = np.asarray(L.link.values).astype(str)
            geo = xr.Dataset(coords={"cml_id": ids, "site_0_x": ("cml_id", x0), "site_0_y": ("cml_id", y0),
                                     "site_1_x": ("cml_id", x1), "site_1_y": ("cml_id", y1)})
            blocks.append(calculate_cml_line(geo, discretization=discretization).values)
            w = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
                x1_line=x0, y1_line=y0, x2_line=x1, y2_line=y1, cml_id=ids,
                x_grid=gx, y_grid=gy, grid_point_location="center")
            da_grid = xr.DataArray(self.radar.reshape((len(self.times),) + self.grid.shape),
                                   dims=("time", "y", "x"), coords={"time": self.times})
            r = plg.spatial.get_grid_time_series_at_intersections(grid_data=da_grid, intersect_weights=w)
            rad.append(r.transpose("cml_id", "time").values)
            obs.append(L.values.astype(float))
        if gauges is not None and gauges.sizes.get("station", 0):
            G = gauges.transpose("station", "time").reindex(time=self.times)
            sx, sy = to_local_xy(G.lat.values, G.lon.values, self.lat0, self.lon0)
            yx = np.stack([sy, sx], axis=1)[:, :, None]
            blocks.append(np.repeat(yx, discretization + 1, axis=2))
            cell = self._nearest_cell(sx, sy)
            rad.append(self.radar[:, cell].T)
            obs.append(G.values.astype(float))
        self.obs = np.concatenate(obs)                      # (n, time)
        self.rad = np.concatenate(rad)                      # radar at each observation
        self.x0 = np.concatenate(blocks)                    # (n, y/x, disc + 1)
        self.mid = self.x0[:, :, discretization // 2]       # midpoints (y, x); needs an even discretization
        self._cov = None

    # ----------------------------------------------------------------- geometry
    def _nearest_cell(self, x, y) -> np.ndarray:
        d = (self.x_grid.ravel()[None, :] - np.asarray(x)[:, None]) ** 2 + \
            (self.y_grid.ravel()[None, :] - np.asarray(y)[:, None]) ** 2
        return d.argmin(axis=1)

    def cells_at(self, lat, lon) -> np.ndarray:
        """Flat indices of the grid cells nearest to the points (lat, lon)."""
        x, y = to_local_xy(np.asarray(lat), np.asarray(lon), self.lat0, self.lon0)
        return self._nearest_cell(x, y)

    def _block_cov(self) -> np.ndarray:
        """``-mean(gamma)`` between every pair of observation blocks, diagonal 0 (mergeplg)."""
        if self._cov is None:
            n = self.x0.shape[0]
            cov = np.empty((n, n))
            for i in range(n):          # rows one at a time: n x d x d distances
                dy = self.x0[i, 0][None, None, :] - self.x0[:, 0][:, :, None]
                dx = self.x0[i, 1][None, None, :] - self.x0[:, 1][:, :, None]
                cov[i] = -spherical(self.variogram, np.hypot(dx, dy)).mean(axis=(1, 2))
            np.fill_diagonal(cov, 0.0)
            self._cov = cov
        return self._cov

    def _target_lengths(self, cells: np.ndarray, keep: np.ndarray) -> np.ndarray:
        """Distance from each target cell to every point of every kept block ``(Q, n, d)``."""
        ty, tx = self.y_grid.ravel()[cells], self.x_grid.ravel()[cells]
        x0 = self.x0[keep]
        return np.hypot(x0[None, :, 1, :] - tx[:, None, None], x0[None, :, 0, :] - ty[:, None, None])

    # ------------------------------------------------------------------ methods
    def adjust(self, method: str, cells: np.ndarray | None = None, obs_mask: np.ndarray | None = None,
               chunk: int = 2000) -> np.ndarray:
        """``(time, target)`` merged field at flat cell indices ``cells`` (default: all).

        ``obs_mask`` (bool per observation) leaves observations out - a held-out fold -
        without rebuilding the geometry."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        cells = np.arange(self.radar.shape[1]) if cells is None else np.asarray(cells)
        use = np.ones(self.obs.shape[0], bool) if obs_mask is None else np.asarray(obs_mask, bool)
        out = np.empty((len(self.times), len(cells)))
        for t in range(len(self.times)):
            rq = self.radar[t, cells]
            out[t] = self._hour(method, t, cells, rq, use, chunk)
        return out

    def _hour(self, method, t, cells, rq, use, chunk) -> np.ndarray:
        obs, rad = self.obs[:, t], self.rad[:, t]
        with np.errstate(invalid="ignore", divide="ignore"):
            if method == "ked":
                ok = use & ~np.isnan(obs) & ~np.isnan(rad) & (obs > 0) & (rad > 0)
            elif method in ("idw_add", "okrig_add"):
                diff = np.where(rad > 0, obs - rad, np.nan)
                ok = use & ~np.isnan(diff)
            else:
                diff = np.where(rad > 0.0, obs / rad, np.nan)
                ok = use & ~np.isnan(diff)
        keep = np.flatnonzero(ok)
        if keep.size <= MIN_OBSERVATIONS:
            return rq.copy()
        if method in ("idw_add", "idw_mul"):
            from mergeplg.radolan.idw import Invdisttree
            q = np.stack([self.y_grid.ravel()[cells], self.x_grid.ravel()[cells]], axis=1)
            nn = min(keep.size, IDW["nnear"])
            f = Invdisttree(self.mid[keep])(q=q, z=diff[keep], nnear=nn, p=IDW["p"],
                                            idw_method=IDW["idw_method"], max_distance=IDW["max_distance"])
            adj = rq + f if method == "idw_add" else f * rq
            return np.where(adj < 0, 0.0, adj)
        cov = self._block_cov()[np.ix_(keep, keep)]
        nn = min(keep.size, N_CLOSEST)
        est = np.full(len(cells), np.nan)
        if method == "okrig_add":
            targets = np.arange(len(cells))
            vals, drift = diff[keep], None
        else:
            rfield = np.where(rq <= 0, np.nan, rq)            # mergeplg: zero radar -> NaN -> 0
            targets = np.flatnonzero(~np.isnan(rfield))
            vals, drift = obs[keep], rad[keep]
        for s in range(0, len(targets), chunk):
            idx = targets[s:s + chunk]
            L = self._target_lengths(cells[idx], keep)                     # (Q, n, d)
            near = L.min(axis=2)
            ind = np.argpartition(near, nn - 1, axis=1)[:, :nn]           # (Q, nn)
            g = -spherical(self.variogram, np.take_along_axis(L, ind[:, :, None], axis=1)).mean(axis=2)
            Q = len(idx)
            if drift is None:
                A = np.zeros((Q, nn + 1, nn + 1))
                A[:, :nn, :nn] = cov[ind[:, :, None], ind[:, None, :]]
                A[:, nn, :nn] = 1.0
                A[:, :nn, nn] = 1.0
                b = np.concatenate([g, np.ones((Q, 1))], axis=1)
            else:
                A = np.zeros((Q, nn + 2, nn + 2))
                A[:, :nn, :nn] = cov[ind[:, :, None], ind[:, None, :]]
                A[:, nn, :nn] = 1.0
                A[:, :nn, nn] = 1.0
                A[:, nn + 1, :nn] = drift[ind]
                A[:, :nn, nn + 1] = drift[ind]
                b = np.concatenate([g, np.ones((Q, 1)), rfield[idx][:, None]], axis=1)
            w = np.einsum("qij,qj->qi", np.linalg.pinv(A), b)[:, :nn]
            est[idx] = np.einsum("qi,qi->q", w, vals[ind])
        if method == "okrig_add":
            adj = rq + est
            return np.where(adj < 0, 0.0, adj)
        return np.where((est < 0) | np.isnan(est), 0.0, est)

    def field(self, method: str, name: str | None = None, obs_mask=None) -> xr.DataArray:
        """The merged map ``(time, lat, lon)`` on the whole grid."""
        out = self.adjust(method, obs_mask=obs_mask).reshape((len(self.times),) + self.grid.shape)
        return xr.DataArray(out.astype("float32"), dims=("time", "lat", "lon"),
                            coords={"time": self.times, "lat": self.grid.lat, "lon": self.grid.lon},
                            name=name or method, attrs={"units": "mm", "merging": f"mergeplg {method}",
                                                        "variogram": str(self.variogram)})


def fit_radar_variogram(radar: xr.DataArray, min_wet_fraction: float = 0.2, wet_mm: float = 0.1,
                        max_lag_m: float = 60_000.0, n_pairs: int = 200_000, seed: int = 0) -> dict:
    """A spherical variogram shape from the radar's own hourly fields (no gauge is used).

    Each hour with at least ``min_wet_fraction`` wet cells is standardised (zero mean, unit
    variance); semivariances of random cell pairs are pooled in distance bins and a
    spherical model is fitted by least squares. Returns ``{"sill", "range", "nugget"}``
    with sill 1 (the scale does not change kriging weights) and the fit's inputs.
    """
    from scipy.optimize import curve_fit

    g = Grid(radar.lat.values, radar.lon.values)
    glat, glon = g.mesh()
    x, y = to_local_xy(glat.ravel(), glon.ravel(), float(np.mean(g.lat)), float(np.mean(g.lon)))
    R = radar.transpose("time", "lat", "lon").values.reshape(radar.sizes["time"], -1)
    rng = np.random.default_rng(seed)
    wet = [h for h in R if np.isfinite(h).mean() > 0.5 and (h[np.isfinite(h)] >= wet_mm).mean() >= min_wet_fraction]
    if not wet:
        return dict(DEFAULT_VARIOGRAM, fitted=False)
    per = max(1, n_pairs // len(wet))
    bins = np.linspace(0, max_lag_m, 31)
    num, den = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    for h in wet:
        ok = np.flatnonzero(np.isfinite(h))
        z = (h[ok] - h[ok].mean()) / (h[ok].std() or 1.0)
        i, j = rng.integers(0, ok.size, per), rng.integers(0, ok.size, per)
        d = np.hypot(x[ok[i]] - x[ok[j]], y[ok[i]] - y[ok[j]])
        b = np.digitize(d, bins) - 1
        m = (b >= 0) & (b < len(num)) & (i != j)
        np.add.at(num, b[m], 0.5 * (z[i[m]] - z[j[m]]) ** 2)
        np.add.at(den, b[m], 1)
    lag = 0.5 * (bins[1:] + bins[:-1])
    ok = den > 50
    gamma = num[ok] / den[ok]

    def model(h, sill, rng_, nugget):
        return spherical({"sill": sill, "range": rng_, "nugget": nugget}, h)
    (sill, range_, nugget), _ = curve_fit(model, lag[ok], gamma, p0=[1.0, 20_000.0, 0.1],
                                          bounds=([0.1, 1_000.0, 0.0], [3.0, 4 * max_lag_m, 1.0]))
    nugget = min(nugget, sill)
    return {"sill": 1.0, "range": float(range_), "nugget": float(nugget / sill), "fitted": True,
            "hours": len(wet)}
