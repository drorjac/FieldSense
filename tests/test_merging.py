"""Radar/link/gauge merging: core.maps.merge and the mergeplg methods of core.maps.mergeplg_methods."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from core.geo import Grid, to_local_xy
from core.maps import merge
from core.maps.mergeplg_methods import DEFAULT_VARIOGRAM, METHODS, Merger, fit_radar_variogram

LAT = np.round(np.arange(40.60, 40.80, 0.01) + 0.005, 3)
LON = np.round(np.arange(-74.00, -73.85, 0.01) + 0.005, 3)
TIMES = np.array(["2024-01-01T01", "2024-01-01T02", "2024-01-01T03"], dtype="datetime64[ns]")


def _inputs(seed=0, n_links=12, n_gauges=9):
    rng = np.random.default_rng(seed)
    glat, glon = np.meshgrid(LAT, LON, indexing="ij")
    base = 2 + np.sin(glat * 40) + np.cos(glon * 30)
    radar = xr.DataArray(np.stack([base * f for f in (0.5, 1.0, 1.5)]) + rng.gamma(1, 0.3, (3,) + base.shape),
                         dims=("time", "lat", "lon"), coords={"time": TIMES, "lat": LAT, "lon": LON})
    radar.values[1, :3, :4] = 0.0                                   # some dry cells
    la0, lo0 = rng.uniform(40.62, 40.78, n_links), rng.uniform(-73.98, -73.87, n_links)
    la1, lo1 = la0 + rng.uniform(-0.02, 0.02, n_links), lo0 + rng.uniform(-0.02, 0.02, n_links)
    links = xr.DataArray(rng.gamma(2, 1.0, (n_links, 3)), dims=("link", "time"),
                         coords={"link": [f"l{i}" for i in range(n_links)], "time": TIMES,
                                 "site_0_lat": ("link", la0), "site_0_lon": ("link", lo0),
                                 "site_1_lat": ("link", la1), "site_1_lon": ("link", lo1),
                                 "mid_lat": ("link", (la0 + la1) / 2), "mid_lon": ("link", (lo0 + lo1) / 2)})
    links.values[0, 0] = np.nan
    gauges = xr.DataArray(rng.gamma(2, 1.0, (n_gauges, 3)), dims=("station", "time"),
                          coords={"station": [f"g{i}" for i in range(n_gauges)], "time": TIMES,
                                  "lat": ("station", rng.uniform(40.61, 40.79, n_gauges)),
                                  "lon": ("station", rng.uniform(-73.99, -73.86, n_gauges))})
    return radar, links, gauges


def _mergeplg_inputs(radar, links, gauges, t):
    """The same data in mergeplg's own layout (projected metres, one time step)."""
    lat0, lon0 = float(LAT.mean()), float(LON.mean())
    glat, glon = np.meshgrid(LAT, LON, indexing="ij")
    gx, gy = to_local_xy(glat, glon, lat0, lon0)
    da_rad = xr.DataArray(radar.values[t].copy(), dims=("y", "x"),
                          coords={"x_grid": (("y", "x"), gx), "y_grid": (("y", "x"), gy)})
    x0, y0 = to_local_xy(links.site_0_lat.values, links.site_0_lon.values, lat0, lon0)
    x1, y1 = to_local_xy(links.site_1_lat.values, links.site_1_lon.values, lat0, lon0)
    da_cml = xr.DataArray(links.values[:, t], dims="cml_id",
                          coords={"cml_id": links.link.values, "site_0_x": ("cml_id", x0),
                                  "site_0_y": ("cml_id", y0), "site_1_x": ("cml_id", x1),
                                  "site_1_y": ("cml_id", y1), "x": ("cml_id", (x0 + x1) / 2),
                                  "y": ("cml_id", (y0 + y1) / 2)})
    sx, sy = to_local_xy(gauges.lat.values, gauges.lon.values, lat0, lon0)
    da_gauge = xr.DataArray(gauges.values[:, t], dims="id",
                            coords={"id": gauges.station.values, "x": ("id", sx), "y": ("id", sy),
                                    "lon": ("id", gauges.lon.values), "lat": ("id", gauges.lat.values)})
    return da_rad, da_cml, da_gauge


def _reference(method, da_rad, da_cml, da_gauge):
    from mergeplg import merge as mm
    if method.startswith("idw"):
        obj = mm.MergeDifferenceIDW()
        out = obj.adjust(da_rad, da_cml=da_cml, da_gauge=da_gauge,
                         method="additive" if method == "idw_add" else "multiplicative")
    elif method == "okrig_add":
        obj = mm.MergeDifferenceOrdinaryKriging()
        out = obj.adjust(da_rad, da_cml=da_cml, da_gauge=da_gauge, variogram_parameters=dict(DEFAULT_VARIOGRAM))
    else:
        obj = mm.MergeKrigingExternalDrift()
        out = obj.adjust(da_rad, da_cml=da_cml, da_gauge=da_gauge, variogram_parameters=dict(DEFAULT_VARIOGRAM))
    return np.asarray(out)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("sources", ["links", "gauges", "both"])
def test_matches_mergeplg(method, sources):
    pytest.importorskip("mergeplg")
    radar, links, gauges = _inputs()
    L = links if sources in ("links", "both") else None
    G = gauges if sources in ("gauges", "both") else None
    m = Merger(radar, L, G)
    ours = m.adjust(method)
    for t in range(len(TIMES)):
        da_rad, da_cml, da_gauge = _mergeplg_inputs(radar, links, gauges, t)
        ref = _reference(method, da_rad, da_cml if L is not None else None, da_gauge if G is not None else None)
        np.testing.assert_allclose(ours[t], ref.ravel(), rtol=1e-6, atol=1e-6, equal_nan=True)


def test_targets_and_held_out():
    radar, links, gauges = _inputs(1)
    m = Merger(radar, links, gauges)
    full = m.adjust("okrig_add")
    cells = np.array([3, 40, 77])
    np.testing.assert_allclose(m.adjust("okrig_add", cells=cells), full[:, cells])
    # holding out observations equals building without them
    mask = np.ones(m.obs.shape[0], bool)
    mask[-3:] = False
    held = m.adjust("ked", obs_mask=mask)
    np.testing.assert_allclose(held, Merger(radar, links, gauges.isel(station=slice(0, -3))).adjust("ked"))


def test_fit_radar_variogram_shape():
    radar, _, _ = _inputs(2)
    v = fit_radar_variogram(radar, n_pairs=20_000)
    assert v["fitted"] and v["sill"] == 1.0 and 0 <= v["nugget"] <= 1 and v["range"] > 0


def test_merge_adjust_family():
    radar, links, gauges = _inputs(3)
    obs, rad = merge.observations(radar, cml=links, gauges=gauges)
    assert obs.sizes["point"] == links.sizes["link"] + gauges.sizes["station"]
    assert set(obs.kind.values) == {"cml", "gauges"}
    # mean-field bias: at every hour, sum over points of the adjusted radar equals the obs sum
    both = np.isfinite(obs.values) & np.isfinite(rad.values)
    mfb = merge.adjust(radar, obs, rad, "mfb", max_factor=100)
    f = mfb.values / radar.values
    for t in range(len(TIMES)):
        expected = np.where(both[:, t], obs.values[:, t], 0).sum() / np.where(both[:, t], rad.values[:, t], 0).sum()
        assert np.nanmedian(f[t]) == pytest.approx(expected, rel=1e-5)
    for method in ("add", "mul"):
        out = merge.adjust(radar, obs, rad, method)
        assert out.shape == radar.shape and np.isfinite(out.values).all() and (out.values >= 0).all()
    m = merge.merge_idw(obs, Grid(LAT, LON))
    assert m.shape == radar.shape
