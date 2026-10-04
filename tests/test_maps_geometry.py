"""core.maps.geometry: fields along link paths and distances to the nearest path."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.geo import haversine_m
from core.maps import geometry as g


def _links(n=6, seed=0):
    rng = np.random.default_rng(seed)
    la0, lo0 = 57.70 + 0.05 * rng.random(n), 11.95 + 0.08 * rng.random(n)
    return pd.DataFrame({"site_0_lat": la0, "site_0_lon": lo0,
                         "site_1_lat": la0 + 0.02 * rng.standard_normal(n),
                         "site_1_lon": lo0 + 0.03 * rng.standard_normal(n)},
                        index=pd.Index([f"l{i}" for i in range(n)], name="link"))


def _field(nt=3, seed=1):
    rng = np.random.default_rng(seed)
    lat, lon = np.linspace(57.6, 57.85, 60), np.linspace(11.8, 12.15, 70)
    return xr.DataArray(rng.gamma(0.8, 2.0, (nt, lat.size, lon.size)), dims=("time", "lat", "lon"),
                        coords={"time": pd.date_range("2015-07-01", periods=nt, freq="h"),
                                "lat": lat, "lon": lon})


def test_path_average_points_uniform_field_is_the_field():
    f = _field() * 0 + 2.5
    out = g.path_average_points(f, _links())
    assert out.dims == ("link", "time")
    np.testing.assert_allclose(out.values, 2.5)


def test_path_average_points_matches_a_hand_written_mean():
    f, links = _field(), _links()
    out = g.path_average_points(f, links, n_samples=7)
    r = links.iloc[2]
    s = np.linspace(0, 1, 7)
    lat = r.site_0_lat + s * (r.site_1_lat - r.site_0_lat)
    lon = r.site_0_lon + s * (r.site_1_lon - r.site_0_lon)
    i = np.abs(f.lat.values[:, None] - lat).argmin(0)
    j = np.abs(f.lon.values[:, None] - lon).argmin(0)
    np.testing.assert_allclose(out.sel(link="l2").values, f.values[:, i, j].mean(1))


def test_mrms_path_average_is_the_shared_one():
    from core.radar.mrms.maps import path_average
    f, links = _field(), _links()
    xr.testing.assert_identical(path_average(f, links), g.path_average_points(f, links))


def test_segment_distance_point_to_segment():
    # unit segment along x from (0,0) to (1000,0) m
    d = g.segment_distance_km([0], [0], [1000], [0], np.array([500, -300, 1400]), np.array([200, 400, 0]))
    np.testing.assert_allclose(d, [0.2, 0.5, 0.4])


def test_segment_distance_keeps_shape_and_takes_the_nearest():
    x, y = np.meshgrid(np.arange(0, 3000, 500.0), np.arange(0, 2000, 500.0))
    d = g.segment_distance_km([0, 0], [0, 1500], [3000, 3000], [0, 1500], x, y, chunk=5)
    assert d.shape == x.shape
    np.testing.assert_allclose(d, np.minimum(np.abs(y), np.abs(y - 1500)) / 1000)


def test_distance_to_links_km_zero_on_the_path_and_monotone_away():
    links = _links(1)
    r = links.iloc[0]
    mid_lat, mid_lon = (r.site_0_lat + r.site_1_lat) / 2, (r.site_0_lon + r.site_1_lon) / 2
    d = g.distance_to_links_km(np.array([mid_lat, mid_lat + 0.01, mid_lat + 0.02]),
                               np.array([mid_lon] * 3), links, step_m=50)
    assert d[0] < 0.05
    assert d[0] < d[1] < d[2]


def test_distance_to_links_km_matches_sampled_haversine():
    links = _links(3)
    lat, lon = np.meshgrid(np.linspace(57.65, 57.8, 9), np.linspace(11.9, 12.1, 11), indexing="ij")
    best = np.full(lat.shape, np.inf)
    for _, r in links.iterrows():
        n = max(2, int(haversine_m(r.site_0_lat, r.site_0_lon, r.site_1_lat, r.site_1_lon) / 100) + 1)
        for s in np.linspace(0, 1, n):
            best = np.minimum(best, haversine_m(lat, lon, r.site_0_lat + s * (r.site_1_lat - r.site_0_lat),
                                                r.site_0_lon + s * (r.site_1_lon - r.site_0_lon)))
    np.testing.assert_array_equal(g.distance_to_links_km(lat, lon, links, step_m=100), best / 1000)


def test_path_average_intersect_rejects_unknown_plane():
    with pytest.raises(ValueError):
        g.path_average_intersect(_field(), xr.Dataset(), plane="polar")


def test_apply_weights_excludes_nan_sources_per_step():
    from core.maps.idw import apply_weights
    W = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    V = np.array([[1.0, np.nan], [3.0, 3.0], [9.0, 9.0]])
    out = apply_weights(W, V)
    np.testing.assert_allclose(out[0], [2.0, 3.0])
    assert np.isnan(out[1]).all()


def test_points_idw_map_reproduces_a_station_and_matches_idw_map():
    from core.geo import Grid
    from core.maps.idw import idw_map, points_idw_map
    st = xr.DataArray([[1.0, 2.0], [5.0, 6.0]], dims=("station", "time"),
                      coords={"station": ["a", "b"], "lat": ("station", [57.70, 57.75]),
                              "lon": ("station", [11.95, 12.00]),
                              "time": pd.date_range("2015-07-01", periods=2, freq="h")})
    grid = Grid(np.array([57.70, 57.725, 57.75]), np.array([11.95, 11.975, 12.00]))
    out = points_idw_map(st, grid, power=2.0, radius_m=None)
    np.testing.assert_allclose(out.sel(lat=57.70, lon=11.95).values, [1.0, 2.0])
    as_links = st.rename(station="link").assign_coords(mid_lat=("link", st.lat.values),
                                                       mid_lon=("link", st.lon.values))
    xr.testing.assert_identical(out, idw_map(as_links, grid, power=2.0, radius_m=None))


def test_standardised_semivariogram_rises_with_lag_for_a_smooth_field():
    from scipy.ndimage import gaussian_filter
    from core.maps.mergeplg_methods import standardised_semivariogram
    rng = np.random.default_rng(0)
    R = np.stack([np.maximum(gaussian_filter(rng.standard_normal((40, 40)), 5) * 10 + 1, 0).ravel()
                  for _ in range(5)])
    x, y = (v.ravel() for v in np.meshgrid(np.arange(40) * 1000.0, np.arange(40) * 1000.0))
    lag, gamma, n = standardised_semivariogram(R, x, y, max_lag_m=30_000.0)
    assert n == 5
    assert gamma[0] < gamma[len(gamma) // 2]


def test_idw_map_link_weights_pull_towards_the_heavier_link():
    from core.geo import Grid
    from core.maps.idw import idw_map
    links = xr.DataArray([[1.0], [5.0]], dims=("link", "time"),
                         coords={"link": ["a", "b"], "time": pd.date_range("2015-07-01", periods=1, freq="h"),
                                 "mid_lat": ("link", [57.70, 57.70]), "mid_lon": ("link", [11.90, 12.10])})
    grid = Grid(np.array([57.70]), np.array([12.0]))
    mid = float(idw_map(links, grid, radius_m=None).squeeze())
    assert mid == pytest.approx(3.0)
    assert float(idw_map(links, grid, radius_m=None, weights=[1.0, 3.0]).squeeze()) == pytest.approx(4.0)
    xr.testing.assert_identical(idw_map(links, grid, radius_m=None),
                                idw_map(links, grid, radius_m=None, weights=None))
