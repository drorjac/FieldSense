"""Algorithm audit of core/maps: each method against its definition or a hand calculation.

Synthetic, fast, no downloads. See docs/algorithm_audit.md for the verdicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.geo import Grid, haversine_m, to_local_xy
from core.maps import merge
from core.maps.geometry import (distance_to_links_km, path_average_intersect, path_average_points,
                                segment_distance_km)
from core.maps.gmz import gmz_map, virtual_gauges
from core.maps.idw import accumulate, apply_weights, idw_map, idw_weights, points_idw_map
from core.maps.scores import scores
from core.maps.wet_area import conditional_idw, masked_idw, wet_area_scores, wet_probability

LAT = np.round(np.arange(40.60, 40.80, 0.01) + 0.005, 3)
LON = np.round(np.arange(-74.00, -73.85, 0.01) + 0.005, 3)
GRID = Grid(LAT, LON)
TIMES = np.array(["2024-01-01T01", "2024-01-01T02"], dtype="datetime64[ns]")


def _stations(lat, lon, values):
    values = np.asarray(values, float)
    return xr.DataArray(values, dims=("station", "time"),
                        coords={"station": [f"g{i}" for i in range(len(lat))],
                                "time": TIMES[:values.shape[1]],
                                "lat": ("station", np.asarray(lat, float)),
                                "lon": ("station", np.asarray(lon, float))})


# --------------------------------------------------------------------------- idw
def test_audit_idw_weights_exact_at_source_and_radius():
    sx, sy = np.array([0.0, 1000.0, 5000.0]), np.zeros(3)
    dx, dy = np.array([0.0, 500.0, 20_000.0]), np.zeros(3)
    w = idw_weights(sx, sy, dx, dy, power=2.0, radius_m=10_000.0)
    np.testing.assert_array_equal(w[0], [1.0, 0.0, 0.0])            # exact interpolation
    np.testing.assert_allclose(w[1], [1 / 500 ** 2, 1 / 500 ** 2, 1 / 4500 ** 2])
    np.testing.assert_array_equal(w[2], 0.0)                        # nothing within radius
    w8 = idw_weights(sx, sy, dx, dy, nnear=2, radius_m=None)
    assert (w8 > 0).sum(1).tolist() == [1, 2, 2]


def test_audit_apply_weights_nan_excluded_and_hand_value():
    W = np.array([[1.0, 1.0, 0.25], [0.0, 0.0, 0.0]])
    V = np.array([[2.0, np.nan], [4.0, 4.0], [10.0, 10.0]])
    out = apply_weights(W, V)
    np.testing.assert_allclose(out[0], [(2 + 4 + 2.5) / 2.25, (4 + 2.5) / 1.25])
    assert np.isnan(out[1]).all()                                    # no source in range
    assert apply_weights(W.astype("float32"), V.astype("float32"), np.float32).dtype == np.float32


def test_audit_points_idw_exact_at_station_and_bounded():
    lat, lon = LAT[[3, 10, 15]], LON[[2, 7, 12]]
    st = _stations(lat, lon, [[1.0, 0.0], [5.0, 2.0], [3.0, np.nan]])
    m = points_idw_map(st, GRID)
    for i in range(3):
        v = m.sel(lat=lat[i], lon=lon[i]).values
        np.testing.assert_allclose(v[0], st.values[i, 0], rtol=1e-6)
    # an IDW value is a convex combination: inside [min, max] of the sources
    assert float(m.min()) >= 0.0 and float(m.isel(time=0).max()) <= 5.0 + 1e-6


def test_audit_idw_nan_source_on_cell_centre_drops_out():
    lat, lon = LAT[[3, 10, 15]], LON[[2, 7, 12]]
    st = _stations(lat, lon, [[1.0], [5.0], [np.nan]])
    m = points_idw_map(st, GRID)
    assert np.isfinite(m.sel(lat=lat[2], lon=lon[2]).values[0])


def test_audit_idw_nnear_counts_valid_sources_only():
    lat = np.array([LAT[10], LAT[10], LAT[10]])
    lon = np.array([LON[5], LON[6], LON[8]])
    st = _stations(lat, lon, [[np.nan], [2.0], [4.0]])
    m = points_idw_map(st, Grid(LAT[[10]], LON[[4]]), nnear=2, radius_m=None)
    # pycomlink: the two nearest valid sources (2.0 at 2 cells, 4.0 at 4 cells)
    d1, d2 = 2.0, 4.0
    expected = (2.0 / d1 ** 2 + 4.0 / d2 ** 2) / (1 / d1 ** 2 + 1 / d2 ** 2)
    np.testing.assert_allclose(float(m.values.ravel()[0]), expected, rtol=1e-3)


def test_audit_accumulate_interval_ending_and_coverage():
    t = pd.date_range("2024-01-01T00:05", "2024-01-01T03:00", freq="5min")
    rate = xr.DataArray(np.full(t.size, 6.0), dims="time", coords={"time": t})
    rate[(t > "2024-01-01T01:00") & (t <= "2024-01-01T02:00")] = 12.0
    acc = accumulate(rate, "1h")
    assert pd.Timestamp(acc.time.values[0]) == pd.Timestamp("2024-01-01T01:00")   # (00:00, 01:00]
    np.testing.assert_allclose(acc.sel(time="2024-01-01T01:00"), 6.0)            # 6 mm/h x 1 h
    np.testing.assert_allclose(acc.sel(time="2024-01-01T02:00"), 12.0)
    np.testing.assert_allclose(acc.sel(time="2024-01-01T03:00"), 6.0)
    half = rate.where(t > "2024-01-01T00:35")                                     # 5 of 12 samples
    assert np.isnan(float(accumulate(half, "1h").sel(time="2024-01-01T01:00")))
    acc15 = accumulate(rate, "15min")
    np.testing.assert_allclose(acc15.sel(time="2024-01-01T00:15"), 1.5)          # 6 mm/h x 0.25 h


# --------------------------------------------------------------------------- gmz
def _links(rows):
    """Links whose virtual gauges (k=5) fall on grid cell centres."""
    la0, lo0, la1, lo1, val = (np.array(c, float) for c in zip(*rows))
    return xr.DataArray(val[:, None], dims=("link", "time"),
                        coords={"link": [f"l{i}" for i in range(len(rows))], "time": TIMES[:1],
                                "site_0_lat": ("link", la0), "site_0_lon": ("link", lo0),
                                "site_1_lat": ("link", la1), "site_1_lon": ("link", lo1),
                                "mid_lat": ("link", (la0 + la1) / 2), "mid_lon": ("link", (lo0 + lo1) / 2),
                                "frequency": ("link", np.full(len(rows), 23.0)),
                                "polarization": ("link", np.array(["V"] * len(rows)))})


GMZ_LINKS = [(LAT[10] + 0.003, LON[1] + 0.0037, LAT[10] + 0.0071, LON[5] + 0.002, 5.0),
             (LAT[12] + 0.004, LON[2] + 0.001, LAT[13] + 0.002, LON[6] + 0.005, 1.0),
             (LAT[8] + 0.002, LON[8] + 0.0045, LAT[12] + 0.006, LON[8] + 0.003, 3.0)]


def test_audit_gmz_virtual_gauges_keep_path_average():
    """Goldshtein et al. (2009): after the correction the virtual gauges of each link
    average, in the attenuation domain R**b, to the link's own R**b.

    The map is a linear IDW of the virtual gauges, so their values are recovered from
    the returned field by least squares on the same weights.
    """
    from core.cml.power_law import itu_ab
    links = _links(GMZ_LINKS)
    lat, lon, owner = virtual_gauges(links, k=5)
    _, b = itu_ab(np.full(3, 23.0), np.array(["V"] * 3))
    lat0, lon0 = float(LAT.mean()), float(LON.mean())
    px, py = to_local_xy(lat, lon, lat0, lon0)
    glat, glon = GRID.mesh()
    gx, gy = to_local_xy(glat.ravel(), glon.ravel(), lat0, lon0)
    W = idw_weights(px, py, gx, gy, 2.0, 10_000.0)
    ok = W.sum(1) > 0
    A = W[ok] / W[ok].sum(1, keepdims=True)
    spread = []
    for n_iter in (0, 1, 10):
        F = gmz_map(links, GRID, n_iter=n_iter, k=5).isel(time=0).values.ravel().astype(float)[ok]
        pts = np.linalg.lstsq(A, F, rcond=None)[0]
        for i in range(3):
            np.testing.assert_allclose(np.mean(pts[owner == i] ** b[i]), links.values[i, 0] ** b[i], rtol=1e-5)
        spread.append(pts[owner == 1].std())
    # n_iter=0 is uniform along each link; the passes let it lean towards the neighbours
    assert spread[0] < 1e-4 < 0.05 < spread[1] < spread[2]


def test_audit_gmz_zero_iterations_is_line_idw():
    links = _links(GMZ_LINKS)
    F = gmz_map(links, GRID, n_iter=0, k=5)
    lat, lon, owner = virtual_gauges(links, k=5)
    pts = xr.DataArray(links.values[owner], dims=("station", "time"),
                       coords={"station": np.arange(owner.size), "time": TIMES[:1],
                               "lat": ("station", lat), "lon": ("station", lon)})
    np.testing.assert_allclose(F.values, points_idw_map(pts, GRID).values, rtol=1e-5)


# --------------------------------------------------------------------------- merge
def _gauge_obs(values):
    lat, lon = LAT[[2, 5, 9, 14, 18]], LON[[1, 4, 8, 11, 13]]
    return merge.as_points(_stations(lat, lon, values), "gauges")


def _radar(value):
    return xr.DataArray(np.broadcast_to(np.asarray(value, float)[:, None, None],
                                        (len(value), LAT.size, LON.size)).copy(),
                        dims=("time", "lat", "lon"), coords={"time": TIMES[:len(value)], "lat": LAT, "lon": LON})


def test_audit_mean_field_bias_hand_calculation():
    radar = _radar([2.0, 0.05])
    obs = _gauge_obs(np.tile([[3.0, 1.0]], (5, 1)))
    rad_obs = obs.copy(data=np.tile([[2.0, 0.05]], (5, 1)))
    out = merge.adjust(radar, obs, rad_obs, "mfb", min_radar_mm=0.5)
    np.testing.assert_allclose(out.isel(time=0), 3.0, rtol=1e-6)             # sum G / sum R = 1.5
    # hour 2: too little radar at the gauges -> event factor sum(G)/sum(R) over both hours
    f_event = (5 * 3.0 + 5 * 1.0) / (5 * 2.0 + 5 * 0.05)
    assert out.attrs["event_factor"] == pytest.approx(min(f_event, 5.0))
    np.testing.assert_allclose(out.isel(time=1), 0.05 * min(f_event, 5.0), rtol=1e-6)
    big = merge.adjust(radar, obs.copy(data=obs.values * 100), rad_obs, "mfb")
    np.testing.assert_allclose(big.isel(time=0), 2.0 * 5.0, rtol=1e-6)       # clipped at max_factor


def test_audit_additive_and_multiplicative_adjustment():
    radar = _radar([2.0, 4.0])
    rad_obs = _gauge_obs(np.tile([[2.0, 4.0]], (5, 1)))
    obs = rad_obs.copy(data=rad_obs.values + 1.5)
    add = merge.adjust(radar, obs, rad_obs, "add")
    np.testing.assert_allclose(add.values, radar.values + 1.5, rtol=1e-6)    # covered and not
    eps = 0.5
    obs2 = rad_obs.copy(data=2 * (rad_obs.values + eps) - eps)              # (O+eps)/(P+eps) = 2
    mul = merge.adjust(radar, obs2, rad_obs, "mul", eps=eps)
    covered = np.isfinite(points_idw_map(_stations(obs.lat.values, obs.lon.values, rad_obs.values), GRID))
    np.testing.assert_allclose(mul.values[covered.values], 2 * radar.values[covered.values], rtol=1e-6)
    neg = merge.adjust(radar, rad_obs.copy(data=rad_obs.values * 0), rad_obs, "add")
    assert float(neg.min()) == 0.0                                          # clipped at 0


def test_audit_merge_idw_exact_at_gauges():
    obs = _gauge_obs([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    m = merge.merge_idw(obs, GRID)
    for p in range(5):
        v = m.sel(lat=obs.lat.values[p], lon=obs.lon.values[p]).values
        np.testing.assert_allclose(v, obs.values[p], rtol=1e-6)


# ------------------------------------------------------------ mergeplg methods
def _merger_case(seed=0, variogram=None):
    from core.maps.mergeplg_methods import Merger
    glat, glon = np.meshgrid(LAT, LON, indexing="ij")
    base = 2 + np.sin(glat * 40) + np.cos(glon * 30)                         # strictly positive
    radar = xr.DataArray(np.stack([base, 1.5 * base]), dims=("time", "lat", "lon"),
                         coords={"time": TIMES, "lat": LAT, "lon": LON})
    rng = np.random.default_rng(seed)
    ii, jj = rng.choice(LAT.size, 9, replace=False), rng.choice(LON.size, 9, replace=False)
    r_at = radar.values[:, ii, jj].T                                        # (station, time)
    return radar, ii, jj, r_at, lambda vals: Merger(radar, gauges=_stations(LAT[ii], LON[jj], vals),
                                                     variogram=variogram)


def test_audit_mergeplg_methods_reproduce_exact_relations():
    radar, ii, jj, r_at, make = _merger_case()
    R = radar.values.reshape(2, -1)
    np.testing.assert_allclose(make(r_at + 1.0).adjust("idw_add"), R + 1.0, rtol=1e-6)
    np.testing.assert_allclose(make(2.0 * r_at).adjust("idw_mul"), 2.0 * R, rtol=1e-6)
    np.testing.assert_allclose(make(r_at + 1.0).adjust("okrig_add"), R + 1.0, rtol=1e-6)
    # KED: obs exactly linear in the radar -> the drift reproduces it everywhere
    np.testing.assert_allclose(make(0.5 + 2.0 * r_at).adjust("ked"), 0.5 + 2.0 * R, rtol=1e-5)


def test_audit_mergeplg_too_few_observations_returns_radar():
    radar, ii, jj, r_at, make = _merger_case()
    vals = r_at + 1.0
    vals[4:] = np.nan                                                        # 4 <= MIN_OBSERVATIONS
    np.testing.assert_array_equal(make(vals).adjust("okrig_add"), radar.values.reshape(2, -1))


def test_audit_kriging_weights_invariant_to_variogram_scale():
    """The module docstring's claim: only the variogram shape matters (diagonal is 0)."""
    radar, ii, jj, r_at, _ = _merger_case()
    obs = r_at * np.random.default_rng(1).uniform(0.5, 1.5, r_at.shape)
    a = _merger_case(variogram={"sill": 0.9, "range": 5000.0, "nugget": 0.1})[-1](obs)
    b = _merger_case(variogram={"sill": 9.0, "range": 5000.0, "nugget": 1.0})[-1](obs)
    for m in ("okrig_add", "ked"):
        np.testing.assert_allclose(a.adjust(m), b.adjust(m), rtol=1e-6, atol=1e-9)


def test_audit_standardised_semivariogram_white_noise_is_one():
    from core.maps.mergeplg_methods import standardised_semivariogram
    rng = np.random.default_rng(0)
    glat, glon = GRID.mesh()
    x, y = to_local_xy(glat.ravel(), glon.ravel(), LAT.mean(), LON.mean())
    R = rng.normal(5.0, 1.0, (40, x.size))                                  # all wet, no correlation
    lag, gamma, n = standardised_semivariogram(R, x, y, max_lag_m=20_000, n_pairs=400_000)
    assert n == 40
    np.testing.assert_allclose(gamma, 1.0, atol=0.12)                        # gamma -> variance = 1
    assert abs(gamma.mean() - 1.0) < 0.02
    # a smooth field: semivariance grows with lag
    smooth = np.stack([np.sin(x / 8000.0 + k) + np.cos(y / 9000.0) + 3 for k in range(5)])
    lag, gamma, _ = standardised_semivariogram(smooth, x, y, max_lag_m=10_000)
    assert gamma[0] < gamma[-1] and np.all(np.diff(gamma[:5]) > -0.02)
    # dry fields are left out
    assert standardised_semivariogram(np.zeros((3, x.size)), x, y)[2] == 0


# --------------------------------------------------------------------------- scores
def test_audit_scores_hand_case():
    est = np.array([2.0, 0.0, 1.0, 0.0, np.nan])
    ref = np.array([1.0, 1.0, 0.0, 0.0, 5.0])
    s = scores(est, ref, wet_threshold=0.1)
    assert s["n"] == 4
    assert s["bias"] == pytest.approx(0.25)                                 # mean(e - r)
    assert s["rel_bias"] == pytest.approx(3 / 2 - 1)                        # sum e / sum r - 1
    assert s["rmse"] == pytest.approx(np.sqrt(3 / 4))
    assert s["nrmse"] == pytest.approx(np.sqrt(3 / 4) / 0.5)                 # / mean(ref)
    assert s["mae"] == pytest.approx(0.75)
    # hits 1, misses 1, false alarms 1, correct negatives 1
    assert (s["pod"], s["far"], s["csi"]) == pytest.approx((0.5, 0.5, 1 / 3))
    assert scores([np.nan], [1.0]) == {"n": 0}


# --------------------------------------------------------------------------- geometry
def test_audit_segment_distance_hand_cases():
    d = segment_distance_km([0.0], [0.0], [1000.0], [0.0], np.array([500.0, -300.0, 1300.0, 500.0]),
                            np.array([400.0, 400.0, -400.0, 0.0]))
    np.testing.assert_allclose(d, [0.4, 0.5, 0.5, 0.0])                     # beside, past ends, on


def test_audit_distance_to_links_matches_projected_segment_distance():
    la0, lo0, la1, lo1 = 40.70, -73.98, 40.74, -73.92                       # ~6.6 km diagonal link
    glat, glon = GRID.mesh()
    links = pd.DataFrame({"site_0_lat": [la0], "site_0_lon": [lo0], "site_1_lat": [la1], "site_1_lon": [lo1]})
    gc = distance_to_links_km(glat, glon, links, step_m=50.0)
    lat0, lon0 = LAT.mean(), LON.mean()
    x0, y0 = to_local_xy(la0, lo0, lat0, lon0)
    x1, y1 = to_local_xy(la1, lo1, lat0, lon0)
    gx, gy = to_local_xy(glat, glon, lat0, lon0)
    pl = segment_distance_km(x0, y0, x1, y1, gx, gy)
    np.testing.assert_allclose(gc, pl, atol=0.03)                           # step/2 + projection error


def test_audit_path_average_points_linear_field_is_midpoint_value():
    glat, glon = GRID.mesh()
    field = xr.DataArray(glon - LON[0], dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    links = pd.DataFrame({"site_0_lat": [LAT[5]], "site_0_lon": [LON[2]],
                          "site_1_lat": [LAT[5]], "site_1_lon": [LON[10]]}, index=["a"])
    out = path_average_points(field, links, spacing_m=50.0)
    assert float(out.sel(link="a")) == pytest.approx((LON[2] + LON[10]) / 2 - LON[0], abs=2e-4)


def test_audit_path_average_intersect_linear_field_is_midpoint_value():
    x = np.arange(0, 10_000, 500.0) + 250.0
    y = np.arange(0, 8_000, 500.0) + 250.0
    xg, yg = np.meshgrid(x, y)
    field = xr.DataArray(np.stack([xg / 1000.0, 2 * xg / 1000.0]), dims=("time", "y", "x"),
                         coords={"time": TIMES, "x_grid": (("y", "x"), xg), "y_grid": (("y", "x"), yg)})
    ds = xr.Dataset(coords={"cml_id": ["a", "b"], "site_0_x": ("cml_id", [1000.0, 2100.0]),
                            "site_0_y": ("cml_id", [1100.0, 3000.0]), "site_1_x": ("cml_id", [9000.0, 2100.0]),
                            "site_1_y": ("cml_id", [1100.0, 7000.0])})
    out = path_average_intersect(field, ds).transpose("time", "cml_id").values
    # a horizontal link through a field linear in x averages to the value at its midpoint
    # (up to the half-cell end effect); a vertical link sees one column
    np.testing.assert_allclose(out[:, 0], [5.0, 10.0], atol=0.15)
    np.testing.assert_allclose(out[:, 1], [2.25, 4.5], rtol=1e-6)


@pytest.mark.xfail(strict=True, reason=(
    "poligrain's intersect weights count a link lying exactly on a cell edge in both "
    "neighbouring cells, so the path average is doubled. Only exactly edge-aligned links "
    "are affected; the fix belongs in poligrain."))
def test_audit_path_average_intersect_link_on_cell_edge():
    x = np.arange(0, 10_000, 500.0) + 250.0
    y = np.arange(0, 8_000, 500.0) + 250.0
    xg, yg = np.meshgrid(x, y)
    field = xr.DataArray(np.ones_like(xg), dims=("y", "x"),
                         coords={"x_grid": (("y", "x"), xg), "y_grid": (("y", "x"), yg)})
    ds = xr.Dataset(coords={"cml_id": ["a"], "site_0_x": ("cml_id", [1000.0]), "site_0_y": ("cml_id", [1000.0]),
                            "site_1_x": ("cml_id", [9000.0]), "site_1_y": ("cml_id", [1000.0])})
    np.testing.assert_allclose(np.asarray(path_average_intersect(field, ds)).ravel(), 1.0)


# --------------------------------------------------------------------------- wet area
def _wet_case():
    lat, lon = LAT[[3, 4, 15, 16]], LON[[2, 3, 11, 12]]
    return _stations(lat, lon, [[4.0, 0.0], [6.0, 0.0], [0.0, 0.0], [0.0, 0.0]])


def test_audit_wet_probability_is_indicator_idw():
    st = _wet_case()
    p = wet_probability(st, GRID)
    for i, want in enumerate([1.0, 1.0, 0.0, 0.0]):
        assert float(p.isel(time=0).sel(lat=st.lat.values[i], lon=st.lon.values[i])) == pytest.approx(want, abs=1e-6)
    assert float(p.isel(time=1).max()) == 0.0
    assert float(p.min()) >= 0 and float(p.max()) <= 1


def test_audit_masked_and_conditional_idw():
    st = _wet_case()
    plain = points_idw_map(st, GRID).isel(time=0)
    m = masked_idw(st, GRID).isel(time=0)
    c = conditional_idw(st, GRID).isel(time=0)
    dry = st.isel(station=2)
    assert float(m.sel(lat=dry.lat, lon=dry.lon)) == 0.0                    # plain IDW leaks rain here
    assert float(plain.sel(lat=dry.lat, lon=dry.lon)) == pytest.approx(0.0, abs=1e-6)
    wet = st.isel(station=0)
    assert float(c.sel(lat=wet.lat, lon=wet.lon)) == pytest.approx(4.0, rel=1e-6)
    # conditional: wet cells interpolate wet observations only, so stay within [4, 6]
    wet_cells = c.values[c.values > 0]
    assert wet_cells.min() >= 4.0 - 1e-5 and wet_cells.max() <= 6.0 + 1e-5
    # NaN where the plain map has no sensor in range, as plain IDW
    assert (np.isnan(c.values) == np.isnan(plain.values)).all()


def test_audit_wet_area_scores_hand_case():
    s = wet_area_scores([0.0, 0.5, 1.0, 2.0], [0.0, 0.0, 1.0, 4.0], wet_threshold=0.1, peak_q=1.0)
    assert (s["war_est"], s["war_ref"]) == (0.75, 0.5)
    assert s["war_ratio"] == pytest.approx(1.5) and s["peak_ratio"] == pytest.approx(0.5)
