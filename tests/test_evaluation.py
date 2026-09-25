"""Matching and scoring with poligrain, including the traps it sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.opensense import conventions as cv
from core.opensense import evaluation as ev
from core.opensense import wet_dry


def _links(x0, y0, x1, y1):
    """Links given directly in projected metres."""
    n = len(x0)
    return xr.Dataset(coords={
        "cml_id": np.arange(n),
        "site_0_x": ("cml_id", np.asarray(x0, float)),
        "site_0_y": ("cml_id", np.asarray(y0, float)),
        "site_1_x": ("cml_id", np.asarray(x1, float)),
        "site_1_y": ("cml_id", np.asarray(y1, float)),
        # km, the way this pipeline stores it - the trap for poligrain
        "length": ("cml_id", np.hypot(np.subtract(x1, x0), np.subtract(y1, y0)) / 1e3),
    })


def _gauges(x, y):
    return xr.Dataset(coords={"id": [f"g{i}" for i in range(len(x))],
                              "x": ("id", np.asarray(x, float)),
                              "y": ("id", np.asarray(y, float))})


def test_closest_gauges_uses_the_whole_path_not_a_km_radius():
    """A gauge beside the far end of a 4 km link, 500 m off the path.

    poligrain searches ``length/2 + max_distance`` from the midpoint in the
    units of the coordinates. With length in km that radius is 1002 m, the
    gauge sits ~2.06 km from the midpoint, and it is silently not found.
    """
    import poligrain as plg

    links = _links([0], [0], [4000], [0])
    gauges = _gauges([3900], [500])

    naive = plg.spatial.get_closest_points_to_line(links, gauges, 1000, 1)
    assert not np.isfinite(naive.distance.values[0, 0])      # the trap

    ours = ev.closest_gauges(links, gauges, max_distance_m=1000)
    assert ours.neighbor_id.values[0, 0] == "g0"
    assert ours.distance.values[0, 0] == pytest.approx(500.0)


def test_gauge_series_at_links_leaves_unmatched_links_nan():
    links = _links([0, 50000], [0, 0], [4000, 54000], [0, 0])
    gauges = _gauges([2000], [300])
    time = pd.date_range("2020-01-01", periods=3, freq="15min").as_unit("ns")
    da = xr.DataArray([[1.0], [2.0], [3.0]], dims=("time", "id"),
                      coords={"time": time, "id": gauges.id})
    out = ev.gauge_series_at_links(da, ev.closest_gauges(links, gauges, 1000))
    np.testing.assert_array_equal(out.isel(cml_id=0).values, [1, 2, 3])
    assert bool(out.isel(cml_id=1).isnull().all())


def test_radar_along_links_of_a_uniform_field_is_that_field():
    time = pd.date_range("2020-01-01", periods=2, freq="5min").as_unit("ns")
    lat = np.linspace(57.60, 57.80, 21)
    lon = np.linspace(11.80, 12.20, 25)
    radar = cv.project_grid(xr.Dataset(
        {"R": (("time", "lat", "lon"), np.stack([np.full((21, 25), 3.0),
                                                  np.full((21, 25), 7.0)]))},
        coords={"time": time, "lat": lat, "lon": lon}), "EPSG:32632")
    xm, ym = float(radar.x_grid.mean()), float(radar.y_grid.mean())
    links = _links([xm - 3000, xm], [ym, ym - 2000], [xm + 3000, xm + 500],
                   [ym + 1000, ym + 2000])
    out = ev.radar_along_links(radar.R, links)
    assert out.dims == ("time", "cml_id")
    # poligrain discretizes the line, so its weights sum to 1 within ~1e-4
    np.testing.assert_allclose(out.values, [[3.0, 3.0], [7.0, 7.0]], rtol=1e-3)


def test_distance_to_network_is_point_to_segment():
    links = _links([0], [0], [10000], [0])
    d = ev.distance_to_network(links, np.array([5000, -3000, 5000]),
                               np.array([2000, 4000, 0]))
    np.testing.assert_allclose(d, [2.0, 5.0, 0.0])
    grid = ev.distance_to_network(links, np.zeros((2, 3)), np.ones((2, 3)) * 1000)
    assert grid.shape == (2, 3)


def test_metrics_pair_dataarrays_by_coordinate_not_memory_order():
    """(cml_id, time) against (time, cml_id): flattening pairs wrong values."""
    rng = np.random.default_rng(1)
    time = pd.date_range("2020-01-01", periods=200, freq="5min").as_unit("ns")
    a = xr.DataArray(rng.gamma(0.5, 2.0, (200, 6)), dims=("time", "cml_id"),
                     coords={"time": time, "cml_id": np.arange(6)})
    b = a.transpose("cml_id", "time")
    m = ev.rainfall_metrics(a, b)
    assert m["r"] == pytest.approx(1.0)
    assert m["pbias"] == pytest.approx(0.0, abs=1e-9)
    assert m["mcc"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        ev.rainfall_metrics(np.ones(5), np.ones(6))


def test_metrics_bias_uses_the_whole_record():
    """Light rain counts toward the bias; only detection uses the threshold."""
    ref = np.array([0.05, 0.05, 0.05, 1.0])
    est = np.array([0.0, 0.0, 0.0, 1.0])
    m = ev.rainfall_metrics(ref, est)
    assert m["ratio"] == pytest.approx(1.0 / 1.15)
    assert m["n"] == 4


def test_radar_wet_mask_dilates_and_keeps_undecided_nan():
    t5 = pd.date_range("2020-01-01", periods=6, freq="5min").as_unit("ns")
    path = xr.DataArray([[0.0], [0.0], [2.0], [0.0], [0.0], [np.nan]],
                        dims=("time", "cml_id"), coords={"time": t5, "cml_id": [0]})
    t1 = pd.date_range("2020-01-01", periods=30, freq="1min").as_unit("ns")
    mask = wet_dry.from_radar(path, t1)
    per5 = mask.values[::5, 0]
    np.testing.assert_array_equal(per5[:5], [0, 1, 1, 1, 0])
    assert np.isnan(per5[5])


def test_fill_undecided_keeps_decisions_and_falls_back_elsewhere():
    from conftest import make_cml, rain_event
    from core.opensense import example_data

    rain = rain_event(n_links=2)
    ds = example_data.normalize_cml(
        make_cml(rain, [[23.0, 23.0], [38.0, 38.0]], [4.0, 5.0], noise_db=0.05),
        "EPSG:32632")
    mask = xr.DataArray(np.full((ds.sizes["time"], 2), np.nan),
                        dims=("time", "cml_id"),
                        coords={"time": ds.time, "cml_id": ds.cml_id})
    mask[:100, 0] = 1.0                   # decided wet, in fact dry
    mask[:100, 1] = 0.0                   # decided dry
    out = wet_dry.fill_undecided(mask, ds)
    assert out.dims == ("time", "cml_id", "sublink_id") and out.dtype == bool
    assert bool(out.isel(time=slice(0, 100), cml_id=0).all())
    assert not bool(out.isel(time=slice(0, 100), cml_id=1).any())
    # undecided samples got the rolling-std flag: the rain edges are wet
    assert bool(out.isel(time=slice(595, 606)).any())
