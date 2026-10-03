"""radar_adjustment on synthetic data: settings, scoring, cropping, the mergeplg calls."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_adjustment.adjust import checks_for, crop, merger_kwargs  # noqa: E402
from radar_adjustment.score import distance_to_links_km, metrics, score_products  # noqa: E402


def _scene(nt=4, ny=30, nx=40, seed=0):
    rng = np.random.default_rng(seed)
    x = 500_000 + 1000.0 * np.arange(nx)
    y = 4_900_000 + 1000.0 * np.arange(ny)
    X, Y = np.meshgrid(x, y)
    t = pd.date_range("2022-06-01 01:00", periods=nt, freq="1h")
    rain = 2 + np.sin(X / 7000)[None] + np.cos(Y / 9000)[None] + rng.gamma(1, 0.3, (nt, ny, nx))
    rad = xr.DataArray(rain * 0.7, dims=("time", "y", "x"), coords={"time": t})
    rad.coords["x_grid"], rad.coords["y_grid"] = (("y", "x"), X), (("y", "x"), Y)
    rad.coords["lon"], rad.coords["lat"] = (("y", "x"), 11 + (X - x[0]) / 80_000), (("y", "x"), 44 + (Y - y[0]) / 111_000)
    n = 15
    i0, j0 = rng.integers(5, ny - 5, n), rng.integers(5, nx - 8, n)
    s0x, s0y, s1x, s1y = x[j0], y[i0], x[j0 + 3], y[i0]
    cml = xr.DataArray([[rain[k, i, j:j + 4].mean() for k in range(nt)] for i, j in zip(i0, j0)],
                       dims=("cml_id", "time"), coords={"cml_id": [f"c{k}" for k in range(n)], "time": t})
    for name, v in (("site_0_x", s0x), ("site_0_y", s0y), ("site_1_x", s1x), ("site_1_y", s1y)):
        cml.coords[name] = ("cml_id", v.astype(float))
    cml.coords["x"], cml.coords["y"] = (cml.site_0_x + cml.site_1_x) / 2, (cml.site_0_y + cml.site_1_y) / 2
    for s, xx, yy in (("0", s0x, s0y), ("1", s1x, s1y)):
        cml.coords[f"site_{s}_lon"] = ("cml_id", 11 + (xx - x[0]) / 80_000)
        cml.coords[f"site_{s}_lat"] = ("cml_id", 44 + (yy - y[0]) / 111_000)
    gi, gj = np.array([10, 20, 15]), np.array([10, 30, 20])
    g = xr.DataArray(rain[:, gi, gj].T, dims=("id", "time"), coords={"id": ["g0", "g1", "g2"], "time": t})
    g.coords["x"], g.coords["y"] = ("id", x[gj]), ("id", y[gi])
    g.coords["lon"], g.coords["lat"] = ("id", 11 + (x[gj] - x[0]) / 80_000), ("id", 44 + (y[gi] - y[0]) / 111_000)
    return rad, cml, g


def test_checks_follow_the_notebook():
    assert checks_for("add_b_ok", "default") == {"diff_check": 10}
    assert checks_for("ked_p", "cc") == {"diff_check": 5}
    assert checks_for("mul_p_idw", "default") == {"ratio_check": (0.1, 15)}
    assert checks_for("mul_b_ok", "nc") == {}
    cls, kw = merger_kwargs("add_p_idw", "default")
    assert cls == "MergeDifferenceIDW" and "idw_method" not in kw and kw["nnear"] == 12
    cls, kw = merger_kwargs("ked_b", None)
    assert kw["full_line"] is True and "range_checks" not in kw


def test_metrics_threshold_and_perfect():
    ref = np.array([0.0, 0.05, 1.0, 2.0, np.nan])
    m = metrics(ref, ref)
    assert m["rmse"] == 0 and m["n_pairs"] == 2 and m["n_all"] == 5
    m = metrics(ref, ref * 2)
    assert m["pbias"] == pytest.approx(100.0)


def test_distance_to_links():
    _, cml, g = _scene()
    d = distance_to_links_km(g, cml)
    assert (d >= 0).all() and d.index.tolist() == ["g0", "g1", "g2"]
    # a gauge on a link's path is at distance 0
    g2 = g.copy()
    g2.coords["x"] = ("id", [float(cml.x[0]), 0.0, 0.0])
    g2.coords["y"] = ("id", [float(cml.y[0]), 0.0, 0.0])
    assert distance_to_links_km(g2, cml).iloc[0] == pytest.approx(0, abs=1e-9)


def test_score_products_tables():
    _, _, g = _scene()
    res = score_products({"perfect": g, "double": g * 2}, g, distance_to_links_km(g, _scene()[1]))
    o = res["overall"].set_index("product")
    assert o.loc["perfect", "rmse"] == 0 and o.loc["double", "pbias"] == pytest.approx(100)
    assert set(res["per_gauge"]["id"]) == {"g0", "g1", "g2"}
    assert len(res["bands"]) > 0 and len(res["classes"]) == 8


def test_crop_keeps_all_observations():
    rad, cml, g = _scene()
    c = crop(rad, cml, g, pad_deg=0.0)
    assert c.sizes["y"] <= rad.sizes["y"] and c.sizes["x"] <= rad.sizes["x"]
    for lat, lon in zip(g.lat.values, g.lon.values):
        assert c.lat.min() - 0.02 <= lat <= c.lat.max() + 0.02
        assert c.lon.min() - 0.02 <= lon <= c.lon.max() + 0.02


def _new_api():
    try:
        from mergeplg import merge
    except ImportError:
        return False
    return "ds_rad" in inspect.signature(merge.MergeDifferenceIDW.__init__).parameters


@pytest.mark.skipif(not _new_api(), reason="needs mergeplg with the constructor-and-call API")
@pytest.mark.parametrize("variant", ["add_p_idw", "add_b_ok", "ked_p", "mul_p_ok"])
def test_crop_does_not_change_values_at_gauges(variant):
    from radar_adjustment.adjust import adjust_series, at_points, build_merger
    rad, cml, g = _scene()
    cls, kw = merger_kwargs(variant, "default")
    full = np.stack(adjust_series(build_merger(cls, kw, rad, cml), rad, cml))
    small_rad = crop(rad, cml, g, pad_deg=0.02)
    small = np.stack(adjust_series(build_merger(cls, kw, small_rad, cml), small_rad, cml))
    np.testing.assert_allclose(at_points(full, rad, g), at_points(small, small_rad, g), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not _new_api(), reason="needs mergeplg with the constructor-and-call API")
def test_adjustment_moves_radar_towards_links():
    from radar_adjustment.adjust import adjust_series, at_points, build_merger
    rad, cml, g = _scene()
    cls, kw = merger_kwargs("add_b_ok", "nc")
    f = np.stack(adjust_series(build_merger(cls, kw, rad, cml), rad, cml))
    err_rad = np.abs(at_points(rad.values, rad, g) - g.values).mean()
    err_adj = np.abs(at_points(f, rad, g) - g.values).mean()
    assert err_adj < err_rad
