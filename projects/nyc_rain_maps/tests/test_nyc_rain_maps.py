"""nyc_rain_maps: retrieval methods, mapping, events, selection, study - no network, no data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_rain_maps import baseline as bl  # noqa: E402
from core.geo import NYC, Domain, Grid, haversine_m  # noqa: E402,F401
from nyc_rain_maps.events import Event, classify, detect_events  # noqa: E402
from nyc_rain_maps.idw import accumulate, idw_map, idw_weights  # noqa: E402
from nyc_rain_maps.power_law import attenuation_from_rain, itu_ab, rain_from_attenuation  # noqa: E402
from nyc_rain_maps.scores import scores  # noqa: E402


def test_itu_coefficients_match_both_libraries():
    a, b = itu_ab(68.04, "v", "ITU_2003")          # PyNNcml / implementation_1
    assert a[0] == pytest.approx(0.7577, abs=1e-4) and b[0] == pytest.approx(0.7985, abs=1e-4)
    a, b = itu_ab(68.04, "v", "ITU_2005")          # pycomlink / implementation_2
    assert a[0] == pytest.approx(0.9939, abs=1e-4) and b[0] == pytest.approx(0.7261, abs=1e-4)


def test_power_law_roundtrip_and_conventions():
    R = np.array([0.5, 2.0, 10.0, 50.0])
    A = attenuation_from_rain(R, 2.0, 68.04, "v")
    assert np.allclose(rain_from_attenuation(A, 2.0, 68.04, "v", r_min=0), R)
    out = rain_from_attenuation(np.array([-1.0, 0.0, np.nan, 1e-4]), 2.0, 68.04, "v", r_min=0.1)
    assert out[0] == 0 and out[1] == 0 and np.isnan(out[2]) and out[3] == 0


def test_power_law_per_link_broadcast():
    A = np.ones((2, 5))
    R = rain_from_attenuation(A, [1.0, 2.0], [68.04, 24.1], ["v", "h"], r_min=0)
    assert R.shape == (2, 5) and not np.allclose(R[0], R[1])


def test_frequency_out_of_table_raises():
    with pytest.raises(ValueError):
        itu_ab(0.5, "v")


def test_trailing_min_matches_naive():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 50))
    w = 7
    naive = np.array([[x[i, max(0, t - w + 1):t + 1].min() for t in range(50)] for i in range(3)])
    assert np.allclose(bl.trailing_min(x, w), naive)


def test_trailing_min_propagates_nan():
    x = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
    out = bl.trailing_min(x, 2)
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2]) and out[0, 3] == 3.0


def test_dynamic_baseline_removes_constant_offset():
    A = np.full((1, 300), 40.0)
    A[0, 250:260] += 10
    A_rain, _ = bl.dynamic_baseline(A, window=200, quantization_delta=1.0)
    assert np.allclose(A_rain[0, :250], -1.0)            # qd bias correction
    assert np.allclose(A_rain[0, 250:260], 9.0)


def test_constant_baseline_holds_while_wet():
    A = np.array([[1.0, 2.0, 5.0, 6.0, 3.0]])
    wet = np.array([[0, 0, 1, 1, 0]])
    assert np.allclose(bl.constant_baseline(A, wet), [[1, 2, 2, 2, 3]])


def test_std_wet_dry_edges_are_dry():
    A = np.random.default_rng(1).normal(scale=5, size=(1, 100))
    wet, sig = bl.std_wet_dry(A, window=20, threshold=1.0)
    assert (wet[0, :9] == 0).all() and (wet[0, -10:] == 0).all() and wet[0, 20:80].max() == 1


def test_estimators_match_pynncml():
    torch = pytest.importorskip("torch")
    pnc = pytest.importorskip("pynncml")
    rng = np.random.default_rng(2)
    att = 40 + np.round(np.cumsum(rng.normal(0, 0.3, 1200)))
    att[500:560] += np.linspace(0, 12, 60)
    A_rain, _ = bl.dynamic_baseline(att[None], 200, 1.0)
    ours = rain_from_attenuation(A_rain, 2.0, 68.04, "v", table="ITU_2003", r_min=0.5)
    m = pnc.scm.rain_estimation.one_step_dynamic_baseline(pnc.scm.power_law.PowerLawType.INSTANCE, 0.5, 200,
                                                          quantization_delta=1)
    ref, _ = m(torch.tensor(att[None], dtype=torch.float32), pnc.datasets.MetaData(68.04, True, 2.0, 10, 10))
    assert np.allclose(ours, ref.numpy(), atol=1e-4)


def _links(lat, lon, vals):
    return xr.DataArray(np.asarray(vals, float)[:, None], dims=("link", "time"),
                        coords={"link": [f"l{i}" for i in range(len(lat))],
                                "time": pd.to_datetime(["2024-01-01"]),
                                "mid_lat": ("link", lat), "mid_lon": ("link", lon)})


def test_idw_exact_at_source_and_bounded():
    grid = Grid(np.array([40.70, 40.71]), np.array([-74.00, -73.99]))
    da = _links([40.70, 40.71], [-74.00, -73.99], [2.0, 8.0])
    m = idw_map(da, grid)
    assert m.sel(lat=40.70, lon=-74.00).item() == pytest.approx(2.0)
    assert m.sel(lat=40.71, lon=-73.99).item() == pytest.approx(8.0)
    assert 2.0 < m.sel(lat=40.70, lon=-73.99).item() < 8.0


def test_idw_nan_policies_and_radius():
    grid = Grid(np.array([40.70]), np.array([-73.995]))
    da = _links([40.70, 40.70], [-74.00, -73.99], [np.nan, 4.0])
    assert idw_map(da, grid, nan_policy="exclude").item() == pytest.approx(4.0)
    assert idw_map(da, grid, nan_policy="zero").item() == pytest.approx(2.0)
    far = idw_map(da, Grid(np.array([41.5]), np.array([-73.99])), radius_m=10_000)
    assert np.isnan(far.item())


def test_idw_nnear():
    W = idw_weights(np.array([0.0, 100.0, 1000.0]), np.zeros(3), np.array([50.0]), np.zeros(1), nnear=2)
    assert W[0, 2] == 0 and W[0, 0] > 0 and W[0, 1] > 0


def test_accumulate_hour_ending():
    t = pd.date_range("2024-01-01 00:01", "2024-01-01 02:00", freq="1min")
    da = xr.DataArray(np.full((1, t.size), 6.0), dims=("link", "time"), coords={"time": t})
    acc = accumulate(da, "1h")
    assert list(pd.DatetimeIndex(acc.time.values)) == list(pd.to_datetime(["2024-01-01 01:00", "2024-01-01 02:00"]))
    assert np.allclose(acc.values, 6.0)


def test_detect_events_gap_rule():
    t = pd.date_range("2024-01-01 01:00", periods=48, freq="h")
    v = np.zeros(48)
    v[[5, 6, 9]] = 2.0       # gap of 2 dry hours: same event
    v[[30, 31]] = 1.0        # 20 h later: new event
    da = xr.DataArray(np.broadcast_to(v[:, None, None], (48, 2, 2)).copy(), dims=("time", "lat", "lon"),
                      coords={"time": t, "lat": [0, 1], "lon": [0, 1]})
    ev = detect_events(da, min_total_mm=1.0)
    assert len(ev) == 2 and ev[0].total_mm == 6.0 and ev[0].duration_h == 5
    assert ev[0].start == str(t[5] - pd.Timedelta("1h"))


def _ev(sf, asos):
    return Event("x", "2024-01-01", "2024-01-02", 1, 1, 1.0, 1.0, "2024-01-01", 1.0, 1.0,
                 snow_fraction=sf, asos=asos)


def test_classify_rules():
    assert classify(_ev(0.9, {"snow": 20})) == "snow"
    assert classify(_ev(0.97, {"snow": 43, "freezing": 27})) == "mix"
    assert classify(_ev(0.0, {"rain": 40})) == "rain"
    assert classify(_ev(0.0, {"rain": 100, "mix": 1})) == "rain"     # lone sleet report
    assert classify(_ev(0.0, {"rain": 30, "mix": 3})) == "mix"
    assert classify(_ev(0.3, {"rain": 10, "snow": 10})) == "mix"
    assert classify(_ev(float("nan"), {})) == "unclassified"


def test_scores():
    s = scores([1, 2, 3, np.nan], [1, 2, 5, 1])
    assert s["n"] == 3 and s["bias"] == pytest.approx(-2 / 3)
    assert s["nrmse"] == pytest.approx(np.sqrt(4 / 3) / (8 / 3))
    assert s["csi"] == 1.0


def test_accumulate_coverage_and_infill():
    t = pd.date_range("2024-01-01 00:01", "2024-01-01 02:00", freq="1min")
    v = np.full(t.size, 6.0)
    v[:30] = np.nan                         # first hour half missing -> NaN
    v[60:66] = np.nan                       # second hour 90 % present -> infilled
    acc = accumulate(xr.DataArray(v, dims="time", coords={"time": t}), "1h")
    assert np.isnan(acc.values[0]) and acc.values[1] == pytest.approx(6.0)


def test_trailing_min_skipna():
    x = np.array([[5.0, np.nan, 3.0, 4.0]])
    assert np.allclose(bl.trailing_min(x, 2, skipna=True), [[5, 5, 3, 3]])


def test_selection_paths_group_same_and_reverse():
    from nyc_rain_maps.link_selection import _paths
    t = pd.DataFrame({"cml_id": [40, 45, 40, 8], "site_0_lat": [40.674, 40.696, 40.674, 40.70],
                      "site_0_lon": [-73.94, -73.9397, -73.94, -73.98],
                      "site_1_lat": [40.696, 40.674, 40.696, 40.73], "site_1_lon": [-73.9399, -73.94, -73.9399, -73.95]},
                     index=["40/sublink_1", "45/sublink_1", "40/sublink_2", "8/sublink_1"])
    p = _paths(t, 100.0)
    assert p["40/sublink_1"] == p["45/sublink_1"] == p["40/sublink_2"] != p["8/sublink_1"]


def test_common_sample_does_not_let_gaps_shrink_everyones_sample():
    from nyc_rain_maps.study import _common_sample
    rows = []
    for m, cells in [("a", range(100)), ("b", range(100)), ("gappy", range(30))]:
        rows += [{"event": "e", "method": m, "cell": c, "est": 1.0, "ref": 1.0} for c in cells]
    out, cov = _common_sample(pd.DataFrame(rows), ["event"], "method", "cell", 0.9)
    n = out.groupby("method").cell.nunique()
    assert n["a"] == n["b"] == 100 and n["gappy"] == 30
    assert cov.set_index("method").loc["gappy", "full"] == False  # noqa: E712


def test_score_vs_official_by_hour_type():
    from nyc_rain_maps.validation import pooled_vs_official, score_vs_official
    t = pd.date_range("2024-01-01 01:00", periods=6, freq="h")
    off = pd.DataFrame({"NYC": [0, 2, 4, 1, 0, np.nan]}, index=t)
    src = pd.DataFrame({"NYC": [0, 1, 2, 2, 0.5, 3]}, index=t)
    pt = pd.DataFrame({"NYC": ["none", "rain", "rain", "snow", "none", "rain"]}, index=t)
    s = score_vs_official({"radar": src}, off, pt).set_index("hours")
    assert s.loc["all", "n"] == 5                         # NaN official hour excluded, not zero-filled
    assert s.loc["rain", "n"] == 2 and s.loc["rain", "total_official_mm"] == 6
    assert s.loc["snow", "total_source_mm"] == 2
    p = pooled_vs_official({"radar": src}, off, pt).set_index("hours")
    assert p.loc["rain", "rel_bias"] == pytest.approx(-0.5)
    gappy = src.copy()
    gappy.iloc[2] = np.nan
    q = pooled_vs_official({"radar": src, "gappy": gappy}, off, pt).set_index(["hours", "source"])
    assert q.loc[("rain", "radar"), "n"] == q.loc[("rain", "gappy"), "n"] == 1     # common sample
    assert q.loc[("rain", "gappy"), "coverage"] == pytest.approx(0.5)


def test_event_analysis_rankings_and_drivers():
    from nyc_rain_maps.all_events import drivers, win_counts
    t = pd.DataFrame({"a_nrmse": [0.5, 0.9, 0.4, 1.2, 0.7, 0.6, 0.8], "a_bias": [0.1, -0.2, 0.0, 0.5, 0.1, -0.1, 0.2],
                      "b_nrmse": [0.6, 0.8, 0.9, 1.0, 0.9, 0.5, 0.9], "b_bias": [-0.5] * 7,
                      "total_mm": [30, 5, 40, 2, 20, 35, 10]})
    t["best_method"] = t[["a_nrmse", "b_nrmse"]].idxmin(axis=1).str[:-6]
    w = win_counts(t)
    assert w.loc["a", "wins"] == 4 and w.loc["b", "share_underestimating"] == 1.0
    d = drivers(t).set_index(["method", "metric"])
    assert d.loc[("a", "NRMSE"), "total_mm"] < -0.8          # bigger events, lower error
