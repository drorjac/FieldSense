"""nyc_rain_maps: events, link selection, study, validation - no network, no data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.geo import NYC, Domain, Grid, haversine_m  # noqa: E402,F401
from nyc_rain_maps.events import Event, classify, detect_events  # noqa: E402


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
