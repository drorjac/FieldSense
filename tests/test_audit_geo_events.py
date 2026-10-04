"""Audit of core.geo (distances, local projection) and core.events (event detection).

References: the WGS84 geodesic (pyproj.Geod) for distances, and hand-built hourly
records for the event rules stated in the core.events docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.events import detect_events
from core.geo import NYC, haversine_m, to_local_xy

pyproj = pytest.importorskip("pyproj")


def _nyc_pairs(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(NYC.lat_min, NYC.lat_max, (2, n))
    lon = rng.uniform(NYC.lon_min, NYC.lon_max, (2, n))
    return lat, lon


# ---------------------------------------------------------------------------
# geo
# ---------------------------------------------------------------------------
def test_haversine_known_values():
    # one degree of longitude on the equator, one of latitude anywhere (sphere)
    r = 6_371_008.8
    assert haversine_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(np.radians(1) * r, rel=1e-12)
    assert haversine_m(40.0, -74.0, 41.0, -74.0) == pytest.approx(np.radians(1) * r, rel=1e-12)
    assert haversine_m(40.7, -74.0, 40.7, -74.0) == 0.0
    # antipodes: half the circumference, no NaN from arcsin(>1) (arcsin near 1 loses
    # digits, which only matters at planetary distances)
    assert haversine_m(10.0, 20.0, -10.0, -160.0) == pytest.approx(np.pi * r, rel=1e-7)


def test_haversine_vs_wgs84_geodesic_over_nyc():
    """A sphere differs from the ellipsoid by < 0.5% at mid-latitudes; here ~0.26%."""
    lat, lon = _nyc_pairs()
    _, _, geod = pyproj.Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])
    rel = np.abs(haversine_m(lat[0], lon[0], lat[1], lon[1]) / geod - 1)
    assert rel.max() < 0.004


def test_to_local_xy_distance_accuracy_over_nyc():
    """Equirectangular around the domain centre vs the great circle and the geodesic.

    Pins the accuracy stated in the docstring: ~0.3% across the whole NYC box,
    < 0.1% for pairs within ~5 km of the origin, ~0.6% vs WGS84.
    """
    lat, lon = _nyc_pairs()
    lat0, lon0 = (NYC.lat_min + NYC.lat_max) / 2, (NYC.lon_min + NYC.lon_max) / 2
    x, y = to_local_xy(lat, lon, lat0, lon0)
    d_xy = np.hypot(x[0] - x[1], y[0] - y[1])
    d_gc = haversine_m(lat[0], lon[0], lat[1], lon[1])
    _, _, d_geod = pyproj.Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])
    assert np.max(np.abs(d_xy / d_gc - 1)) < 0.004
    assert np.max(np.abs(d_xy / d_geod - 1)) < 0.007

    # pairs within ~5 km of the origin: 0.03 deg lat x 0.04 deg lon box
    rng = np.random.default_rng(1)
    la = lat0 + rng.uniform(-0.03, 0.03, (2, 2000))
    lo = lon0 + rng.uniform(-0.04, 0.04, (2, 2000))
    x, y = to_local_xy(la, lo, lat0, lon0)
    d_xy = np.hypot(x[0] - x[1], y[0] - y[1])
    d_gc = haversine_m(la[0], lo[0], la[1], lo[1])
    ok = d_gc > 500
    assert np.max(np.abs(d_xy[ok] / d_gc[ok] - 1)) < 0.001


def test_to_local_xy_axes_and_origin():
    x, y = to_local_xy(40.7, -74.0, 40.7, -74.0)
    assert (x, y) == (0.0, 0.0)
    x, y = to_local_xy(40.7, -73.99, 40.7, -74.0)       # east -> +x
    assert x > 0 and y == 0
    x, y = to_local_xy(40.71, -74.0, 40.7, -74.0)       # north -> +y
    assert y > 0 and x == 0


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def _hourly(mean_mm, start="2024-07-01 01:00", valid=None):
    """(time, lat, lon) hour-ending record with the given domain mean per hour."""
    mean_mm = np.asarray(mean_mm, float)
    t = pd.date_range(start, periods=mean_mm.size, freq="h")
    data = np.repeat(mean_mm[:, None, None], 4, axis=1).reshape(-1, 2, 2).copy()
    if valid is not None:
        for i, frac in valid.items():
            data[i].flat[: int(round((1 - frac) * 4))] = np.nan
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": t, "lat": [40.6, 40.7], "lon": [-74.0, -73.9]})


def test_events_gap_rule_labels_and_totals():
    #        h1   h2   h3 h4 h5 h6   h7 ... h14 (gap of 7 dry) h15
    rain = [2.0, 1.0, 0, 0, 0, 0.5] + [0] * 7 + [3.0]
    ev = detect_events(_hourly(rain), min_gap_h=6)
    assert len(ev) == 2
    first, second = ev.iloc[0], ev.iloc[1]
    # hour-ending labels: the first wet hour is stamped 01:00 and covers 00:00-01:00
    assert first.start == "2024-07-01 00:00:00"
    assert first.end == "2024-07-01 06:00:00"
    assert first.duration_h == 6 and first.wet_hours == 3
    assert first.total_mm == pytest.approx(3.5)
    assert first.peak_hour == "2024-07-01 01:00:00"
    assert second.start == "2024-07-01 13:00:00" and second.total_mm == pytest.approx(3.0)


def test_events_gap_of_exactly_min_gap_splits():
    """``fewer than min_gap_h dry hours`` merge; exactly min_gap_h split."""
    rain = [2.0] + [0] * 6 + [2.0]
    assert len(detect_events(_hourly(rain), min_gap_h=6)) == 2
    rain = [2.0] + [0] * 5 + [2.0]
    assert len(detect_events(_hourly(rain), min_gap_h=6)) == 1


def test_events_min_total_and_wet_threshold():
    ev = detect_events(_hourly([0.3, 0.3, 0.3]), min_total_mm=1.0)
    assert ev.empty and list(ev.columns)[0] == "event_id"
    ev = detect_events(_hourly([0.09, 2.0]), wet_mm=0.1)            # 0.09 is dry
    assert ev.iloc[0].start == "2024-07-01 01:00:00"
    ev = detect_events(_hourly([0.1, 2.0]), wet_mm=0.1)             # threshold is >=
    assert ev.iloc[0].start == "2024-07-01 00:00:00"


def test_events_low_coverage_and_archive_gaps_are_dry():
    # hour 2 has only half the domain: treated as dry, splitting nothing here but
    # not contributing to the total
    da = _hourly([2.0, 5.0, 2.0], valid={1: 0.5})
    ev = detect_events(da)
    assert ev.iloc[0].total_mm == pytest.approx(4.0)
    # a missing hour (archive gap) never bridges: 7 missing hours split the events
    da = xr.concat([_hourly([2.0]), _hourly([2.0], start="2024-07-01 09:00")], "time")
    assert len(detect_events(da, min_gap_h=6)) == 2
