"""core.opensense.networks helpers, core.events and measured-TSL total loss - no data files."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.cml.preprocess import total_loss
from core.events import detect_events
from core.geo import Domain, Grid
from core.opensense.networks import points_near_links, regrid, to_hourly


def _series(values, freq="15min", start="2024-01-01 00:15"):
    t = pd.date_range(start, periods=len(values), freq=freq)
    return xr.DataArray(np.asarray(values, float)[None, :], dims=("station", "time"),
                        coords={"station": ["a"], "time": t})


def test_to_hourly_end_and_start_labels():
    # four 15-min amounts stamped 00:15 .. 01:00
    da = _series([1, 2, 3, 4])
    end = to_hourly(da, "end")                     # (00:00, 01:00] -> labelled 01:00
    assert float(end.sel(time="2024-01-01 01:00").item()) == 10
    start = to_hourly(_series([1, 2, 3, 4], start="2024-01-01 00:00"), "start")
    assert float(start.sel(time="2024-01-01 01:00").item()) == 10


def test_to_hourly_needs_coverage():
    da = _series([1, np.nan, np.nan, 4])
    assert np.isnan(to_hourly(da, "end", min_coverage=0.8).sel(time="2024-01-01 01:00").item())
    assert float(to_hourly(da, "end", min_coverage=0.5).sel(time="2024-01-01 01:00").item()) == 5


def test_regrid_block_mean_and_nearest_fill():
    grid = Grid(np.array([40.0, 40.02]), np.array([-74.0, -73.98]))
    # three source pixels: two inside the first cell, one inside the last
    lat = np.array([40.001, 39.999, 40.02])
    lon = np.array([-74.0, -74.001, -73.98])
    out = regrid(np.array([[2.0, 4.0, 8.0]]), lat, lon, grid)
    assert out.shape == (1, 2, 2)
    assert out[0, 0, 0] == pytest.approx(3.0)          # mean of the two
    assert out[0, 1, 1] == pytest.approx(8.0)
    assert np.isfinite(out).all()                     # empty cells take the nearest pixel


def test_points_near_links_radius():
    t = pd.date_range("2024-01-01 01:00", periods=2, freq="1h")
    pts = xr.DataArray([[1.0, 2.0], [10.0, 20.0]], dims=("station", "time"),
                       coords={"station": ["near", "far"], "time": t,
                               "lat": ("station", [40.70, 41.5]), "lon": ("station", [-74.0, -74.0])})
    links = pd.DataFrame({"site_0_lat": [40.69], "site_0_lon": [-74.01], "site_1_lat": [40.71],
                          "site_1_lon": [-73.99]}, index=pd.Index(["l1"], name="link"))
    out = points_near_links(pts, links, radius_km=3.0)
    assert out.sel(link="l1").values.tolist() == [1.0, 2.0]


def test_detect_events_gap_rule_and_total():
    t = pd.date_range("2024-01-01 01:00", periods=40, freq="h")
    rain = np.zeros(40)
    rain[[2, 3, 6]] = 1.0          # gap of 2 dry hours: one event
    rain[[20, 21]] = 2.0           # 13 dry hours later: a second event
    field_ = xr.DataArray(np.broadcast_to(rain[:, None, None], (40, 2, 2)).copy(),
                          dims=("time", "lat", "lon"), coords={"time": t})
    ev = detect_events(field_, min_gap_h=6, min_total_mm=1.0)
    assert len(ev) == 2
    assert ev.total_mm.tolist() == [3.0, 4.0]
    assert ev.start.iloc[0] == str(t[2] - pd.Timedelta("1h"))


def test_total_loss_uses_measured_tsl():
    t = pd.date_range("2024-01-01", periods=4, freq="1min")
    links = xr.Dataset({"rsl": (("link", "time"), [[-50.0, -52.0, np.nan, -51.0]]),
                        "tsl": (("link", "time"), [[10.0, np.nan, 12.0, 12.0]])},
                       coords={"link": ["l"], "time": t})
    tl = total_loss(links)
    # a missing TSL sample is bridged; a missing RSL sample stays missing
    assert tl.values[0].tolist()[:2] == [60.0, 62.0] and np.isnan(tl.values[0, 2]) and tl.values[0, 3] == 63.0
    assert total_loss(links.drop_vars("tsl"), tsl=0.0).values[0, 0] == 50.0


def test_domain_padding_contains():
    d = Domain(40.0, 41.0, -74.0, -73.0).pad(0.1)
    assert d.contains([39.95], [-74.05])[0]
