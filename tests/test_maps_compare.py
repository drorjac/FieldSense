"""core.maps.merge (observations, merge_idw) and core.maps.scores (compare_maps, compare_links)."""

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid
from core.maps.merge import merge_idw, observations
from core.maps.scores import compare_links, compare_maps

TIMES = pd.date_range("2015-07-01 01:00", periods=4, freq="h").as_unit("ns")
GRID = Grid(np.round(np.arange(57.60, 57.80, 0.01), 4), np.round(np.arange(11.85, 12.15, 0.01), 4))


def _links(values):
    """Three links inside the grid, hourly values ``(link, time)``."""
    la0, lo0 = np.array([57.65, 57.70, 57.74]), np.array([11.90, 12.00, 12.05])
    la1, lo1 = la0 + 0.02, lo0 + 0.03
    return xr.DataArray(np.asarray(values, float), dims=("link", "time"),
                        coords={"link": ["a", "b", "c"], "time": TIMES,
                                "site_0_lat": ("link", la0), "site_0_lon": ("link", lo0),
                                "site_1_lat": ("link", la1), "site_1_lon": ("link", lo1),
                                "mid_lat": ("link", (la0 + la1) / 2), "mid_lon": ("link", (lo0 + lo1) / 2)})


def _gauges(values):
    return xr.DataArray(np.asarray(values, float), dims=("station", "time"),
                        coords={"station": ["g1", "g2"], "time": TIMES,
                                "lat": ("station", [57.66, 57.75]), "lon": ("station", [11.95, 12.10])})


def _radar(value=2.0):
    return xr.DataArray(np.full((len(TIMES),) + GRID.shape, value), dims=("time", "lat", "lon"),
                        coords={"time": TIMES, "lat": GRID.lat, "lon": GRID.lon}, attrs={"units": "mm"})


def _link_table(links):
    return pd.DataFrame({c: links[c].values for c in ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon")},
                        index=pd.Index(links.link.values, name="link"))


def test_observations_stacks_links_and_gauges_with_their_kind():
    links, gauges = _links(np.ones((3, 4))), _gauges(np.full((2, 4), 3.0))
    obs, rad = observations(cml=links, gauges=gauges)
    assert obs.dims == ("point", "time") and obs.sizes["point"] == 5
    assert list(obs.kind.values) == ["cml"] * 3 + ["gauges"] * 2
    np.testing.assert_allclose(obs.lat.values[:3], links.mid_lat.values)
    assert rad is None


def test_observations_reads_the_radar_at_each_point():
    links, gauges = _links(np.ones((3, 4))), _gauges(np.full((2, 4), 3.0))
    _, rad = observations(radar=_radar(2.0), cml=links, gauges=gauges)
    np.testing.assert_allclose(rad.values, 2.0)


def test_merge_idw_of_a_uniform_value_is_that_value_where_covered():
    obs, _ = observations(cml=_links(np.full((3, 4), 1.5)), gauges=_gauges(np.full((2, 4), 1.5)))
    m = merge_idw(obs, GRID)
    assert m.dims == ("time", "lat", "lon")
    v = m.values[np.isfinite(m.values)]
    assert v.size > 0
    np.testing.assert_allclose(v, 1.5, rtol=1e-6)


def test_merge_idw_kind_weights_pull_towards_the_heavier_source():
    obs, _ = observations(cml=_links(np.full((3, 4), 1.0)), gauges=_gauges(np.full((2, 4), 5.0)))
    light = float(merge_idw(obs, GRID, kind_weights={"cml": 0.1}).mean())
    heavy = float(merge_idw(obs, GRID, kind_weights={"cml": 10.0}).mean())
    assert heavy < light


def test_compare_maps_perfect_estimate():
    r = _radar(2.0) + xr.DataArray(np.arange(len(TIMES), dtype=float), dims="time", coords={"time": TIMES})
    per_hour, pooled = compare_maps(r, r)
    assert len(per_hour) == len(TIMES)
    assert pooled["rmse"] == 0 and pooled["bias"] == 0
    assert pooled["hours"] == len(TIMES)
    assert pooled["event_total_scores"]["rmse"] == 0


def test_compare_maps_only_wet_hours_drops_dry_hours():
    r = _radar(2.0).copy()
    r[0] = 0.0
    per_hour, pooled = compare_maps(r, r, only_wet_hours=True)
    assert pooled["hours"] == len(TIMES) - 1


def test_compare_links_against_a_uniform_radar():
    links = _links(np.full((3, 4), 2.5))
    out = compare_links(links, _radar(2.0), _link_table(links))
    assert list(out.index) == ["a", "b", "c"]
    np.testing.assert_allclose(out["bias"], 0.5)
    np.testing.assert_allclose(out["total_radar"], 8.0)
