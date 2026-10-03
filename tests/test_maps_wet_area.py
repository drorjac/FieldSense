"""core.maps.wet_area: wet probability, masked and conditional IDW, wet-area scores."""

import numpy as np
import pandas as pd
import xarray as xr

from core.geo import Grid
from core.maps import wet_area as wa
from core.maps.idw import idw_map

GRID = Grid(np.round(np.arange(57.60, 57.80, 0.01), 4), np.round(np.arange(11.80, 12.20, 0.01), 4))


def _stations(values):
    """Four stations in a row west to east; ``values (station, time)``."""
    return xr.DataArray(np.asarray(values, float), dims=("station", "time"),
                        coords={"station": list("abcd"),
                                "time": pd.date_range("2015-07-01", periods=np.shape(values)[1], freq="h"),
                                "lat": ("station", [57.70] * 4), "lon": ("station", [11.85, 11.95, 12.05, 12.15])})


def test_wet_probability_is_one_and_zero_at_wet_and_dry_stations():
    p = wa.wet_probability(_stations([[5.0], [4.0], [0.0], [0.0]]), GRID, radius_m=None)
    assert float(p.sel(lat=57.70, lon=11.85).squeeze()) == 1.0
    assert float(p.sel(lat=57.70, lon=12.15).squeeze()) == 0.0
    assert ((p >= 0) & (p <= 1)).all()


def test_masked_idw_dries_the_dry_side_and_keeps_the_wet_side():
    st = _stations([[5.0], [4.0], [0.0], [0.0]])
    plain = idw_map(_as_links(st), GRID, radius_m=None)
    m = wa.masked_idw(st, GRID, radius_m=None)
    east = m.sel(lon=slice(12.08, 12.20))
    assert float(east.max()) == 0.0
    assert float(plain.sel(lon=slice(12.08, 12.20)).max()) > 0          # plain IDW smears rain east
    xr.testing.assert_equal(m.sel(lat=57.70, lon=11.85), plain.sel(lat=57.70, lon=11.85))


def test_conditional_idw_keeps_the_intensity_where_it_rains():
    st = _stations([[4.0], [4.0], [0.0], [0.0]])
    plain = idw_map(_as_links(st), GRID, radius_m=None)
    c = wa.conditional_idw(st, GRID, radius_m=None)
    wet = c.values[c.values > 0]
    np.testing.assert_allclose(wet, 4.0)                                 # not diluted by the dry stations
    assert float(plain.sel(lat=57.70, lon=11.98).squeeze()) < 4.0


def test_all_dry_gives_a_dry_map():
    c = wa.conditional_idw(_stations(np.zeros((4, 2))), GRID, radius_m=None)
    assert float(c.max()) == 0.0


def test_wet_area_scores():
    ref = np.array([0, 0, 0, 1, 2, 3.0])
    s = wa.wet_area_scores(np.array([0.5, 0.5, 0.5, 1, 1, 1.0]), ref)
    assert s["war_ref"] == 0.5 and s["war_est"] == 1.0 and s["war_ratio"] == 2.0
    assert s["peak_ratio"] < 1


def _as_links(st):
    return st.rename(station="link").assign_coords(mid_lat=("link", st.lat.values), mid_lon=("link", st.lon.values))
