"""Unit and naming conventions across OpenSense sources."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.opensense import conventions as cv
from core.opensense import example_data


def _da(values, units=None):
    return xr.DataArray(np.asarray(values, float),
                        attrs={} if units is None else {"units": units})


@pytest.mark.parametrize("values, units, expected", [
    ([2500.0], "m", 2.5), ([2.5], "km", 2.5),
    ([2500.0], None, 2.5), ([2.5], None, 2.5),          # undeclared: magnitude
])
def test_to_km(values, units, expected):
    assert cv.to_km(_da(values, units))[0] == pytest.approx(expected)


@pytest.mark.parametrize("values, units, expected", [
    ([23000.0], "MHz", 23.0), ([23e9], "Hz", 23.0), ([23.0], "GHz", 23.0),
    ([7456.0], None, 7.456),                             # the example-subset trap
    ([23.0], None, 23.0),
])
def test_to_ghz(values, units, expected):
    assert cv.to_ghz(_da(values, units))[0] == pytest.approx(expected)


def test_unknown_units_raise_rather_than_guess():
    with pytest.raises(ValueError):
        cv.to_ghz(_da([23.0], "furlongs"))
    with pytest.raises(ValueError):
        cv.to_km(_da([2.0], "miles"))


def test_polarization_spellings():
    got = cv.normalize_polarization(["Vertical", "v", "H", "horizontal", "?"])
    assert list(got) == ["vertical", "vertical", "horizontal", "horizontal",
                         "vertical"]


def _radar(units, accum_h=None, lat_lon_1d=True):
    time = pd.date_range("2020-06-01", periods=4, freq="15min").as_unit("ns")
    lat = np.linspace(57.6, 57.8, 5)
    lon = np.linspace(11.8, 12.1, 6)
    attrs = {"units": units}
    if accum_h is not None:
        attrs["accum_time_h"] = accum_h
    return xr.Dataset({"R": (("time", "lat", "lon"), np.ones((4, 5, 6)), attrs)},
                      coords={"time": time, "lat": lat, "lon": lon})


def test_radar_accumulation_is_converted_to_a_rate():
    """OpenRainER's subset calls a 15-minute accumulation ``R`` (units mm)."""
    out = example_data.normalize_radar(_radar("mm", accum_h=0.25), "EPSG:32632")
    assert float(out.R.max()) == pytest.approx(4.0)
    assert out.R.attrs["units"] == "mm h-1"
    # a rate is left alone
    rate = example_data.normalize_radar(_radar("mm/h"), "EPSG:32632")
    assert float(rate.R.max()) == pytest.approx(1.0)
    # and the step is used when the accumulation period is not declared
    step = example_data.normalize_radar(_radar("mm"), "EPSG:32632")
    assert float(step.R.max()) == pytest.approx(4.0)


def test_project_grid_attaches_what_mergeplg_and_poligrain_read():
    out = cv.project_grid(_radar("mm/h"), "EPSG:32632")
    assert out.R.dims == ("time", "y", "x")
    for c in ("x_grid", "y_grid", "lon", "lat"):
        assert out[c].dims == ("y", "x")
    assert out.x.dims == ("x",) and out.y.dims == ("y",)
    # projected metres, increasing eastward
    assert np.all(np.diff(out.x.values) > 0)
    assert 6.3e6 < float(out.y_grid.mean()) < 6.5e6


def test_check_format_reports_what_is_missing():
    from conftest import make_cml, rain_event
    ds = make_cml(rain_event(n_links=1), [[23.0, 23.0]], [4.0])
    ds = ds.assign_coords(cml_id=ds.cml_id.astype(str))      # the convention: strings
    ds.rsl.attrs["units"] = "dBm"
    ds.tsl.attrs["units"] = "dBm"
    table = cv.check_format(ds)
    assert table.passed.all(), table[~table.passed]
    broken = ds.drop_vars("length")
    broken.frequency.attrs.pop("units")
    bad = cv.check_format(broken).set_index("check")
    assert not bad.loc["site coordinates, frequency, length", "passed"]
    assert "length" in bad.loc["site coordinates, frequency, length", "detail"]
    assert not bad.loc["frequency units declared", "passed"]
    ints = cv.check_format(make_cml(rain_event(n_links=1), [[23.0, 23.0]], [4.0])).set_index("check")
    assert not ints.loc["cml_id is a string", "passed"]       # OpenMRG's subset does this
