"""MRMS radar, ASOS and the lat/lon grid helpers - no network, no data files."""

import numpy as np
import pandas as pd
import pytest

from core.asos import classify_wxcodes
from core.geo import NYC, Domain, Grid, haversine_m
from core.radar.mrms.client import _GridSpec, file_url, valid_times


def test_mrms_urls():
    t = pd.Timestamp("2024-01-09 12:00")
    assert file_url("MultiSensor_QPE_01H_Pass2", t, "aws") == (
        "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00/20240109/"
        "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20240109-120000.grib2.gz")
    assert file_url("PrecipFlag", t, "iem").endswith("/2024/01/09/mrms/ncep/PrecipFlag/PrecipFlag_00.00_20240109-120000.grib2.gz")


def test_valid_times():
    assert len(valid_times("MultiSensor_QPE_01H_Pass2", "2024-01-01 00:30", "2024-01-01 03:00")) == 3
    assert len(valid_times("PrecipRate", "2024-01-01 00:00", "2024-01-01 00:10")) == 6
    assert len(valid_times("PrecipFlag", "2024-01-01 00:00", "2024-01-01 01:00", freq="10min")) == 7
    with pytest.raises(ValueError):
        valid_times("PrecipFlag", "2024-01-01", "2024-01-02", freq="3min")


def test_mrms_grid_window_covers_domain():
    spec = _GridSpec(7000, 3500, 54.995, 230.005 - 360, -0.01, 0.01)
    rs, cs = spec.window(NYC)
    lat = spec.lat0 + spec.dlat * np.arange(rs.start, rs.stop)
    lon = spec.lon0 + spec.dlon * np.arange(cs.start, cs.stop)
    assert lat.min() >= NYC.lat_min - 1e-9 and lat.max() <= NYC.lat_max + 1e-9
    assert lat.min() - NYC.lat_min < 0.01 and NYC.lon_max - lon.max() < 0.01


@pytest.mark.parametrize("code,expected", [
    ("-RA BR", "rain"), ("-SN", "snow"), ("-PLRA", "mix"), ("RASN", "mix"), ("-FZRA BR", "freezing"),
    ("BR", "none"), (None, "none"), ("UP", "unknown"), ("-DZ", "rain"), ("PL", "mix")])
def test_wxcodes(code, expected):
    assert classify_wxcodes(code) == expected


def test_domain_helpers():
    assert haversine_m(40.7, -74.0, 40.8, -74.0) == pytest.approx(11_119, rel=1e-3)
    g = Grid.from_domain(Domain(40.6, 40.7, -74.0, -73.9), 0.01)
    assert np.allclose(np.round(g.lat * 1000) % 10, 5)      # MRMS-aligned centres


def test_grid_centres_inside_domain_exactly():
    from core.geo import OPENMESH
    for dom in (NYC, OPENMESH, Domain(40.6, 40.7, -74.0, -73.9)):
        g = Grid.from_domain(dom, 0.01)
        assert g.lat.min() >= dom.lat_min and g.lat.max() <= dom.lat_max
        assert g.lon.min() >= dom.lon_min and g.lon.max() <= dom.lon_max
        assert dom.lat_min + 0.01 > g.lat.min() and dom.lat_max - 0.01 < g.lat.max()
    assert Grid.from_domain(NYC).shape == (45, 59)          # = the MRMS crop


def test_mrms_empty_marker_does_not_break_day(tmp_path, monkeypatch):
    from core.radar.mrms import client as mrms
    c = mrms.MRMSClient(cache_dir=tmp_path, processes=0)
    lat, lon = np.array([40.5, 40.51]), np.array([-74.0, -73.99])

    def fake(p, times, domain):
        times = list(times)
        ok = [t for t in times if t.hour != 3]
        gone = [t for t in times if t.hour == 3]
        if not ok:
            return None, gone, []
        return mrms._to_dataarray(np.ones((len(ok), 2, 2)), lat, lon, ok, p), gone, []

    monkeypatch.setattr(c, "_fetch_many", fake)
    with pytest.raises(mrms.MRMSNotFound):
        c.load("MultiSensor_QPE_01H_Pass2", "2020-01-01 03:00", "2020-01-01 03:00", NYC)
    da = c.load("MultiSensor_QPE_01H_Pass2", "2020-01-01 01:00", "2020-01-01 05:00", NYC)
    assert da.sizes["time"] == 4 and da.attrs["missing_times"] == [str(pd.Timestamp("2020-01-01 03:00"))]
