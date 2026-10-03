"""The Dutch CML loader (core.opensense.netherlands) on synthetic inputs.

RAINLINK-format text: time conversion, path and sub-link naming, the monthly split and
streaming out of a zip. KNMI hourly text: station header, hour-ending times, RH codes.
The RAINLINK retrieval: dry gives zero rain, a known attenuation gives the rain the
power law says, and the vectorised wet/dry step equals pycomlink's own.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.cml.power_law import itu_ab
from core.opensense import netherlands as nl

HEADER = "YStart XStart YEnd XEnd Frequency DateTime ES SES Pmin Pmax PathLength Vendor ID\n"


def _row(ys, xs, ye, xe, f, t, pmin, pmax, length, vendor, i):
    return f"{ys} {xs} {ye} {xe} {f} {t} 0 0 {pmin} {pmax} {length} {vendor} {i}\n"


A = (52.1, 5.1)        # path A: sub-link 7 from A to B, sub-link 3 from B to A
B = (52.0, 5.2)
C = (51.9, 4.9)        # path C-D: one sub-link (9), C is site 0


def _text() -> str:
    rows = [HEADER]
    for t in ("201206302330", "201206302345", "201207010000", "201207010015"):
        rows.append(_row(*A, *B, 38.0, t, -50, -48, 13.1, "NOKIA", 7))
        rows.append(_row(*B, *A, 37.5, t, -52, -51, 13.1, "NOKIA", 3))
        rows.append(_row(*C, 52.05, 5.0, 23.0, t, -60.5, -59.9, 18.0, "NEC", 9))
    return "".join(rows)


def test_parse_times_and_sublinks():
    df = nl.read_rainlink(io.BytesIO(_text().encode()))
    assert len(df) == 12
    ds = nl.to_dataset(df)
    # DateTime is the interval END in UTC, kept as the time label
    assert ds.time.values[0] == np.datetime64("2012-06-30T23:30")
    assert ds.sizes == {"cml_id": 2, "sublink_id": 2, "time": 4}
    ab = "52.00000_5.20000_52.10000_5.10000"          # site 0 = B (smaller latitude)
    assert ab in ds.cml_id.values
    # sublink_0 is the direction from site 0 (B) to site 1: RAINLINK ID 3
    assert ds.rainlink_id.sel(cml_id=ab).values.tolist() == [3, 7]
    assert ds.direction.sel(cml_id=ab).values.tolist() == [0, 1]
    assert ds.frequency.sel(cml_id=ab, sublink_id="sublink_1").item() == 38.0
    assert ds.rsl_min.sel(cml_id=ab, sublink_id="sublink_0").values.tolist() == [-52] * 4
    cd = [c for c in ds.cml_id.values if c != ab][0]
    assert ds.rainlink_id.sel(cml_id=cd).values.tolist() == [9, -1]
    assert np.isnan(ds.rsl_max.sel(cml_id=cd, sublink_id="sublink_1")).all()
    np.testing.assert_allclose(ds.rsl_max.sel(cml_id=cd, sublink_id="sublink_0"), -59.9, rtol=1e-6)
    assert ds.length.attrs["units"] == "km" and ds.frequency.attrs["units"] == "GHz"
    assert ds.rsl_min.attrs["units"] == "dBm"


def test_period_filter_and_small_blocks():
    src = _text().encode()
    full = nl.read_rainlink(io.BytesIO(src))
    # interval ends in (23:45, 00:15]: two steps, whatever the block size
    for block in (64, 200, 1 << 20):
        part = pd.concat(list(nl.iter_rainlink(io.BytesIO(src), "2012-06-30 23:45",
                                               "2012-07-01 00:15", block_bytes=block)))
        assert sorted(part.DateTime.unique()) == [201207010000, 201207010015]
        assert len(part) == 6
    assert len(full) == 12


def test_monthly_split_from_zip(tmp_path):
    z = tmp_path / "IDRawCMLdata.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("IDRawCMLdata.dat", _text())
    out = tmp_path / "monthly"
    files = nl.convert_months("2012-06-30", "2012-07-02", source=z, out_dir=out)
    assert [f.name for f in files] == ["cml_2012-06.nc", "cml_2012-07.nc"]
    june, july = (xr.open_dataset(f) for f in files)
    # the interval ending 1 July 00:00 is June's last
    assert june.time.values[-1] == np.datetime64("2012-07-01T00:00")
    assert july.sizes["time"] == 1 and july.time.values[0] == np.datetime64("2012-07-01T00:15")
    ds = nl.open_months("2012-06-30", "2012-07-02", out_dir=out, convert=False)
    assert ds.sizes["time"] == 4 and ds.sizes["cml_id"] == 2


KNMI = """# SOURCE: ROYAL NETHERLANDS METEOROLOGICAL INSTITUTE (KNMI)
# STN         LON(east)   LAT(north)  ALT(m)      NAME
# 260         5.180       52.100      1.90        De Bilt
# 380         5.762       50.906      114.30      Maastricht Airport
# YYYYMMDD  : datum (YYYY=jaar;MM=maand;DD=dag)
# RH        : Uursom van de neerslag (in 0.1 mm) (-1 voor <0.05 mm)
# STN,YYYYMMDD,HH,   RH
  260,20120701,    1,   12
  260,20120701,    2,   -1
  260,20120701,   24,    0
  380,20120701,    1,
  380,20120701,    2,    3
  380,20120701,   24,   55
"""


def test_knmi_parse():
    st, rain = nl.parse_knmi_hourly(KNMI)
    assert st.loc[260, "name"] == "De Bilt" and st.loc[380, "lat"] == pytest.approx(50.906)
    assert rain.dims == ("station", "time")
    # HH is the hour ending at HH UT; 24 is midnight of the next day
    assert rain.time.values.tolist() == pd.DatetimeIndex(
        ["2012-07-01 01:00", "2012-07-01 02:00", "2012-07-02 00:00"]).as_unit("ns").asi8.tolist()
    np.testing.assert_allclose(rain.sel(station=260), [1.2, 0.0, 0.0])   # -1 (<0.05 mm) -> 0
    v = rain.sel(station=380).values
    assert np.isnan(v[0]) and v[1] == pytest.approx(0.3) and v[2] == pytest.approx(5.5)
    assert float(rain.lat.sel(station=260)) == pytest.approx(52.1)


# ------------------------------------------------------------------ retrieval

def _cluster(n=6, steps=192, length=5.0, freq=38.0, seed=None):
    """``n`` parallel 5 km links a few km apart, Pmin = Pmax = -50 dBm (dry)."""
    lat0 = 52.0 + 0.01 * np.arange(n)
    lon0 = np.full(n, 5.0)
    lon1 = lon0 + length / (111.32 * np.cos(np.radians(52.0)))
    time = pd.date_range("2012-07-01 00:15", periods=steps, freq="15min")
    p = np.full((n, steps), -50.0)
    if seed is not None:
        p = p + np.random.default_rng(seed).integers(-1, 2, size=p.shape)
    ids = [f"l{i}" for i in range(n)]
    return xr.Dataset(
        {"pmin": (("cml_id", "time"), p.copy()), "pmax": (("cml_id", "time"), p.copy())},
        coords={"cml_id": ids, "time": time, "length": ("cml_id", np.full(n, length)),
                "frequency": ("cml_id", np.full(n, freq)),
                "site_0_lat": ("cml_id", lat0), "site_0_lon": ("cml_id", lon0),
                "site_1_lat": ("cml_id", lat0), "site_1_lon": ("cml_id", lon1)})


def test_retrieval_dry_is_zero():
    out = nl.rainlink_retrieval(_cluster())
    r = out.rain.isel(time=slice(96, None))          # after the 24-h spin-up
    assert r.notnull().all() and float(abs(r).max()) == 0.0


def test_retrieval_known_attenuation():
    sub = _cluster()
    event = slice(150, 158)                           # 2 h of rain on every link
    a_max, a_min = 12.0, 6.0                          # dB below the -50 dBm reference
    sub.pmin[:, event] = -50.0 - a_max
    sub.pmax[:, event] = -50.0 - a_min
    out = nl.rainlink_retrieval(sub)
    assert (out.wet[:, event] == 1).all()
    a, b = itu_ab([38.0], ["v"])
    aa = nl.RETRIEVAL["waa_max"]
    r = lambda A: ((A - aa) / (a[0] * 5.0)) ** (1 / b[0])   # noqa: E731
    expected = 0.33 * r(a_max) + 0.67 * r(a_min)
    np.testing.assert_allclose(out.rain[:, event], expected, rtol=1e-6)
    assert float(out.rain[:, 100:148].max()) == 0.0   # dry before the event


def test_fast_wetdry_equals_pycomlink():
    from pycomlink.processing.wet_dry.nearby_wetdry import (
        calc_distance_between_cml_endpoints, nearby_wetdry)
    sub = _cluster(n=7, steps=260, seed=1)
    sub.pmin[:3, 150:160] -= 8.0                      # rain on part of the cluster
    sub.pmin[2, 40:60] = np.nan                       # a gap
    sub["pmin"] = sub.pmin.where(np.arange(260) != 200)
    pmin = sub.pmin
    dist = calc_distance_between_cml_endpoints(sub.cml_id.values, sub.site_0_lat.values,
                                               sub.site_0_lon.values, sub.site_1_lat.values,
                                               sub.site_1_lon.values)
    wet_ref, F_ref = nearby_wetdry(pmin, dist, **nl.WETDRY)
    within = nl.nearby_within(sub.site_0_lat, sub.site_0_lon, sub.site_1_lat, sub.site_1_lon)
    wet, F = nl.nearby_wetdry_fast(pmin, within, **nl.WETDRY)
    xr.testing.assert_allclose(wet.transpose(*wet_ref.dims), wet_ref.astype(float))
    xr.testing.assert_allclose(F.transpose(*F_ref.dims), F_ref)
    assert (wet == 1).any()
