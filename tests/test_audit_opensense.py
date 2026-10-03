"""Audit of core.opensense: unit conventions, projection, scoring, the intercomparison
chain's time conventions, and the PWS checks.

References: pyproj for the projection, poligrain.validation and hand calculations for
the scores, the conventions stated in the module docstrings for the time labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.itu_p838 import get_k_alpha
from core.opensense import conventions as cv
from core.opensense import evaluation as ev
from core.opensense import intercomparison_chain as ic
from core.opensense import pws_qc


# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------
def test_to_ghz_declared_and_undeclared_units():
    f = xr.DataArray([18000.0, 23000.0], attrs={"units": "MHz"})
    np.testing.assert_allclose(cv.to_ghz(f), [18.0, 23.0])
    np.testing.assert_allclose(cv.to_ghz(xr.DataArray([1.8e10], attrs={"units": "Hz"})), [18.0])
    np.testing.assert_allclose(cv.to_ghz(xr.DataArray([38.0], attrs={"units": "GHz"})), [38.0])
    # undeclared: MHz recognised by magnitude, including E-band (80 GHz = 80000 MHz)
    np.testing.assert_allclose(cv.to_ghz(xr.DataArray([7456.0, 80000.0])), [7.456, 80.0])
    np.testing.assert_allclose(cv.to_ghz(xr.DataArray([7.456, 80.0])), [7.456, 80.0])
    with pytest.raises(ValueError):
        cv.to_ghz(xr.DataArray([1.0], attrs={"units": "THz"}))


def test_to_km_declared_and_undeclared_units():
    np.testing.assert_allclose(cv.to_km(xr.DataArray([2500.0], attrs={"units": "m"})), [2.5])
    np.testing.assert_allclose(cv.to_km(xr.DataArray([2.5], attrs={"units": "km"})), [2.5])
    np.testing.assert_allclose(cv.to_km(xr.DataArray([300.0, 4200.0])), [0.3, 4.2])
    np.testing.assert_allclose(cv.to_km(xr.DataArray([0.3, 4.2])), [0.3, 4.2])


def test_undeclared_length_of_a_short_link_network_is_read_as_km():
    """RISK: with no units, a network whose median link is < 100 m is taken as km."""
    assert cv.to_km(xr.DataArray([60.0, 80.0, 90.0]))[0] == 60.0


def test_normalize_polarization_spellings():
    out = cv.normalize_polarization(["V", "h", "Horizontal", "vertical", " H ", "x", "nan"])
    assert list(out) == ["vertical", "horizontal", "horizontal", "vertical",
                         "horizontal", "vertical", "vertical"]


def test_itu_coefficients_per_link_and_scalar_polarization():
    f = np.array([[18.0, 23.0], [38.0, 15.0]])
    pol = np.array([["horizontal", "vertical"], ["vertical", "horizontal"]])
    k, a = cv.itu_coefficients(f, pol)
    assert k.shape == f.shape
    for i in range(2):
        for j in range(2):
            assert (k[i, j], a[i, j]) == get_k_alpha(f[i, j], pol[i, j])
    # one polarization for every link is used, not replaced by the vertical default
    k, a = cv.itu_coefficients(f, np.array("horizontal"))
    assert k[0, 0] == get_k_alpha(18.0, "horizontal")[0]
    assert k[0, 0] != get_k_alpha(18.0, "vertical")[0]


def test_project_points_matches_pyproj_and_preserves_distance():
    pyproj = pytest.importorskip("pyproj")
    lon = np.array([11.95, 11.99, 12.02])
    lat = np.array([57.69, 57.71, 57.72])
    ds = xr.Dataset(coords={"id": ["a", "b", "c"], "lon": ("id", lon), "lat": ("id", lat)})
    ds = cv.project_points(ds, "EPSG:32632")
    x, y = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform(lon, lat)
    np.testing.assert_allclose(ds.x, x, atol=1e-6)
    np.testing.assert_allclose(ds.y, y, atol=1e-6)
    # Gothenburg sits 3 deg off zone 32's meridian: scale error ~1e-5, irrelevant
    _, _, d = pyproj.Geod(ellps="WGS84").inv(lon[0], lat[0], lon[2], lat[2])
    assert np.hypot(x[2] - x[0], y[2] - y[0]) / d == pytest.approx(1.0, abs=5e-4)


def test_project_cml_midpoint_and_sites():
    ds = xr.Dataset(coords={"cml_id": ["l1", "l2"],
                            "site_0_lon": ("cml_id", [11.95, 11.90]),
                            "site_0_lat": ("cml_id", [57.69, 57.60]),
                            "site_1_lon": ("cml_id", [11.99, 11.80]),
                            "site_1_lat": ("cml_id", [57.71, 57.70])})
    out = cv.project_cml(ds)
    np.testing.assert_allclose(out.x, (out.site_0_x + out.site_1_x) / 2)
    np.testing.assert_allclose(out.y, (out.site_0_y + out.site_1_y) / 2)
    assert out.site_1_x[0] > out.site_0_x[0]            # east is +x
    assert out.site_1_y[0] > out.site_0_y[0]            # north is +y


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def test_rainfall_metrics_against_hand_calculation():
    ref = np.array([0.0, 0.0, 0.05, 0.5, 1.0, 2.0, 4.0, 0.0, 3.0, np.nan])
    est = np.array([0.0, 0.2, 0.00, 0.4, 1.5, 1.0, 5.0, 0.0, np.nan, 1.0])
    m = ev.rainfall_metrics(ref, est)
    ok = np.isfinite(ref) & np.isfinite(est)
    r, e = ref[ok], est[ok]
    assert m["n"] == 8
    assert m["r"] == pytest.approx(np.corrcoef(r, e)[0, 1])
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((e - r) ** 2)))
    assert m["mae"] == pytest.approx(np.mean(np.abs(e - r)))
    # pbias over the whole record (zeros kept): 100 * (sum e - sum r) / sum r
    assert m["pbias"] == pytest.approx(100 * (e.sum() - r.sum()) / r.sum())
    assert m["ratio"] == pytest.approx(e.sum() / r.sum())
    # detection at >= 0.1 mm/h: ref wet {0.5,1,2,4}, est wet {0.2,0.4,1.5,1,5}
    tp, fp, fn, tn = 4, 1, 0, 3
    assert m["tpr"] == pytest.approx(tp / (tp + fn))
    assert m["fpr"] == pytest.approx(fp / (fp + tn))
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert m["mcc"] == pytest.approx(mcc)


def test_rainfall_metrics_pairs_by_coordinates_not_memory_order():
    t = pd.date_range("2024-01-01", periods=5, freq="min")
    rng = np.random.default_rng(0)
    a = xr.DataArray(rng.gamma(1, 2, (5, 3)), dims=("time", "cml_id"),
                     coords={"time": t, "cml_id": ["a", "b", "c"]})
    b = a.transpose("cml_id", "time").isel(cml_id=[2, 0, 1])
    assert ev.rainfall_metrics(a, b)["r"] == pytest.approx(1.0)


def test_aggregate_label_conventions():
    t = pd.date_range("2024-01-01 00:01", periods=30, freq="min")   # 00:01 .. 00:30
    da = xr.DataArray(np.arange(1, 31, dtype=float), dims="time", coords={"time": t})
    end = ev.aggregate(da, "15min", label="end")                  # (t-15, t] stamped t
    assert end.sel(time="2024-01-01 00:15").item() == pytest.approx(np.mean(range(1, 16)))
    assert end.sel(time="2024-01-01 00:30").item() == pytest.approx(np.mean(range(16, 31)))
    start = ev.aggregate(da, "15min", label="start")              # [t, t+15) stamped t
    assert start.sel(time="2024-01-01 00:00").item() == pytest.approx(np.mean(range(1, 15)))


# ---------------------------------------------------------------------------
# intercomparison chain
# ---------------------------------------------------------------------------
def test_hourly_from_mean_is_hour_ending_and_closed_right():
    # 15-min depths stamped at interval end: 00:15 ... 02:00
    t = pd.date_range("2024-01-01 00:15", periods=8, freq="15min")
    da = xr.DataArray([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], dims="time",
                      coords={"time": t})
    h = ic.hourly_from_mean(da, 4)
    np.testing.assert_allclose(h.sel(time=["2024-01-01 01:00", "2024-01-01 02:00"]), [4.0, 8.0])
    assert h.time.values[0] == np.datetime64("2024-01-01 01:00")
    # a missing step is filled by the mean of the hour, not counted as zero
    da[1] = np.nan
    assert h.sel(time="2024-01-01 01:00").item() == pytest.approx(4.0)
    assert ic.hourly_from_mean(da, 4).sel(time="2024-01-01 01:00").item() == pytest.approx(4.0)


def test_hourly_links_mean_of_one_minute_depths():
    t = pd.date_range("2024-01-01 00:01", periods=60, freq="min")
    r = xr.DataArray(np.full((1, 2, 60), 6.0), dims=("cml_id", "sublink_id", "time"),
                     coords={"time": t, "cml_id": ["l"], "sublink_id": list(ic.SUBLINKS)})
    out = ic.hourly_links(xr.Dataset({"R_acc": r / 60}))
    assert out.sel(time="2024-01-01 01:00").item() == pytest.approx(6.0)   # 6 mm/h for 1 h


def test_threshold_radar_zeroes_small_values_and_nan():
    out = ic.threshold_radar(xr.DataArray([0.0, 0.01, 0.011, np.nan, 2.0]))
    np.testing.assert_allclose(out, [0, 0, 0.011, 0, 2.0])


def test_openmrg_radar_rate_marshall_palmer_200_16():
    # Z = 200 R^1.6 -> R = (Z/200)^(1/1.6); 23 dBZ is ~1 mm/h
    dbz = 10 * np.log10(200 * 5.0 ** 1.6)
    assert ic.openmrg_radar_rate(dbz) == pytest.approx(5.0)


def _cml_with_spike(minutes=600):
    rng = np.random.default_rng(0)
    t = pd.date_range("2024-07-01", periods=minutes, freq="min")
    tl = np.empty((1, 2, minutes))
    for s in range(2):
        noise = rng.normal(0, 0.1, minutes)
        noise[:120] = rng.normal(0, 1.5, 120)     # noisy for 20% of the record
        tl[0, s] = 50 + noise
    tl[0, 1, 400] += 40                            # spike on channel2 only
    return xr.Dataset(
        {"tsl": (("cml_id", "sublink_id", "time"), np.full_like(tl, 10.0)),
         "rsl": (("cml_id", "sublink_id", "time"), 10.0 - tl)},
        coords={"cml_id": ["l1"], "sublink_id": list(ic.SUBLINKS), "time": t})


def test_preprocess_spike_removes_the_higher_channel_only():
    kept, removed = ic.preprocess(_cml_with_spike())
    assert list(kept.cml_id.values) == ["l1"] and removed.sizes["cml_id"] == 0
    tl = kept.tl.sel(cml_id="l1")
    assert np.isnan(tl.sel(sublink_id="channel2").isel(time=400))
    assert np.isfinite(tl.sel(sublink_id="channel1").isel(time=400))
    assert np.isnan(tl).sum() == 1


def test_preprocess_flat_link_is_removed():
    ds = _cml_with_spike()
    ds["rsl"] = xr.full_like(ds.rsl, -40.0)
    kept, removed = ic.preprocess(ds)
    assert kept.sizes["cml_id"] == 0 and list(removed.cml_id.values) == ["l1"]


@pytest.mark.parametrize("rad_freq", [1, 2, 5, 8, 10, 15])
def test_wet_from_radar_window_is_interval_plus_five_minutes(rad_freq):
    """A radar stamp t (interval end) wets t - (rad_freq - 1) ... t + 5.

    The notebook's ``origin=int(n / 2 - 6)`` gives this only for odd rad_freq (and
    rad_freq < 8); 5 and 15 min - the two datasets - are unaffected by the fix.
    """
    t = pd.date_range("2024-07-01", periods=90, freq="min")
    ds = xr.Dataset({"tl": (("cml_id", "sublink_id", "time"), np.zeros((1, 2, 90)))},
                    coords={"cml_id": ["l1"], "sublink_id": list(ic.SUBLINKS), "time": t})
    stamp = t[45]
    rad = xr.DataArray([[1.0]], dims=("cml_id", "time"),
                       coords={"cml_id": ["l1"], "time": [stamp]})
    wet = ic.wet_from_radar(ds, rad, rad_freq).isel(cml_id=0)
    for s in range(2):
        idx = np.flatnonzero(wet.isel(sublink_id=s).values)
        assert (idx.min() - 45, idx.max() - 45) == (-(rad_freq - 1), 5)
        assert idx.size == rad_freq + 5


# ---------------------------------------------------------------------------
# PWS QC
# ---------------------------------------------------------------------------
def _pws(n_steps=24):
    t = pd.date_range("2024-07-01", periods=n_steps, freq="5min")
    ids = ["p0", "p1", "p2", "far"]
    x = np.array([0.0, 1000.0, 2000.0, 50_000.0])
    y = np.zeros(4)
    amount = np.zeros((4, n_steps))
    amount[0] = 0.2
    amount[1] = 0.4
    amount[2] = 0.6
    amount[3] = 9.0
    rate = np.zeros((4, n_steps))
    rate[0, 3] = 12.0                               # 1 mm in 5 min ...
    amount[0, 3] = 0.0                              # ... but the bucket saw none
    rate[0, 5] = 5.0                                # 0.42 mm: below two tips
    amount[0, 5] = 0.0
    return xr.Dataset(
        {"rainfall_amount": (("id", "time"), amount),
         "rainfall_rate": (("id", "time"), rate)},
        coords={"id": ids, "time": t, "x": ("id", x), "y": ("id", y),
                "lon": ("id", 11.97 + x / 6e4), "lat": ("id", np.full(4, 57.7))})


def test_regularize_sums_amounts_and_keeps_gaps_nan():
    pws = _pws()
    pws["rainfall_amount"][1, 2] = np.nan
    ds = pws_qc.regularize(pws, "10min")
    assert ds.rainfall.dims == ("id", "time") and ds.sizes["time"] == 12
    assert ds.rainfall.sel(id="p2").isel(time=0).item() == pytest.approx(1.2)
    assert ds.rainfall.sel(id="p1").isel(time=1).item() == pytest.approx(0.4)   # one of two
    pws["rainfall_amount"][1, 3] = np.nan
    assert np.isnan(pws_qc.regularize(pws, "10min").rainfall.sel(id="p1").isel(time=1))


def test_neighbour_reference_excludes_self_and_far_stations():
    ds = pws_qc.regularize(_pws())
    dist = pws_qc.distances(ds)
    ref = pws_qc.neighbour_reference(ds, dist, max_distance_m=1500)
    r = ref.reference.isel(time=0)
    assert r.sel(id="p0").item() == pytest.approx(0.4)          # only p1 within 1.5 km
    assert r.sel(id="p1").item() == pytest.approx(0.4)          # median of p0 and p2
    assert np.isnan(r.sel(id="far").item())
    assert ref.nbrs_not_nan.isel(time=0).sel(id="p1").item() == 2
    assert ref.nbrs_not_nan.isel(time=0).sel(id="far").item() == 0


def test_rate_flag_two_tips_rule():
    ds = pws_qc.regularize(_pws())
    ds.attrs["step"] = "5min"
    f = pws_qc.rate_flag(ds).sel(id="p0")
    assert f.isel(time=3).item() == 1                 # 12 mm/h x 5 min = 1 mm, bucket 0
    assert f.isel(time=5).item() == 0                 # 0.42 mm < 0.5 mm
    assert f.isel(time=4).item() == 0                 # no rate, no amount
