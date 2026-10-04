"""The OpenSense intercomparison chain on synthetic links (core.opensense.intercomparison_chain)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.opensense import intercomparison_chain as ic

pytest.importorskip("pycomlink")


def _links(n_links=3, minutes=2 * 24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2022-06-01", periods=minutes, freq="1min")
    tl = 40 + rng.normal(0, 0.3, (n_links, 2, minutes))
    rain = np.zeros(minutes)
    rain[600:660] = 3.0                       # one hour of rain, 10:00-11:00
    tl += rain[None, None, :]
    tsl = np.full_like(tl, 10.0)
    ds = xr.Dataset({"tsl": (("cml_id", "sublink_id", "time"), tsl),
                     "rsl": (("cml_id", "sublink_id", "time"), tsl - tl)},
                    coords={"cml_id": [f"l{i}" for i in range(n_links)],
                            "sublink_id": list(ic.SUBLINKS), "time": t,
                            "frequency": (("cml_id", "sublink_id"), np.full((n_links, 2), 23000.0)),
                            "polarization": (("cml_id", "sublink_id"), np.full((n_links, 2), "v")),
                            "length": ("cml_id", np.full(n_links, 5000.0)),
                            "site_0_lat": ("cml_id", 44.0 + 0.01 * np.arange(n_links)),
                            "site_0_lon": ("cml_id", np.full(n_links, 11.0)),
                            "site_1_lat": ("cml_id", 44.0 + 0.01 * np.arange(n_links)),
                            "site_1_lon": ("cml_id", np.full(n_links, 11.05))})
    return ds


def _radar_along(ds, step=15):
    t = pd.date_range(ds.time.values[0], ds.time.values[-1], freq=f"{step}min")
    v = np.where((t >= "2022-06-01 10:00") & (t <= "2022-06-01 11:00"), 1.0, 0.0)
    return xr.DataArray(np.tile(v, (ds.sizes["cml_id"], 1)), dims=("cml_id", "time"),
                        coords={"cml_id": ds.cml_id.values, "time": t})


def test_openmrg_radar_rate_uses_b_1_6():
    # 10 * log10(200 * R^1.6) dBZ gives back R
    r = 7.0
    assert ic.openmrg_radar_rate(10 * np.log10(200 * r ** 1.6)) == pytest.approx(r)


def test_hourly_from_mean_is_right_closed_and_fills_gaps():
    t = pd.date_range("2022-06-01 00:15", periods=8, freq="15min")
    da = xr.DataArray([1.0, 1, 1, 1, 2, np.nan, 2, 2], dims="time", coords={"time": t})
    h = ic.hourly_from_mean(da, 4)
    assert list(pd.DatetimeIndex(h.time.values).hour) == [1, 2]
    np.testing.assert_allclose(h.values, [4.0, 8.0])          # the gap filled by the mean


def test_threshold_radar():
    da = xr.DataArray([0.005, 0.01, 0.02, np.nan])
    np.testing.assert_allclose(ic.threshold_radar(da).values, [0, 0, 0.02, 0])


def test_spike_filter_removes_the_higher_channel():
    ds = _links(1)
    rsl = ds.rsl.values.copy()
    rsl[0, 0, 100] -= 50                         # channel1 total loss +50 dB
    ds["rsl"] = (ds.rsl.dims, rsl)
    kept, _ = ic.preprocess(ds)
    assert np.isnan(kept.tl.values[0, 0, 100])
    assert np.isfinite(kept.tl.values[0, 1, 100])


def test_flat_link_is_dropped():
    ds = _links(2)
    rsl = ds.rsl.values.copy()
    rsl[1] = -30.0                               # no fluctuation at all on link 1
    ds["rsl"] = (ds.rsl.dims, rsl)
    kept, removed = ic.preprocess(ds)
    assert list(removed.cml_id.values) == ["l1"]
    assert list(kept.cml_id.values) == ["l0"]


def test_wet_mask_follows_radar_and_dilation():
    ds = _links(1)
    ds["tl"] = ds.tsl - ds.rsl
    wet = ic.wet_from_radar(ds, _radar_along(ds, 15), 15)
    w = wet.isel(cml_id=0, sublink_id=0).to_series()
    assert w.loc["2022-06-01 10:30"] and not w.loc["2022-06-01 06:00"]
    assert (wet.isel(sublink_id=0).values == wet.isel(sublink_id=1).values).all()
    # 20-minute structuring element with origin 4: a stamp t marks t-14 ... t+5 min, the
    # radar's 15-min interval and 5 minutes after it
    assert w.loc["2022-06-01 09:46"] and not w.loc["2022-06-01 09:45"]
    assert w.loc["2022-06-01 11:05"] and not w.loc["2022-06-01 11:06"]


def test_chain_finds_the_rain_hour():
    ds = _links(3)
    hourly, removed = ic.chain(ds, _radar_along(ds), 15)
    assert removed == []
    s = hourly.mean("cml_id").to_series()
    assert s.idxmax() == pd.Timestamp("2022-06-01 11:00")       # hour ending 11:00
    assert s.loc["2022-06-01 11:00"] > 1.0
    assert s.drop(pd.Timestamp("2022-06-01 11:00")).max() < 0.05 * s.max()
