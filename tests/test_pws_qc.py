"""pws_qc: each filter catches the fault it is named for, and nothing else."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.opensense import pws_qc

N_STATIONS, N_STEPS = 10, 4 * 288          # four days at 5 min
STUCK, SPIKE, RATE = "pws_2", "pws_5", "pws_7"


@pytest.fixture(scope="module")
def network():
    rng = np.random.default_rng(0)
    time = pd.date_range("2021-07-01", periods=N_STEPS, freq="5min")
    rain = np.zeros(N_STEPS)
    for start in (200, 600, 1000):                        # three showers of 4 h
        rain[start:start + 48] = rng.gamma(2.0, 0.3, 48)
    amount = np.repeat(rain[None, :], N_STATIONS, axis=0) * rng.uniform(0.8, 1.2, (N_STATIONS, 1))
    ids = [f"pws_{i}" for i in range(N_STATIONS)]
    amount[ids.index(STUCK), 550:] = 0.0                   # a clogged funnel
    amount[ids.index(SPIKE), 400] = 80.0                   # a counter reset, in dry weather
    rate = amount * 12.0
    rate[ids.index(RATE), 450:453] = 150.0                 # a stuck rate, empty bucket
    return xr.Dataset(
        {"rainfall_amount": (("id", "time"), amount), "rainfall_rate": (("id", "time"), rate)},
        coords={"id": ids, "time": time,
                "x": ("id", 1000.0 * np.arange(N_STATIONS) % 4000),
                "y": ("id", 1000.0 * (np.arange(N_STATIONS) // 4))})


@pytest.fixture(scope="module")
def qc(network):
    return pws_qc.flag(network, so_evaluation_period=288, so_mmatch=20)


def test_each_fault_is_caught_by_its_filter(qc):
    t = pws_qc.summary(qc)
    assert t.loc[STUCK, "fz"] > 0.05 and (t.drop(STUCK).fz == 0).all()
    assert qc.hi_flag.sel(id=SPIKE).isel(time=400) == 1
    assert (qc.rate_flag.sel(id=RATE).isel(time=slice(450, 453)) == 1).all()
    assert (t.drop(RATE).rate == 0).all()


def test_flagged_steps_leave_rainfall_qc_and_healthy_stations_are_untouched(qc):
    assert np.isnan(qc.rainfall_qc.sel(id=SPIKE).isel(time=400))
    healthy = [i for i in qc.id.values if i not in (STUCK, SPIKE, RATE)]
    np.testing.assert_allclose(qc.rainfall_qc.sel(id=healthy), qc.rainfall.sel(id=healthy))
    usable = pws_qc.usable(qc)
    assert STUCK not in usable and set(healthy) <= set(usable)


def test_regular_axis_and_the_amsterdam_variable_name(network):
    jittered = network.assign_coords(time=network.time + pd.to_timedelta(
        np.random.default_rng(1).integers(0, 60, N_STEPS), unit="s"))
    ds = pws_qc.regularize(jittered.rename(rainfall_amount="rainfall").drop_vars("rainfall_rate"))
    assert ds.sizes["time"] == N_STEPS and "rate_max" not in ds
    np.testing.assert_allclose(ds.rainfall.sum("time"), network.rainfall_amount.sum("time"))


def test_so_is_not_evaluated_on_a_record_shorter_than_its_window(network):
    short = pws_qc.flag(network.isel(time=slice(0, 300)))       # default window: 28 days
    assert (short.so_flag == -1).all()


def test_an_isolated_station_is_not_evaluated_rather_than_breaking(network):
    far = network.assign_coords(x=network.x.where(network.id != "pws_9", 1e6))
    qc = pws_qc.flag(far, so_evaluation_period=288, so_mmatch=20)
    alone = qc.sel(id="pws_9")
    assert int(alone.nbrs_not_nan.max()) == 0
    assert (alone.fz_flag == -1).all() and (alone.hi_flag == -1).all()


def test_openmrg2_file_quirks_are_normalized():
    from core.opensense import openmrg2

    raw = xr.Dataset(
        {"rainfall": (("id", "time"), np.zeros((2, 3)))},
        coords={"id": ["0", "1"], "time": pd.date_range("2015-06-01", periods=3, freq="5min"),
                "elevation": ("elevation", [119.0, 73.0]),       # a dimension of its own
                "latitude": ("id", [57.70, 57.71]), "longitude": ("id", [11.97, 11.98])})
    ds = openmrg2._points(raw)
    assert "elevation" not in ds.dims and list(ds.elevation.values) == [119.0, 73.0]
    assert {"lon", "lat", "x", "y", "rainfall_amount", "R"} <= set(ds.variables)
    assert 600_000 < float(ds.x[0]) < 700_000                  # UTM 32N metres
