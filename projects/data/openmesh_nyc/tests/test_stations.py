"""Per-station rain summaries on synthetic point data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stations  # noqa: E402


def _points(rates, step="5min"):
    t = pd.date_range("2024-01-12", periods=len(rates), freq=step).as_unit("ns")
    r = np.asarray(rates, dtype=float)
    return xr.Dataset({"R": (("time", "id"), r)},
                      coords={"time": t, "id": [f"s{i}" for i in range(r.shape[1])]})


def test_totals_integrate_rate_over_time():
    # 12 samples of 6 mm/h at 5 min = 6 mm; a dead station; one half missing
    rates = np.zeros((24, 3))
    rates[:12, 0] = 6.0
    rates[:, 2] = np.nan
    rates[12:, 2] = 2.0
    t = stations.totals(_points(rates))
    assert t.loc["s0", "total_mm"] == pytest.approx(6.0)
    assert t.loc["s1", "total_mm"] == 0.0
    assert t.loc["s2", "missing"] == pytest.approx(0.5)
    assert t.loc["s0", "wet_hours"] == pytest.approx(1.0)
    assert list(t.index) == ["s0", "s2", "s1"]            # sorted by total


def test_reporting_gaps_do_not_count_as_rain_time():
    ds = _points(np.full((4, 1), 6.0))
    gap = ds.time.values.copy()
    gap[3] = gap[2] + np.timedelta64(6, "h")               # a 6 h silence
    ds = ds.assign_coords(time=gap)
    acc = stations.accumulation(ds).isel(id=0).values
    assert acc[-1] < 6.0 * 1.0                             # not 6 h of 6 mm/h
