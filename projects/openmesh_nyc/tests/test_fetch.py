"""The fetch pipelines' offline pieces: ASOS summaries and resampling, WU reshaping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fetch import asos_functions as asos, wu_functions as wu  # noqa: E402


def _minute(precip, station="JFK"):
    t = pd.date_range("2024-01-01", periods=len(precip), freq="1min")
    return pd.DataFrame({"datetime": t, "station_id": station,
                         "precip_mm": precip, "temp_c": 2.0, "wind_speed_ms": 5.0})


def test_station_summary_counts_totals_and_wet_minutes():
    precip = np.zeros(60)
    precip[10:20] = 0.1
    s = asos.station_summary({"JFK": _minute(precip)})
    assert s.loc["JFK", "precip_mm"] == pytest.approx(1.0)
    assert s.loc["JFK", "wet_rows"] == 10
    assert s.loc["JFK", "rows"] == 60


@pytest.mark.parametrize("interval", ["5min", "15min", "1h"])
def test_resampling_conserves_precipitation(interval):
    rng = np.random.default_rng(0)
    df = _minute(rng.exponential(0.05, 180) * (rng.random(180) < 0.3))
    out = asos.resample_data(df, interval)
    assert out["precip_mm"].sum() == pytest.approx(df["precip_mm"].sum())


def test_station_frames_has_the_shape_the_plotters_take():
    clean = pd.DataFrame({"precip_rate": [0.0, 1.0]})
    results = {"dataframes": {"S1": {"raw": clean, "clean": clean}},
               "metadata": {"S1": pd.DataFrame()}}
    frames = wu.station_frames(results)
    assert set(frames["S1"]) == {"raw", "clean", "metadata"}


def test_api_key_explains_how_to_set_it(monkeypatch):
    monkeypatch.delenv("WU_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WU_API_KEY"):
        wu.api_key()
