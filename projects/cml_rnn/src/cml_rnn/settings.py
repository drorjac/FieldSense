"""Paths, periods and the train / validation / test split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.opensense.fetch import DATA_ROOT

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = DATA_ROOT / "_cml_rnn"                   # built datasets and models, not tracked

PERIODS = {
    "openmrg": [("2015-06-01", "2015-09-01")],
    "openrainer": [("2021-05-01", "2021-12-01"), ("2022-05-01", "2022-12-01")],
    "openmesh": [("2023-11-01", "2024-07-01")],
}
CHUNK = "10D"                    # links are loaded and processed this much at a time
SPINUP = "1D"                    # history loaded before each chunk (baselines, the RNN's context)

# Point sets averaged with the radar into the training target.
TARGET_POINTS = {"openmrg": ("pws", "city"), "openrainer": ("gauges",), "openmesh": ("pws",)}
GAUGE_RADIUS_KM = 3.0

PL_METHODS = ("dynamic", "constant", "pycomlink", "nearby")


def split(times: pd.DatetimeIndex, test_windows=()) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test, per hour.

    Weeks (ISO) cycle train, train, validation, test, so every season is in every split;
    ``test_windows`` (the multisensor_maps study events, +-1 day) are always test, so
    the RNN's maps can join that comparison without having seen those storms.
    """
    week = np.asarray(times.isocalendar().week, dtype=int)
    out = np.select([week % 4 == 2, week % 4 == 3], [1, 2], 0)
    for s, e in test_windows:
        sel = (times > pd.Timestamp(s) - pd.Timedelta("1D")) & (times <= pd.Timestamp(e) + pd.Timedelta("1D"))
        out[sel] = 2
    return out
