"""Paths and the study design."""

from __future__ import annotations

from pathlib import Path

from core.opensense.fetch import DATA_ROOT

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
EVENT_INPUTS = DATA_ROOT / "_multisensor_maps" / "merging" / "inputs"   # built by maps/multisensor

LENGTH_BINS_KM = (0.0, 1.0, 2.0, 4.0, 8.0, 100.0)
IDW = {"power": 2.0, "radius_m": 10_000.0}           # as maps/multisensor
RETRIEVALS = ("rnn", "dynamic")
GAUGES = {"openmrg": "city", "openrainer": "gauges", "openmesh": "asos"}  # held out: never mapped
WET_MM = 0.1
MIN_SHORT_KM = 1.0                                   # the "drop short links" variant
