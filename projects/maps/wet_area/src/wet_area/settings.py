"""Paths and the study design."""

from __future__ import annotations

from pathlib import Path

from core.opensense.fetch import DATA_ROOT

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
# the per-event inputs of projects/maps/multisensor (links of every retrieval, gauges, radar)
EVENT_INPUTS = DATA_ROOT / "_multisensor_maps" / "merging" / "inputs"

WET_MM = 0.1                       # hourly total (mm) that counts as wet, maps and scores
P_CUTS = (0.3, 0.5, 0.7)           # wet-probability cuts tried; 0.5 is the default
IDW = {"power": 2.0, "radius_m": 10_000.0}      # as maps/multisensor
RETRIEVALS = ("rnn", "dynamic")    # the best link retrieval and the best power law there
GAUGES = {"openmrg": "city", "openrainer": "gauges", "openmesh": "asos"}   # held out: never mapped
N_SYNTHETIC = 16                   # random scenarios (core.simulation.scenario.random_scenario)
