"""Paths and the study design: which period of each network, which methods."""

from __future__ import annotations

from pathlib import Path

from core.opensense.fetch import DATA_ROOT

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = DATA_ROOT / "_multisensor_maps"          # intermediate files, not tracked

# Period searched for events, per network. OpenRainER: two warm seasons of the two-year
# record (its radar months are ~1 GB each unpacked); the others: the whole record.
PERIODS = {
    "openmrg": ("2015-06-01", "2015-09-01"),
    "openrainer": ("2021-05-01", "2021-12-01"),
    "openmesh": ("2023-11-01", "2024-07-01"),
}
OPENRAINER_SECOND_SEASON = ("2022-05-01", "2022-12-01")

# Events per network in the study: the largest by domain-mean radar total.
N_EVENTS = 10

# Point sets used in maps, and the one used for the leave-one-out point check.
MAP_POINTS = {"openmrg": ("pws", "city"), "openrainer": ("gauges",), "openmesh": ("pws",)}
CHECK_POINTS = {"openmrg": "city", "openrainer": "gauges", "openmesh": "pws"}

METHODS = ("dynamic", "constant", "pycomlink", "nearby")
# every link retrieval is mapped three ways (see event.interpolate)
INTERPOLATORS = ("idw", "line", "gmz")
# the RNN trained in projects/retrieval/rnn_three_networks, used as one more retrieval when present
RNN_MODEL = DATA_ROOT / "_cml_rnn" / "models" / "gru_physics_nbr"
