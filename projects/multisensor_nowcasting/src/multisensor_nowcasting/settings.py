"""Paths and the study design: networks, grids, splits, products, methods, models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.geo import Domain
from core.opensense.fetch import DATA_ROOT

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = DATA_ROOT / "_multisensor_nowcasting"        # cubes, forecasts, models; not tracked
RNN_DIR = DATA_ROOT / "_cml_rnn"                          # projects/cml_rnn: features and models
RNN_MODEL = "gru_physics_nbr"                             # its best model (projects/cml_rnn README)

STEP_MIN = 5                     # every network on a 5-minute step
N_LEADS = 12                     # 5 ... 60 min
N_HISTORY = 6                    # 30 min of past fields per forecast
PIXEL_KM = 2.0


@dataclass(frozen=True)
class NetworkSetup:
    key: str
    domain: Domain               # nowcasting domain (radar coverage, sensors well inside)
    period: tuple                # (start, end) of the record used
    gauges: tuple                # independent reference point sets, never an input
    label: str


NETWORKS = {
    # all of the SMHI composite (as projects/os_nowcasting): 44 x 35 pixels of 2 km
    "openmrg": NetworkSetup("openmrg", Domain(57.24, 58.03, 11.44, 12.63, "msn_openmrg"),
                            ("2015-06-01 00:00", "2015-08-31 23:55"), ("city", "smhi"),
                            "Gothenburg (OpenMRG + OpenMRG2), JJA 2015"),
    # the 240 x 240 km MRMS box of core.radar.mrms_nyc_wide: 120 x 120 pixels of 2 km
    "openmesh": NetworkSetup("openmesh", Domain(39.6382, 41.7941, -75.3823, -72.5378, "msn_openmesh"),
                             ("2023-11-01 00:00", "2024-07-01 00:00"), ("asos",),
                             "New York City (OpenMesh), Nov 2023 - Jun 2024, rain events"),
}

# Products: what each sensor combination makes of the rain field, every 5 minutes.
#   R      radar
#   C      links (line IDW of 5-min link rain)        P   PWS (IDW)
#   CP     links and PWS in one IDW
#   RC/RP/RCP  radar adjusted with links / PWS / both (additive residual IDW, core.maps.merge)
PRODUCTS = ("R", "C", "P", "CP", "RC", "RP", "RCP")
SENSOR_ONLY = ("C", "P", "CP")
LINE_IDW = {"power": 2.0, "radius_m": 15_000.0, "per_km": 1.0}
PWS_IDW = {"power": 2.0, "radius_m": 15_000.0}
MERGE_IDW = {"power": 2.0, "radius_m": 15_000.0, "nnear": 12}

# Physics nowcasts (pysteps, core.nowcast.methods)
PHYSICS_METHODS = ("persistence", "extrapolation", "sprog")
ENSEMBLE_PRODUCTS = ("R", "RCP", "CP")     # STEPS on these only (cost)
N_MEMBERS = 10

# Issue times: every ISSUE_EVERY in the test split, when > MIN_WET_FRACTION of the radar
# domain has > 0.1 mm/h.
ISSUE_EVERY = "30min"
MIN_WET_FRACTION = 0.05

# Learning
INPUT_SETS = {                      # channel ablation; each sensor channel comes with its mask
    "R": ("R",), "C": ("C",), "P": ("P",), "CP": ("C", "P"),
    "RC": ("R", "C"), "RP": ("R", "P"), "RCP": ("R", "C", "P"),
}
TARGETS = ("R", "RCP")              # future radar; future radar adjusted with links and PWS
SEED = 20151

# Verification
CAT_THRESHOLDS = (0.5, 1.0, 5.0)
FSS_SCALES_KM = (2, 10, 30)
HEADLINE_LEADS = (15, 30, 60)
N_BOOT = 1000
