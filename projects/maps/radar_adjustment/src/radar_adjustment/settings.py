"""Paths and the study design."""

from __future__ import annotations

from pathlib import Path

from core.data_paths import OUTPUTS

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = OUTPUTS / "_radar_adjustment"            # prepared inputs and adjusted fields
PREPARED = DATA_DIR / "prepared"
FIELDS = DATA_DIR / "fields"
# the mergeplg commit the OpenSense repository pins as a submodule, installed here with
# ``pip install --no-deps --target`` (see README)
MERGEPLG_PINNED = DATA_DIR / "mergeplg_9894b9c"

# The intercomparison's periods: three summer months of each network.
MONTHS = {
    "openmrg": ("2015-06", "2015-07", "2015-08"),
    "openrainer": ("2022-06", "2022-07", "2022-08"),
}
RAD_FREQ = {"openmrg": 5, "openrainer": 15}          # radar step (min), for wet/dry

# Method settings shared by every variant (3_adjust_radar / 3b_adjust_radar).
NNEAR = 12
VARIOGRAM = {"sill": 1.0, "range": 30000, "nugget": 0.3}

# Range checks: the published run ("default", 3_adjust_radar), the conservative run
# ("cc") and none ("nc"), both from 3b_adjust_radar.
CHECKS = {
    "default": {"diff_check": 10, "ratio_check": (0.1, 15)},
    "cc": {"diff_check": 5, "ratio_check": (0.2, 8)},
    "nc": {},
}

# The eight adjusted products: key -> (class, method, full_line).
# Additive variants and KED take the difference check, multiplicative ones the ratio check.
VARIANTS = {
    "add_p_idw": ("MergeDifferenceIDW", "additive", None),
    "add_p_ok": ("MergeDifferenceOrdinaryKriging", "additive", False),
    "add_b_ok": ("MergeDifferenceOrdinaryKriging", "additive", True),
    "ked_p": ("MergeKrigingExternalDrift", None, False),
    "ked_b": ("MergeKrigingExternalDrift", None, True),
    "mul_p_idw": ("MergeDifferenceIDW", "multiplicative", None),
    "mul_p_ok": ("MergeDifferenceOrdinaryKriging", "multiplicative", False),
    "mul_b_ok": ("MergeDifferenceOrdinaryKriging", "multiplicative", True),
}
LABELS = {
    "radar": "radar", "add_p_idw": "add. IDW", "add_p_ok": "add. point OK",
    "add_b_ok": "add. block OK", "ked_p": "point KED", "ked_b": "block KED",
    "mul_p_idw": "mul. IDW", "mul_p_ok": "mul. point OK", "mul_b_ok": "mul. block OK",
}

# Scoring (4_analysis): threshold, intensity classes (mm/h, lower bound inclusive).
THRESHOLD = 0.2
CLASSES = ((0.2, 2.5), (2.5, 10), (10, 50), (50, 1000))
DISTANCE_BANDS_KM = (0, 2, 5, 10, 20, 1000)
