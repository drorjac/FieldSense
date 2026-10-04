"""Paths and the designs of the studies."""

from __future__ import annotations

from pathlib import Path

from core.opensense.fetch import DATA_ROOT

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_DIR / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELS = RESULTS / "models"          # trained networks and their simulated training data

# The real radar of os_nowcasting, read from the event windows it caches (native time step,
# its 2 km nowcasting grid); this project reads the files, it does not import that project.
REAL_RADAR_CACHE = DATA_ROOT / "_os_nowcasting"
REAL_STEP_MIN = {"openmrg": 5, "openrainer": 15}

GALLERY = ["convective_cells", "clustered_storms", "squall_line", "stratiform_matern",
           "banded_anisotropic", "scale_free", "rainfarm", "cascade", "multifractal", "frontal"]
CELL_MODELS = ("convective_cells", "clustered_storms", "squall_line")

# regional study: 64 km domain, 0.5 km / 1 min truth, 1 km / 5 min products
MOTION_PRODUCTS = ("truth", "radar", "idw links", "ked [links+gauges]")
NOWCAST_PRODUCTS = ("truth", "radar", "ked [links+gauges]")

# city study: 25.6 km domain, 0.1 km / 1 min truth, 0.2 km analysis grid
CITY_N, CITY_DX_KM, CITY_FACTOR = 256, 0.1, 2
CITY_DURATION_MIN = 120
LINK_BANDS_KM = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9)


def ensure_dirs():
    for d in (FIGURES, TABLES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
