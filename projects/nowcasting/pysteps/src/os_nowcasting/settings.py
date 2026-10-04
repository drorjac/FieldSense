"""Paths and the study design: networks, events, products, nowcasting methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.geo import Domain
from core.opensense.fetch import DATA_ROOT

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = DATA_ROOT / "_os_nowcasting"           # per-event fields, not tracked


@dataclass(frozen=True)
class NetworkSetup:
    """How one network is nowcast: the grid, the native step and the lead times."""

    key: str
    domain: Domain              # nowcasting domain (inside the radar's coverage)
    step_min: int               # native time step of the radar (and of every product)
    n_leadtimes: int            # lead times, in steps
    pixel_km: float = 2.0       # square pixels in a local equirectangular projection
    gauges: str = "gauges"      # point set used as the independent reference
    pws: str | None = None      # point set mapped as a PWS product, if any

    @property
    def max_lead_min(self) -> int:
        return self.step_min * self.n_leadtimes


NETWORKS = {
    # the network's own domain padded by 0.25 deg: room upstream of the links
    "openrainer": NetworkSetup("openrainer", Domain(43.50, 45.25, 9.00, 12.90, "os_nowcast_openrainer"),
                               step_min=15, n_leadtimes=6, gauges="gauges"),
    # as much of the SMHI composite as exists (57.21-58.06 N, 11.41-12.66 E)
    "openmrg": NetworkSetup("openmrg", Domain(57.24, 58.03, 11.44, 12.63, "os_nowcast_openmrg"),
                            step_min=5, n_leadtimes=18, gauges="city", pws="pws"),
}

# Events (start, end of the radar's wet period, hour-ending labels; core.events.detect_events
# on the hourly radar). OpenRainER: the training school's event (2022-08-17..19) and the next
# three largest of June-August 2022, all in August. OpenMRG: the five largest of JJA 2015
# (projects/maps/multisensor/results/events.csv).
EVENTS = {
    "openrainer": [
        ("20220817T10", "2022-08-17 10:00", "2022-08-19 16:00"),
        ("20220806T09", "2022-08-06 09:00", "2022-08-09 06:00"),
        ("20220830T11", "2022-08-30 11:00", "2022-08-31 23:00"),
        ("20220811T22", "2022-08-11 22:00", "2022-08-13 04:00"),
    ],
    "openmrg": [
        ("20150827T01", "2015-08-27 01:00", "2015-08-28 01:00"),
        ("20150825T02", "2015-08-25 02:00", "2015-08-25 23:00"),
        ("20150728T03", "2015-07-28 03:00", "2015-07-29 11:00"),
        ("20150707T17", "2015-07-07 17:00", "2015-07-09 09:00"),
        ("20150602T04", "2015-06-02 04:00", "2015-06-03 10:00"),
    ],
}

# Products nowcast on each network. ``cml_idw<d>``: link rain by IDW within d km (the
# session's ``rainfall_interpolateIDW_10/20/40``); ``merged``: radar adjusted with the links
# (mergeplg difference block kriging, core.maps.mergeplg_methods); ``pws_idw``: PWS by IDW.
CML_IDW_KM = (10, 20, 40)
PRODUCTS = {
    "openrainer": ("radar", "cml_idw10", "cml_idw20", "cml_idw40", "merged"),
    "openmrg": ("radar", "cml_idw10", "cml_idw20", "cml_idw40", "merged", "pws_idw"),
}

# Nowcasting methods (pysteps). Deterministic ones run on every product; LINDA is slow and
# runs on the radar only; STEPS is the ensemble.
DETERMINISTIC = ("persistence", "extrapolation", "sprog", "anvil")
LINDA_PRODUCTS = ("radar",)
ENSEMBLE = "steps"
N_ENS_MEMBERS = 12

# Issue times: every ISSUE_EVERY of the event at which the radar is wet over at least
# MIN_WET_FRACTION of the domain (> 0.1 mm/h); the first needs N_PAST steps of history.
ISSUE_EVERY = "1h"
MIN_WET_FRACTION = 0.05
from core.nowcast.methods import N_PAST  # noqa: E402,F401  {"LK": 4, "VET": 3, "DARTS": 8, "proesmans": 2}

# Transform (the session's): mm/h -> dBR with 0.1 mm/h threshold and -15 dBR fill.
from core.nowcast.grid import DB_THRESHOLD, DB_ZEROVALUE  # noqa: E402,F401  (0.1 mm/h, -15 dBR)

# Verification
from core.nowcast.verify import (CAT_THRESHOLDS, FSS_SCALES_KM, FSS_THRESHOLD,  # noqa: E402,F401
                                 PROB_THRESHOLD)
