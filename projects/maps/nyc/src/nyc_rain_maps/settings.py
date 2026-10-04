"""Where this project reads and writes.

``events/``    the event catalog (tracked): every detected event and its classification
``results/``   generated summaries (tracked): link selection, the study, all events, validation
``DATA_DIR``   intermediate files (not tracked), under ``dataset/open_datasets/OpenMesh_NYC/``

Inputs are read from the shared data store ``~/data/cml`` (or ``data/interim/``), see
``core/data_paths.py`` and ``DATA.md``: MRMS crops, ASOS/METAR reports and the OpenMesh
record all live under ``~/data/cml/openmesh/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.opensense.fetch import DATA_ROOT

VERSION = "1.0"

PROJECT_DIR = Path(__file__).resolve().parents[2]
EVENTS_DIR = PROJECT_DIR / "events"
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = DATA_ROOT / "OpenMesh_NYC" / "nyc_rain_maps"


def repo_events_dir() -> Path:
    return EVENTS_DIR


def repo_reports_dir() -> Path:
    return RESULTS_DIR


@dataclass(frozen=True)
class _Paths:
    root: Path

    @property
    def events(self) -> Path:
        return self.root / "events"


def get_paths() -> _Paths:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _Paths(DATA_DIR)
