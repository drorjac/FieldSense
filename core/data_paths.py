"""Where FieldSense reads its input data - the one place an input path is written down.

Every input file is looked up in two roots, in this order:

1. ``INTERIM`` = ``<repo>/data/interim/`` - a local override. Gitignored and
   empty by default; drop a file here (same relative path) to use it instead
   of the shared copy.
2. ``SHARED``  = ``~/data/cml/`` - the machine's shared data store, one copy
   of each dataset used by several projects (override with ``$CML_DATA_ROOT``).

Paths below are relative to those roots, in the ``~/data/cml`` layout. What
each file is, and which code reads it, is listed in ``DATA.md``.

Downloads (Zenodo, MRMS, NEXRAD, IEM) are raw inputs too, so they are written
to ``SHARED``. Files FieldSense *produces* - ``processed/``, ``_cml_rnn/``,
model outputs - stay under ``OUTPUTS`` (``dataset/open_datasets/``).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM = REPO_ROOT / "data" / "interim"
SHARED = Path(os.environ.get("CML_DATA_ROOT", Path.home() / "data" / "cml")).expanduser()
OUTPUTS = REPO_ROOT / "dataset" / "open_datasets"


def data_path(rel: str | Path) -> Path:
    """An input file or folder: ``INTERIM/rel`` if present, else ``SHARED/rel``.

    Returns the ``SHARED`` path when neither exists, so a caller can download
    into it or raise a clear "not found" naming the canonical location.
    """
    local = INTERIM / rel
    return local if local.exists() else SHARED / rel


# --- OpenMRG (Gothenburg, JJA 2015; zenodo 7107689) --------------------------
OPENMRG_DOWNLOAD = "openmrg/_download"                    # OpenMRG.zip
OPENMRG_CML = "openmrg/cml/openmrg_cml_full.nc"           # archive cml/cml.nc
OPENMRG_CML_META = "openmrg/cml/openmrg_cml_metadata.csv"  # archive cml/cml_metadata.csv
OPENMRG_RADAR = "openmrg/weather/openmrg_radar_full.nc"   # archive radar/radar.nc
OPENMRG_GAUGES = "openmrg/weather/gauges"                 # archive gauges/{city,smhi}/
OPENMRG2 = "openmrg2/weather"                             # OpenMRG2 PWS preview (Drive)

# --- OpenRainER (Emilia-Romagna, 2021-2022; zenodo 22829808) -----------------
OPENRAINER_DOWNLOAD = "openrainer/_download"              # CML.tar, AWS.tar, RADrain.tar
#: monthly files, one folder per product (tar name -> extracted folder)
OPENRAINER_MONTHLY = {
    "CML": "openrainer/cml",
    "AWS": "openrainer/weather/aws",
    "RADrain": "openrainer/weather/rad_rain",
    "RADadj": "openrainer/weather/rad_adj",
    "RADref": "openrainer/weather/rad_ref",
}

# --- OpenMesh (NYC Mesh; zenodo 15287692 + 17508286 for the PWS) -------------
OPENMESH_RELEASE = "openmesh/release_v1"                  # ds_openmesh.nc, pws_wu_os.nc, metadata
OPENMESH_ZIPS = "openmesh/release_v1/zips"                # OpenMesh.zip, PWS_NYC_WU.zip
IEM_CACHE = "openmesh/weather/raw_fetch/iem_cache"        # ASOS 1-min + METAR pulls (IEM)
MRMS_CACHE = "openmesh/weather/radar/mrms_cache"          # MRMS QPE crops
NEXRAD_CACHE = "openmesh/weather/radar/nexrad_okx"        # NEXRAD OKX level-3 event files

# --- OpenSense example subsets (small demo extracts, never a result basis) ---
SAMPLES = {
    "OpenMRG": "openmrg/_sample_8d",
    "OpenMesh": "openmesh/_sample_20d",
    "OpenRainER": "openrainer/_sample_8d",
    "AMS_PWS": "ams_pws/_sample",
}
