"""
OpenMRG2 preview: Gothenburg's personal weather stations for the OpenMRG period.

OpenMRG (Andersson et al. 2022) has links, radar and 11 gauges for
Gothenburg, June-August 2015. OpenMRG2 adds 30 Netatmo PWS for the same
months (SMHI, CC-BY-4.0), distributed from Google Drive until its Zenodo
record exists; ``python -m core.opensense.fetch --dataset openmrg2_pws``
downloads and verifies them.

    from core.opensense import openmrg2
    d = openmrg2.load()          # {"pws", "gauge_city", "gauge_smhi"}

Normalized like ``example_data``: ``lon``/``lat``, projected ``x``/``y``
(EPSG:32632, the OpenMRG zone), per-step ``rainfall_amount`` in mm and
``R`` in mm/h. The PWS file stores ``elevation`` as a dimension of its own;
it becomes a coordinate on ``id``.
"""

from __future__ import annotations

import xarray as xr

from core.opensense import example_data
from core import data_paths as dp

CRS = "EPSG:32632"
RAW = dp.data_path(dp.OPENMRG2)     # ~/data/cml (or data/interim): see core/data_paths.py


def _points(ds: xr.Dataset) -> xr.Dataset:
    ren = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon"),
                             ("rainfall", "rainfall_amount")) if k in ds.variables}
    ds = ds.rename(ren)
    if "elevation" in ds.dims:
        elev = ds.elevation.values
        ds = ds.drop_vars("elevation").assign_coords(elevation=("id", elev))
    ds["rainfall_amount"].attrs.update(units="mm", long_name="rainfall amount per time step")
    return example_data._normalize_points(ds, CRS)


def load(fetch_missing: bool = True) -> dict:
    """PWS, city gauges and the SMHI gauge, normalized; fetched if missing."""
    files = {"pws": "OpenMRGplus_rain.nc", "gauge_city": "city_gauges.nc",
             "gauge_smhi": "smhi_gauges.nc"}
    if fetch_missing and not all((RAW / f).exists() for f in files.values()):
        from core.opensense.fetch import fetch
        fetch("openmrg2_pws")
    out = {}
    for key, name in files.items():
        with xr.open_dataset(RAW / name) as ds:
            out[key] = _points(ds.load())
    return out
