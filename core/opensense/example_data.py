"""
Curated OpenSense example subsets - the fast tier of data acquisition.

Two ways into the same datasets live in this project:

``fetch.py``
    the full published records from Zenodo. OpenMRG is a 318 MB zip holding a
    4.6 GB ``cml.nc``; OpenRainER is 4.6 GB of tars. Weeks to months of data,
    raw instrument units, and the retrieval chain has to run.
``example_data.py`` (this module)
    the small curated subsets the OpenSense community publishes at
    ``OpenSenseAction/opensense_example_data``. Days rather than months,
    already on the OpenSense-1.0 convention, 0.01-27 MB per file. Enough to
    exercise the whole pipeline in seconds.

Ported from ``poligrain.example_data``, with four changes:

* **Every subset is exposed.** ``poligrain.load_openmesh`` hardcodes ``20d``
  and documents it as the only option, but ``1d`` and ``1w`` also exist. They
  are CML + PWS only - ASOS is published for ``20d`` alone - which is why they
  were awkward to expose from a function returning a fixed 3-tuple. Here a
  subset simply declares which components it has.
* **Verified downloads.** ``poligrain`` uses a bare ``urlretrieve`` with no
  timeout, no resume and no size check; a truncated file is indistinguishable
  from a good one. This reuses the resumable, size-checked downloader in
  ``fetch.py``.
* **Named returns.** A dict keyed by component, not a positional tuple whose
  order differs per dataset (``load_openmrg`` returns radar first, and
  ``load_openmesh`` returns CML first).
* **Normalized on load.** Units and polarization spellings differ between
  sources and are sometimes undeclared - see ``conventions.py``.

    python -m core.opensense.example_data --list
    python -m core.opensense.example_data --dataset openmrg --subset 8d
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import xarray as xr

from core.opensense import conventions as cv
from core.opensense.fetch import download_url

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://github.com/OpenSenseAction/opensense_example_data"
VERSION = "main"

# The subsets live in ~/data/cml (or data/interim), one folder per dataset:
# see core/data_paths.py SAMPLES. Derived files (CNN masks) go to OUTPUTS.
from core import data_paths as dp  # noqa: E402
CACHE = dp.OUTPUTS / "_example_subsets"      # derived files only (e.g. _masks/)


def sample_dir(key: str) -> Path:
    """Folder holding one dataset's example subsets, e.g. ``sample_dir("openmrg")``."""
    return dp.data_path(dp.SAMPLES[DATASETS[key].folder])


@dataclass(frozen=True)
class ExampleDataset:
    """One dataset in the example-data repository."""

    key: str
    name: str
    folder: str                # folder in the example-data repo
    crs: str                   # projected CRS appropriate to the region
    # subset -> {component: filename template}
    subsets: dict = field(default_factory=dict)
    note: str = ""
    # Which edge of its interval an accumulation's timestamp names: "start"
    # or "end". Not declared in the files - established by lagging the 1-min
    # CML signals against each reference; see evaluation.aggregate.
    accumulation_label: str = "start"

    def files(self, subset: str) -> dict:
        if subset not in self.subsets:
            raise SystemExit(
                f"{self.name}: unknown subset {subset!r}. "
                f"available: {sorted(self.subsets)}")
        return self.subsets[subset]


DATASETS = {
    "openmrg": ExampleDataset(
        key="openmrg", name="OpenMRG", folder="OpenMRG", crs="EPSG:32632",
        note="Gothenburg, Sweden. The 8d subset carries a reference "
             "retrieval 'R' alongside raw tsl/rsl.",
        subsets={
            "8d": {"cml": "openmrg_cml_8d.nc",
                   "radar": "openmrg_rad_8d.nc",
                   "gauge_municipal": "openmrg_municp_gauge_8d.nc",
                   "gauge_smhi": "openmrg_smhi_gauge_8d.nc"},
            "5min_2h": {"cml": "openmrg_cml_5min_2h.nc",
                        "radar": "openmrg_rad_5min_2h.nc",
                        "gauge_municipal": "openmrg_municp_gauge_5min_2h.nc",
                        "gauge_smhi": "openmrg_smhi_gauge_5min_2h.nc"},
        }),
    "openrainer": ExampleDataset(
        key="openrainer", name="OpenRainER", folder="OpenRainER",
        crs="EPSG:32632", accumulation_label="end",
        note="Emilia-Romagna, Italy. Gauge and radar 15-min accumulations "
             "are stamped at the END of their interval.",
        subsets={
            "8d": {"cml": "openrainer_cml_8d.nc",
                   "radar": "openrainer_radar_8d.nc",
                   "gauge": "openrainer_gauges_8d.nc"},
        }),
    "openmesh": ExampleDataset(
        key="openmesh", name="OpenMesh", folder="OpenMesh", crs="EPSG:32618",
        note="New York City. RSL only - no tsl. Up to 3 sublinks per link, "
             "but in the 20d subset 51 of 75 links report one, 20 two (mostly "
             "both directions of one band) and 4 three; bands ~5-6, 24 and "
             "58-69 GHz. ASOS is published for 20d only.",
        subsets={
            "1d": {"cml": "openmesh_cml_1d.nc",
                   "pws": "openmesh_wu_pws_1d.nc"},
            "1w": {"cml": "openmesh_cml_1w.nc",
                   "pws": "openmesh_wu_pws_1w.nc"},
            "20d": {"cml": "openmesh_cml_20d.nc",
                    "pws": "openmesh_wu_pws_20d.nc",
                    "asos": "openmesh_asos_ws_20d.nc"},
        }),
    "ams_pws": ExampleDataset(
        key="ams_pws", name="Amsterdam PWS", folder="AMS_PWS",
        crs="EPSG:32631", note="Amsterdam personal weather stations, 25 months.",
        subsets={
            "full_period": {"pws": "ams_pws_full_period.nc",
                            "gauge": "ams_gauges_full_period.nc"},
        }),
}


# --------------------------------------------------------------------------
def download(dataset: str, subset: str, cache_dir: Path | None = None,
             force: bool = False, verbose: bool = True) -> dict:
    """Fetch every component of one subset. Returns {component: local path}."""
    spec = DATASETS[dataset]
    files = spec.files(subset)
    dest = Path(cache_dir) / spec.folder if cache_dir else sample_dir(dataset)
    dest.mkdir(parents=True, exist_ok=True)

    out = {}
    for component, filename in files.items():
        url = f"{BASE_URL}/raw/{VERSION}/{spec.folder}/{filename}"
        out[component] = download_url(url, dest / filename, force=force,
                                      verbose=verbose)
    return out


def load(dataset: str, subset: str | None = None,
         cache_dir: Path | None = None, normalize: bool = True,
         time: slice | None = None, components=None,
         verbose: bool = True) -> dict:
    """Download if needed, then open one subset.

    Returns a dict keyed by component - ``cml``, ``radar``, ``gauge``,
    ``pws``, ``asos`` - so callers do not depend on a positional order that
    differs per dataset.

    With ``normalize`` (the default), CML datasets gain ``length_km``,
    ``frequency_ghz`` and a normalized ``polarization``, plus projected
    endpoint coordinates; point datasets gain projected ``x``/``y``; radar
    gains ``x_grid``/``y_grid`` (see ``conventions.project_grid``).

    ``time`` selects a window before anything is read into memory, which
    matters for the OpenMRG ``8d`` CML file: 1.2 GB once loaded, but a
    single day of it is 150 MB. ``components`` restricts which files are
    opened, e.g. ``components=("cml",)``.

    >>> data = load("openmrg", "8d", time=slice("2015-07-28", "2015-07-28"))
    >>> sorted(data)
    ['cml', 'gauge_municipal', 'gauge_smhi', 'radar']
    """
    spec = DATASETS[dataset]
    subset = subset or next(iter(spec.subsets))
    paths = download(dataset, subset, cache_dir, verbose=verbose)

    # load_dataset, not open_dataset: these are small files and callers open
    # several in one process, which exhausts netCDF4's handle cache and fails
    # with "NetCDF: HDF error" rather than anything that names the cause.
    if components is not None:
        unknown = set(components) - set(paths)
        if unknown:
            raise ValueError(f"{spec.name}/{subset} has no {sorted(unknown)}; "
                             f"available: {sorted(paths)}")
        paths = {c: p for c, p in paths.items() if c in components}
    out = {c: _open(p, time) for c, p in paths.items()}
    if not normalize:
        return out

    if "cml" in out:
        out["cml"] = normalize_cml(out["cml"], spec.crs)
    if "radar" in out:
        out["radar"] = normalize_radar(out["radar"], spec.crs)
    for component in ("gauge", "gauge_municipal", "gauge_smhi", "pws", "asos"):
        if component in out:
            out[component] = _normalize_points(out[component], spec.crs)
    return out


def _open(path: Path, time: slice | None) -> xr.Dataset:
    """Read one file fully into memory, optionally only a time window."""
    if time is None:
        return xr.load_dataset(path)
    with xr.open_dataset(path) as ds:
        return ds.sel(time=time).load()


def normalize_radar(ds: xr.Dataset, crs: str) -> xr.Dataset:
    """Radar on a projected grid, with the rain variable named ``R``.

    Both subsets call the variable ``R``, but only OpenMRG's is a rate
    (``mm/h``). OpenRainER's ``R`` is a **15-minute accumulation** declared
    ``units: mm`` - read as a rate, it is four times too low, silently.
    Accumulations are converted to mm/h from the declared ``accum_time_h``,
    or from the time step when that is absent.
    """
    from core.opensense.retrieval import sampling_interval_s

    if "R" not in ds.data_vars and "rainfall_amount" in ds.data_vars:
        ds = ds.rename({"rainfall_amount": "R"})
    units = str(ds.R.attrs.get("units", "")).strip().lower()
    # OpenMRG's 5min_2h radar declares its accumulation as "sum 5min"
    summed = re.fullmatch(r"sum\s*(\d+)\s*min", units)
    if summed:
        ds.R.attrs["accum_time_h"] = int(summed.group(1)) / 60.0
        units = "mm"
    if units == "mm":
        hours = float(ds.R.attrs.get("accum_time_h",
                                     sampling_interval_s(ds.time) / 3600.0))
        attrs = dict(ds.R.attrs)
        ds["R"] = ds.R / hours
        attrs.pop("valid_max", None)
        ds.R.attrs.update(attrs, units="mm h-1", long_name="rain rate",
                          standard_name="rainfall_rate",
                          comment=f"converted from {hours:g} h accumulation in mm")
    elif units not in ("mm/h", "mm h-1", "mm hr-1", ""):
        raise ValueError(f"unexpected radar units {units!r}")
    lon = next(c for c in ("lon", "longitude", "longitudes") if c in ds.variables)
    lat = next(c for c in ("lat", "latitude", "latitudes") if c in ds.variables)
    return cv.project_grid(ds, crs, lon=lon, lat=lat)


def normalize_cml(ds: xr.Dataset, crs: str) -> xr.Dataset:
    """Attach km / GHz / canonical polarization and projected geometry."""
    if "length" in ds.coords or "length" in ds.data_vars:
        ds.coords["length_km"] = (ds.length.dims, cv.to_km(ds.length))
    if "frequency" in ds.coords or "frequency" in ds.data_vars:
        ds.coords["frequency_ghz"] = (ds.frequency.dims, cv.to_ghz(ds.frequency))
    if "polarization" in ds.coords or "polarization" in ds.data_vars:
        pol = cv.normalize_polarization(ds.polarization.values)
        ds.coords["polarization"] = (ds.polarization.dims,
                                     pol.reshape(ds.polarization.shape))
    if "site_0_lon" in ds.coords:
        ds = cv.project_cml(ds, crs)
    return ds


def _normalize_points(ds: xr.Dataset, crs: str) -> xr.Dataset:
    """Projected x/y, and ``R`` in mm/h next to per-step ``rainfall_amount``.

    Gauge and PWS files store the accumulation per time step (1 min for
    OpenMRG's municipal gauges, 15 min for OpenRainER's), so every caller
    used to convert by hand with its own hardcoded factor.

    OpenMRG's 5min_2h gauges name the station dimension ``station_id``; it is
    renamed to the ``id`` every other point file uses.
    """
    if "station_id" in ds.dims and "id" not in ds.dims:
        ds = ds.rename({"station_id": "id"})
    lon = next((c for c in ("lon", "longitude") if c in ds.coords), None)
    lat = next((c for c in ("lat", "latitude") if c in ds.coords), None)
    if lon and lat:
        ds = cv.project_points(ds, crs, lon=lon, lat=lat)
    if "rainfall_rate" in ds.data_vars and "R" not in ds.data_vars:
        # a declared rate beats one derived from per-step amounts, which
        # assumes a regular time step (PWS reporting is not)
        ds["R"] = ds.rainfall_rate.transpose("time", ...)
        ds.R.attrs.update(units="mm h-1", long_name="rain rate",
                          comment="rainfall_rate as published")
    if "rainfall_amount" in ds.data_vars and "R" not in ds.data_vars \
            and ds.sizes.get("time", 0) > 1:
        from core.opensense.retrieval import sampling_interval_s

        per_hour = 3600.0 / sampling_interval_s(ds.time)
        ds["R"] = (ds.rainfall_amount * per_hour).transpose("time", ...)
        ds.R.attrs.update(units="mm h-1", long_name="rain rate",
                          comment=f"rainfall_amount x {per_hour:g}")
    return ds


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    ap.add_argument("--subset", help="default: the first subset listed")
    ap.add_argument("--all", action="store_true",
                    help="fetch every subset of every dataset")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list:
        for spec in DATASETS.values():
            print(f"\n{spec.name}  ({spec.key})  {spec.crs}")
            if spec.note:
                print(f"  {spec.note}")
            for subset, files in spec.subsets.items():
                print(f"    {subset:12s} {', '.join(sorted(files))}")
        print(f"\ncached under {dp.SHARED}/<dataset>/_sample*/ (see core/data_paths.py)")
        return

    targets = ([(d, s) for d in DATASETS for s in DATASETS[d].subsets]
               if args.all else
               [(args.dataset, args.subset or next(iter(DATASETS[args.dataset].subsets)))]
               if args.dataset else None)
    if targets is None:
        ap.error("pass --dataset, --all, or --list")

    for dataset, subset in targets:
        spec = DATASETS[dataset]
        print(f"\n{spec.name} / {subset}")
        data = load(dataset, subset)
        for component, ds in sorted(data.items()):
            dims = dict(ds.sizes)
            print(f"  {component:16s} {dims}")


if __name__ == "__main__":
    main()
