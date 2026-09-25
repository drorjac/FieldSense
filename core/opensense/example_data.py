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
from dataclasses import dataclass, field
from pathlib import Path

import xarray as xr

from core.opensense import conventions as cv
from core.opensense.fetch import download_url

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://github.com/OpenSenseAction/opensense_example_data"
VERSION = "main"

CACHE = REPO_ROOT / "dataset" / "open_datasets" / "_example_subsets"


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
        crs="EPSG:32632", note="Emilia-Romagna, Italy.",
        subsets={
            "8d": {"cml": "openrainer_cml_8d.nc",
                   "radar": "openrainer_radar_8d.nc",
                   "gauge": "openrainer_gauges_8d.nc"},
        }),
    "openmesh": ExampleDataset(
        key="openmesh", name="OpenMesh", folder="OpenMesh", crs="EPSG:32618",
        note="New York City. RSL only - no tsl - and 3 sublinks per link "
             "across three bands. ASOS is published for 20d only.",
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
             force: bool = False) -> dict:
    """Fetch every component of one subset. Returns {component: local path}."""
    spec = DATASETS[dataset]
    files = spec.files(subset)
    dest = Path(cache_dir or CACHE) / spec.folder
    dest.mkdir(parents=True, exist_ok=True)

    out = {}
    for component, filename in files.items():
        url = f"{BASE_URL}/raw/{VERSION}/{spec.folder}/{filename}"
        out[component] = download_url(url, dest / filename, force=force)
    return out


def load(dataset: str, subset: str | None = None,
         cache_dir: Path | None = None, normalize: bool = True) -> dict:
    """Download if needed, then open one subset.

    Returns a dict keyed by component - ``cml``, ``radar``, ``gauge``,
    ``pws``, ``asos`` - so callers do not depend on a positional order that
    differs per dataset.

    With ``normalize`` (the default), CML datasets gain ``length_km``,
    ``frequency_ghz`` and a normalized ``polarization``, plus projected
    endpoint coordinates; point datasets gain projected ``x``/``y``.
    """
    spec = DATASETS[dataset]
    subset = subset or next(iter(spec.subsets))
    paths = download(dataset, subset, cache_dir)

    # load_dataset, not open_dataset: these are small files and callers open
    # several in one process, which exhausts netCDF4's handle cache and fails
    # with "NetCDF: HDF error" rather than anything that names the cause.
    out = {c: xr.load_dataset(p) for c, p in paths.items()}
    if not normalize:
        return out

    if "cml" in out:
        out["cml"] = normalize_cml(out["cml"], spec.crs)
    for component in ("gauge", "gauge_municipal", "gauge_smhi", "pws", "asos"):
        if component in out:
            out[component] = _normalize_points(out[component], spec.crs)
    return out


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
    lon = next((c for c in ("lon", "longitude") if c in ds.coords), None)
    lat = next((c for c in ("lat", "latitude") if c in ds.coords), None)
    if lon and lat:
        ds = cv.project_points(ds, crs, lon=lon, lat=lat)
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
        print(f"\ncached under {CACHE.relative_to(REPO_ROOT)}/")
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
