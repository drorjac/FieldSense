# Data this project reads

FieldSense keeps **no input data in git and none in the repository folder**.
Every input file is looked up in two places, in this order
([`core/data_paths.py`](core/data_paths.py), the only file that writes a path down):

1. **`data/interim/`** — a local override, git-ignored and empty by default. Put a
   file here under the same relative path (e.g. `data/interim/openmrg/cml/openmrg_cml_full.nc`)
   to use it instead of the shared copy.
2. **`~/data/cml/`** — the shared data store on this machine: one copy of each
   dataset, used by several projects and documented in the
   `data` index project (`~/PycharmProjects/data`). Set `CML_DATA_ROOT` to
   point somewhere else.

Downloads (Zenodo, MRMS, NEXRAD, IEM) are inputs too, so they are written into
`~/data/cml`. What FieldSense *produces* — `processed/`, `_cml_rnn/`,
`_multisensor_maps/`, nowcast caches, the PyNNcml symlink view — stays under
`dataset/open_datasets/` (git-ignored).

On a new machine: clone, then either copy the files below into `~/data/cml/`
(same layout) or into `data/interim/`, or let the fetchers download them
(`python -m core.opensense.fetch --dataset <name>`).

## Files, by dataset

Paths are relative to `~/data/cml/` (or `data/interim/`).

### OpenMRG — Gothenburg, Sweden, Jun–Aug 2015
Commercial microwave links (CML), weather radar and rain gauges, published by SMHI.
Andersson et al. (2022), [doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689), CC BY-SA 4.0.

| file | what it is | read by |
|---|---|---|
| `openmrg/_download/OpenMRG.zip` | the Zenodo archive as published (318 MB) | `fetch --dataset openmrg` writes it; PyNNcml view |
| `openmrg/cml/openmrg_cml_full.nc` | archive `cml/cml.nc`: 364 links / 728 sublinks, 10-s TSL + RSL (dBm), 92 days | `networks.OpenMRG.links`, `ingest_openmrg.py`, PyNNcml |
| `openmrg/cml/openmrg_cml_metadata.csv` | archive `cml/cml_metadata.csv`: per-sublink sites, length (km), frequency (GHz), polarization | `networks.OpenMRG.links_table`, `ingest_openmrg.py`, PyNNcml |
| `openmrg/weather/openmrg_radar_full.nc` | archive `radar/radar.nc`: SMHI 5-min reflectivity composite (pseudo-dBZ, Z = 200 R^1.5) | `networks.OpenMRG` radar, `ingest_openmrg.py`, `spatial_interpolation` |
| `openmrg/weather/gauges/city/CityGauges-2015JJA.csv`, `CityGauges-metadata.csv` | 11 municipal gauges, 1-min rain (mm) + coordinates | `ingest_openmrg.py`, PyNNcml |
| `openmrg/weather/gauges/smhi/GbgA-71420-*.csv` | the SMHI Gothenburg-A gauge, 15-min | PyNNcml |
| `openmrg2/weather/OpenMRGplus_rain.nc` | OpenMRG2 preview: Netatmo PWS rain over Gothenburg, 5-min | `networks.OpenMRG.points`, `core.opensense.openmrg2` |
| `openmrg2/weather/city_gauges.nc`, `smhi_gauges.nc` | the same city (1-min) and SMHI (15-min) gauges as NetCDF | same |

OpenMRG2 is not on Zenodo yet; `fetch --dataset openmrg2_pws` downloads it from the
authors' Google Drive links, checksummed in `fetch.py`.

### OpenRainER — Emilia-Romagna, Italy, 2021–2022
CML (1-min TSL + RSL), automatic weather stations and radar. Covi & Roversi,
[doi:10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808), CC BY 4.0.

| file | what it is | read by |
|---|---|---|
| `openrainer/_download/CML.tar`, `AWS.tar`, `RADrain.tar` | the Zenodo archives (monthly `.nc.gz` inside) | `fetch --dataset openrainer` writes them; a missing month is unpacked from here |
| `openrainer/cml/CML_<YYYYMM>…nc` | one month of CML: 151 links × 2 sublinks, 1-min RSL + TSL | `networks.OpenRainER`, `ingest_openrainer.py` |
| `openrainer/weather/aws/AWS_<YYYYMM>.nc` | one month of rain gauges (~300 stations), 15-min accumulation, stamped at interval **end** | same |
| `openrainer/weather/rad_rain/RADrain_<YYYYMM>.nc` | one month of radar 15-min accumulated rain (mm), stamped at interval **end** | same (radar reference) |

Months are unpacked on demand into the folders above, so after a run the shared
store may hold more months than before.

### OpenMesh — New York City, Oct 2023 – Jul 2024
NYC Mesh community-network links (RSL only, no TSL) and Weather Underground PWS.
Jacoby et al. (2026), ESSD, [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026);
links [doi:10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692) (CC BY 4.0),
PWS [doi:10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286) (CC BY-NC 4.0).

| file | what it is | read by |
|---|---|---|
| `openmesh/release_v1/zips/OpenMesh.zip`, `PWS_NYC_WU.zip` | the two Zenodo packages | `fetch --dataset openmesh[_pws]` writes them |
| `openmesh/release_v1/ds_openmesh.nc` | 75 links × 3 sublinks, 1-min RSL (dBm), 2023-10-29 → 2024-07-01 | `core.opensense.openmesh`, `networks.OpenMesh`, `nyc_rain_maps` |
| `openmesh/release_v1/links_metadata.csv` | per-sublink sites, frequency, length, polarization | same |
| `openmesh/release_v1/pws_wu_os.nc`, `pws_metadata.csv` | 37 WU PWS, ~5-min rain (mm), one group per station | same |
| `openmesh/weather/raw_fetch/iem_cache/asos1min_*.csv`, `metar_*.csv`, `asos_*.csv` | ASOS 1-min and METAR reports (NYC, LGA, JFK, EWR) pulled from IEM, per request window | `core.asos`, `core.radar.nexrad` (download on miss) |
| `openmesh/weather/radar/mrms_cache/<product>/<domain>/<YYYYMMDD>.nc` | NOAA MRMS QPE / precip-type crops over the domain, one file per day | `core.radar.mrms` (download on miss) |
| `openmesh/weather/radar/nexrad_okx/nexrad_OKX_N0B_<date>_<kind>.nc` | KOKX NEXRAD base reflectivity frames for 8 rain/snow events | `core.radar.nexrad` (download on miss) |

### OpenSense example subsets (small demos)
From [OpenSenseAction/opensense_example_data](https://github.com/OpenSenseAction/opensense_example_data).
A few days each — fine for tutorials and tests, **never a basis for a result**.

| folder | files | read by |
|---|---|---|
| `openmrg/_sample_8d/` | `openmrg_{cml,rad,municp_gauge,smhi_gauge}_{8d,5min_2h}.nc` | `example_data.load("openmrg", …)`, `examples/`, `opensense_pipeline` |
| `openrainer/_sample_8d/` | `openrainer_{cml,radar,gauges}_8d.nc` | `example_data.load("openrainer", "8d")` |
| `openmesh/_sample_20d/` | `openmesh_{cml,wu_pws}_{1d,1w,20d}.nc`, `openmesh_asos_ws_20d.nc` | `example_data.load("openmesh", …)` |
| `ams_pws/_sample/` | `ams_pws_full_period.nc`, `ams_gauges_full_period.nc` — 134 Amsterdam Netatmo PWS + reference gauges (de Vos et al. 2019) | `example_data.load("ams_pws", "full_period")` |
