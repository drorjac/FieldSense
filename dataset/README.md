# Datasets

The open datasets FieldSense works with. **No data is stored in git**: this
folder holds documentation only, and the files are downloaded on request
into the git-ignored folders described below.

## Two ways to get the data

| | example subsets | full records |
|---|---|---|
| what | a few days of each dataset, already on the OpenSense conventions | months to years, as published |
| size | about 60 MB in all | hundreds of MB to several GB per dataset |
| source | [OpenSenseAction/opensense_example_data](https://github.com/OpenSenseAction/opensense_example_data) | Zenodo (below) |
| code | `core/opensense/example_data.py` | `core/opensense/fetch.py` |
| lands in | `open_datasets/_example_subsets/` | `open_datasets/<dataset>/raw/` |
| list | `python -m core.opensense.example_data --list` | `python -m core.opensense.fetch --list` |

Start with the example subsets; the notebooks in [`examples/`](../examples/)
read them. The full records are for work that needs more than a few days:
`projects/opensense_pipeline/src/ingest_*.py` turns them into OpenSense
NetCDF files under `open_datasets/<dataset>/processed/`.

## The datasets

| dataset | region, period | sensors | example subset | full record | license |
|---|---|---|---|---|---|
| **OpenMRG** | Gothenburg, Sweden; Jun-Aug 2015 | 364 CMLs, radar, 11 gauges | `openmrg` (`8d`, `5min_2h`) | `fetch --dataset openmrg`, 318 MB | CC BY-SA 4.0 |
| **OpenRainER** | Emilia-Romagna, Italy; 2021-2022 | CMLs, radar, gauges | `openrainer` (`8d`) | `fetch --dataset openrainer`, 1.4 GB by default | CC BY 4.0 |
| **OpenMesh** | New York City; Oct 2023 - Jul 2024 | NYC Mesh community-network links, PWS, ASOS | `openmesh` (`1d`, `1w`, `20d`) | `fetch --dataset openmesh` (+ `openmesh_pws`) | CC BY 4.0 (links), CC BY-NC 4.0 (PWS) |
| **Amsterdam PWS** | Amsterdam, Netherlands; 2016-2018 | 134 Netatmo PWS, radar reference | `ams_pws` (`full_period`) | - | CC BY 4.0 |
| **CML Netherlands** | Netherlands; 2011-2015 | nationwide CMLs | - | not automated yet | see source |

### Sources and citations

Cite the original dataset in anything that uses it.

- **OpenMRG** - Andersson, J. et al. (2022). The OpenMRG data set, v1.1.
  [doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689).
  [OpenSense page](https://opensenseaction.eu/datasets/openmrg-open-data-from-microwave-links-radar-and-gauges/).
- **OpenRainER** - Covi, E. and Roversi, G. OpenRainER, v2.0.3.
  [doi:10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808).
  [OpenSense page](https://opensenseaction.eu/news/new-open-cml-dataset-from-italy-openrainer/).
- **OpenMesh** - Jacoby, D. et al. (2026). OpenMesh: Wireless Signal Dataset for
  Opportunistic Urban Weather Sensing in New York City. *Earth System Science
  Data* 18, 5817-5836. [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026).
  Links: [doi:10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692);
  PWS: [doi:10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286);
  code: [drorjac/OpenMesh](https://github.com/drorjac/OpenMesh);
  network: [NYC Mesh](https://www.nycmesh.net/).
- **Amsterdam PWS** - de Vos, L. et al. (2019). Quality control for crowdsourced
  personal weather stations to enable operational rainfall monitoring.
  *Geophysical Research Letters*. [doi:10.1029/2019GL083731](https://doi.org/10.1029/2019GL083731).
  Data at [4TU](https://data.4tu.nl/articles/dataset/Rainfall_observations_datasets_from_Personal_Weather_Stations/12703250).
- **CML Netherlands** - Overeem, A., Walraven, B. and Leijnse, H. (2024).
  Four-year commercial microwave link dataset for the Netherlands.
  [doi:10.4121/be252844-b672-471e-8d69-27269a862ec1.v1](https://doi.org/10.4121/be252844-b672-471e-8d69-27269a862ec1.v1).

```bibtex
@article{jacoby2026openmesh,
  title={OpenMesh: Wireless Signal Dataset for Opportunistic Urban Weather Sensing in New York City},
  author={Jacoby, Dror and Yu, Shuyue and Hu, Qianfei and Hine, Zachary and Johnson, Rob and Ostrometzky, Jonatan and Kadota, Igor and Zussman, Gil and Messer, Hagit},
  journal={Earth System Science Data},
  volume={18},
  pages={5817--5836},
  year={2026},
  doi={10.5194/essd-18-5817-2026}
}
```

## Folder layout

```
dataset/open_datasets/
├── _example_subsets/     # example subsets, one folder per dataset       (downloaded)
├── OpenMRG_Sweden/       # README, SMHI's readme, example reader scripts
│   ├── raw/              #   the Zenodo archive                          (downloaded)
│   └── processed/        #   OpenSense NetCDF made by ingest_openmrg.py  (generated)
├── OpenRainER_Italy/     # README; raw/ and processed/ as above
├── OpenMesh_NYC/         # the Zenodo package's README.txt, network maps (HTML)
└── CML_Netherlands/      # README only
```

## The OpenSense data format

All NetCDF files here follow, or are converted to, the conventions of the
[OpenSense](https://opensenseaction.eu/) community: CML data on dimensions
`cml_id`, `sublink_id` and `time`, with `rsl` (received signal level) and,
where published, `tsl` (transmitted), and link geometry as `site_0_lat`,
`site_0_lon`, `site_1_lat`, `site_1_lon`, `length`, `frequency`,
`polarization`. Point sensors (gauges, PWS) use `id` and `time`.

The published files still differ in units and names - frequency in MHz
without a unit, radar accumulations named like rates, `station_id` instead of
`id`. `example_data.load()` normalizes these; `core/opensense/conventions.py`
documents each case.

## Adding a dataset

1. Create `open_datasets/<Name_Region>/README.md`: what it is, the official
   source and DOI, the license.
2. Add a `Source` to `core/opensense/fetch.py` (for a full record) or an
   `ExampleDataset` to `core/opensense/example_data.py` (for a subset).
3. Add it to the tables above.
