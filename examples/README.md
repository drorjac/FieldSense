# Example datasets

Notebooks that read the OpenSense example datasets: small, curated subsets of
open opportunistic-sensing datasets, published by the OpenSense community at
[OpenSenseAction/opensense_example_data](https://github.com/OpenSenseAction/opensense_example_data)
and used by the examples of `poligrain`, `pypwsqc` and `mergeplg`. One
notebook per folder of that repository.

| notebook | dataset | sensors | shows |
|---|---|---|---|
| [`00_overview`](00_overview.ipynb) | all four | | every file, its size and span; the four networks side by side |
| [`01_openmrg`](01_openmrg.ipynb) | OpenMRG, Gothenburg, July 2015 | CML, radar, gauges | frequency stored in MHz without units; a link against the radar along its path; radar against gauges; the 5-minute `mergeplg` subset |
| [`02_openrainer`](02_openrainer.ipynb) | OpenRainER, Emilia-Romagna, August 2022 | CML, radar, gauges | radar `R` is a 15-min accumulation stamped at the interval end; raw and gauge-adjusted radar against gauges |
| [`03_openmesh`](03_openmesh.ipynb) | OpenMesh, New York City, January 2024 | CML, PWS, ASOS | received power only; up to three sublinks on 5-69 GHz; snow seen by links and ASOS but not by PWS; PWS quality control |
| [`04_ams_pws`](04_ams_pws.ipynb) | Amsterdam PWS, 2016-2018 | PWS, radar-derived reference | monthly totals against the reference; what each PWS QC filter catches on one faulty station |

## Running

From the repository root, with the `opensense` and `notebooks` extras installed
(see the root README):

```bash
jupyter lab examples/
```

The first run downloads the files (about 60 MB in all) into
`~/data/cml/<dataset>/_sample*/` (outside the repo, see `DATA.md`); later runs
read them from there. To fetch everything up front:

```bash
python -m core.opensense.example_data --all
```

## Reading the data yourself

```python
from core.opensense import example_data

data = example_data.load("openrainer", "8d")   # {"cml": ..., "radar": ..., "gauge": ...}
cml = example_data.load("openmrg", "8d", components=("cml",),
                        time=slice("2015-07-25", "2015-07-25"))["cml"]
```

`load()` returns a dict keyed by sensor type and normalizes on the way in:
link length in km and frequency in GHz whatever unit the file stored, one
polarization spelling, projected `x`/`y` for every sensor, radar and
point-sensor rain as a rate `R` in mm/h. `normalize=False` returns the files
as published.

## Sources and licenses

Cite the original dataset when you use one.

| dataset | reference | license |
|---|---|---|
| OpenMRG | Andersson et al. (2022), [doi:10.5281/zenodo.6673750](https://doi.org/10.5281/zenodo.6673750) | CC BY-SA 4.0 |
| OpenRainER | Covi et al., [doi:10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808) | CC BY 4.0 |
| OpenMesh CML | Jacoby et al. (2026), [ESSD 18, 5817-5836](https://doi.org/10.5194/essd-18-5817-2026); data [doi:10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692) | CC BY 4.0 |
| OpenMesh PWS | [doi:10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286) | CC BY-NC 4.0 |
| Amsterdam PWS | de Vos et al. (2019), [doi:10.1029/2019GL083731](https://doi.org/10.1029/2019GL083731) | CC BY 4.0 |
