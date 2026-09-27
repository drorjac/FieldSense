# OpenMesh NYC

Everything about the **OpenMesh** dataset in one place: the manuscript, the
data-collection pipelines that built it, and the notebooks that read and
explore it.

OpenMesh is a public dataset of wireless signal measurements from the
[NYC Mesh](https://www.nycmesh.net/), a community-run network in New York
City. Links provisioned for affordable internet access are repurposed
opportunistically for high-resolution urban weather monitoring.

- **Coverage:** 103 links in Lower Manhattan and Brooklyn, November 2023 – June 2024
- **Bands:** 5–6 GHz (C), 24 GHz (K), 58–70 GHz (V, mmWave)
- **Ground truth:** 37 Weather Underground PWS (5/15-min) + 3 NOAA ASOS stations
- **Study period:** ~900 mm total rainfall, from intense rain to the winter
  2023–24 snowstorms; attenuation up to 30 dB with frequent V-band outages
- **Data DOI:** https://doi.org/10.5281/zenodo.15268340
- **Code:** https://github.com/drorjac/OpenMesh

## Structure

```
openmesh_nyc/
├── paper/
│   ├── camera_ready.tex        # Working manuscript — revisions marked \blue{...}
│   └── camera_ready_clean.tex  # Generated: same text, markup stripped
├── notebooks/
│   ├── openmesh_data.ipynb               # START HERE: pull, format check, links, PWS, ASOS (concise)
│   ├── asos_pipeline.ipynb               # NOAA ASOS collection pipeline
│   ├── wu_pipeline.ipynb                 # Weather Underground collection pipeline
│   ├── nexrad_rain_vs_snow.ipynb         # radar over the network, rain vs snow
│   └── archive/                          # the three notebooks openmesh_data.ipynb replaces
├── src/
│   ├── mop.py                  # strips \blue{} markup, camera_ready → *_clean
│   ├── stations.py             # PWS / ASOS totals, accumulation, dead-station check
│   └── fetch/                  # the collection pipelines the notebooks drive
│       ├── asos_functions.py
│       └── wu_functions.py     wu_plotting.py
├── tests/test_stations.py
├── requirements.txt
└── README.md

`openmesh_data.ipynb` covers what the three archived notebooks did, in ~25 lines
of calls into `core.opensense` and `src/stations.py`, and runs offline on the
curated 20-day subset. The full records: `python -m core.opensense.fetch
--dataset openmesh` (CML, CC-BY-4.0) and `--dataset openmesh_pws` (PWS,
CC-BY-NC-4.0, non-commercial). The collection pipelines need network access,
and `wu_pipeline.ipynb` a Weather Underground API key read from
`src/config.py`, which is not in the repository.

Two facts from the 20-day subset worth knowing before using it: most links
report one sublink (51 of 75; 20 report two, 4 three), and 8 of the 37 PWS
record under 1 mm in 20 days against ~70 mm at the rest. Seven of those are
89-100% missing; one (KNYNEWYO1622) reports throughout but stays at zero,
and pypwsqc's faulty-zero filter flags 88% of its record.
```

The published data itself stays in `dataset/open_datasets/OpenMesh_NYC/`,
alongside the other open datasets — see *Data* below.

## Data

Network maps and the dataset description live in
`dataset/open_datasets/OpenMesh_NYC/`. The measurement files are not in git;
`python -m core.opensense.fetch --dataset openmesh` fetches them from Zenodo.

The notebooks locate `src/` by walking up from their own directory, so they
run from anywhere inside this project without path fiddling.

## Radar

OpenMesh ships CMLs, PWS and ASOS but **no radar**, so there is no gridded
reference to score a CML-derived field against — the one thing OpenMRG and
OpenRainER both have. `core/radar/nexrad.py` (shared, since `opensense_pipeline`
uses it too) supplies one from **KOKX**
(Upton, NY), the NEXRAD covering New York City, through the Iowa
Environmental Mesonet archive.

```bash
python -m core.radar.nexrad --classify        # which days
python -m core.radar.nexrad --date 2024-01-16 # fetch one
python -m core.radar.nexrad --best 3 --kind snow
```

Days are classified from METAR present-weather codes at the four NYC ASOS
stations — `SN`/`SG`/`IC`/`PL` for frozen, `RA`/`DZ`/`TS` for liquid — which
is more reliable than a temperature threshold near the melting layer.

| | strongest days | daily precip |
|---|---|---|
| snow | 2024-01-16, 02-13, 02-17 | 8.8, 16.8, 6.8 mm |
| rain | 2024-03-23, 2023-12-18, 11-22 | 84.1, 60.2, 50.9 mm |

**2024-01-16 and 01-19 fall inside the OpenMesh 20-day example subset**, so
radar and CML cover the same hours and can be compared directly.

### Rain and snow need different conversions

Reflectivity becomes a rate through a power law whose coefficients are not
the same for the two — snow of a given liquid-water-equivalent rate scatters
much more than rain:

| | relation | |
|---|---|---|
| rain | `Z = 200 R^1.6` | Marshall-Palmer |
| snow | `Z = 180 S^2.0` | Sekhon-Srivastava, the WSR-88D operational pair |

At 40 dBZ the rain relation gives 11.5 mm/h against snow's 7.5 — applying it
to a snow day overstates the rate by half. The module picks the relation from
the day's classification.

Two caveats that matter for any comparison against the links: the beam is
several hundred metres above the city, and that gap between beam and ground
is larger in snow than in rain; and published snow relations disagree with
each other by a factor of two or more, so `Z = 180 S^2.0` is a defensible
choice rather than a settled one.

`notebooks/nexrad_rain_vs_snow.ipynb` works through all of it.

## The manuscript

`camera_ready.tex` is the file to edit. `camera_ready_clean.tex` is
**generated** — regenerate it rather than editing it by hand, or the two drift
apart.

```bash
# strip \blue{} markup: paper/camera_ready.tex -> paper/camera_ready_clean.tex
python projects/openmesh_nyc/src/mop.py

# or with explicit paths
python projects/openmesh_nyc/src/mop.py in.tex out.tex
```

Building the PDF needs a LaTeX distribution with the `copernicus` class:

```bash
cd projects/openmesh_nyc/paper
pdflatex camera_ready_clean.tex && bibtex camera_ready_clean \
  && pdflatex camera_ready_clean.tex && pdflatex camera_ready_clean.tex
```

## Citation

```bibtex
@article{jacoby2025openmesh,
  title={OpenMesh: Wireless Signal Dataset for Opportunistic Urban Weather
         Sensing in New York City},
  author={Jacoby, Dror and Yu, Shuyue and Hu, Qianfei and Hine, Zachary and
          Johnson, Rob and Ostrometzky, Jonatan and Kadota, Igor and
          Zussman, Gil and Messer, Hagit},
  journal={Earth System Science Data},
  year={2025}
}
```
