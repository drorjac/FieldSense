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
│   ├── download_and_read_openmesh.ipynb  # fetch the published dataset and open it
│   ├── openmesh_dataset_example.ipynb    # explore the link data
│   ├── read_pws_sample.ipynb             # personal weather station sample
│   ├── asos_pipeline.ipynb               # NOAA ASOS collection pipeline
│   └── wu_pipeline.ipynb                 # Weather Underground collection pipeline
├── src/
│   ├── mop.py                  # strips \blue{} markup, camera_ready → *_clean
│   └── fetch/                  # the collection pipelines the notebooks drive
│       ├── asos_functions.py   asos_plotting.py
│       └── wu_functions.py     wu_plotting.py
├── requirements.txt
└── README.md
```

The published data itself stays in `dataset/open_datasets/OpenMesh_NYC/`,
alongside the other open datasets — see *Data* below.

## Data

Network maps and the dataset description live in
`dataset/open_datasets/OpenMesh_NYC/`. The measurement files are not in git;
`notebooks/download_and_read_openmesh.ipynb` fetches them from Zenodo.

The notebooks locate `src/` by walking up from their own directory, so they
run from anywhere inside this project without path fiddling.

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
