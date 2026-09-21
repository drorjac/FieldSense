# OpenMesh NYC — Dataset Paper

## Overview

Manuscript for **"OpenMesh: Wireless Signal Dataset for Opportunistic Urban
Weather Sensing in New York City"**, the dataset paper accompanying
`dataset/open_datasets/OpenMesh_NYC/`.

OpenMesh is a publicly available dataset of wireless signal measurements from
the NYC Mesh, a community-run network in New York City. Links provisioned for
affordable internet access are repurposed opportunistically for high-resolution
urban weather monitoring, at 1-minute sampling with dense spatial coverage.

- **Coverage:** 103 links in Lower Manhattan and Brooklyn, November 2023 – June 2024
- **Bands:** 5–6 GHz (C), 24 GHz (K), 58–70 GHz (V, mmWave)
- **Ground truth:** 37 Weather Underground PWS (5/15-min) + 3 NOAA ASOS stations
- **Study period:** ~900 mm total rainfall, from intense rain to the winter 2023–24 snowstorms;
  attenuation up to 30 dB with frequent outages on V-band links
- **Data DOI:** https://doi.org/10.5281/zenodo.15268340
- **Format:** Copernicus (`\documentclass[manuscript]{copernicus}`)

## Structure

```
openmesh_nyc_paper/
├── paper/
│   ├── camera_ready.tex        # Working manuscript — revisions marked with \blue{...}
│   └── camera_ready_clean.tex  # Generated: same text with \blue{} markup stripped
├── src/
│   └── mop.py                  # Strips \blue{} markup, camera_ready → camera_ready_clean
├── requirements.txt            # No pip dependencies; LaTeX toolchain required
└── README.md                   # This file
```

`camera_ready.tex` is the file to edit. `camera_ready_clean.tex` is **generated**
— regenerate it rather than editing it by hand, or the two will drift apart.

## Regenerating the clean manuscript

```bash
# from anywhere; defaults to paper/camera_ready.tex -> paper/camera_ready_clean.tex
python projects/openmesh_nyc_paper/src/mop.py

# or with explicit paths
python projects/openmesh_nyc_paper/src/mop.py in.tex out.tex
```

`mop.py` removes `\blue{...}` wrappers while keeping their contents, applied
repeatedly so nested markup is handled.

## Building the PDF

Requires a LaTeX distribution with the `copernicus` class:

```bash
cd projects/openmesh_nyc_paper/paper
pdflatex camera_ready_clean.tex && bibtex camera_ready_clean && pdflatex camera_ready_clean.tex && pdflatex camera_ready_clean.tex
```

## Related

- `dataset/open_datasets/OpenMesh_NYC/` — the dataset itself, fetch pipelines
  (NOAA ASOS, Weather Underground), link metadata, and coverage maps
