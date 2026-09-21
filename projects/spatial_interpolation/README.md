# Spatial Interpolation — Rainfall Nowcasting from CML Networks

## Overview

Commercial Microwave Link (CML) networks attenuate during rainfall, which turns
backhaul infrastructure already in the ground into an opportunistic rain-sensing
layer at zero hardware cost. CML-based rainfall *estimation* is mature; this
project targets the less settled problem of short-term *nowcasting*
(15–60 min ahead) from CML observations.

The work is framed as a two-stage spatiotemporal learning problem:

1. **Estimation** — a recurrent network maps raw CML attenuation to per-link
   rain rates, which are interpolated onto a regular grid.
2. **Dynamics** — a downstream dynamical model propagates the resulting
   CML-derived rain fields forward in time.

A central proposition is a **CML self-supervised** forecasting objective: the
Stage-2 model is trained against future CML-derived fields with no external rain
reference, preserving the opportunistic-sensing premise end to end. Three
families of dynamical models are benchmarked on equal footing — Transformer,
POD-SINDy, and a Mamba-style state-space model — across single-step and
multi-horizon settings, evaluated both point-to-pixel against gauges and
grid-to-grid against radar.

## Structure

```
spatial_interpolation/
├── notebooks/
│   ├── advanced_models_colab_v2.ipynb  # Main working notebook — source of truth for all results
│   └── version_v1.ipynb                # Earlier iteration, kept for provenance
├── paper/
│   ├── paper.tex                       # 5-page IEEE conference paper (IEEEtran) — target artifact
│   └── full_report.tex                 # Long-form report, earlier experiment scope
├── src/
│   └── main.py                         # Placeholder entry point
├── CLAUDE.md                           # Working rules for AI assistance in this project
├── requirements.txt                    # Project-specific dependencies
└── README.md                           # This file
```

## Getting Started

```bash
# from the repository root
pip install -r requirements.txt
pip install -r projects/spatial_interpolation/requirements.txt

jupyter lab projects/spatial_interpolation/notebooks/advanced_models_colab_v2.ipynb
```

The notebooks were developed in Google Colab and mount Drive for data access;
running locally means repointing those paths at a local copy of the dataset.

## Results

`notebooks/advanced_models_colab_v2.ipynb` is the source of truth for every
number and figure. `paper/full_report.tex` describes an **earlier and different**
experiment scope (IDW vs GMZ map inputs, pySTEPS baseline, full-period training),
so its result tables do not match the notebook — treat it as methodology
background, not as a source of numbers. See `CLAUDE.md` for the full working rules.

## Team

- Ben Yehoshua S.
- Jacoby D.
- Salganik Y.

Tel Aviv University
