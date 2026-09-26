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
│   ├── nowcasting.ipynb                # the pipeline from src/, faithful vs corrected (concise)
│   ├── advanced_models_colab_v2.ipynb  # Main working notebook — source of truth for all results
│   └── archive/version_v1.ipynb        # Earlier iteration, kept for provenance
├── paper/
│   ├── paper.tex                       # 5-page IEEE conference paper (IEEEtran) — target artifact
│   └── full_report.tex                 # Long-form report, earlier experiment scope
├── src/
│   ├── config.py                       # NowcastConfig: splits, budgets, faithful/corrected
│   ├── data.py                         # radar, gauges, links, per-link rain on one 15-min axis
│   ├── maps.py                         # IDW and GMZ maps on the radar grid
│   ├── forecasting.py                  # Transformer, GRU, POD-SINDy, persistence, pySTEPS
│   ├── evaluation.py                   # metrics, aligned scoring vs gauges and radar
│   ├── pipeline.py                     # the stages, each cached by the settings it uses
│   └── plots.py
├── tests/test_nowcast.py
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

## The pipeline as code

`src/` is `advanced_models_colab_v2.ipynb` taken apart into modules, and
`notebooks/nowcasting.ipynb` runs it in ~35 lines of calls. The Colab, Drive
and v1-migration code is gone: OpenMRG is read from the repository's archive
(`python -m core.opensense.fetch --dataset openmrg`). W&B sweeps are not
run; hyper-parameters come from `NowcastConfig`. It covers the CML
self-supervised setting the v2 notebook runs (forecasters trained on
CML-derived maps).

```python
from config import NowcastConfig
import pipeline
res = pipeline.run(NowcastConfig(faithful=False))   # three months: hours on a GPU
res = pipeline.run(NowcastConfig.smoke())           # three weeks, few epochs: minutes on a CPU
```

`faithful=True` reproduces the notebook's choices; `faithful=False` applies
the corrections below. The committed `nowcasting.ipynb` is a **smoke run** -
enough to show every stage works and which way each fix pushes, not to rank
the models or reproduce the paper. pySTEPS is optional (it did not build on
the machine this was written on).

## Issues found in the original pipeline

Found while extracting the code; the original notebook is unchanged. Each is
switchable in `config.py`, and the smoke run measures its direction.

1. **pySTEPS is scored one frame out of line.** Every model sees frames
   `[i, i + lookback)` and is scored on frame `i + lookback - 1 + h`; pySTEPS
   extrapolated from frame `i + lookback`, which at h = 15 min *is* the
   frame it is scored against. Persistence of that frame alone scores HSS
   1.00 at 15 min in the smoke run, against 0.73 for honest persistence.
2. **GMZ ran with pynncml 0.3.7's interpolation bug** (a bilinear corner
   read from the wrong cell; see `core/scientific_packages/pynncml_compat.py`)
   **and was misregistered**: its grid is normalized to `(utm - min)/scale`,
   and the notebook spread it over the bounding box of link midpoints
   instead. Corrected, 4 of 5 GMZ-trained models gain in the smoke run (up
   to +0.14 HSS vs radar).
3. **IDW measured distance in degrees**, which at 57.7 N weighs east-west
   distance at half its true size. Corrected to km; small effect.
4. **"Mamba" is a GRU.** `SimpleMamba` is a GRU over flattened frames, not a
   state-space model; the paper's "Mamba-style SSM" overstates it. Renamed
   `GRUForecaster` here, with the old name kept as an alias.

Also worth knowing: "Model 1 (physical)" is the CML estimate *at the target
time* - a nowcast reference, not a forecast; the committed
`advanced_models_colab_v2.ipynb` has no saved outputs, and its cell numbers
no longer match the ones the paper's figures were taken from (72-80).

## Results

`notebooks/advanced_models_colab_v2.ipynb` is the source of truth for every
number and figure. `paper/full_report.tex` describes an **earlier and different**
experiment scope (IDW vs GMZ map inputs, pySTEPS baseline, full-period training),
so its result tables do not match the notebook — treat it as methodology
background, not as a source of numbers.

## Team

- Ben Yehoshua S.
- Jacoby D.
- Salganik Y.

Tel Aviv University
