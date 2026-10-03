# cml_retrieval: PyNNcml on OpenMRG

How well do the two families of CML rain retrieval in
[PyNNcml](https://github.com/haihabi/PyNNcml) work on a real network: the model-driven chain
(wet/dry classification, baseline, power law) and the data-driven two-step RNN of Habi and
Messer? And how much does the choice of map method change the rainfall field built from the
same links? Everything here runs on OpenMRG.

## Data

**OpenMRG** (Gothenburg, Sweden, June-August 2015): 364 links (728 sublinks) at 10 s with
transmitted and received power, the SMHI C-band radar composite at 5 min, 11 municipal gauges
at 1 min and the SMHI gauge at 15 min. Andersson et al. (2022),
[doi:10.5194/essd-14-5411-2022](https://doi.org/10.5194/essd-14-5411-2022); data
[doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689), CC BY-SA 4.0.

The notebooks read OpenMRG from the shared data store `~/data/cml/openmrg/` (see
[`DATA.md`](../../DATA.md)). PyNNcml's own loader would download 318 MB into `./data/` next to
each notebook and re-extract the 4.6 GB archive on every call;
`pynncml_compat.openmrg_data_path()` points it at the shared copy instead.

## Method

- **Model-driven chain** (`src/model_driven.py`): wet/dry classification, dry-weather
  baseline, the ITU-R P.838-3 power law, then IDW and GMZ maps.
- **Two-step RNN** (`src/rnn_retrieval.py`, code in `core/scientific_packages/`): PyNNcml's
  wet/dry detector and rain-rate regressor, trained against the gauges.
- **Five map methods** (`src/rain_maps.py`): one API over four packages - IDW from
  [pycomlink](https://github.com/pycomlink/pycomlink) and PyNNcml, GMZ from PyNNcml, IDW and
  block kriging from [mergeplg](https://github.com/OpenSenseAction/mergeplg).

The notebooks are narrative: each cell calls into `src/`. They replace the two upstream PyNNcml
tutorials, which predate PyNNcml 0.3.7 (`DNNType` is now `RNNType`) and do not run under
NumPy 2 without `pynncml_compat`.

```python
from rnn_retrieval import TrainConfig, openmrg_dataset, build_model, train, predict
cfg = TrainConfig(n_epochs=40)
train_loader, val_loader, stats = openmrg_dataset(slice("2015-06-01", "2015-06-10"), cfg)
model = build_model(cfg, train_loader)
history = train(model, train_loader, cfg, stats["exp_gamma"])
result = predict(model, val_loader, cfg)
```

`rnn_retrieval.ipynb` trains for 40 epochs (about 8 min on a CPU) so that it executes end to
end; the paper and the upstream tutorial use 200 (`TrainConfig()`). For training the RNN on
three networks against radar and gauges, see [`cml_rnn`](../cml_rnn/).

## Findings: rainfall maps, five ways

The five reconstructions use the same links at the same time step, so the differences come
from the methods, not the input:

| | package | geometry | peak (mm/h) | grid mean |
|---|---|---|---|---|
| IDW | `pycomlink` | midpoint | 85.9 | 2.63 |
| IDW | `pynncml` | midpoint | 84.6 | 1.30 |
| **GMZ** | `pynncml` | **line**, 3 pts/link | 66.6 | 1.68 |
| IDW | `mergeplg` | **line** | 72.9 | 2.13 |
| block kriging | `mergeplg` | **line** | 80.5 | 2.19 |

- **Midpoint against line is the split that matters.** A CML measures a path average over
  kilometres; collapsing it to its midpoint discards that. GMZ (Goldshtein, Messer and
  Zinevich) and both `mergeplg` methods keep the path. GMZ has the highest 99th percentile of
  the five (42.8 against 30-35 mm/h) and the *lowest* peak, because it spreads intensity along
  the paths instead of concentrating it at midpoints.
- **The grid means differ by a factor of two, mostly through coverage:** `pynncml` fills only a
  radius around each link and masks the rest, while the others interpolate across the whole
  domain.

## How to run

```bash
pip install -e ".[opensense,notebooks,dev]" && pip install -r projects/cml_retrieval/requirements.txt
python -m core.opensense.fetch --dataset openmrg      # once; the notebooks read it in place
python -m pytest projects/cml_retrieval/tests
```

| notebook | what it does |
|---|---|
| `notebooks/model_driven_retrieval.ipynb` | wet/dry, baselines, power law, IDW and GMZ |
| `notebooks/rnn_retrieval.ipynb` | the two-step RNN: training and validation |
| `notebooks/rainfall_maps_idw_gmz.ipynb` | the five reconstructions side by side |

## Layout

```
cml_retrieval/
├── notebooks/                         # the three notebooks above
├── src/
│   ├── model_driven.py                # the model-driven chain + gauge_reference
│   ├── rain_maps.py                   # one API over four map packages
│   ├── plots.py                       # every figure the notebooks draw
│   └── rnn_retrieval.py, pynncml_compat.py   # shims: the code is in core/scientific_packages/
└── tests/test_cml_retrieval.py
```

## Caveats: PyNNcml 0.3.7 workarounds

All are in `core/scientific_packages/pynncml_compat.py`, applied in memory and reversible; each
belongs upstream at [haihabi/PyNNcml](https://github.com/haihabi/PyNNcml).

- **GMZ: the ceiling index is unclamped.** `i_ceiling` reaches `len(grid)` whenever a link
  point lands on the edge of the bounding box built from the links themselves, so it raises
  `IndexError` on any full network.
- **GMZ: the ceiling-ceiling corner reads the wrong axis.**
  `cc = in_rain_map[:, j_ceiling, j_ceiling]` uses the *y* index for both dimensions where it
  should be `[i_ceiling, j_ceiling]`. It does not raise; it silently samples the wrong cell for
  one of the four bilinear corners wherever `i_ceiling != j_ceiling`, which is almost
  everywhere. Every GMZ result produced with this version is affected.
- **NumPy 2: nearest-gauge search.** `PointSet.find_near_gauge(s)` calls `math.sqrt` on the
  one-element arrays `PointSensor` stores its coordinates in, which NumPy 2 refuses;
  `loader_open_mrg_dataset` fails on its first link.
- **The upstream tutorial's confusion matrix is transposed.** It passes the detector as the
  reference to `sklearn.metrics.confusion_matrix`. `rnn_retrieval.detection_scores` puts the
  gauge on the rows. F1 is unaffected.

## Related

| | |
|---|---|
| [`tutorials/02_cml_signal_to_rain.ipynb`](../../tutorials/), [`03_training_a_retrieval_network.ipynb`](../../tutorials/), [`05_2d_rain_maps.ipynb`](../../tutorials/) | the retrieval, training and map methods, step by step |
| [`cml_rnn`](../cml_rnn/) | PyNNcml's RNN trained on three networks against radar and gauges |
| [`opensense_pipeline`](../opensense_pipeline/) | the same problem at scale: retrieval chain, poligrain scoring, `mergeplg` merging, two networks |
| [`rainfall_field_sim`](../rainfall_field_sim/) | synthetic rain fields and CML sampling, with known ground truth |
| `core/scientific_packages/PYNNcml.md` | PyNNcml reference notes |

## References

1. Habi, H. V., and Messer, H. (2021). Recurrent neural network for rain estimation using
   commercial microwave links. *IEEE Transactions on Geoscience and Remote Sensing*, 59(5),
   3672-3681. [doi:10.1109/TGRS.2020.3010305](https://doi.org/10.1109/TGRS.2020.3010305)
2. Andersson, J. C. M., Olsson, J., van de Beek, R. (C. Z.), and Hansryd, J. (2022). OpenMRG:
   Open data from Microwave links, Radar, and Gauges for rainfall quantification in
   Gothenburg, Sweden. *Earth System Science Data*, 14, 5411-5426.
   [doi:10.5194/essd-14-5411-2022](https://doi.org/10.5194/essd-14-5411-2022)
3. Goldshtein, O., Messer, H., and Zinevich, A. (2009). Rain rate estimation using
   measurements from commercial telecommunications links. *IEEE Transactions on Signal
   Processing*, 57(4), 1616-1625.
   [doi:10.1109/TSP.2009.2012554](https://doi.org/10.1109/TSP.2009.2012554)
4. ITU-R P.838-3 (2005). Specific attenuation model for rain for use in prediction methods.
   <https://www.itu.int/rec/R-REC-P.838-3-200503-I/en>
