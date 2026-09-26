# CML Retrieval — PyNNcml on OpenMRG

Rain retrieval from commercial microwave links with
[PyNNcml](https://github.com/haihabi/PyNNcml), model-driven and data-driven,
plus rainfall maps from five reconstruction methods, all on OpenMRG.

```
cml_retrieval/
├── notebooks/
│   ├── model_driven_retrieval.ipynb   # wet/dry, baselines, power law, IDW/GMZ   (concise)
│   ├── rnn_retrieval.ipynb            # two-step RNN: train and validate       (concise)
│   ├── rainfall_maps_idw_gmz.ipynb    # five reconstructions, side by side
│   ├── model_driven_tutorial.ipynb    # upstream PyNNcml tutorial, kept as is
│   └── data_driven_tutorial.ipynb     # upstream PyNNcml tutorial, kept as is
├── src/
│   ├── model_driven.py                # the model-driven chain + gauge_reference
│   ├── rnn_retrieval.py               # loss, windowing, train, predict, scores
│   ├── rain_maps.py                   # one API over four map packages
│   ├── plots.py                       # every figure the notebooks draw
│   └── pynncml_compat.py              # PyNNcml 0.3.7 workarounds, in one place
└── tests/test_cml_retrieval.py
```

The notebooks are narrative: each cell is a call into `src/`. The two
upstream tutorials are kept unchanged for reference; they predate PyNNcml
0.3.7 (`DNNType` is now `RNNType`) and do not run under NumPy 2 without
`pynncml_compat`.

## Running

```bash
pip install -e ".[opensense,notebooks,dev]" && pip install -r projects/cml_retrieval/requirements.txt
python -m core.opensense.fetch --dataset openmrg      # once; the notebooks read it in place
python -m pytest projects/cml_retrieval/tests
```

The notebooks read OpenMRG from `dataset/open_datasets/OpenMRG_Sweden/raw/`.
PyNNcml's own loader would download 318 MB into `./data/` next to each
notebook and re-extract the 4.6 GB archive on every call;
`pynncml_compat.openmrg_data_path()` points it at the shared copy instead.

```python
from rnn_retrieval import TrainConfig, openmrg_dataset, build_model, train, predict
cfg = TrainConfig(n_epochs=40)
train_loader, val_loader, stats = openmrg_dataset(slice("2015-06-01", "2015-06-10"), cfg)
model = build_model(cfg, train_loader)
history = train(model, train_loader, cfg, stats["exp_gamma"])
result = predict(model, val_loader, cfg)
```

`rnn_retrieval.ipynb` trains for 40 epochs (~8 min on a CPU) so it executes
end to end; the paper and upstream tutorial use 200 (`TrainConfig()`).

## Rainfall maps, five ways

`src/rain_maps.py` runs five reconstructions over the same links at the same
timestep, so the differences are the methods rather than the input:

| | package | geometry | peak (mm/h) | grid mean |
|---|---|---|---|---|
| IDW | `pycomlink` | midpoint | 85.9 | 2.63 |
| IDW | `pynncml` | midpoint | 84.6 | 1.30 |
| **GMZ** | `pynncml` | **line**, 3 pts/link | 66.6 | 1.68 |
| IDW | `mergeplg` | **line** | 72.9 | 2.13 |
| block kriging | `mergeplg` | **line** | 80.5 | 2.19 |

The split that matters is midpoint versus line. A CML measures a path average
over kilometres; collapsing it to its midpoint throws that away. GMZ
(Goldshtein–Messer–Zinevich) and both `mergeplg` methods keep it, and it
shows: GMZ has the highest 99th percentile of the five (42.8 against
30–35 mm/h) while having the *lowest* peak, because it spreads intensity
along the paths instead of piling it onto midpoints.

The grid means span a factor of two, mostly a coverage difference: `pynncml`
only fills a radius around each link and leaves the rest masked, while the
others interpolate across the whole domain.

## PyNNcml 0.3.7 workarounds

All in `src/pynncml_compat.py`, applied in memory; each belongs upstream at
[haihabi/PyNNcml](https://github.com/haihabi/PyNNcml).

**GMZ: the ceiling index is unclamped.** `i_ceiling` reaches `len(grid)`
whenever a link point lands on the edge of the bounding box built from the
links themselves, so it raises `IndexError` on any full network.

**GMZ: the ceiling-ceiling corner reads the wrong axis.**
`cc = in_rain_map[:, j_ceiling, j_ceiling]` uses the *y* index for both
dimensions where it should be `[i_ceiling, j_ceiling]`. It does not raise; it
silently samples the wrong cell for one of the four bilinear corners wherever
`i_ceiling != j_ceiling`, which is almost everywhere. Any GMZ result produced
with this version is affected.

**NumPy 2: nearest-gauge search.** `PointSet.find_near_gauge(s)` calls
`math.sqrt` on the one-element arrays `PointSensor` stores its coordinates
in, which NumPy 2 refuses; `loader_open_mrg_dataset` fails on its first link.

**The upstream tutorial's confusion matrix is transposed.** It passes the
detector as the reference to `sklearn.metrics.confusion_matrix`.
`rnn_retrieval.detection_scores` puts the gauge on the rows. F1 is unaffected.

## Related

| | |
|---|---|
| `projects/opensense_pipeline/` | the same problem at scale: retrieval chain, poligrain scoring, `mergeplg` merging, two networks |
| `projects/rainfall_field_sim/` | synthetic rain fields and CML sampling, with known ground truth |
| `core/scientific_packages/PYNNcml.md` | PyNNcml reference notes |

## References

1. Habi, H. V., & Messer, H. (2020). Recurrent neural network for rain
   estimation using commercial microwave links. *IEEE TGRS*, 59(5), 3672–3681.
2. van de Beek, R. C. Z., et al. (2022). OpenMRG: Open data from microwave
   links, radar, and gauges for rainfall quantification in Gothenburg, Sweden.
   *Earth System Science Data*, 14, 5411–5426.
3. Goldshtein, O., Messer, H., & Zinevich, A. (2009). Rain rate estimation
   using measurements from commercial telecommunications links. *IEEE Trans.
   Signal Processing*, 57(4), 1616–1625.
