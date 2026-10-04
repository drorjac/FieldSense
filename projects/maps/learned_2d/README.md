# learned_2d: learned 2-D rain maps from link attenuation (proposed project: starter)

A starting point for the B.Sc. proposal
[`docs/pre_projects/2d_project.md`](../../../docs/pre_projects/2d_project.md): learn a mapping
$g_\theta$ from the attenuations of a commercial microwave link (CML) network and its metadata
(endpoints, frequency, polarization, length) to a gridded rain field, supervised by radar, and
compare it with the model-driven baselines IDW, ordinary kriging and GMZ on storm events it never
saw, with RMSE, MAE, Pearson correlation, cumulative event error, POD/FAR/CSI and the error at
independent gauges.

**FieldSense does not implement the proposal** (see
[`docs/pre_projects/README.md`](../../../docs/pre_projects/README.md)). This folder gives it a
solid base: the data load, the inputs and targets are built, the baselines are scored the way
the proposal will be judged, and the notebooks stop where the project's method goes.

**Data.** The proposal's own data, the lab's Israeli CML network with its radar and gauges, is
private. The starters use **OpenMRG** (Gothenburg, June-August 2015; Andersson et al., 2022,
[doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689)): 364 links with radar and
gauges, the same open benchmark used by the diffusion-prior paper the proposal cites as its
performance target (Moufad et al., ICML 2026). The code is written for any network of
`core.opensense.networks` (OpenRainER and OpenMesh have the same interface), and a lab dataset
in the same structures would run unchanged.

## What FieldSense provides

| the proposal needs | where it is |
|---|---|
| CML, radar and gauge data, aligned hourly | `core.opensense.networks` (`links`, `radar_hourly`, `points_hourly`) |
| attenuation from RSL/TSL, baseline, wet antenna, power law | `core.cml` (`link_qc`, `estimators`, `power_law`) |
| storm events and a split by event | `core.events.detect_events`, `core.maps.learning.split_events` |
| links rasterized as channels on the target grid; the link table for graph models; radar target; held-out gauges | `core.maps.learning` (`build_network_dataset`, `assemble`, `arrays`) |
| the path-integral forward operator (field to path averages) | `core.maps.learning` (`path_weights`, `sample_paths`) |
| IDW, GMZ, kriging with external drift | `core.maps.idw`, `core.maps.gmz`, `core.maps.mergeplg_methods` |
| ordinary kriging of the links | `src/learned_2d/baselines.py` (pykrige) |
| the proposal's metric table | `core.maps.map_skill` (`skill_table`, `pcc_per_step`, `event_totals`, `detection`, `gauge_point_error`) on top of `core.maps.scores` |
| a simulated truth | `core.simulation.scenario` and `core.maps.learning.from_synthetic` |
| a minimal learned model (interface only) | `src/learned_2d/unet.py` |

## Notebooks

| notebook | what it does | runtime |
|---|---|---|
| [`01_dataset`](notebooks/01_dataset.ipynb) | builds the OpenMRG dataset: 27 radar storm events, 341 hours, 354 links on the 0.02 deg radar grid (35 x 51); the four input channels, the link table, the radar target, the 41 gauges and the split by event (17 / 5 / 5); checks that rasterization conserves the link values | 2 min first time, then seconds |
| [`02_baselines`](notebooks/02_baselines.ipynb) | IDW, ordinary kriging and GMZ on the 5 test events: the full metric table against the radar (cells within 2 km of a link) and at 11 independent gauges; detection at three thresholds; event totals; maps of the largest test event | 30 s |
| [`03_minimal_unet`](notebooks/03_minimal_unet.ipynb) | a tiny U-Net on the four channels, 25 epochs on a CPU, scored with the same table; labelled as an interface example, not a result; then the "your method goes here" cell with pointers for a GNN and a diffusion prior | 2.5 min |
| [`04_simulated_truth`](notebooks/04_simulated_truth.ipynb) | the same pipeline on 12 simulated storms where the true field is known: the baselines and the radar against the truth, and whether scoring against the radar ranks the methods as the truth does | 4 min |

The dataset is cached under `dataset/open_datasets/_learned_2d/` (not tracked); notebooks 02-04
read it. Run the notebooks from the repository root's environment (`pip install -e .[opensense]`,
plus `torch` for 03).

Baseline scores on the held-out events (notebook 02, hourly mm, 86 659 cell-hours, 1 122
gauge-hours):

| map | RMSE | PCC pooled | PCC per hour | event abs. rel. error | CSI (0.1 mm) | RMSE at gauges |
|---|---|---|---|---|---|---|
| IDW | 1.03 | 0.68 | 0.37 | 0.59 | 0.38 | 0.74 |
| OK | 1.04 | 0.68 | 0.38 | 0.59 | 0.39 | 0.88 |
| GMZ | 1.03 | 0.68 | 0.37 | 0.58 | 0.38 | 0.73 |

The radar itself scores RMSE 0.87 at the same gauges: the target is a noisy reference, as the
proposal's risk table says. All three baselines read every test event's total low (by 22% on
the largest, by 59% on average over the five): the links' retrieval (constant baseline, ITU power law) is the weak step, not the interpolation,
which agrees with `projects/maps/multisensor`. The retrieval is a parameter of
`build_network_dataset` (`method=`).

## What is yours to do

- **The model.** Pick the families (KED, GNN, CNN/U-Net, diffusion prior), build, train and tune
  them. The U-Net of notebook 03 only shows the interface.
- **The inputs.** Raw attenuation instead of power-law rain, wet-antenna handling, a finer time
  step, other channels (link density, frequency), several sublinks per path.
- **The methodology choices** of WP3: grid, resolution, aggregation window, how many events go
  to which split, event-level cross-validation.
- **The evaluation breakdowns** of WP5: by rain intensity, link density and storm type;
  `core.maps.map_skill` gives the metrics, the breakdown is the project's.
- **Visualization** (WP6) and the report.

If a piece of the base is missing, it belongs in `core/` with tests, not in a notebook.

## Layout

```
src/learned_2d/baselines.py   ordinary kriging of the links; IDW / OK / GMZ maps of a dataset
src/learned_2d/unet.py        the minimal U-Net: transform, masked training loop, prediction
tests/test_learned_2d.py      kriging and U-Net on synthetic data
notebooks/                    01-04 above, committed with their outputs
```
