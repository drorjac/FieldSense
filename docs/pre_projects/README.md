# Proposed projects: what FieldSense gives them

This folder holds proposals for projects that build on FieldSense: a 1-D
project on the laws of link attenuation (`1d_project.md`), a 2-D project on
learned rain maps (`2d_project.md`), and more to come.

**FieldSense does not implement these projects.** The research is the
project's own work. FieldSense's job is to make sure each one starts on solid
ground: the data load, the chain runs, a baseline is scored, and a starter
notebook shows where the project's method plugs in. A project should spend its
time on its question, not on rebuilding data loaders, power laws or scoring.

## What "a solid base" means

For every proposal, FieldSense provides five things:

| # | Base | Where it lives |
|---|---|---|
| 1 | **Data**: every dataset the proposal names loads with one call, documented in `DATA.md` | `core/data_paths.py`, `core/opensense/` |
| 2 | **The chain up to the project's input**: processing, retrieval, maps, whatever the project consumes, tested | `core/` |
| 3 | **A baseline with scores**: the standard method the project must beat, scored the way the proposal will be judged | `core/` + the starter notebook |
| 4 | **Known truth**: a simulated version of the problem, so the method can be checked before real data | `core/simulation/` |
| 5 | **Starter notebooks**: load the data, build inputs and targets, run the baseline, score it, and stop where the project's method goes | `examples/<project>/`, or the project's folder under `projects/<stage>/` |

If a proposal needs a piece that is missing, the piece is added to `core/`
with tests. It is not written inside a starter notebook.

## Starter notebooks

- **Purpose:** a student or collaborator opens one and has a working pipeline
  in minutes. They replace one marked cell with their method.
- **Allowed:** a minimal working example of the method family (a small U-Net
  trained for a few epochs, a PySR run on a toy), so the interface is clear.
  It must run in a few minutes on a laptop.
- **Not allowed:** the project's actual research: tuning, the full
  experiment, conclusions. Those belong to the project.
- **Each notebook ends with:** the baseline's scores, the evaluation function
  the project should reuse, and a short "your method goes here" cell.
- **Difference from `tutorials/`:** tutorials teach a method FieldSense uses.
  Starter notebooks hand over a problem FieldSense does not solve.

## Adding a proposal

1. Put the proposal in this folder (`<name>_project.md`).
2. Add a section below: a table mapping each thing the proposal needs to the
   FieldSense piece that provides it, marked **ready**, **partial** or **gap**.
3. Close the gaps in `core/` (with tests), then write the starter notebooks.
4. The section lists the starter notebooks when they exist.

## 1-D: the laws of microwave-link attenuation (`1d_project.md`)

Learn readable equations for the path law $f(R, L)$ and for the non-rain term
$\delta$ (wet antenna, with memory), with PySR and weak-form SINDy.

| Proposal needs | FieldSense piece | Status |
|---|---|---|
| OpenMRG, OpenRainER | `core/data_paths.py`, `core/opensense/`, tutorial 01 | ready |
| Processing chain: QC, wet/dry, baseline, wet antenna, power law | `core/cml/`, `core/opensense/` (`wet_dry`, `retrieval`, `intercomparison_chain`), tutorial 02 | ready |
| Linear law $aR^bL$ as baseline | `core/cml/power_law.py` (`itu_ab`, `rain_from_attenuation`) | ready |
| Literature wet-antenna models (Schleiss, Pastorek) | `core/cml/estimators.py`, `core/opensense/retrieval.py` | ready |
| Radar along each link, nearest gauge | three separate `radar_along_links` functions | partial: to become one `path_sample` (plan T7) |
| Event split into training and test | `core/events.py` (`detect_events`) | ready |
| Simulator: moving 2-D rain, links of many lengths, noise, quantization | `core/simulation/` (`generators`, `flows`, `cml_network.forward_model`) | ready, but `cml_network` has no tests (plan T13) |
| Exact path integral vs linear law | `cml_network.sample_along_paths`, `path_averaging_bias` | ready |
| $\delta$ with memory (wet antenna that builds up and dries) | `cml_network.wet_antenna_db` is static in rain | **gap**: add a dynamic wet-antenna model to the simulator |
| SINDy and PySR set-up | `projects/physics_ml/discovery` (`01_sindy_basics`, `02_pysr_basics`, `discover_itu.py`) | ready |

Starter notebooks (planned, `examples/1d_path_law/`):
1. `01_simulated_path_law`: simulated links of many lengths; exact integral vs
   $aR^bL$; the gap by length and regime, as a table ready for PySR.
2. `02_real_links_and_radar`: OpenMRG links with radar along the path and the
   nearest gauge; event split; linear-law baseline scored by link length.
3. `03_wet_antenna_tails`: post-event drying tails; $\hat\delta = A - f(R, L)$;
   literature models fitted per link as the baseline for SINDy.

## 2-D: learned rain maps (`2d_project.md`)

Learn a mapping $g_\theta$ from link attenuations and network metadata to a gridded rain field,
supervised by radar; compare with IDW, ordinary kriging and GMZ on held-out storm events, with
RMSE, MAE, PCC (per step and pooled), cumulative event error, POD/FAR/CSI and gauge-point error.
The proposal's data (an Israeli network) is private; the starters use OpenMRG, the open
benchmark of the diffusion-prior paper it cites.

| Proposal needs | FieldSense piece | Status |
|---|---|---|
| CML, radar and gauges, aligned in time (WP2) | `core/opensense/networks.py` (OpenMRG, OpenRainER, OpenMesh on one interface) | ready |
| Attenuation from RSL/TSL, dry baseline, wet antenna (WP2) | `core/cml/` (`link_qc`, `baseline`, `estimators`, `power_law`), tutorial 02 | ready |
| Grid, projection, aggregation window (WP3) | `core/geo.py` (`Grid`), each network's radar grid, hourly | ready |
| Split by storm event, not by sample (WP3) | `core/events.py` (`detect_events`), `core/maps/learning.py` (`split_events`) | ready |
| Links rasterized as input channels; link table with metadata for a GNN; radar target; held-out gauges | `core/maps/learning.py` (`build_network_dataset`, `assemble`, `arrays`), tested | ready |
| Exact path-integral forward operator (for a physics loss or posterior sampling) | `core/maps/learning.py` (`path_weights`, `sample_paths`), `core/cml/power_law.py` (`itu_ab`) | ready |
| Baselines: IDW, GMZ | `core/maps/idw.py`, `core/maps/gmz.py`, tutorial 05 | ready |
| Baseline: ordinary kriging of the links | `projects/maps/learned_2d/src/learned_2d/baselines.py` (pykrige); block kriging only inside radar merging (`core/maps/mergeplg_methods.py`) | partial: project code, not in `core/` |
| KED (candidate family) | `core/maps/mergeplg_methods.py` (`Merger`, `ked`) with radar as drift | partial: drift is the radar only |
| GNN, CNN/U-Net, diffusion prior (candidate families) | a minimal U-Net as interface example: `projects/maps/learned_2d/src/learned_2d/unet.py`, tutorial 08 | the project's work; PyTorch Geometric not installed |
| Metrics of the proposal's Table 1 | `core/maps/scores.py` (RMSE, MAE, bias, POD/FAR/CSI) and `core/maps/map_skill.py` (PCC per step and pooled, event totals, detection at any threshold, gauge-point error, one common-sample table), tested | ready |
| Breakdown by intensity, link density, storm type (WP5) | `map_skill` functions take a mask; `distance_km` and `coverage` are in the dataset | partial: the breakdown itself is the project's |
| Maps with known truth | `core/simulation/` (`scenario`, `sensors`) and `core/maps/learning.py` (`from_synthetic`) | ready |
| Animation over a storm (WP6) | `core/opensense/plots.py`, matplotlib | partial: no animation helper |

Starter notebooks (`projects/maps/learned_2d/notebooks/`, committed with outputs; see
[`projects/maps/learned_2d/README.md`](../../projects/maps/learned_2d/README.md)):
1. `01_dataset`: the OpenMRG dataset, 27 storm events, the input channels, link table, radar
   target, gauges and the split by event.
2. `02_baselines`: IDW, ordinary kriging and GMZ on the test events; the full metric table
   against the radar and at independent gauges; maps of one event.
3. `03_minimal_unet`: a tiny U-Net on the channels, scored with the same table, as an interface
   example; then "your method goes here" with pointers for a GNN and a diffusion prior.
4. `04_simulated_truth`: the same pipeline on simulated storms where the truth is known.
