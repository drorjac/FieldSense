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
| 5 | **Starter notebooks**: load the data, build inputs and targets, run the baseline, score it, and stop where the project's method goes | `examples/<project>/` |

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

Proposal not written yet. Likely base, to be checked against the proposal:

| Likely need | FieldSense piece | Status |
|---|---|---|
| Classical maps as baselines: IDW, GMZ, kriging, KED, merging | `core/maps/`, tutorials 05 and 06 | ready |
| Every sensor on one grid as a training set | `multisensor_nowcasting` `cube.py`, tutorial 09 | partial: project code, not in `core/` |
| A training loop without leakage, a U-Net | tutorial 08 | partial: tutorial code, not in `core/` |
| Maps with known truth | `core/simulation/` (`scenario`, `sensors`, `benchmark`), tutorial 10 | ready |
| Scoring against gauges and radar | `core/maps/scores.py` | ready (five scoring copies to merge, plan T10) |

Starter notebooks: to be listed once the proposal is written.
