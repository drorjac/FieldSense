# Reorganization and mapping plan

Status (2026-10-03): T0-T20 done, T21 (the PR) pending; algorithm audit merged (`docs/algorithm_audit.md`) on branch `reorg`; see the notes under each task.

One continuous run on one branch (`reorg`), no pauses between parts. Commit
after each task so any task can be reverted on its own; one PR at the end.
Stop only if a check fails and the cause is not clear.

## Organization

### Target layout

```
projects/
├── data/
│   └── openmesh_nyc/            ← openmesh_nyc
├── retrieval/
│   ├── openmrg/                 ← cml_retrieval
│   └── rnn_three_networks/      ← cml_rnn
├── maps/
│   ├── multisensor/             ← multisensor_maps
│   ├── radar_adjustment/        ← radar_adjustment
│   ├── nyc/                     ← nyc_rain_maps
│   └── archive_pipeline/        ← opensense_pipeline (frozen)
├── nowcasting/
│   ├── pysteps/                 ← os_nowcasting
│   └── multisensor/             ← multisensor_nowcasting
├── simulation/
│   ├── regimes/                 ← rainfall_field_sim
│   └── testbed/                 ← synthetic_testbed
├── physics_ml/                  (unchanged)
└── spatial_interpolation/       (unchanged; listed under nowcasting)
```

### Rules

- Each stage folder has a `README.md`: its question, a table of subprojects,
  the headline numbers. The top-level README follows the chain: data →
  retrieval → maps → nowcasting, plus simulation and physics_ml.
- Package names in `src/` stay (`multisensor_maps`, `os_nowcasting`, ...).
- Cache folders `dataset/open_datasets/_*` keep their names; nothing recomputes.
- `git mv` for committed folders; uncommitted ones are moved, then committed
  in their new place, so the old names never enter history.
- `spatial_interpolation` is not touched (its own rules). `opensense_pipeline`
  is frozen: paths fixed so it runs, nothing else.
- Repo root is always `core.data_paths.REPO_ROOT`, never `parents[N]`.

## Tasks

Each task ends with the checks in the last section, then a commit.

### Baseline
- [x] T0. Branch `reorg`. Save `pytest` pass/fail list and checksums of every
      `projects/*/results/` to the scratch folder.

### Move
- [x] T1. Create `data/ retrieval/ maps/ nowcasting/ simulation/`; move the 12
      subprojects per the layout.
- [x] T2. Replace depth-based repo roots with `core.data_paths.REPO_ROOT`
      (14 scripts: `opensense_pipeline/src/*` x8, `physics_ml/src/discover_*.py`,
      `cml_retrieval/src/rain_maps.py`, `rainfall_field_sim/src/run_demo.py`, ...).
- [x] T3. Fix cross-project paths: `multisensor_nowcasting/.../links.py:64`
      (→ `maps/multisensor/results`), 4 `multisensor_nowcasting` notebooks and
      tutorial 09 (`ROOT / "projects" / "multisensor_nowcasting"`), the
      `radar_adjustment` notebook fallback path.
- [x] T4. `pyproject.toml` testpaths; `python projects/<old>/src/run.py` usage
      lines in docstrings.
- [x] T5. Links in `README.md`, `GETTING_STARTED.md`, `CONTRIBUTING.md`,
      `DATA.md`, `core/README.md`, `core/radar/MRMS.md`,
      `core/scientific_packages/UPSTREAM_ISSUES.md`, `dataset/README.md`,
      `dataset/open_datasets/*/README.md`, `tutorials/` (README + 8 notebooks),
      `docs/references.md`, and project READMEs.
- [x] T6. Stage READMEs and the top-level README by stage.

### One copy of each 2-D piece in `core/maps`
- [x] T7. `core/maps/geometry.py`: one `path_sample` (rain along each link,
      replaces about 7 copies incl. the three `radar_along_links`) and one
      `distance_to_links` (replaces about 7 copies).
- [x] T8. One IDW: `core/maps/idw.py`; point `core/maps/merge._interp`,
      `multisensor_nowcasting/cube.py:91` and `core/simulation/reconstruct._idw`
      at it.
- [x] T9. Move into `core/maps`: `interpolate`, `points_map` from
      `multisensor_maps/event.py`; `points_to_map` from `nyc_rain_maps/compare.py`;
      `merged_map` from `os_nowcasting/products.py`; the variogram fit from
      `radar_adjustment/extend.py` (merge with `mergeplg_methods.fit_radar_variogram`).
- [x] T10. One scoring module: `core/maps/scores.py` takes over
      `reconstruct.score`, `radar_adjustment/score.py` and
      `core/opensense/evaluation.rainfall_metrics` (keep names as thin aliases
      where notebooks call them).
- [x] T11. `radar_adjustment` calls `core/maps/mergeplg_methods.Merger` instead
      of mergeplg directly.
- [x] T12. Tests: `merge_idw`, `observations`, `compare_maps`, `compare_links`,
      `path_sample`, `distance_to_links`.

**Notes on T7-T12.** Every change keeps results identical (old and new code compared).
Left apart on purpose, because sharing them would change numbers or meaning:
`core/simulation/reconstruct._idw` (smoothed, weighted IDW on the simulation grid);
`core/simulation/benchmark.link_distance_km` (km-unit simulation geometry); the nowcasting
cube's `path_weights` (a sparse matrix, not a field average); the four scorers (T10: their
definitions differ on purpose, documented in `core/maps/scores.py`); `radar_adjustment`
calling mergeplg directly (T11: it reproduces the OpenSense intercomparison, which needs
mergeplg's own API). `physics_ml` is now a stage folder (`physics_ml/discovery`).

### 1-D
- [x] T13. Tests: `cml_network.forward_model`, `path_averaging_bias`,
      `retrieve_rain`, `reconstruct.idw_path`, `fields_1d.metagaussian_1d`,
      `cascade_1d`, `series_stats`.
- [x] T14. Study in `simulation/regimes`: path-averaging bias against
      along-path variability (correlation length, wet fraction, intermittency)
      for each network's link lengths and frequencies. Output: an error model,
      bias and spread per link given length, frequency and regime.
- [x] T15. Real-data check on OpenMRG and NYC: radar along each link
      (`path_sample`) against link rain; does the error model predict the
      scatter?
- [x] T16. Feed the per-link spread into merging as observation error
      (KED/kriging noise term); compare with the current merged maps.

T13-T15 were done as the 1-D proposal's base (`physics_ml/path_law_1d`, `core/simulation/wet_antenna.py`, `tests/test_simulation_1d.py`). T16 became `maps/link_weights`: the error by length learned on real storms (path averaging alone is a small part of it), used as IDW weights.

### 2-D studies (in `maps/`)
- [x] T17. Maps that can be dry: wet/dry mask from links and PWS before
      interpolation, and indicator kriging; score on the synthetic truth, then
      on the three networks.
- [x] T18. (done as `maps/netherlands`: RAINLINK on the whole network, JJA 2012, against KNMI hourly gauges; KNMI adjusted radar needs an API key) Add `CML_Netherlands` to the multi-network comparison.
- [x] T19. Adjusted ARPAE radar as the Emilia-Romagna baseline in
      `radar_adjustment`.

### Finish

**T20 result (2026-10-03).** All 63 notebooks outside `spatial_interpolation` executed without error from their new folders (re-run into a scratch copy; the committed outputs were kept). The numbers they print match the committed ones to 0.1% in 56; the other 7 differ only in run-time lines, third-decimal jitter in two `path_law_1d` tables, PySR's stochastic search (`02_pysr_basics`), and live external data (`wu_pipeline`, one day fewer in `nexrad_rain_vs_snow`'s IEM listing).

- [x] T20. Re-run every notebook (except `spatial_interpolation`); numbers
      match the baseline or the difference is explained in the PR.
- [ ] T21. Final check below, then the PR.

## Checks after every task

- `pytest` matches the baseline (new tests pass).
- `grep -rn "projects/<old_name>"` finds nothing outside this file.
- Notebooks touched by the task run their setup cells (imports, paths, data
  load) from their `notebooks/` folder.
- `results/` checksums unchanged, except where a study task writes new results.
- Before the PR: `git log reorg --format=%B` and `git grep` contain no
  assistant attribution (see the local rules).

## Estimate

| Part | Wall clock |
|---|---|
| T0–T6 move | 4–6 h |
| T7–T12 core/maps | 6–8 h |
| T13–T16 1-D | 1–2 days |
| T17–T19 2-D studies | 1–2 days |
| T20–T21 re-runs, PR | half a day |
