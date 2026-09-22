# OpenSense Pipeline — Plan

Status: plan committed 2026-09-23, executed in the same session.

## Goal

An end-to-end, reproducible path from **raw open CML data** to **merged rainfall
maps**, using the OpenSense software ecosystem, across two independent datasets
(OpenMRG / Sweden and OpenRainER / Italy) — with a **synthetic-truth benchmark**
in the middle that makes the algorithm comparison meaningful.

## Why the synthetic stage exists

Merge algorithms cannot be ranked on real data alone: there is no ground truth
rainfall field, only other sensors with their own errors. Validating "CML+radar
merge" against gauges measures agreement at ~10 points, which is not the same as
field accuracy.

So the pipeline runs the same algorithms twice:

1. **On synthetic fields with the real network geometry.** The fields come from
   `projects/rainfall_field_sim`, so the truth is known exactly. The CML layout,
   lengths, frequencies and the radar grid come from the real OpenMRG data. This
   gives a defensible ranking of the merge methods, with error attributable to
   the method rather than to reference uncertainty.
2. **On the real data**, where the ranking from step 1 is what justifies the
   choice of method, and gauges serve as an independent (if sparse) check.

## Stages

| # | Stage | Module | Output |
|---|---|---|---|
| 1 | Fetch | `src/fetch.py` | raw archives from Zenodo, resumable, checksummed |
| 2 | Ingest | `src/ingest_openmrg.py`, `src/ingest_openrainer.py` | OpenSense-1.0 `xarray` datasets |
| 3 | Retrieve | `src/cml_retrieval.py` | CML attenuation → path rain rate |
| 4 | Merge | `src/merging.py` | thin wrapper over `mergeplg` methods |
| 5 | Benchmark | `src/synthetic_benchmark.py` | method ranking against known truth |
| 6 | Apply | `src/run_openmrg.py`, `src/run_openrainer.py` | rainfall maps + scores |
| 7 | Visualize | `src/figures.py` | figures in `results/` |

## Methods compared

From `mergeplg`, all sharing the `update()` / `adjust()` contract:

- `InterpolateIDW` — CML only, inverse distance weighting
- `InterpolateOrdinaryKriging` — CML only, block kriging along the link line
- `MergeDifferenceIDW` — radar + CML, additive and multiplicative
- `MergeDifferenceOrdinaryKriging` — radar + CML, kriged difference field
- `MergeKrigingExternalDrift` — radar as external drift

Plus a radar-only and a CML-only baseline so the merge gain is measurable.

## Data

| Dataset | Source | Size | License | Status at plan time |
|---|---|---|---|---|
| OpenMRG (Sweden) | Zenodo 10.5281/zenodo.7107689 | 318 MB zip | CC-BY-SA-4.0 | radar + gauges on disk; **`cml.nc` missing**, must fetch |
| OpenRainER (Italy) | Zenodo 10.5281/zenodo.22829808 | 4.6 GB total | CC-BY-4.0 | nothing on disk; fetch `CML.tar`, `RADrain.tar`, `AWS.tar` (~1.44 GB) |

Large files stay untracked — the root `.gitignore` already excludes `*.nc`,
`*.zip`, `*.tar`. Only code, configs, small derived summaries and figures are
committed.

## Priorities

- **P0** — OpenMRG: fetch, ingest, retrieve, merge, maps. This is the spine.
- **P1** — Synthetic benchmark on the real OpenMRG geometry.
- **P2** — OpenRainER: second dataset, confirms the pipeline is not
  OpenMRG-specific.

If P2 runs out of night, P0+P1 still stand alone as a complete result.

## Environment

The repo's existing venv at `~/enviorments/FieldSense` has
`include-system-site-packages = true`, and system `cftime`/`netCDF4` compiled
against NumPy 1.x collide with the venv's NumPy 2.4.6 — `import netCDF4` fails
there. Rather than mutate a working environment, this project uses an isolated
`.venv` at the repo root (already gitignored), pinned to `numpy<2`.

```bash
.venv/bin/python projects/opensense_pipeline/src/run_openmrg.py
```

## Non-goals

- Not re-implementing merge algorithms — `mergeplg` is the reference.
- Not a nowcasting study; `projects/spatial_interpolation` covers that.
- Not touching `projects/spatial_interpolation` or `projects/physics_ml`.
