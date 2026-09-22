# OpenSense Pipeline — from raw CML data to merged rainfall maps

End-to-end path from **published open data** to **merged rainfall maps**, built
on the [OpenSense](https://opensenseaction.eu/) software ecosystem
([`poligrain`](https://github.com/OpenSenseAction/poligrain),
[`mergeplg`](https://github.com/OpenSenseAction/mergeplg)) and run on two
independent datasets from two countries.

```
fetch  →  ingest  →  CML retrieval  →  [synthetic benchmark]  →  merge  →  maps
```

The bracketed stage is the point of the design. See *Why a synthetic stage*.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r projects/opensense_pipeline/requirements.txt

.venv/bin/python projects/opensense_pipeline/src/fetch.py --dataset openmrg
.venv/bin/python projects/opensense_pipeline/src/fetch.py --dataset openrainer
.venv/bin/python projects/opensense_pipeline/src/run_pipeline.py
```

`fetch.py --list` shows what each Zenodo record holds without downloading.
Downloads resume and are md5-verified against the published checksum, so
re-running is cheap. `run_pipeline.py --skip-benchmark` runs only the
application half.

## The data

| | OpenMRG (Sweden) | OpenRainER (Italy) |
|---|---|---|
| region | Gothenburg | Emilia-Romagna |
| DOI | [10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689) | [10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808) |
| license | CC-BY-SA-4.0 | CC-BY-4.0 |
| CML | 728 sublinks → 364 links, raw TSL/RSL at **10 s** | 151 links × 2 sublinks, raw TSL/RSL at **1 min** |
| radar | 48 × 37 at ~2 km, 5 min, pseudo-dBZ | 290 × 373 at ~1 km, 15 min, mm |
| gauges | 10 municipal | 319 AWS |
| download | 318 MB zip (4.6 GB unpacked) | 1.44 GB of 4.6 GB |

The two have **opposite sensor balances** — OpenMRG is link-rich and
gauge-poor, OpenRainER the reverse — which is exactly why running both is
worth the effort.

Neither ships rainfall. Both ship raw transmitted and received power, so the
retrieval chain is part of the pipeline, not a preprocessing step someone else
did.

## CML retrieval

Per sublink: total loss `A = TSL − RSL` → rolling-standard-deviation wet/dry
→ dry-weather baseline → wet-antenna removal → `R = (A_rain/(kL))^(1/α)` with
ITU-R P.838-3 coefficients from `core/itu_p838.py` → average the two
directions of each link.

**Validation against gauges**, matched pairs within 3 km (OpenMRG) / 5 km
(OpenRainER), same retrieval parameters for both:

| | median CML/gauge ratio | corr(CML, gauge) | corr(radar, gauge) |
|---|---|---|---|
| OpenMRG, 25 Aug 2015 | 0.79 | 0.80 | 0.52 |
| OpenRainER, 26 Sep 2021 | **1.01** | 0.81 | 0.60 |

On both datasets the CML retrieval tracks the gauges **better than the radar
does**, which is the entire premise of opportunistic sensing.

### Three traps worth recording

These each produced a silently wrong answer rather than an error.

**Radar was being scaled twice.** OpenMRG stores pseudo-dBZ with
`scale_factor=0.4`, `add_offset=-30`, and the dataset readme documents the
transform — but xarray applies it automatically on read. Applying it again
drove the radar field to a flat 0.0 mm/h while gauges read 24 mm/h. The file
also carries its own Z-R coefficients (`zr_b = 1.5`, not the textbook 1.6),
now preferred over the default.

**Units differ between the two datasets.** OpenRainER declares `length` in
metres and `frequency` in MHz; OpenMRG uses km and GHz. Feeding metres into
`k·L` and MHz into the ITU-R table retrieves exactly zero rain everywhere —
no error, no warning. The loaders now read the declared `units` attribute
rather than assuming a convention.

**Zeroing rain on the dry flag deletes steady rain.** The rolling-standard-
deviation classifier keys on *fluctuation*, and widespread stratiform rain
attenuates steadily, so hours of real rain get labelled dry. Subtracting the
baseline everywhere is sufficient; that alone moved CML/gauge totals from 0.24
to 0.65.

### What actually sets the magnitude

Two parameters dominate, and both are recorded with their sensitivity rather
than quietly tuned. Measured on OpenMRG, 25 August:

| baseline window | wet-antenna max | CML/gauge ratio | corr |
|---|---|---|---|
| 1 h | 0.5 dB | 0.68 | 0.80 |
| 3 h | 0.0 dB | 0.94 | 0.80 |
| 3 h | **0.5 dB** ← used | **0.79** | **0.79** |
| 3 h | 1.0 dB | 0.64 | 0.78 |
| 3 h | 2.3 dB (Schleiss et al. 2013) | 0.35 | 0.75 |

A one-hour centred median absorbs a multi-hour event into its own dry
reference. And the literature 2.3 dB wet-antenna value is far too aggressive
for this network: a 2 km link at 23 GHz develops only ~2.7 dB of rain
attenuation at 10 mm/h, so subtracting 2.3 dB removes most of the signal.

Correlation is flat at 0.75–0.80 across the entire range. **These knobs move
magnitude, never skill** — which is why the merge comparison downstream is
robust to them.

## Why a synthetic stage

Merge methods cannot be honestly ranked on real data. There is no true
rainfall field — only other sensors, each with its own error. Scoring against
gauges compares a field to ten points (OpenMRG), says nothing about the space
between them, and is circular whenever the gauges were an input.

So the methods are scored first where truth is exact:

- **rainfall** — synthetic, from `projects/rainfall_field_sim`: stratiform,
  convective and frontal regimes, rescaled to the mesoscale domain the radar
  actually covers.
- **sampling** — real: the actual OpenMRG link endpoints, lengths,
  frequencies and polarizations, and the actual radar grid.
- **degradation** — physical: CMLs get the ITU-R forward model and back
  (so path-averaging bias survives), radar gets beam smoothing, a
  range-dependent multiplicative bias and multiplicative noise.

A note on that last point, because it decided the result. An earlier version
simulated an **unbiased** radar. With nothing to correct, radar-only won every
regime by construction and the benchmark measured nothing. Radar QPE bias is
the entire reason merging exists, so the synthetic radar now reads ~35% low
with the bias worsening by range.

## Results

### Synthetic truth — the ranking

Field RMSE (mm/h), averaged over 3 field realizations. "Blow-up" marks methods
producing physically impossible rates (> 300 mm/h).

| method | stratiform | convective | frontal |
|---|---|---|---|
| Merge: difference IDW (additive) | 0.98 | **4.74** | 6.64 |
| Merge: difference kriging (additive) | **0.97** | 5.09 | 6.77 |
| Merge: difference IDW (multiplicative) | 0.98 | 41,022 ⚠ | **4.10** |
| Merge: kriging with external drift | 1.16 | 53.1 ⚠ | 5.81 |
| Radar only | 1.65 | 5.81 | 6.29 |
| CML only, IDW | 1.55 | 7.85 | 10.42 |
| CML only, block kriging | 1.64 | 8.67 | 10.79 |

**Merging beats radar-only in every regime**, and beats CML-only by more.
That is the result the pipeline exists to establish.

**Additive difference IDW is the method to reach for.** It is top-two on two
of three regimes, never worst, and never blows up.

**Multiplicative merging is a coin flip.** Best of all methods on the frontal
band (4.10), catastrophic on convective cells — RMSE 41,022 mm/h with 8.6% of
pixels physically impossible. It divides by the radar field, so one near-zero
radar pixel under a raining link sends the estimate to five figures. The
artifacts are visible as dark blotches in the top-left of its panel in
`openmrg_2_maps.png`.

**Kriging with external drift underperforms** consistently here and
occasionally blows up, despite being the most sophisticated method on offer.

## Figures

| file | what it shows |
|---|---|
| `benchmark_ranking.png` | the synthetic ranking, per regime |
| `*_1_sensors.png` | the three sensor geometries — grid, lines, points |
| `*_2_maps.png` | one rainfall map per method |
| `*_3_gauge_validation.png` | each method against held-out gauges |

## Layout

```
opensense_pipeline/
├── src/
│   ├── fetch.py                # Zenodo download, resumable + verified
│   ├── ingest_openmrg.py       # raw TSL/RSL -> OpenSense-1.0 + retrieval
│   ├── ingest_openrainer.py    # same, second dataset
│   ├── merging.py              # uniform wrapper over mergeplg + baselines
│   ├── synthetic_benchmark.py  # real geometry, synthetic truth
│   ├── plots.py                # figures
│   └── run_pipeline.py         # entry point
├── results/
├── PLAN.md
└── README.md
```

## Notes

- Raw archives and derived netCDFs are gitignored; `fetch.py` and the ingest
  modules reproduce them.
- This project uses an isolated `.venv` at the repo root. The pre-existing
  venv at `~/enviorments/FieldSense` sets
  `include-system-site-packages = true`, and system `cftime`/`netCDF4` built
  against NumPy 1.x collide with its NumPy 2.4.6 — `import netCDF4` fails
  there.
- `poligrain`'s `GridAtPoints.__call__` reads `da_point_data.lon`
  unconditionally, even when constructed for projected coordinates, so gauge
  arrays must carry lon/lat even though nothing reads them numerically.
- Every number here comes from `run_pipeline.py`; all randomness is seeded.
