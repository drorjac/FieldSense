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
```

There are **two tiers of data acquisition**, and which you want depends on
whether you are exercising the code or doing the science.

### Example subsets — seconds, ~57 MB

The curated subsets the OpenSense community publishes. Already on the
OpenSense-1.0 convention, so no retrieval chain is needed to look at them.

```bash
.venv/bin/python projects/opensense_pipeline/src/example_data.py --list
.venv/bin/python projects/opensense_pipeline/src/example_data.py --dataset openmrg --subset 8d
.venv/bin/python projects/opensense_pipeline/src/example_data.py --all
```

```python
from example_data import load
data = load("openmesh", "20d")     # dict: cml, pws, asos
data["cml"].frequency_ghz          # normalized on load
```

`notebooks/01_read_opensense_data.ipynb` walks through reading all four
datasets and plotting them with `poligrain`, including the units trap below.

| dataset | subsets | components |
|---|---|---|
| `openmrg` | `8d`, `5min_2h` | cml, radar, gauge_municipal, gauge_smhi |
| `openrainer` | `8d` | cml, radar, gauge |
| `openmesh` | `1d`, `1w`, `20d` | cml, pws, asos *(20d only)* |
| `ams_pws` | `full_period` | pws, gauge |

### Full records — 8.4 GB, the actual experiments

```bash
.venv/bin/python projects/opensense_pipeline/src/fetch.py --list        # look, don't fetch
.venv/bin/python projects/opensense_pipeline/src/fetch.py --dataset openmrg
.venv/bin/python projects/opensense_pipeline/src/run_pipeline.py
```

Downloads resume and are md5-verified, and an already-verified file is
skipped, so re-running costs nothing.

### Choosing where the pipeline reads from

```bash
# the full archives already on disk — never touches the network
.venv/bin/python projects/opensense_pipeline/src/run_pipeline.py --source raw --offline

# the small subsets instead; downloads ~57 MB on first use, then cached
.venv/bin/python projects/opensense_pipeline/src/run_pipeline.py --source example
```

`--source raw` (the default) reads the extracted Zenodo archives.
`--source example` runs the identical pipeline on the curated subsets — same
retrieval chain, same merge methods, same figures. `--offline` refuses to
download anything and fails with the command to run instead, so a machine that
already has the data never re-fetches it.

Results are namespaced by source (`openmrg_2_maps.png` vs
`openmrg_2_maps_example.png`), because the subsets cover a different period
from the curated events and would otherwise overwrite them.

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

**Units differ between every source, and are sometimes undeclared.**
OpenRainER's raw files declare `length` in metres and `frequency` in MHz;
OpenMRG's use km and GHz. Feeding metres into `k·L` and MHz into the ITU-R
table retrieves exactly zero rain everywhere — no error, no warning. Worse,
the *example subsets* ship MHz with **no units attribute at all**, and
polarization is spelled `Vertical`, `vertical` or `v` depending on the file.

`conventions.py` holds the full table of who disagrees with whom, reads the
declared units where present, and falls back on magnitude where absent —
nothing terrestrial transmits at 7,456 GHz, so that value is MHz.

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

## Checking the retrieval against OpenSense's own

The retrieval had no independent check: it was scored against rain gauges,
which are sparse point sensors measuring something different from a path
average, so a disagreement could not be attributed to either side.

The OpenMRG `8d` subset closes that gap. It ships raw `tsl`/`rsl` **and** a
reference rain rate `R` that the OpenSense community retrieved from exactly
those signals — 364 links at 10 s over 2015-07-22 to 07-29, a window that
contains the Torslanda event. Both retrievals see identical input.

```bash
.venv/bin/python projects/opensense_pipeline/src/validate_retrieval.py --offline
.venv/bin/python projects/opensense_pipeline/src/validate_retrieval.py --sweep
```

**The chain has the right skill and the wrong magnitude.** Over the full 8 days
(22.5 M link-timesteps):

| | ours | reference |
|---|---|---|
| correlation with each other | 0.878 | — |
| wet/dry agreement | 82.7% | — |
| mean rain | 0.490 mm/h | 0.266 mm/h |
| ratio to municipal gauges, matched pairs | **2.28** | **1.05** |
| correlation with gauges | 0.72 | 0.74 |

The reference is essentially unbiased against gauges; ours over-reads by a
factor of ~2, with the same correlation. `retrieval_vs_reference_torslanda.png`
shows why: the time series have the same shape and timing, uniformly scaled up,
and the exceedance curves run parallel. That is a calibration error, not a
skill error.

### The wet-antenna default is not transferable

`--sweep` varies the wet-antenna term against the reference. On the Torslanda
window correlation *peaks* at 1.5–2.0 dB and the ratio reaches 1.0 at 2.3 dB —
the Schleiss et al. (2013) literature value:

| waa_max_db | ratio to reference (8d) | ratio (Torslanda) | corr |
|---|---|---|---|
| 0.5 *(current default)* | 1.75 | 1.74 | 0.896 |
| 1.5 | 1.11 | 1.25 | 0.907 |
| 2.3 *(literature)* | 0.78 | 0.98 | 0.906 |

But the same sweep against the aug25 **gauges** pulls the other way: 0.5 dB
gives 0.79 and 1.5 dB gives 0.53. No single static value satisfies both, and
the spread across events is about 2× (aug25 0.79, Torslanda 1.48, 8-day 2.28
at the current default).

Two things are worth saying plainly. The 0.5 dB default was fitted to one
event against 10 gauges — 200 paired values — and does not generalise; the
reference comparison is 22.5 M paired values and favours 1.5–2.3 dB, which is
also where the literature sits. And a static saturating wet-antenna model
cannot track an effect that depends on the wetting and drying history of the
radome, which is the likely reason no constant works everywhere.

**The default is left at 0.5 dB** so the results elsewhere in this README stay
reproducible. Treat the magnitude as uncertain to a factor of ~2 and the
ranking of merge methods — which is what the pipeline exists to establish — as
unaffected, since every method receives the same CML input.

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

Field RMSE (mm/h), averaged over 3 field realizations. "⚠" marks methods
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

Against synthetic truth, **merging beats radar-only in every regime**, and
beats CML-only by more. **Additive difference IDW** is the method to reach for:
top-two on two of three regimes, never worst, never unstable.

**Multiplicative merging is a coin flip.** Best of all methods on the frontal
band (4.10), catastrophic on convective cells — RMSE 41,022 mm/h with 8.6% of
pixels physically impossible. It divides by the radar field, so one near-zero
radar pixel under a raining link sends the estimate to five figures. The
artifacts are visible as dark blotches in its panel of `openmrg_2_maps.png`.

### Real data — the ranking does not hold

Scored against held-out gauges, pooled over the 20 wettest timesteps:

| | radar only | best CML-only | best merged | winner |
|---|---|---|---|---|
| **OpenMRG** (n=200) | RMSE 6.16, r=0.35 | RMSE **4.45**, r=0.66 | RMSE 4.69, r=0.64 | CML |
| **OpenRainER** (n=4,642) | RMSE **8.62**, r=0.72 | RMSE 10.50, r=0.35 | RMSE 8.98, r=0.58 | radar |

On real data **merging never beat the better single sensor**. It landed between
its two inputs in both cases — which is what a hedge does.

The two datasets are near mirror images. The Swedish radar is poor against
gauges (r=0.35) while its dense 364-link network is good (r=0.66); the Italian
radar is good (r=0.72) while its sparse 151-link network is poor (r=0.35). The
correlation gap is −0.31 one way and +0.37 the other, and in both cases the
better sensor wins outright.

**So the practical answer is: find out which of your sensors is better before
merging.** Merging is insurance against the possibility that it is the other
one, not a free improvement. That is a different claim from the one the
synthetic benchmark supports, and the disagreement is the most useful thing
this pipeline produced.

### Why synthetic and real disagree

The synthetic benchmark imposed a ~35% low radar bias with a relatively clean
CML retrieval — errors that are *complementary*, which is the situation
merging is designed for and where it duly wins. Real error structure is less
obliging: the Italian radar has little bias to correct (−1.05 mm/h), so
merging mostly injects CML noise into an already-good field.

This is a caveat on synthetic benchmarking generally, this one included: **its
ranking is only as good as the error model you assume.** The synthetic result
here describes the OpenMRG regime well and the OpenRainER regime not at all.

### A hypothesis that turned out to be wrong

The obvious explanation for OpenRainER was geometry: only 22% of its gauges
lie within 2 km of a link (median 5.9 km, max 30.5 km), against 100% within
0.7 km for OpenMRG. So merging should help near the network and hurt far from
it.

It does not. Stratifying the OpenRainER gauges by distance to the nearest link
path (`openrainer_4_coverage.png`), radar-only wins in **every** band,
including 0–2 km where 1,420 gauge-timesteps sit right beside links:

| band | n | best method | radar-only RMSE |
|---|---|---|---|
| 0–2 km | 1,420 | Radar only, 8.85 | 8.85 |
| 2–5 km | 1,380 | Radar only, 8.53 | 8.53 |
| 5–10 km | 1,740 | Radar only, 6.67 | 6.67 |
| >10 km | 1,840 | Radar only, 9.99 | 9.99 |

The figure is kept because it is the evidence that ruled the explanation out.
Relative sensor skill, not network geometry, is what decides this.

## Figures

| file | what it shows |
|---|---|
| `benchmark_ranking.png` | the synthetic ranking, per regime |
| `*_1_sensors.png` | the three sensor geometries — grid, lines, points |
| `*_2_maps.png` | one rainfall map per method |
| `*_3_gauge_validation.png` | each method against held-out gauges |
| `*_4_coverage.png` | RMSE by gauge distance to the link network |

## Layout

```
opensense_pipeline/
├── src/
│   ├── fetch.py                # Zenodo full records, resumable + verified
│   ├── example_data.py         # curated OpenSense subsets (ported from poligrain)
│   ├── conventions.py          # unit / polarization normalization across sources
│   ├── retrieval.py            # the CML retrieval chain, source-independent
│   ├── validate_retrieval.py   # ours vs the OpenSense reference retrieval
│   ├── ingest_openmrg.py       # raw TSL/RSL -> OpenSense-1.0 + retrieval
│   ├── ingest_openrainer.py    # same, second dataset
│   ├── merging.py              # uniform wrapper over mergeplg + baselines
│   ├── synthetic_benchmark.py  # real geometry, synthetic truth
│   ├── plots.py                # figures
│   └── run_pipeline.py         # entry point
├── notebooks/
│   └── 01_read_opensense_data.ipynb   # reading all four datasets
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
- Runtime is dominated by block kriging, which scales with grid cells x links.
  OpenRainER's full 160 x 285 grid takes ~35 min for a 20-timestep
  validation, so `--val-max-cells` (default 6000) coarsens the grid for the
  pooled validation only; the published maps stay at full resolution and the
  coarsening applies identically to every method.
