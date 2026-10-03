# radar_adjustment: the OpenSense radar-adjustment intercomparison, reproduced and extended

OpenSense's [`radar_adjustment_intercomparison`](https://github.com/OpenSenseAction/radar_adjustment_intercomparison)
compares the radar-CML merging methods of [`mergeplg`](https://github.com/OpenSenseAction/mergeplg)
on two open networks: weather radar adjusted with commercial microwave links, hourly, over
three summer months, scored at rain gauges that no method uses. This project runs that study
again from the raw archives, checks it against the table the OpenSense notebooks print, and
asks what it leaves out: weather stations as adjusters, maps without radar, the DWD's RADOLAN
method, the variogram, and New York City.

| network | links | radar | independent gauges | period |
|---|---|---|---|---|
| Gothenburg, OpenMRG | 364 links (350 after QC), TSL+RSL | SMHI C-band, 5 min, 2 km | 10 municipal + 1 SMHI | JJA 2015 |
| Emilia-Romagna, OpenRainER | 151 links (127 after QC), TSL+RSL, 1 min | ARPAE composite, 15 min, ~1 km | 319 ARPAE | JJA 2022 |

Data: OpenMRG [doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689)
(Andersson et al., 2022) and OpenRainER [doi:10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808),
read from the shared store (see [`DATA.md`](../../../DATA.md)); for the extensions, the OpenMRG2
Netatmo PWS preview and New York's OpenMesh links, WU PWS, MRMS and ASOS. Every series is
hour-ending (UTC), built as the OpenSense notebooks build it.

## In short

- **The published result reproduces exactly**, from the raw archives: the radar row to six
  decimals, and all eight adjustments within 0.0006 in RMSE and 0.0003 in correlation - but
  only with a newer `mergeplg` than the commit the OpenSense repository records. With that
  commit the point methods still match; IDW and the whole-line (block) methods do not.
- **Every adjustment beats the radar at independent gauges, by little in Gothenburg and more in
  Emilia-Romagna.** Hourly RMSE: Gothenburg 1.44 mm (radar) to 1.27 (additive block kriging);
  Emilia-Romagna 3.58 to 3.08 (block KED), with the radar's +107% bias down to -7%.
- **The range checks decide whether multiplicative adjustment works.** Without them it breaks
  down in Emilia-Romagna (RMSE 15.5-16.0); with conservative checks it is the best method in
  Gothenburg (1.24).
- **The gain is a near-link effect.** In Emilia-Romagna, block KED cuts RMSE by 22% at gauges
  within 2 km of a link and by 5-16% further away. At 10 mm/h and more, the adjustments do no
  better than the radar, and KED does worse (10.2 vs 8.5).
- **Beyond the OpenSense set:**
  - *Gothenburg:* an IDW map of 26 quality-controlled Netatmo PWS, with no radar, is the best
    product (RMSE 1.01, r 0.82). RADOLAN with the links (1.19) beats all eight methods of the
    intercomparison. A link map without radar (1.27) does as well as those eight methods
    (best 1.27).
  - *Emilia-Romagna:* the links alone, with no radar, cannot map the region (RMSE 4.1-5.1).
  - *New York:* MRMS radar-only adjusted with the WU PWS by RADOLAN (1.39) beats NOAA's
    gauge-corrected MRMS (1.59). Adding links to the PWS makes every product worse.
  - *Variogram:* mergeplg's 5-km default costs 0.05-0.11 RMSE. A variogram fitted to the radar
    (range 60-68 km) and the nugget from the link geometry change nothing against the study's
    30 km.

# The replication

## Pipeline

Every step follows the OpenSense notebooks, on the same raw files (via `core.data_paths`):

1. **Inputs** (`1_data_preparation`, `prepare.py`). OpenMRG: the 10-s CML record taken to
   1 min with the first valid sample of each minute (the OpenSense transformer's
   `resample("1min").first()`); the radar's dBZ turned into rain with
   `(10**(dBZ/10)/200)**(5/8)`; city gauges summed to 15 min, then every source made
   hour-ending by the mean of its sub-hourly depths times the steps per hour; radar of
   0.01 mm or less set to 0. OpenRainER: the monthly CML, AWS and 15-min radar files, links
   and gauges present in all three months.
2. **CML chain** (`2_cml_processing`, `core/opensense/intercomparison_chain.py`): total loss,
   a ±30 dB channel-difference spike filter, three link-QC flags on the whole period (diurnal
   cycle, noisy, flat), wet/dry from the radar along the path, a constant baseline over wet
   spells, the Pastorek (2021) wet-antenna model (`A_max=6, zeta=0.7, d=0.15`), the ITU k-R
   relation, hourly totals and the mean of the two channels.
3. **Adjustment** (`3_adjust_radar`, `3b_adjust_radar`, `adjust.py`): eight products, each
   built per month and called hour by hour with the links only: additive and multiplicative
   difference IDW (`p_idw`), difference ordinary kriging of points (`p_ok`) and of whole lines
   (`b_ok`, block kriging), and kriging with external drift of points and lines (`ked_p`,
   `ked_b`); 12 nearest observations, spherical variogram (sill 1, range 30 km, nugget 0.3).
   Each with three range-check settings: the published run (`default`: |link - radar| <= 10 mm,
   link/radar in 0.1-15), conservative (`cc`: 5 mm, 0.2-8) and none (`nc`); additive methods and
   KED take the difference check, multiplicative ones the ratio check.
4. **Scores** (`4_analysis`, `score.py`): the adjusted field at each gauge's cell (poligrain
   `GridAtPoints`, nearest), all gauge-hours pooled, poligrain's `calculate_rainfall_metrics`
   with a 0.2 mm threshold on both sides; and, beyond the notebook, by intensity class of the
   gauge value, per gauge, and by the gauge's distance to the nearest link.

Two versions of `mergeplg` are run, both in `.venv-mergeplg-main`:

- `pinned` - commit `9894b9c`, the submodule the OpenSense repository records;
- `main` - the commit `pyproject.toml` pins as `mergeplg-main` (`dd380b1`).

They differ in two ways that matter: `main`'s block kriging and block KED use another
right-hand side (`0.5 (var_within + nugget)` for `var_within`; the same for points), and its
IDW default is `standard` (1/d²) where `9894b9c`'s is `radolan`. The notebooks pass no
`idw_method`, so each version runs with its own default.

## Does it reproduce?

Hourly, at the 11 Gothenburg gauges (24,277 gauge-hours), range checks of `3_adjust_radar`;
`published` is the table printed in `4_analysis.ipynb` of the OpenSense repository:

| product | RMSE published | pinned | main | r published | pinned | main | bias % published | pinned | main |
|---|---|---|---|---|---|---|---|---|---|
| radar | 1.4401 | 1.4401 | 1.4401 | 0.5524 | 0.5524 | 0.5524 | -18.81 | -18.81 | -18.81 |
| add. IDW | 1.3057 | 1.2774 | **1.3057** | 0.6826 | 0.6932 | **0.6826** | -10.18 | -10.70 | **-10.18** |
| add. point OK | 1.2666 | **1.2666** | **1.2666** | 0.6960 | **0.6960** | **0.6960** | -10.31 | **-10.31** | **-10.31** |
| add. block OK | 1.2662 | 1.2666 | **1.2662** | 0.6966 | 0.6952 | **0.6966** | -10.39 | -12.72 | **-10.39** |
| point KED | 1.2902 | **1.2902** | **1.2902** | 0.6869 | **0.6869** | **0.6869** | -11.28 | **-11.28** | **-11.28** |
| block KED | 1.2897 | 1.2891 | **1.2903** | 0.6876 | 0.6861 | **0.6873** | -11.34 | -13.38 | **-11.35** |
| mul. IDW | 1.3180 | 1.2794 | **1.3180** | 0.6775 | 0.6914 | **0.6775** | -6.68 | -7.29 | **-6.68** |
| mul. point OK | 1.2830 | **1.2830** | **1.2830** | 0.6918 | **0.6918** | **0.6918** | -6.20 | **-6.20** | **-6.20** |
| mul. block OK | 1.2822 | 1.2795 | **1.2822** | 0.6924 | 0.6904 | **0.6924** | -6.32 | -9.24 | **-6.32** |

- **Inputs:** the radar row matches to every printed digit, so the data preparation, the
  hour-ending conventions and the scoring are the same.
- **Methods:** `main` matches all eight (block KED within 0.0006); `9894b9c` matches only
  the point methods, which do not depend on either change between the two. The published
  numbers were therefore made with a `mergeplg` from November 2025 or later (the right-hand
  side change and the `standard` IDW default both landed then), not with the submodule the
  repository records.
- OpenRainER's table is not printed in the OpenSense notebooks, so only Gothenburg can be
  checked.

# Results

All with `main` (the version that reproduces the published table), hourly, at gauges no method
uses; every table, also for `pinned`, is in [`results/report.md`](results/report.md).

| method | Gothenburg RMSE | r | bias % | Emilia-Romagna RMSE | r | bias % |
|---|---|---|---|---|---|---|
| radar | 1.440 | 0.55 | -18.8 | 3.576 | 0.75 | +107.1 |
| add. IDW | 1.306 | 0.68 | -10.2 | 3.189 | 0.76 | +32.9 |
| add. point OK | 1.267 | 0.70 | -10.3 | 3.196 | 0.76 | +33.5 |
| add. block OK | **1.266** | 0.70 | -10.4 | 3.186 | 0.76 | +32.5 |
| point KED | 1.290 | 0.69 | -11.3 | 3.082 | 0.73 | -6.2 |
| block KED | 1.290 | 0.69 | -11.4 | **3.080** | 0.73 | -7.0 |
| mul. IDW | 1.318 | 0.68 | -6.7 | 3.581 | 0.65 | +24.7 |
| mul. point OK | 1.283 | 0.69 | -6.2 | 3.572 | 0.65 | +27.5 |
| mul. block OK | 1.282 | 0.69 | -6.3 | 3.505 | 0.65 | +24.5 |

(published range checks; 24,277 gauge-hours in Gothenburg, 704,033 in Emilia-Romagna of which
66,287 have no gauge value; RMSE and bias over the gauge-hours where gauge or estimate reach
0.2 mm.)

- **Gothenburg: all eight methods are close.** RMSE spans 1.27-1.32 against the radar's 1.44,
  correlation 0.68-0.70 against 0.55. Kriging is a little better than IDW (1.27 vs 1.31), and
  points and lines give the same results. The links remove about half of the radar's
  underestimate (-19% to -6..-11%).
- **Emilia-Romagna: KED is clearly best.** It is the only method that removes the radar's bias,
  which doubles the rain (+107%), down to -7%. The additive methods keep +33% and the
  multiplicative ones lose correlation (0.65).
- **Range checks** (RMSE; published / conservative / none):

  | | Gothenburg | Emilia-Romagna |
  |---|---|---|
  | add. block OK | 1.27 / 1.30 / 1.27 | 3.19 / 3.41 / 3.90 |
  | block KED | 1.29 / 1.31 / 1.30 | 3.08 / 3.27 / 5.10 |
  | mul. block OK | 1.28 / **1.24** / 1.29 | 3.51 / 3.41 / **15.93** |

  In Gothenburg they hardly matter, except that the conservative ratio check (0.2-8) makes
  the multiplicative methods the best of all (1.234-1.263 with `pinned`, 1.238-1.268 with
  `main`). In Emilia-Romagna, without checks a link that sees rain where the radar sees
  almost none produces huge ratios, and the multiplicative maps break down.
- **By intensity** (main, published checks): in Gothenburg every method improves every class.
  All still underestimate heavy rain, by about half at 10-50 mm/h (radar -71%). In
  Emilia-Romagna the gain is in light and moderate rain: below 2.5 mm/h KED gives RMSE 1.35
  against the radar's 3.18. At 10-50 mm/h no adjustment beats the radar (8.55); the additive
  methods tie with it (8.55-8.59) and KED is worse (10.25).
- **By distance to the nearest link** (Emilia-Romagna; Gothenburg's 11 gauges are all within
  2 km):

  | band | gauges | radar | block KED | change |
  |---|---|---|---|---|
  | 0-2 km | 67 | 3.48 | 2.70 | -22% |
  | 2-5 km | 67 | 3.57 | 3.03 | -15% |
  | 5-10 km | 88 | 3.74 | 3.15 | -16% |
  | 10-20 km | 73 | 3.51 | 3.35 | -5% |
  | >20 km | 24 | 3.39 | 3.02 | -11% |

  The additive methods behave the same way (-18% within 2 km). The multiplicative ones are
  worse than the radar beyond 5 km (3.8-4.0).

# Extensions

All with `main`, scored as above (tables in `results/report.md`, section 5-6).

**Weather stations as adjusters (Gothenburg).** Of OpenMRG2's 30 Netatmo PWS, 26 pass
pypwsqc's faulty-zero, high-influx and station-outlier filters (`core.opensense.pws_qc`,
`results/pws_qc_openmrg.csv`). They serve as point observations; the 11 official gauges stay
independent.

| product | RMSE | r | bias % |
|---|---|---|---|
| IDW map of the PWS, no radar | **1.008** | 0.82 | -13.7 |
| kriging map of the PWS, no radar | 1.030 | 0.81 | -11.7 |
| RADOLAN, links + PWS | 1.177 | 0.72 | -10.2 |
| RADOLAN, links | 1.187 | 0.72 | -9.9 |
| radar adjusted with the PWS (block KED / add. IDW / add. block OK) | 1.21-1.23 | 0.72-0.73 | -16..-19 |
| kriging map of links + PWS, no radar | 1.247 | 0.71 | -10.7 |
| best intercomparison method (add. block OK, links) | 1.266 | 0.70 | -10.4 |
| kriging map of the links, no radar | 1.268 | 0.70 | -10.1 |
| radar | 1.440 | 0.55 | -18.8 |

- In Gothenburg the radar is the weakest sensor. A map of the PWS without it is best, and a
  link map without it (1.27) does as well as the eight intercomparison merges (1.27-1.32); only
  RADOLAN gets more out of the links with the radar (1.19).
- Mixing links with PWS drags the PWS map down (1.01 to 1.25). Multiplicative adjustment with
  the PWS is the one bad case (1.54).
- RADOLAN beats every method of the intercomparison with the same links. It is an IDW
  adjustment like `add_p_idw`, but its gauge-radar factors are smoothed and checked against
  audit stations.
- The PWS are close to the check gauges: the nearest is 0.2-3.2 km away for 10 of the 11, 7 km
  for the 11th. That favours them.

**Maps without radar, and RADOLAN (Emilia-Romagna).** With 127 links over Emilia-Romagna (~22,000 km²), link
maps without radar fail: RMSE 4.14 (block kriging), 4.23 (point kriging), 5.15 (IDW); r
0.37-0.43. RADOLAN has the lowest RMSE of all products (2.97, r 0.74) but keeps +33% bias;
block KED (3.08) keeps -7%.

**Variogram and nugget** (additive block kriging; the study uses sill 1, range 30 km,
nugget 0.3):

| variogram | Gothenburg | Emilia-Romagna |
|---|---|---|
| study (30 km, nugget 0.3) | 1.266 | 3.186 |
| mergeplg's default (5 km, nugget 0.1) | 1.316 | 3.295 |
| fitted to the radar (60 km / 0.15; 68 km / 0.32) | 1.267 | 3.196 |
| study, nugget from link geometry (`c0_within`) | 1.267 | 3.197 |

The fit (`extend.fit_variogram_xy`) pools semivariances of standardised wet radar hours, with
no gauge. Only mergeplg's 5-km default is wrong for hourly summer rain; any range of 30 km or
more gives the same maps.

**New York City.** Here the radar is MRMS radar-only (hourly), adjusted with the NYC Mesh
links (rain from the RNN of `projects/retrieval/rnn_three_networks`, or from the nearby-link power law), with the
37 WU PWS, or with both. Inputs are those `projects/maps/multisensor` cached for its 10
storms; 9 have every product, giving 663 station-hours at the 4 ASOS gauges.

| product | RMSE | r | bias % |
|---|---|---|---|
| RADOLAN, PWS | **1.39** | 0.91 | -5.2 |
| IDW map of the PWS, no radar | 1.50 | 0.90 | -6.1 |
| MRMS Pass 2 (NOAA's gauge correction; reference) | 1.59 | 0.88 | -3.6 |
| RADOLAN, links (RNN) + PWS | 1.65 | 0.89 | -8.5 |
| radar adjusted with the PWS (add. IDW / block OK / block KED) | 1.70-1.77 | 0.86-0.88 | -11..-13 |
| RADOLAN, links (RNN) | 2.37 | 0.74 | -13.7 |
| add. IDW, links (RNN) | 2.41 | 0.73 | -13.4 |
| radar only | 2.53 | 0.76 | -29.5 |
| add. IDW, links (nearby power law) | 3.05 | 0.76 | -53.6 |

- With the PWS, RADOLAN beats NOAA's own gauge correction at the ASOS gauges.
- The RNN links alone halve the radar's bias and gain a little RMSE. The power-law links make
  the radar worse; they read low, as in `multisensor_maps`.
- Adding either kind of link to the PWS makes every product worse.

## ARPAE's gauge-adjusted radar as the baseline

ARPAE publishes its composite corrected by kriging the gauge/radar ratio of its own gauges
(OpenRainER `RADadj`). Prepared exactly as the unadjusted radar and scored at the same 319
gauges with the same metric (`run.py arpae`, `arpae.py`; tables
`results/arpae_adjusted_scores.csv` and `results/arpae_adjusted_by_distance.csv`):

| Emilia-Romagna, JJA 2022 | PCC | RMSE (mm) | PBIAS (%) |
|---|---|---|---|
| radar, unadjusted (as in the intercomparison) | 0.749 | 3.58 | +107 |
| best link adjustment (block KED, mergeplg main) | 0.735 | 3.08 | -7 |
| ARPAE's gauge-adjusted radar | 0.869 | 2.26 | +6 |

ARPAE's product is better at every distance from the links, including the cells within 2 km
of one (RMSE 2.13 against 2.70 for block KED). It is not an independent baseline - its
correction uses the gauges it is scored at - so it is an upper reference for what dense gauge
adjustment reaches, not a fair competitor. What it shows is that the intercomparison's gains
are measured against a radar that a national service would not use unadjusted: links improve
the raw composite, and a dense gauge network improves it much more.

# Caveats

- **The gauges are the reference, and they are points.** A gauge is compared with the 1-2 km
  cell it falls in. Gothenburg has only 11 gauges, all in the city, all within 2 km of a link
  and up to 3 km from a PWS. Emilia-Romagna's 319 gauges cover every distance band. New York
  has 4 ASOS gauges and 663 station-hours.
- **The metric is OpenSense's.** Gauge-hours where both gauge and estimate are below 0.2 mm are
  left out, and a value below it on one side counts as 0. So `n_pairs` differs between
  products, although every product is scored on the same gauge-hours (`n_all`, `n_nan` equal;
  the Emilia-Romagna maps without radar have 459 more missing hours, of 704,033).
- **OpenRainER's radar here is the unadjusted product** (RADrain), which reads double the
  gauges in summer 2022. ARPAE's adjusted product is scored above as a (non-independent) baseline.
- **OpenMRG's radar uses Z = 200 R^1.6**, as the notebook computes it, where the file says
  R^1.5. Kept, so the inputs match the published ones; `core.opensense.networks` uses 1.5.
- **mergeplg versions.** The `main` run reuses the `pinned` OpenRainER point-variant files.
  The two versions are identical there (shown on OpenMRG: < 1e-4 mm at every gauge in all 27
  files), so recomputing would only cost 36 tasks. The `pinned` OpenRainER run covers the
  published checks, plus the point variants for the other two.
- **Cropped grids.** Merging runs on the radar cells within 0.1° of the links and gauges
  (Emilia-Romagna: 173 x 300 of 290 x 373 cells). Each cell's estimate depends only on the
  observations, so the values at the gauges are unchanged
  (`tests/test_radar_adjustment.py::test_crop_does_not_change_values_at_gauges`); the
  3-month total maps cover the cropped area only.
- **RADOLAN in mergeplg needs care:**
  - The grid must be regular in the stations' projection, so its runs re-project grid, links
    and points: OpenMRG into the composite's polar stereographic projection, the lat/lon
    grids into an equirectangular one (`extend.regular_plane`).
  - It fails on an hour with no valid observation; such hours keep the radar.
  - Its audit stations start at a fixed index (0), not a random one.
  - Inputs are cast to float64 (the upstream issue noted in `core/scientific_packages/UPSTREAM_ISSUES.md`).
- **Only three summer months** per network, as in the intercomparison. No winter, no snow.

# Running it

## How to run

```bash
# inputs (the default environment)
.venv/bin/python projects/maps/radar_adjustment/src/run.py prepare

# the merging needs mergeplg's constructor-and-call API: a second environment, as
# projects/maps/archive_pipeline/README.md sets up (pysteps left out: it does not build there)
python3.11 -m venv .venv-mergeplg-main
.venv/bin/pip freeze | grep -v -e '^mergeplg' -e '^-e' -e '^pysteps' > /tmp/freeze.txt
.venv-mergeplg-main/bin/pip install -r /tmp/freeze.txt && .venv-mergeplg-main/bin/pip install --no-deps -e .
.venv-mergeplg-main/bin/pip install --no-deps "mergeplg @ git+https://github.com/OpenSenseAction/mergeplg.git@dd380b1ed9b5b6bd38fdb3dfdfcca1659ceb8128"
# the commit the OpenSense repository records, beside it (used by the "pinned" runs)
.venv-mergeplg-main/bin/pip install --no-deps --target dataset/open_datasets/_radar_adjustment/mergeplg_9894b9c \
    "mergeplg @ git+https://github.com/OpenSenseAction/mergeplg.git@9894b9c028280ba12da79d3a2a2bcb1956e44e8e"

.venv-mergeplg-main/bin/python projects/maps/radar_adjustment/src/run.py adjust --workers 6
.venv-mergeplg-main/bin/python projects/maps/radar_adjustment/src/run.py extend --workers 6
.venv/bin/python projects/maps/radar_adjustment/src/run.py report        # -> results/report.md, figures, CSVs
```

Every task (version, network, checks, product, month) is saved under
`dataset/open_datasets/_radar_adjustment/fields/` and skipped when present, so runs resume.
On an 8-core laptop: `prepare` ~5 min; `adjust` ~3 h (OpenMRG's 144 tasks ~25 min, OpenRainER's
kriging ~10 min per month and variant); `extend` ~2.5 h (kriging maps without radar take ~40 min per OpenRainER month). Tests:
`.venv/bin/python -m pytest tests/test_intercomparison_chain.py projects/maps/radar_adjustment/tests`
(the mergeplg-specific ones run in `.venv-mergeplg-main` and are skipped elsewhere).

## Layout

```
radar_adjustment/
├── src/run.py                       # prepare / adjust / extend / report
├── src/radar_adjustment/
│   ├── settings.py                  # periods, methods, range checks, paths
│   ├── prepare.py                   # inputs as 1_data_preparation + 2_cml_processing build them
│   ├── adjust.py                    # the eight adjustments, per month, both mergeplg versions
│   ├── extend.py                    # weather stations, maps, RADOLAN, variogram, New York
│   ├── score.py                     # 4_analysis's metrics, by intensity, gauge, distance
│   └── report.py                    # results/report.md, CSVs, figures
├── notebooks/01_intercomparison.ipynb
├── results/                         # report.md, scores*.csv, figures/
└── tests/
core/opensense/intercomparison_chain.py   # the OpenSense CML chain and radar conventions
```

# Related

- [`multisensor_maps`](../multisensor/): links, gauges and radar merged every way on ten
  storms per network, scored at held-out gauges (mergeplg 0.1.0 methods, re-implemented in
  `core.maps.mergeplg_methods`).
- [`opensense_pipeline`](../archive_pipeline/): the first FieldSense merging benchmark, and
  the instructions for the `.venv-mergeplg-main` environment.
- [`os_nowcasting`](../../nowcasting/pysteps/): nowcasting the radar merged with links.
- [`tutorials/06_radar_adjustment.ipynb`](../../../tutorials/): each adjustment method
  implemented step by step.

# References

1. OpenSense radar-adjustment intercomparison:
   <https://github.com/OpenSenseAction/radar_adjustment_intercomparison>;
   merging package [mergeplg](https://github.com/OpenSenseAction/mergeplg)
   ([documentation](https://mergeplg.readthedocs.io)); scoring with
   [poligrain](https://github.com/OpenSenseAction/poligrain); CML processing with
   [pycomlink](https://github.com/pycomlink/pycomlink); PWS QC with
   [pypwsqc](https://github.com/OpenSenseAction/pypwsqc).
2. Andersson, J. C. M., Olsson, J., van de Beek, R. (C. Z.), and Hansryd, J. (2022). OpenMRG.
   *Earth System Science Data*, 14, 5411-5426.
   [doi:10.5194/essd-14-5411-2022](https://doi.org/10.5194/essd-14-5411-2022)
3. Pastorek, J., Fencl, M., Rieckermann, J., and Bares, V. (2022). Precipitation estimates from
   commercial microwave links: practical approaches to wet-antenna correction. *IEEE TGRS*, 60,
   1-9. [doi:10.1109/TGRS.2021.3110004](https://doi.org/10.1109/TGRS.2021.3110004)
4. Fencl, M., et al. (2023). Data formats and standards for opportunistic rainfall sensors.
   *Open Research Europe*, 3, 169.
   [doi:10.12688/openreseurope.16068.1](https://doi.org/10.12688/openreseurope.16068.1)
5. de Vos, L. W., Leijnse, H., Overeem, A., and Uijlenhoet, R. (2019). Quality control for
   crowdsourced personal weather stations to enable operational rainfall monitoring.
   *Geophysical Research Letters*, 46, 8820-8829.
   [doi:10.1029/2019GL083731](https://doi.org/10.1029/2019GL083731)
6. Jacoby, D., et al. (2026). OpenMesh. *Earth System Science Data*, 18, 5817-5836.
   [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026)
