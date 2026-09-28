# Retrieval methods: the two implementations and how they differ

The methods in this project come from two student implementations of the same task:
**implementation 1** by Gabriela and **implementation 2** by Jeries. Both turn OpenMesh link
signals into path rain rates and an IDW rain map over Brooklyn/Manhattan, checked against MRMS
radar, but they make different choices at almost every step.

Both were first unified in the `pcpn_maps` repository (`drorjac/pcpn_maps`,
private; branch `unified-package`). That repository keeps their original code in `archive/` and checks that
every unified method reproduces their committed outputs (`tests/test_reproduction.py`):

| committed output (in `pcpn_maps/archive/`) | method here | agreement |
|---|---|---|
| `comparison/map1.pkl` (impl. 1, 27–28 Dec 2023) | `DynamicBaseline(gap_fill="gauge_max", nan_to_zero=True)` | exact (max diff < 1e-5 mm/h) |
| `comparison/map2.pkl` (impl. 1, 21–22 Nov 2023) | same | exact |
| `implementation_2/results/R_classical_dynamic.csv` | `DynamicBaseline(gap_fill="min_rsl")` | max 0.035 mm/h, totals equal |
| `R_classical_constant.csv` | `ConstantBaselineSTD(gap_fill="min_rsl")` | max 0.030 mm/h |
| `R_manual.csv` | `ManualWindows(dry=…, rain=…, gap_fill="min_rsl")` | max 0.035 mm/h |
| `R_pycomlink.csv` | `PycomlinkRSD(threshold_links=<8-month record>)` | max 0.029 mm/h |
| `R_nearby.csv` | `NearbyLinks()` on the 8-month record | max 0.023 mm/h |
| `R_pnncml.csv` | `PyNNcmlGRU()` | max 5e-6 mm/h |

(The ~0.03 mm/h residuals are CSV float formatting; event totals agree to 0.1 mm.) The port into
FieldSense was checked against `pcpn_maps` in turn: identical scores for every method on the
events compared, and an identical link selection.

## Shared ground

Both use the same **8 sublinks**, picked by eye from attenuation plots:
`3/sublink_3, 8/sublink_1, 12/sublink_2, 20/sublink_2, 31/sublink_1, 34/sublink_1,
40/sublink_1, 45/sublink_1` — all vertically polarised, 65.9 or 68.0 GHz, 1.4–6.9 km
(`nyc_rain_maps.pipeline.SHARED8`). Both treat OpenMesh RSL as 1-min data with constant TSL (no
TSL is recorded, so total loss = −RSL), both map with IDW (power 2, 10 km radius) from link
midpoints, and both use MRMS `MultiSensor_QPE_01H_Pass2` as the radar reference.

## Where they differ

### 1. Gaps in the signal (the most consequential difference)

| | implementation 1 | implementation 2 |
|---|---|---|
| Which gaps are filled | only while PWS gauges within 5 km report rain (15-min mean > 0) | **all** gaps in the event window |
| Fill value | 99th percentile of the link's attenuation (current code); the committed maps used the **maximum** — `pynncml_changes.md` and the README say q99, but only `max` reproduces `map1/map2.pkl` | link's minimum RSL in the window (= maximum attenuation) |
| Gaps in dry weather | stay NaN → rain set to 0 before mapping | become deep fades → the method sees an attenuation spike |
| Package | `gap_fill="gauge_q99"` / `"gauge_max"` | `gap_fill="min_rsl"` |

Why it matters: a link that loses sync in heavy rain reports nothing, so a gap is evidence of
rain. Implementation 2 turns *every* gap into maximum attenuation, including outages in dry
weather, which the dynamic baseline then converts into rain. Implementation 1 needs an
independent gauge network to decide. In the 27–28 Dec and 21–22 Nov events every gap fell in a
wet period, so `impl1_dynamic_maxfill` and `impl2_classical_dynamic` give *identical* maps there;
they diverge on events with dry-weather outages.

### 2. Rain retrieval

| Method (package name) | From | Wet/dry | Baseline | Wet antenna | Power law | Time step |
|---|---|---|---|---|---|---|
| `impl1_dynamic` | impl. 1 (PyNNcml `OneStepDynamic`) | none | trailing 200-min min of A + ½qd, qd = 1 dB | none (qd only) | **ITU 2003**, r_min 0.5 | 1 min |
| `impl2_classical_dynamic` | impl. 2 `pynncml_classical` | none | same as above | none | ITU 2003, r_min 0.5 | 1 min |
| `impl2_classical_constant` | impl. 2 `pynncml_classical` | rolling σ (240 min, centred), wet if σ > 1 dB | constant during wet | flat 3 dB | ITU 2003, r_min 0.5 | 1 min |
| `impl2_manual` | impl. 2 `manual` | hand-picked windows | median over a hand-picked dry window | none | **ITU 2005**, r_min 0.1 | 1 min |
| `impl2_pycomlink` | impl. 2 `pycomlink_basic` | rolling σ (240 min) > per-link q90 of the 8-month σ, or gap | constant (mean of last 5 dry) | Leijnse 2008 | ITU 2005, r_min 0.1 | 1 min |
| `impl2_pycomlink_linear` | same | same | linear | Leijnse 2008 | ITU 2005 | 1 min |
| `impl2_nearby` | impl. 2 `pycomlink_basic` §3 | nearby-link (Overeem 2016), 15 km, ≥3 links | 24-h median of dry intervals | 2.3 dB max, α 0.33 | ITU 2005 | 15 min min/max |
| `impl2_gru` | impl. 2 `PyNNCML_GRU` | learned | learned | learned | learned | 15 min min/max |

**Power-law tables.** PyNNcml (and so implementation 1 and implementation 2's
`pynncml_classical`) uses ITU-R P.838-2 (2003); pycomlink uses P.838-3 (2005). At 68 GHz V:
a = 0.758, b = 0.799 (2003) vs a = 0.994, b = 0.726 (2005). For the same attenuation over 2 km,
the 2003 coefficients give ~20–40 % more rain between 1 and 10 mm/h. This alone explains part
of the systematic gap between the PyNNcml-based and pycomlink-based methods. Both tables are
embedded in `nyc_rain_maps.itu_tables` and selectable per method (`table=`).

**Dynamic baseline window.** 200 samples = 200 minutes (a comment in `pynncml_classical`
says "1 h"). Implementation 1 chose 200 from a sweep against radar over windows 60–420 min; in
the two committed sweeps the lowest MAE was at 210 and the zero-bias point at 150 (Dec) / 290
(Nov) — 200 is a compromise, not an optimum, and the sweep used an unpadded grid unlike the
final maps.

### 3. Mapping

| | implementation 1 | implementation 2 |
|---|---|---|
| Library | PyNNcml `InverseDistanceWeighting` (with local changes) | pycomlink `IdwKdtreeInterpolator` |
| Neighbours | all links within 10 km | 8 nearest within 10 km (= all 8 here) |
| NaN link values | set to 0 mm/h | excluded per time step |
| Grid | UTM box of the link endpoints, normalised by its longer side, ⌈side/1 km⌉ pixels + 2 padding pixels per side → 13 × 11 nodes (~1 km); node lon/lat mapped linearly onto the endpoints' lon/lat box | 12 × 12 km box around (40.69, −73.96), 1 km cells, UTM 18N |
| Output | 1-min rain-rate maps | hourly-mean rate maps (hour-ending) |
| Package | `implementation1_map()` (exact replica) | `idw_map(nnear=8)` on `implementation2_grid()` |

The unified pipeline maps every method with the same `idw_map` on the MRMS grid (0.01°) so the
comparison isolates the retrieval method.

### 4. Evaluation against radar

| | implementation 1 | implementation 2 |
|---|---|---|
| MRMS handling | all `*.grib2` in a folder (event 2 loaded 20 files though 9 were for the event); linear interpolation onto the CML grid | per-window cache; nearest frame, no regridding |
| Metrics | MAE, bias, total ratio, corr(err, rain) over a subset of hours (`display_times`) | visual only in `CML_maps`; RMSE in `roi_sensitivity` (never executed) |
| here | `compare_maps` / `compare_links`: hourly, hour-ending, common grid, only jointly valid cells, NRMSE + bias + corr + POD/FAR/CSI, pooled and per hour, plus per-link scores against radar averaged **along each link path** |

## Issues found in the original code (all fixed or avoided here)

**implementation_1**
- Windows-only absolute paths (`C:\Users\gabir\…`) in the notebooks; `sys.path` points above the folder.
- Stock PyNNcml cannot run it: the required local changes (auto UTM zone, `filled_attenuation`,
  `use_filled`, padded grid) are documented in `pynncml_changes.md` but are not in the installed
  PyNNcml — the methods here need none of them.
- Committed maps use max fill while the docs say q99 (see §1).
- Metrics use `display_times` (a subset of hours) — Nov used 16 of 20 hours in the sweep.
- PWS reference map: lon/lat mapping skewed when padding is on; gauge totals assume exact 5-min sampling (7 % of intervals are irregular).
- Links 40 and 45 are the same path in opposite directions and are both kept (double weight in IDW).

**implementation_2**
- `roi_sensitivity.ipynb` pairs link values with the wrong coordinates for links 34/40/45 (order of `meta_links` vs `sublink_choice`); `CML_maps` has the fix.
- MRMS cache key ignores hours and extent (`mrms_nyc_{YYYYMMDD}_{YYYYMMDD}.nc`): a cached 19-hour file is silently reused for a 48-hour window; with `END = 23:59` the last hour is never fetched; MRMS no-coverage (−3) is not masked; corrupt downloads are never re-fetched.
- ASOS fetcher drops the last day of each month (`<= chunk_end` at 00:00) and discards `ptype`.
- 15-min products (nearby, GRU) are left-labelled but resampled hour-ending → 15-min shift vs MRMS. Here they are stamped interval-ending.
- `CML_maps.ipynb` loads `results/R_pycomlink_linear.csv`, which is not committed; `run_pipeline.sh` sources a `venv/` that is not in the repo; `CML_maps` fails without `WU_API_KEY`.
- Committed state is internally inconsistent: `shared_config.json`/results are for 27–28 Dec, notebook outputs for 21–22 Nov, `rsl_shared.nc` covers 27 Dec 16:00–28 Dec 16:00 only.
- PyNNcml GRU: pretrained on 18–25 GHz OpenMRG links, applied at 65–68 GHz; its length normalisation (mean 7466) suggests metres while km are passed — results should be read with that in mind (`PyNNcmlGRU(length_unit="m")` tests the alternative).
- The enum `pnc.neural_networks.RNNType` is `DNNType` in PyNNcml 0.3.7; both are handled.

**Both**
- Links are selected by eye. The automatic QC procedure (`nyc_rain_maps.link_qc`):
  on 27–28 Dec it keeps 25 of 103 sublinks (51 rejected as < 10 GHz, 13 as < 0.3 km, 14 as
  duplicate paths — including 45 as the reverse of 40). More links is not automatically better:
  on the 9–10 Jan 2024 rainstorm the 25 QC links give hourly NRMSE ≈ 0.9–1.0 against ≈ 0.4 for the
  8 hand-picked ones, because QC removes *impossible* links but keeps noisier 24/58 GHz paths that
  the eye-based selection had dropped. Retrieval-stage QC (flags in `qc_log.csv`) is where the
  next improvement lies.

## Results on the common footing

`python src/run.py study` runs every catalogued event inside the OpenMesh record (`events/`) through the
same pipeline — 6 h spin-up, hourly accumulations, IDW (power 2, 10 km) on the MRMS 0.01° grid,
scored against MRMS — for three link sets. Full report:
[`results/study/report.md`](../results/study/report.md).

**Fair sample.** Scores are pooled over the cell-hours every *full-coverage* method has. A method
that leaves gaps (here the nearby-link method: ~75 % coverage in rain, most of it lost in the
17–18 Dec storm) is scored on its share and its coverage is shown; it is not ranked. An earlier
version of this table used the strict intersection of all methods, which silently dropped most of
that storm for every method and made the wet/dry-gated methods look clearly best (NRMSE 0.45) —
the corrected numbers below replace it.

8 shared links, all cells:

| type | method | events | coverage | hourly NRMSE | rel. bias | corr | CSI |
|---|---|---|---|---|---|---|---|
| rain | impl2_nearby | 5 | 76% | 0.59 | -0.29 | 0.85 | 0.86 |
| rain | impl2_classical_dynamic | 5 | 100% | 0.62 | -0.09 | 0.80 | 0.92 |
| rain | impl1_dynamic | 5 | 100% | 0.62 | -0.14 | 0.80 | 0.92 |
| rain | impl2_pycomlink | 5 | 100% | 0.64 | -0.29 | 0.82 | 0.86 |
| rain | impl2_classical_constant | 5 | 100% | 0.67 | -0.36 | 0.83 | 0.75 |
| mix | impl2_nearby | 3 | 100% | 1.78 | +0.25 | 0.38 | 0.56 |
| mix | impl1_dynamic | 3 | 100% | 2.70 | +0.85 | 0.48 | 0.61 |
| mix | impl2_classical_constant | 3 | 100% | 3.34 | +1.02 | 0.53 | 0.53 |
| mix | impl2_pycomlink | 3 | 100% | 3.59 | +1.43 | 0.51 | 0.66 |
| mix | impl2_classical_dynamic | 3 | 100% | 3.79 | +1.58 | 0.58 | 0.66 |
| snow | impl2_pycomlink | 2 | 100% | 1.13 | -0.48 | 0.39 | 0.49 |
| snow | impl1_dynamic | 2 | 100% | 1.21 | +0.11 | 0.52 | 0.68 |
| snow | impl2_classical_dynamic | 2 | 100% | 1.21 | +0.11 | 0.52 | 0.68 |
| snow | impl2_classical_constant | 2 | 100% | 1.27 | -0.71 | 0.22 | 0.33 |
| snow | impl2_nearby | 2 | 100% | 1.37 | -0.44 | 0.07 | 0.48 |

What this says:

- **Rain — the two families are close; they fail differently.** Over all five rain events the
  dynamic-baseline methods (implementation 1; implementation 2's classical dynamic) and the
  pycomlink RSD method are within 0.02 NRMSE (0.62–0.64), but the dynamic-baseline methods are
  nearly unbiased (−9 to −14 %) while the wet/dry-gated methods under-estimate by 29–36 %.
  Event by event they diverge: in the long 17–18 Dec storm the wet/dry-gated methods fail (NRMSE
  ≈ 1.3) and the dynamic baseline holds (0.6); in the 9–10 Jan storm the opposite happens — after
  ~3 h of continuous heavy rain the 200-min trailing minimum sits inside the rain and the dynamic
  baseline absorbs it (hourly CML ≈ 1 mm vs radar ≈ 8 mm at 04–06 UTC). Part of the pycomlink
  under-estimate is the ITU 2005 table (see §2).
- **Implementation 1 vs implementation 2's dynamic method** differ only in gap filling; across the
  five rain events they have the same NRMSE (0.62) and bias within 5 % — the gauge-gated fill rarely changes the answer on
  these 8 links, whose outages are almost all during rain.
- **Mixed precipitation — all methods over-estimate** (+25 % to +158 %): wet snow, ice pellets and
  the melting layer attenuate far more than the liquid-rain power law assumes. The nearby-link
  method degrades least (NRMSE 1.78, +25 %).
- **Snow — no method is usable.** The wet/dry-gated methods under-estimate strongly (−44 to
  −71 %); the dynamic-baseline ones land near zero bias (+11 %) only by accident of noise
  (corr 0.52). Dry snow barely attenuates at 65–68 GHz and radar snow QPE is itself uncertain;
  only two OpenMesh-period snow events exist, so treat these rows as indicative.
- **Short links and the flat wet-antenna offset.** Per link against radar along the path, link 31
  (1.36 km) loses 95 % of its rain under the PyNNcml constant baseline — the 3 dB offset exceeds
  most of its rain attenuation — the caveat noted in implementation 2's README, now quantified
  (`results/study/figures/link_bias_rain.png`).
- **Hand-picked vs cherry-picked vs automatic-QC links.** The gauge-calibrated selection
  (`link_selection`, below) reproduces the hand-picked set almost exactly and scores within 0.05
  NRMSE of it in 14 of 15 (type, method) cases.
- **Hand-picked vs automatic-QC links.** The 25-ish links surviving automatic QC beat the 8
  hand-picked ones in only 4 of 15 (type, method) cases; for rain every method is worse with
  them (NRMSE 0.83–1.01 vs 0.62–0.67), and the dynamic-baseline methods turn from −10 % to +45 %
  bias because QC keeps short and 24/58 GHz links whose baselines drift. QC removes impossible
  links but not noisy ones — that is what the rain-response and accumulation stages of the
  cherry-picking pipeline add.
- The pretrained GRU (not in the default set; `impl2_gru`) scored NRMSE 1.66 on 27–28 Dec — it
  was trained on 18–25 GHz links and does not transfer to 65–68 GHz.

Per-event scores: `results/study/event_scores.csv`. Walk-through: `notebooks/03_one_event.ipynb`.

## Across every event of the record

The study above pools the 10 catalog events. `python src/run.py all-events`
([`results/all_events/`](../results/all_events/README.md)) runs **all 52 events** of the OpenMesh
period (47 rain, 3 mix, 2 snow) through every method, on the gauge-selected links, with the
sensors alongside. Two March 2024 storms had no link data (network outage) and are listed, not
ranked. Over the 45 scorable rain events:

| method | events won (lowest NRMSE) | median NRMSE | median bias | events under-estimated |
|---|---|---|---|---|
| impl1_dynamic | 16 | 0.98 | +1% | 49% |
| impl2_classical_dynamic | 9 | 1.02 | +7% | 44% |
| impl2_pycomlink | 15 | 1.00 | -49% | 96% |
| impl2_classical_constant | 1 | 1.17 | -61% | 98% |
| impl2_nearby | 4 | 1.00 | -60% | 100% |

- **The dynamic baseline is the only unbiased family.** Both implementations' dynamic-baseline
  methods have median bias within ±7 %, over- and under-estimating about equally often; every
  wet/dry-gated method under-estimates in 96–100 % of events (median −49 % to −61 %), and loses
  light events almost entirely (−66 % to −100 % below 10 mm near the links).
- **On large events the gated methods catch up in pattern.** Above 10 mm near the links,
  pycomlink has the lowest hourly NRMSE (0.68) but still reads 21 % low; the dynamic baseline is
  at 0.78 with no bias.
- **Warm, small, short events are hardest for everyone.** NRMSE falls with event total
  (Spearman −0.53) and duration (−0.35) and rises with temperature (+0.55): summer convective cells
  are smaller than the link spacing, which no retrieval method can fix — it is a network-density
  limit.
- **Held-out vs calibration events give the same ordering**, so the gauge-calibrated link selection
  does not flatter the results.

## Link selection: automating the authors' cherry-picking

Both implementations chose the same 8 sublinks by eye. `nyc_rain_maps.link_selection` (`python src/run.py
select`) chains implementation 1's own screening functions into a reproducible pipeline —
contact loss during gauge rain (`check_link_contact_loss`), rain response against gauges
(`rain_detection` + `roc_threshold`), accumulation against gauges
(`filter_links_by_accumulation`) and one sublink per path (`keep_links`, implementation 2's
`sublink_choice`). It uses **only PWS gauges**, on the **31 rain events of the OpenMesh record that
are not in the evaluation catalog**, so its scores against MRMS are out-of-sample.

Result (`results/link_selection/`): 103 → 7 sublinks — `3/sublink_3, 8/sublink_1, 12/sublink_2,
20/sublink_2, 31/sublink_1, 34/sublink_1, 40/sublink_1`, i.e. **the authors' 8 minus link 45**,
which it identifies as the same path as link 40 (reverse direction). Rejections: 64 at metadata
(51 below 10 GHz, 13 shorter than 0.3 km), 4 for contact loss in rain, 19 for weak rain response
(all 24 GHz links among them), 6 for accumulation 1.5–10× off the gauges, 3 as second sublinks
of a kept path. The independent procedure confirms the hand selection.

## Multi-sensor comparison

`nyc_rain_maps.compare` (`python src/run.py compare`) puts every sensor on the same grid and hour — CML maps,
MRMS, a PWS gauge map (implementation 1's 5th–95th-percentile gauge filter, same IDW) — and
compares them pairwise, following implementation 1's CML/PWS/radar table and implementation 2's
snapshot grid with a PWS row; ASOS stays out of every map as the independent point check.
Pooled over the study events, cells within 2 km of a link, `selected` links, against MRMS:

| type | map | coverage | NRMSE | rel. bias | corr |
|---|---|---|---|---|---|
| rain | PWS | 100% | 0.33 | +0.00 | 0.95 |
| rain | CML impl2_nearby | 74% | 0.55 | -0.27 | 0.88 |
| rain | CML impl2_classical_dynamic | 100% | 0.57 | -0.09 | 0.84 |
| rain | CML impl1_dynamic | 100% | 0.58 | -0.13 | 0.83 |
| rain | CML impl2_pycomlink | 100% | 0.70 | -0.32 | 0.79 |
| rain | CML impl2_classical_constant | 100% | 0.72 | -0.37 | 0.80 |
| mix | PWS | 100% | 1.08 | -0.52 | 0.50 |
| mix | CML impl2_nearby | 98% | 1.89 | +0.36 | 0.38 |
| mix | CML impl1_dynamic | 100% | 2.73 | +0.97 | 0.50 |
| mix | CML impl2_classical_constant | 100% | 3.42 | +1.12 | 0.54 |
| mix | CML impl2_pycomlink | 100% | 3.77 | +1.58 | 0.51 |
| mix | CML impl2_classical_dynamic | 100% | 3.84 | +1.69 | 0.58 |
| snow | CML impl2_pycomlink | 100% | 1.11 | -0.45 | 0.43 |
| snow | CML impl2_classical_constant | 100% | 1.25 | -0.68 | 0.27 |
| snow | CML impl1_dynamic | 100% | 1.26 | +0.22 | 0.56 |
| snow | CML impl2_classical_dynamic | 100% | 1.26 | +0.22 | 0.56 |
| snow | PWS | 100% | 1.40 | -1.00 | -0.06 |
| snow | CML impl2_nearby | 100% | 1.42 | -0.38 | 0.09 |

The PWS gauge map sets the scale: in rain it differs from MRMS by NRMSE 0.33 with no bias, so a
CML map at 0.57–0.72 is within a factor of two of the gauge–radar disagreement. In snow the PWS
map is useless (heated gauges are rare among personal stations: bias −100 %), and in mixed
events gauges under-catch (−52 %) while CMLs over-read (+36 % or more) — the two sensors fail in
opposite directions.
