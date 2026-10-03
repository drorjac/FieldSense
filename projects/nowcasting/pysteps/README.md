# os_nowcasting: pysteps nowcasts of radar, link and merged rain maps

The nowcasting session of the OpenSense training school on merging and application of
opportunistic rainfall data (2025, Ritvanen & Imhoff;
[`OpenSenseAction/TrainingSchoolMergingApplication/nowcasting-session`](https://github.com/OpenSenseAction/TrainingSchoolMergingApplication/tree/main/nowcasting-session))
taught pysteps on OpenRainER: optical flow, extrapolation, S-PROG, STEPS, verification - on
the radar and on link maps for one issue time of one storm. This project runs the same chain
on FieldSense's data and turns it into a study: every method, on every product, at every
hour of nine storms on two networks, scored against the product itself, the radar and
independent gauges.

**Question:** can a rain map made from commercial microwave links (alone, or merged into the
radar) be nowcast with pysteps, and how does that nowcast compare with nowcasting the radar?

| network | radar (step) | links | gauges (reference only) | storms | issue times |
|---|---|---|---|---|---|
| OpenRainER, Emilia-Romagna (Aug 2022) | ARPAE-SIMC 15-min rain depth | 151 links, nearby-link retrieval | 319 ARPAE gauges, 15 min | 4 (incl. the session's 17-19 Aug) | 143 |
| OpenMRG, Gothenburg (JJA 2015) | SMHI 5-min reflectivity | 364 links, nearby-link retrieval | 10 city gauges, 1 min | 5 | 128 |

Data: OpenRainER [doi:10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808)
(radar, links and gauges; 15-min accumulations stamped at the interval end); OpenMRG
[doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689) (Andersson et al., 2022)
and the OpenMRG2 Netatmo PWS preview. File locations: [`DATA.md`](../../../DATA.md). Nowcasting
uses [pysteps](https://pysteps.readthedocs.io) 1.21 (Pulkkinen et al., 2019).

## In short

- **The radar nowcasts as the session taught, and the cascade methods add a little.** At 60
  min, CSI at 1 mm/h against the radar is 0.27 for persistence, 0.40 for extrapolation and
  0.44 for S-PROG (OpenRainER; OpenMRG 0.21 / 0.33 / 0.35). LINDA is best at 60-90 min where
  it ran (0.47 vs 0.42 for extrapolation on the same forecasts) but its fields are too peaked
  (SAL structure +0.8) and on OpenMRG's 44 x 35-pixel grid its MAE doubles. ANVIL without the
  radar's VIL is below extrapolation everywhere.
- **Link maps cannot be nowcast by extrapolation.** Their motion is unrelated to the rain's:
  LK on a link map differs from the radar's motion by about the radar's speed (OpenRainER
  21 vs 22 km/h, OpenMRG 47 vs 51 km/h), and extrapolating a link map is no better than
  persisting it (own-field CSI 0.16 vs 0.17 at 60 min). The IDW blobs are anchored to the
  links; they do not move.
- **Merged into the radar, links keep the radar's motion** (3 km/h from it on OpenRainER),
  and the merged field nowcasts like the radar: at the independent ARPAE gauges its
  extrapolation and STEPS nowcasts are level with the radar's (RMSE 4.36 vs 4.42 mm/h, corr
  0.22 vs 0.25; STEPS CRPS 0.60 vs 0.71).
- **Motion method barely matters, except DARTS.** LK, VET and Proesmans drive extrapolation
  nowcasts within 0.01-0.03 CSI of each other; DARTS is 0.06-0.10 lower; LK without the dB
  transform loses 0.01.
- **STEPS is close to reliable** for 1 mm/h at 60 min, a little over-confident at high
  probabilities, and under-dispersive (the radar falls above every member in 13% of cases).
- **Advection interpolation helps hourly totals only slightly** at OpenMRG's city gauges
  (RMSE 1.19 to 1.15 mm, corr 0.72 to 0.73).
- **Domain size decides what can be verified.** On OpenMRG's 88 x 70 km radar domain, rain
  moving at 50 km/h leaves 45% of the domain verifiable at 45 min and 19% at 90 min; scoring
  the rest as forecast zeros made persistence look like the best method (see Caveats).

## The pipeline

1. **Grid** (`grid.py`). Square 2 km pixels in a local equirectangular projection around the
   domain centre - a regular lat/lon grid with `dlon = dlat / cos(lat0)` - so pysteps gets a
   projection in metres and `kmperpixel`, and the same grid serves the link maps and the
   merging code. (The session passed OpenRainER's native 0.0127 x 0.0090 deg grid to pysteps
   in degrees.) OpenRainER: 97 x 155 pixels around the network; OpenMRG: 44 x 35 pixels, all
   of the SMHI composite there is.
2. **Products** at the radar's native step (`data.py`, `products.py`):
   `radar`; `cml_idw10/20/40` - link rain by IDW within 10, 20, 40 km (the session's three
   link maps; cells out of range are zero); `merged` - the radar adjusted with the links by
   mergeplg's difference block kriging (`core.maps.mergeplg_methods`, variogram fitted to the
   event's radar, radar kept beyond 60 km from any link as in mergeplg `main`); and on OpenMRG
   `pws_idw` - quality-controlled Netatmo PWS (`core.opensense.pws_qc`) by IDW within 20 km.
   Links: metadata and time-series QC (`core.cml.link_qc`; availability lowered to 60%,
   because OpenRainER's record misses whole minutes) and the nearby-link retrieval
   (`core.cml.estimators.NearbyLinks`, 15-min min/max RSL), sublinks averaged.
3. **Motion** (`nowcast.py`): Lucas-Kanade (4 fields), VET (3), DARTS (8), Proesmans (2) on
   dBR (threshold 0.1 mm/h, -15 dBR below) - the session's settings.
4. **Nowcasts** to 90 min: Eulerian persistence; extrapolation (Lagrangian persistence);
   S-PROG; ANVIL; LINDA (radar, every third issue time - it takes a minute or more); the
   STEPS ensemble, 12 members, nonparametric noise, CDF matching, incremental mask; LINDA-P
   (radar, every sixth issue time).
5. **Issue times**: every hour of each storm with rain over at least 5% of the domain.
6. **Verification** (`verify.py`), pooled over all issue times with pysteps' accumulators:
   POD/FAR/CSI/ETS/bias at 0.5, 1, 5 mm/h; ME/MAE/RMSE/correlation; FSS at 2-50 km; SAL
   (written on `scipy.ndimage`, as pysteps' needs scikit-image); CRPS, ROC area, reliability,
   rank histogram for the ensembles. Against the product's own later field (`own`), the radar
   (`radar`), and the gauges at their cells (`gauges`). **Only cells the product's motion
   reaches from inside the domain are scored** - rain advected in from outside is unknown to
   every method, and counting it as a zero forecast makes persistence look best wherever the
   domain is small.
7. Side studies: the four motion methods (and LK without the dB transform) ranked by the
   extrapolation nowcast they drive; each product's motion against the radar's; and on
   OpenMRG, hourly totals from 5-min scans with and without advection interpolation (session
   block 03a), at the city gauges.

## Findings

All numbers from [`results/report.md`](results/report.md) (pooled over every issue time;
scores on the cells each product's motion can reach from inside the domain).

### The nowcasting methods, on the radar

Against the radar, 1 mm/h; OpenRainER (143 issue times) / OpenMRG (128):

| method | CSI 30 min | CSI 60 min | CSI 90 min | FSS 10 km, 60 min | MAE 60 min (mm/h) |
|---|---|---|---|---|---|
| persistence | 0.41 / 0.30 | 0.27 / 0.21 | 0.20 / 0.12 | 0.51 / 0.52 | 1.14 / 1.03 |
| extrapolation | 0.56 / 0.47 | 0.40 / 0.33 | 0.29 / 0.18 | 0.68 / 0.71 | 0.92 / 0.76 |
| S-PROG | **0.60 / 0.49** | 0.44 / 0.35 | 0.33 / 0.20 | 0.70 / 0.70 | 0.92 / 0.89 |
| ANVIL (rain rate as VIL) | 0.50 / 0.35 | 0.36 / 0.25 | 0.27 / 0.20 | 0.64 / 0.60 | 0.94 / 0.86 |
| LINDA (every 3rd issue time) | 0.59 / 0.49 | **0.47 / 0.41** | **0.38 / 0.27** | 0.72 / 0.73 | 1.00 / 1.74 |

On LINDA's own issue times the others score 0.42 (extrapolation) and 0.47 (S-PROG) at 60
min on OpenRainER, so LINDA's edge is at 90 min (0.38 vs 0.32 / 0.36) and in correlation
(0.47 vs 0.37). Its SAL structure is +0.77 / +1.15 against near 0 for extrapolation: it
concentrates rain into too few, too intense objects, which its MAE shows. At the gauges the
ranking holds on OpenRainER (correlation at 60 min: persistence 0.12, extrapolation 0.25,
LINDA 0.36); on OpenMRG no method correlates with the city gauges at 60 min (0.06-0.14).

Ensembles: STEPS CRPS at 60 min is 0.63 / 0.52 mm/h, ROC area 0.86 / 0.78; LINDA-P on its
subset 0.59 / 0.65 against STEPS 0.60 / 0.52 on the same forecasts - no better, at roughly
70 times the cost.

### Every product

Extrapolation at 60 min (OpenRainER; OpenMRG in the report):

| product | own field: CSI | own field: CSI of persistence | radar: CSI | radar: FSS 10 km | gauges: corr | gauges: RMSE | gauges: ME |
|---|---|---|---|---|---|---|---|
| radar | 0.40 | 0.27 | 0.40 | 0.68 | 0.25 | 4.42 | +0.30 |
| merged | 0.26 | 0.19 | 0.30 | 0.56 | 0.22 | 4.36 | +0.11 |
| cml_idw40 | 0.16 | 0.17 | 0.13 | 0.27 | 0.15 | 4.19 | -0.14 |
| cml_idw20 | 0.16 | 0.17 | 0.07 | 0.15 | 0.08 | 4.19 | -0.25 |
| cml_idw10 | 0.14 | 0.16 | 0.03 | 0.06 | 0.05 | 4.04 | -0.41 |

- A link map's nowcast is no better than its persistence, because its motion field does not
  describe the rain: the median vector difference between LK on the link map and LK on the
  radar equals the radar's own speed on both networks. S-PROG and STEPS inherit the same
  motion; STEPS on link maps has ROC areas of 0.52-0.60 (OpenRainER), barely above chance.
- The lower RMSE of the link maps at the gauges is their dry bias, not skill: they read low
  (ME -0.14 to -0.41 mm/h) with correlations of 0.05-0.15; a forecast of less rain loses less
  on the few heavy gauge-steps.
- The merged field keeps the radar's motion (3 km/h apart on OpenRainER, 19 on OpenMRG,
  where the links cover more of the domain) and its nowcast loses to the radar's only
  against the radar itself. At the gauges it is level with the radar and less biased.
- On OpenMRG the PWS map nowcasts better than the link maps (own-field CSI 0.29, radar CSI
  0.26 against 0.15-0.19) - 30 Netatmo stations spread over the city - but also without gain
  over its persistence (0.30).

### Motion, transforms, accumulation

- Extrapolation CSI at 60 min with LK / VET / Proesmans / DARTS: OpenRainER 0.40 / 0.41 /
  0.40 / 0.30; OpenMRG 0.34 / 0.31 / 0.33 / 0.28. DARTS's field is weak and rotating, as the
  session warned (12 km/h against 75 for LK on the session's case, notebook 03).
- LK on rain rate instead of dBR: within 0.01 CSI on both networks; the transform does not
  matter for these storms.
- Hourly totals at the 10 city gauges, 1,489 gauge-hours: RMSE 1.19 -> 1.15 mm, MAE 0.44 ->
  0.43, corr 0.72 -> 0.73, bias -5.2% -> -4.0%; at gauge-hours of 1 mm or more, RMSE 2.25 ->
  2.18. The smoothing is visible in the maps (notebook 03a); at point gauges it is small.

## Caveats

- **Verifiable area.** Every score counts only cells the product's motion reaches from
  inside the domain (`study.reachable`), the same mask for all methods of a product at an
  issue time and lead. Without it, inflow cells count as forecast zeros and persistence
  "wins" on OpenMRG (an earlier run over two of the storms: CSI 0.43 vs 0.16 for
  extrapolation at 60 min). Different products have different motion, so their masks differ; comparisons across
  products are on different cells. On OpenMRG the 60-90 min radar scores rest on 19-30% of
  an already small domain.
- **The radar is one of the products and also a reference.** Scores "against the radar"
  favour the radar's own nowcast; the gauges are the independent check, and at 2 km cells and
  15-min steps against point gauges every product scores low (correlations 0.05-0.36).
- **Link retrieval.** One retrieval (nearby links), which reads low on these networks
  (`projects/maps/multisensor`). The link maps change every 15 min on OpenMRG (the
  retrieval's step), against the radar's 5 min. Link QC availability was lowered to 60%.
- **The merged field** uses mergeplg 0.1.0's difference block kriging with a radar-fitted
  variogram; within 60 km of the links it removes light radar rain where the links read
  lower, which shows in its SAL amplitude (-0.20).
- **Subsampled methods.** LINDA ran on every third issue time and LINDA-P on every sixth;
  compare them only with the "LINDA subset" rows. LINDA used Shi-Tomasi features (pysteps'
  default blob detector needs scikit-image, not installed). S-PROG/STEPS fail on near-empty
  link maps (singular AR fits): those forecasts are skipped, so link-map ensembles have fewer
  forecasts on OpenMRG.
- **Radar products differ from the session's.** The session nowcast OpenRainER's
  gauge-adjusted radar (RADadj); FieldSense has the unadjusted 15-min rain depth (RADrain).
  Nine storms, all summer.
- **Rerun after two fixes in `core/` (October 2026).** The link and PWS maps now use the 12
  nearest *valid* links or stations at each step (`core.maps.idw.IDW`; before, a missing
  one among the 12 nearest was simply lost), and CRPS is the exact ensemble CRPS
  (`core.nowcast.verify.crps_ensemble`; pysteps' version drops ties and read 1-5% low at
  dry pixels). Radar and merged deterministic scores are unchanged; link and PWS scores
  move by about 1% (up to 0.05 CSI on OpenMRG's link maps). CRPS rises by 0-8% (median per
  product; more at the longest leads, where 6-8 forecasts remain). No finding changes.

## How to run

```bash
pip install -e ".[opensense,notebooks,dev]" && pip install -r projects/nowcasting/pysteps/requirements.txt
python projects/nowcasting/pysteps/src/run.py study --network openrainer   # ~2.5 h; resumes per event
python projects/nowcasting/pysteps/src/run.py study --network openmrg      # ~35 min
python projects/nowcasting/pysteps/src/run.py report                       # -> results/report.md
python -m pytest projects/nowcasting/pysteps/tests                         # synthetic checks, ~10 s
```

Inputs come through `core.data_paths` (`~/data/cml`, or `data/interim`); the OpenRainER radar
months are unpacked from `RADrain.tar` on first use. Per-event radar fields and the study state
are cached under `dataset/open_datasets/_os_nowcasting/`.

## Layout

```
os_nowcasting/
├── src/
│   ├── run.py                    # command line: study, report
│   └── os_nowcasting/
│       ├── settings.py           # networks, events, products, methods
│       ├── grid.py               # square-pixel grid, xarray <-> pysteps, dB transform
│       ├── data.py               # native-step radar and gauges on the grid
│       ├── products.py           # link maps (IDW 10/20/40 km), merged, PWS
│       ├── nowcast.py            # motion, extrapolation, persistence, S-PROG, ANVIL, LINDA, STEPS, LINDA-P
│       ├── interp.py             # advection interpolation for accumulations
│       ├── verify.py             # pooled scores, SAL
│       ├── study.py              # events x issue times x products x methods
│       ├── report.py             # tables, figures, results/report.md
│       └── plots.py              # map panels for the notebooks
├── notebooks/                    # the session's blocks on our data, and the results
│   ├── 02_input_data.ipynb
│   ├── 03_optical_flow_and_extrapolation.ipynb
│   ├── 03a_advection_interpolation.ipynb
│   ├── 04_deterministic_nowcasts.ipynb
│   ├── 04a_probabilistic_nowcasts.ipynb
│   └── 05_results.ipynb
├── results/                      # report.md, score tables (CSV), figures
└── tests/                        # synthetic moving-cell checks
```

## Related

- [`tutorials/07_nowcasting_with_pysteps.ipynb`](../../../tutorials/): the methods used here,
  implemented step by step on the example subsets; [`08_learned_nowcasting.ipynb`](../../../tutorials/):
  a small neural nowcaster trained and scored against pysteps.
- [`spatial_interpolation`](../../spatial_interpolation/): learned nowcasting of CML maps
  (Transformer, GRU, POD-SINDy) with a pysteps Lucas-Kanade baseline, on OpenMRG.
- [`radar_adjustment`](../../maps/radar_adjustment/) and [`multisensor_maps`](../../maps/multisensor/):
  how the merged and link maps nowcast here compare with the radar and gauges as estimates.

## References

1. Pulkkinen, S., et al. (2019). Pysteps: an open-source Python library for probabilistic
   precipitation nowcasting (v1.0). *Geoscientific Model Development*, 12, 4185-4219.
   [doi:10.5194/gmd-12-4185-2019](https://doi.org/10.5194/gmd-12-4185-2019)
2. Germann, U., and Zawadzki, I. (2002). Scale-dependence of the predictability of
   precipitation from continental radar images. Part I. *Monthly Weather Review*, 130,
   2859-2873. [doi:10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2](https://doi.org/10.1175/1520-0493(2002)130%3C2859:SDOTPO%3E2.0.CO;2)
3. Seed, A. W. (2003). A dynamic and spatial scaling approach to advection forecasting.
   *Journal of Applied Meteorology*, 42, 381-388.
   [doi:10.1175/1520-0450(2003)042<0381:ADASSA>2.0.CO;2](https://doi.org/10.1175/1520-0450(2003)042%3C0381:ADASSA%3E2.0.CO;2)
4. Bowler, N. E., Pierce, C. E., and Seed, A. W. (2006). STEPS: a probabilistic precipitation
   forecasting scheme which merges an extrapolation nowcast with downscaled NWP. *Quarterly
   Journal of the Royal Meteorological Society*, 132, 2127-2155.
   [doi:10.1256/qj.04.100](https://doi.org/10.1256/qj.04.100)
5. Pulkkinen, S., Chandrasekar, V., von Lerber, A., and Harri, A.-M. (2020). Nowcasting of
   convective rainfall using volumetric radar observations (ANVIL). *IEEE TGRS*, 58,
   7845-7859. [doi:10.1109/TGRS.2020.2984594](https://doi.org/10.1109/TGRS.2020.2984594)
6. Pulkkinen, S., Chandrasekar, V., and Niemi, T. (2021). Lagrangian integro-difference
   equation model for precipitation nowcasting (LINDA). *Journal of Atmospheric and Oceanic
   Technology*, 38, 2125-2145. [doi:10.1175/JTECH-D-21-0013.1](https://doi.org/10.1175/JTECH-D-21-0013.1)
7. Roberts, N. M., and Lean, H. W. (2008). Scale-selective verification of rainfall
   accumulations from high-resolution forecasts of convective events (FSS). *Monthly Weather
   Review*, 136, 78-97. [doi:10.1175/2007MWR2123.1](https://doi.org/10.1175/2007MWR2123.1)
8. Wernli, H., Paulat, M., Hagen, M., and Frei, C. (2008). SAL - a novel quality measure for
   the verification of quantitative precipitation forecasts. *Monthly Weather Review*, 136,
   4470-4487. [doi:10.1175/2008MWR2415.1](https://doi.org/10.1175/2008MWR2415.1)
9. OpenSense Training School on merging and application of opportunistic rainfall data
   (2025), nowcasting session (J. Ritvanen, R. Imhoff):
   <https://github.com/OpenSenseAction/TrainingSchoolMergingApplication/tree/main/nowcasting-session>
