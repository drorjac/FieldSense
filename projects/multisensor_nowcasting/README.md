# multisensor_nowcasting: nowcasting rain with links, radar and weather stations together

**Question.** Commercial microwave links (CMLs) and personal weather stations (PWS) measure
rain closer to the ground than radar, but only where they are. When all three are used to
forecast the next hour - by pysteps or by a neural network - what does each sensor add, and
does learning add anything over physics?

Earlier projects answer parts of this. [`os_nowcasting`](../os_nowcasting/) found that a link
map cannot be nowcast along its own motion and that the radar merged with links nowcasts like
the radar; [`multisensor_maps`](../multisensor_maps/) and
[`radar_adjustment`](../radar_adjustment/) found what links and stations add to a *map*. This
project puts the three sensors into one 5-minute data cube and forecasts from it every way:
pysteps on each product along its own motion or the radar's, STEPS ensembles, and 15 neural
networks - one per combination of sensors and target, and a hybrid that corrects the pysteps
extrapolation - all scored on the same issue times, cells and independent gauges.

## In short

Gothenburg: 645 issue times on 36 storm days; New York: 694 on 34. CSI at 1 mm/h against
the radar at 60 min; next-hour RMSE at the independent gauges; brackets are 95% intervals of
the difference to the radar's own pysteps extrapolation (paired bootstrap over storm days).

| | Gothenburg | New York |
|---|---|---|
| radar extrapolation: CSI 60 min / gauge next-hour RMSE | 0.35 / 1.20 mm | 0.62 / 1.88 mm |
| radar STEPS mean, CSI difference | +0.014 [-0.004, +0.035] | +0.029 [+0.015, +0.042] |
| radar + links + PWS (RCP) along the radar's motion, gauge RMSE difference | **-0.19 [-0.36, -0.03]** | -0.09 [-0.29, +0.13] |
| RCP, STEPS mean, gauge RMSE difference | -0.17 [-0.34, -0.03] | -0.14 [-0.28, +0.03] |
| best learned model at the gauges (U-Net, inputs R+P -> merged RCP) | -0.15 [-0.31, +0.00] | -0.09 [-0.23, +0.09] |
| best learned model against the radar (hybrid), CSI difference | +0.002 [-0.022, +0.023] | -0.017 [-0.028, -0.007] |
| U-Net inputs R -> radar / inputs R+C+P -> radar, CSI difference | -0.054 / -0.054 | -0.070 / -0.115 |

- **Links and PWS improve the forecast where it is measured on the ground, not where it is
  compared with the radar.** Merged into the radar and moved along the radar's motion, they
  lower the next-hour error at Gothenburg's city gauges from 1.20 to 1.02 mm; in New York the
  direction is the same (1.88 to 1.75-1.78 mm) but three ASOS stations cannot resolve it.
  Against the radar every merged product scores slightly *lower* than the radar's own
  extrapolation (CSI -0.05 in Gothenburg, -0.003 to -0.01 in New York): it is compared with the
  field it corrected.
- **A sensor map must move along the radar's motion - when there is room for it.** In Gothenburg,
  where the links and PWS cover most of the domain, the link map along the radar's motion beats
  the same map along its own (CSI 0.26 vs 0.21 at 60 min; gauge RMSE 1.03 vs 1.27 mm); the
  own motion of a link or PWS map differs from the radar's by 42-95 km/h, that of a merged map
  by 19-27 km/h (New York: 63-69 vs 3-4 km/h). In New York the sensors cover 2% of a 240-km box:
  moved along the radar's motion, the city's rain leaves the sensor area within minutes and
  nothing replaces it, so sensor-only maps carry no nowcast skill there (CSI 0.03).
- **Learning adds nothing measurable over physics.** No U-Net beats the radar's pysteps
  extrapolation against the radar on either network; the hybrid that corrects it ties in
  Gothenburg and loses 0.017 in New York. Giving a network the links and PWS does not raise its
  CSI over a radar-only network (Gothenburg -0.054 both; New York -0.070 vs -0.115). At the
  gauges the best learned models - radar and PWS in, the merged field as target - come within
  the intervals of the physics merged product but not past it. Networks without radar (links
  and/or PWS only) are the worst of all; in New York one of them blows up at the gauges.
- **Among the physics methods, the ensemble mean is the safest.** STEPS on the radar has the
  lowest MAE in both networks and the highest CSI in New York; S-PROG adds +0.018 CSI in New
  York and nothing significant in Gothenburg. Over the whole domain the STEPS ensemble's CRPS
  at 60 min is 0.40 (Gothenburg) and 0.59 mm/h (New York) for the radar.
- **The networks' validation-week calibration helps in Gothenburg and hurts in New York.**
  Trained on a rain-weighted error, the raw networks blur and over-forecast (Gothenburg radar
  target: +13% at 60 min; merged target: +93%); matching wet area and volume on the validation
  weeks brings the Gothenburg biases to -9% and +44%, but in New York turns the radar-only
  network's +7% into -43%. Both versions are in the tables (`learned` and `learned_raw`).

## Data

Only two open datasets have links, radar and PWS at the same place and time:

| network | links | PWS | radar | independent reference |
|---|---|---|---|---|
| Gothenburg, JJA 2015 (OpenMRG + OpenMRG2) | 364 links, TSL + RSL ([doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689); Andersson et al. 2022, [doi:10.5194/essd-14-5411-2022](https://doi.org/10.5194/essd-14-5411-2022)) | 30 Netatmo, 5 min ([OpenSenseAction/OpenMRG2](https://github.com/OpenSenseAction/OpenMRG2)) | SMHI C-band composite, 5-min scans | 10 city gauges, 1 min (the SMHI gauge is in the cube, not scored) |
| New York City, Nov 2023 - Jun 2024, 48 rain events (OpenMesh) | NYC Mesh links, RSL ([doi:10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692); Jacoby et al. 2026, [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026)) | 37 Weather Underground PWS ([doi:10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286)) | NOAA MRMS `PrecipRate`, 2 min, over a 240 x 240 km box ([NSSL MRMS](https://www.nssl.noaa.gov/projects/mrms/)) | ASOS 1-min at EWR, JFK, LGA ([IEM](https://mesonet.agron.iastate.edu/request/download.phtml)) |

OpenRainER has no PWS; the Dutch link data (2011-2015) and the Amsterdam PWS (2016-2018) do
not overlap. Files are read through `core.data_paths` (see [DATA.md](../../DATA.md)). New
York's radar is fetched by `python -m core.radar.mrms_nyc_wide` into the shared MRMS cache:
`PrecipRate` at 2 min for every rain event of
`projects/nyc_rain_maps/events/all_detected_events.csv` in the OpenMesh period (padded 3 h
before, 1 h after; mixed events kept when at most 10% of their samples are snow), with
`PrecipFlag` and the ASOS 1-minute record.

## Pipeline

1. **Link rain every 5 minutes** (`links.py`). The best link retrieval in FieldSense is the
   two-step RNN of [`cml_rnn`](../cml_rnn/) (`gru_physics_nbr`), which gives hourly totals. Its
   inputs, cached for every link and hour, include the 60 one-minute values of each link's
   excess loss. The saved model is run on them for hourly totals; the ITU-R power law of each
   minute's excess loss (less a 0.3 dB noise floor) gives the timing within the hour; the
   5-minute rates are that pattern scaled to the RNN's total. Sublinks are averaged.
2. **One cube per network** (`cube.py`): square 2-km pixels (`core.nowcast.grid`; Gothenburg
   44 x 35 over the SMHI composite, New York 120 x 120 over the MRMS box), 5-minute steps
   labelled by their end (UTC). Channels: radar `R` (New York: MRMS cells typed as snow
   blanked); links `C` by line IDW (virtual gauges every km, power 2, within 15 km) with the
   sparse values and an indicator of reporting links; PWS `P` (pypwsqc FZ/HI/SO and a rate
   check, `core.opensense.pws_qc`) by IDW within 15 km, sparse values and indicator; links and
   PWS in one IDW `CP`; and the radar adjusted with links `RC`, PWS `RP` or both `RCP`
   (additive residual IDW as in `core.maps.merge`: the radar along each path for a link, its
   cell for a station; 12 nearest within 15 km; the radar unchanged farther away). Missing
   sensors drop out of every sum, step by step.
3. **Split** (`links.split`): the RNN's own - ISO weeks cycle train, train, validation, test,
   and the `multisensor_maps` storms are always test - so no scored forecast is in a week the
   RNN (whose target included the gauges near each link) or any network here was trained on.
4. **Physics nowcasts** (`physics.py`, pysteps via `core.nowcast`): every 30 minutes of the
   test weeks when more than 5% of the radar domain is wet; for each product persistence,
   extrapolation and S-PROG along its own Lucas-Kanade motion, extrapolation and S-PROG along
   the radar's motion, and 10-member STEPS on `R`, `RCP` and `CP`. 60 minutes ahead in 5-minute
   steps. S-PROG fails on nearly dry sensor maps; there its forecast is the extrapolation
   (counted in the report).
5. **Learned nowcasts** (`learn.py`): a U-Net (~120k weights; RainNet's architecture, Ayzel et
   al. 2020, much smaller) from the last 30 minutes to the next 60, all 12 lead times at once.
   Inputs per step: the `log1p` rate and a mask for each sensor used, plus static coverage.
   Models: inputs `R`, `C`, `P`, `CP`, `RC`, `RP`, `RCP` x target future radar or future `RCP`
   (14), and a hybrid predicting a correction to the pysteps extrapolation of the radar.
   Weighted MSE of `log1p` rates, early stopping on the validation weeks; each model trained
   plain and with random flips and weight decay, the variant with the lower validation loss
   scored.
6. **Verification** (`scoring.py`, `evaluate.py`): every forecast reduced per issue time to
   additive sums - contingency tables at 0.5, 1 and 5 mm/h, the sums behind MAE, RMSE, bias and
   correlation, FSS at 2, 10 and 30 km - against the future radar on the cells the radar's
   motion can reach (one mask for all methods), over the whole domain and within 10 km of a
   link or PWS; and the values at the independent gauges, for 5-minute rates and the next-hour
   total. 95% intervals by a paired bootstrap over storm days.

## How to run

```bash
pip install -e ".[opensense,dev]"                       # repository root
python -m core.radar.mrms_nyc_wide                      # New York radar, flags, ASOS (about 2 h, once)
python projects/multisensor_nowcasting/src/run.py build    --network openmrg   # the cube (~1 min)
python projects/multisensor_nowcasting/src/run.py physics  --network openmrg   # pysteps (~8 min, 7 workers)
python projects/multisensor_nowcasting/src/run.py train    --network openmrg   # 15 models x 2 variants
python projects/multisensor_nowcasting/src/run.py evaluate --network openmrg   # scores at the issue times
python projects/multisensor_nowcasting/src/run.py report                       # results/report.md, figures
```

The same with `--network openmesh`. Everything is cached under
`dataset/open_datasets/_multisensor_nowcasting/` (git-ignored); `--refresh` recomputes.

Runtimes on an 8-core laptop (Apple M2, CPU only): New York radar prefetch 1.9 h; cubes 1 min
(Gothenburg) and 14 min (New York); physics 8 min (Gothenburg, 7 workers) and 60 min (New York,
4 workers); training 2 x 1 h for Gothenburg's 15 models in two variants and 2.6 h for New York's
15 (four at a time); evaluation about 20 min per network.

## Layout

```
multisensor_nowcasting/
├── src/
│   ├── run.py                    # command line: build, physics, train, evaluate, report
│   └── multisensor_nowcasting/
│       ├── settings.py           # networks, grids, products, methods, models
│       ├── links.py              # 5-minute link rain from the cml_rnn RNN; the split
│       ├── cube.py               # radar, links, PWS and merged products on one grid
│       ├── physics.py            # pysteps nowcasts of every product, scored per issue time
│       ├── learn.py              # U-Nets: sensor ablation, two targets, the hybrid
│       ├── scoring.py            # additive scores, pooled scores, day bootstrap
│       ├── evaluate.py           # all forecasts in one table
│       └── report.py             # results/report.md and figures
├── notebooks/                    # 01 data, 02 physics, 03 learning, 04 results
├── results/                      # report.md, CSV tables, figures
└── tests/                        # synthetic checks
```

The pysteps layer (grid and metadata, motion, extrapolation, S-PROG, STEPS, the reachable-cell
mask, pooled verification) is shared with `os_nowcasting` in [`core/nowcast`](../../core/nowcast/).

## Caveats

- **The 5-minute link rain is a disaggregation.** The RNN is trained on hourly totals; the timing
  within the hour comes from the power law of the excess loss. Lagging the link maps against the
  radar puts their best correlation within one 5-minute step of zero (Gothenburg 0.85, New York
  0.79), but 5-minute link rain is not validated against a 5-minute reference.
- **The gauges are few.** Gothenburg: the 10 city gauges at 5 minutes (the SMHI gauge is in the
  cube, hourly, but not scored); New York: the three ASOS stations with 1-minute data (EWR, JFK,
  LGA; Central Park has none). The gauge intervals are wide, and in New York every sensor
  difference at the gauges includes zero.
- **The radar is a reference, not the truth.** Scored against it, the radar's own nowcasts are
  favoured; scored at the gauges, the products carrying the sensors' amounts are. Both are
  reported, never one alone.
- **Small networks, little data.** About 120k weights, 2,300 training samples in Gothenburg and
  1,800 in New York (every second window, 64 x 64 crops, the augmented variant only - fixed
  before any test score was seen, from Gothenburg's validation weeks). Larger models, longer
  records and other losses may change the learning result.
- **S-PROG fails on nearly dry sensor maps** (Gothenburg PWS: 177 of 645 issue times); there its
  forecast is the extrapolation. Counts are in the report.
- **New York's radar is MRMS `PrecipRate`** with cells typed as snow blanked; the events are rain
  or mostly rain. The OpenMesh PWS report every ~5.2 minutes and are binned to 5 minutes.

## Related

- [`tutorials/09_multisensor_nowcasting.ipynb`](../../tutorials/09_multisensor_nowcasting.ipynb):
  the chain step by step on the OpenMRG example subset.
- [`os_nowcasting`](../os_nowcasting/): pysteps on radar, link and merged maps, per product.
- [`spatial_interpolation`](../spatial_interpolation/): Transformer, GRU and POD-SINDy nowcasts of
  link maps.
- [`cml_rnn`](../cml_rnn/): the link retrieval used here; [`radar_adjustment`](../radar_adjustment/)
  and [`multisensor_maps`](../multisensor_maps/): merging links, gauges and radar.

## References

- Andersson, J. C. M. et al. (2022). OpenMRG. *Earth Syst. Sci. Data* 14, 5411-5426. [doi:10.5194/essd-14-5411-2022](https://doi.org/10.5194/essd-14-5411-2022)
- Jacoby et al. (2026). OpenMesh. *Earth Syst. Sci. Data* 18, 5817. [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026)
- de Vos, L. W. et al. (2019). Quality control for crowdsourced personal weather stations. *Geophys. Res. Lett.* [doi:10.1029/2019GL083731](https://doi.org/10.1029/2019GL083731)
- Habi, H. V. & Messer, H. - PyNNcml: [github.com/haihabi/PyNNcml](https://github.com/haihabi/PyNNcml)
- Pulkkinen, S. et al. (2019). Pysteps v1.0. *Geosci. Model Dev.* 12, 4185-4219. [doi:10.5194/gmd-12-4185-2019](https://doi.org/10.5194/gmd-12-4185-2019)
- Seed, A. W. (2003). S-PROG. *J. Appl. Meteor.* 42, 381-388. [doi:10.1175/1520-0450(2003)042<0381:ADASSA>2.0.CO;2](https://doi.org/10.1175/1520-0450(2003)042%3C0381:ADASSA%3E2.0.CO;2)
- Bowler, N. E., Pierce, C. E. & Seed, A. W. (2006). STEPS. *Q. J. R. Meteorol. Soc.* 132, 2127-2155. [doi:10.1256/qj.04.100](https://doi.org/10.1256/qj.04.100)
- Ayzel, G., Scheffer, T. & Heistermann, M. (2020). RainNet v1.0. *Geosci. Model Dev.* 13, 2631-2644. [doi:10.5194/gmd-13-2631-2020](https://doi.org/10.5194/gmd-13-2631-2020)
- Roberts, N. M. & Lean, H. W. (2008). Scale-selective verification of rainfall accumulations (FSS). *Mon. Wea. Rev.* 136, 78-97. [doi:10.1175/2007MWR2123.1](https://doi.org/10.1175/2007MWR2123.1)
- All FieldSense references: [docs/references.md](../../docs/references.md)
