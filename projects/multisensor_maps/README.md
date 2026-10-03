# multisensor_maps: links, gauges and radar on three networks

The comparison `nyc_rain_maps` makes for New York, made on all three open networks with all
three kinds of sensor. Two studies on the same 10 largest storms of each network:

1. **Maps** (`run.py study`, [`results/report.md`](results/report.md)) - every way of making a
   rain map from the links: which retrieval turns each link's signal into rain (four power laws
   and the RNN of `projects/cml_rnn`), and which interpolation turns the links into a field
   (midpoint IDW, line IDW, GMZ); each against the radar and the gauges.
2. **Merging** (`run.py merge`, [`results/merging/report.md`](results/merging/report.md)) - the
   best map from all three sensors together: radar adjusted with links, gauges or both by seven
   methods, from `pcpn_maps` and from the OpenSense package `mergeplg`, every product scored at
   gauges it never saw.

| network | links | point gauges | radar |
|---|---|---|---|
| Gothenburg, OpenMRG (Jun-Aug 2015) | 728 sublinks, TSL+RSL at 10 s | 30 Netatmo PWS, 10 municipal, 1 SMHI | SMHI C-band, 5 min |
| Emilia-Romagna, OpenRainER (2021-22) | 151 links x 2 channels, 1 min | 319 ARPAE gauges, 15 min | ARPAE composite, 15-min totals |
| New York City, OpenMesh (2023-24) | 103 sublinks, RSL at 1 min | 37 WU PWS, 4 ASOS | MRMS Pass 2 and radar-only, hourly |

Sources: OpenMRG [doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689)
(Andersson et al., 2022) and the OpenMRG2 Netatmo preview; OpenRainER
[doi:10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808); OpenMesh
[doi:10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692) and
[doi:10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286) (Jacoby et al., 2026);
MRMS from [NOAA's open-data archive](https://registry.opendata.aws/noaa-mrms-pds/) and ASOS from
the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/). File locations:
[`DATA.md`](../../DATA.md).

## In short

- **Best link map: the RNN**, on every network, against the radar and at held-out gauges, on
  storms it never saw; the power laws read 20-55% low. Retrieval matters several times more
  than interpolation.
- **Best map overall depends on the network** (hourly NRMSE at held-out gauges):
  Gothenburg - the radar adjusted with the RNN links and the gauges (1.01; radar alone 1.47);
  Emilia-Romagna - the radar adjusted with the gauges by kriging with external drift (1.27;
  radar alone 2.21, gauges alone 1.58); New York - the radar adjusted with the PWS (0.88;
  radar alone 0.93).
- **Links help where the radar is weak and the gauges are few** (Gothenburg); where the gauges
  are dense (Emilia-Romagna, New York) they add nothing or make a merged map worse.
- **No sensor is ground truth.** The radar is worse than the gauges at the gauges in Gothenburg
  and Emilia-Romagna, so every product is scored against both.

# Part 1 - Maps: retrieval x interpolation

## The pipeline

For each network, the 10 largest storms of its study period (by radar domain-mean total):

1. **Links.** Metadata QC (10-100 GHz, 0.3-20 km, inside the domain, one sublink per path),
   then time-series QC over the event (availability, stuck feeds, impossible values).
2. **Retrieval**, link by link, hour-ending totals: the four power-law methods of `core.cml`
   (`dynamic`, `constant`, `pycomlink`, `nearby`), and the RNN trained in `projects/cml_rnn`
   (`rnn`) when a model exists.
3. **Interpolation**, each retrieval three ways onto the radar's grid (`core.maps`): `idw` from
   link midpoints, `line` from virtual gauges along each path, `gmz` the same virtual gauges
   corrected so every link keeps its own average (Goldshtein, Messer & Zinevich). Power 2
   within 10 km for all three, so the difference is the geometry.
4. **Point gauges** mapped with the same IDW; the **radar** as it is.
5. **Scores**: every map against the radar and against every gauge map, on the cell-hours both
   have (cells within 2 km of a link, and all cells); and at each gauge of the check set, the
   radar, every link map and a gauge map rebuilt *without* that gauge.

Everything is hour-ending (UTC). The data layer, `core/opensense/networks.py`, gives the three
networks one interface (`links`, `points`, `radar_hourly`); each source's time-label
convention was checked by lagging it against the links.

## Findings (maps)

From `results/report.md` (10 storms per network; every number computed by `run.py study`).
NRMSE is hourly, against the radar on cells within 2 km of a link, or at held-out gauges within
5 km of one:

| network | gauge map vs radar | best power-law map vs radar | RNN map vs radar | at gauges: other gauges / radar / best power law / RNN |
|---|---|---|---|---|
| Gothenburg | 1.48 (PWS) | 1.33 (`nearby line`, -23%) | **1.31** (`rnn idw`, +16%) | 1.10 / 1.47 / 1.04 / **1.03** |
| Emilia-Romagna | 1.59 (ARPAE) | 1.93 (`nearby idw`, -55%) | **1.57** (`rnn idw`, -10%) | 1.46 / 2.26 / 2.05 / **2.01** |
| New York City | 0.57 (PWS) | 0.98 (`nearby line`, -41%) | **0.66** (`rnn idw`, -11%) | 0.98 / 0.91 / 1.29 / **1.11** |

- **The RNN of `projects/cml_rnn` gives the best link maps on every network**, against the radar
  and at held-out gauges - on storms it never saw. The power-law maps read 20-55% low on these
  storms; the RNN's are within 10-16% against the radar. At Gothenburg's ten municipal gauges the
  margin is thin: lower RMSE and bias for the RNN, slightly higher correlation for nearby-link
  (0.87 vs 0.85).
- **Link maps reach the gauges' own level.** In Gothenburg the best link maps match a gauge
  network at its own gauges (1.03-1.04 against 1.10 for the other municipal gauges), and every
  link map except the dynamic baseline's beats the radar there (1.47). In Emilia-Romagna the RNN
  map is as close to the radar as the 319 ARPAE gauges are (1.57 vs 1.59); in New York it comes
  within 0.1 of the PWS map.
- **Interpolation matters less than retrieval.** Line IDW - virtual gauges along each path - is
  the best interpolation for almost every power-law retrieval, by up to 0.5 NRMSE (the most for
  the noisier methods, whose errors it averages along the path); GMZ's correction to each link's
  own average adds nothing on top of it here; for the RNN, midpoint IDW is best or within 0.03.
  Between the best and worst retrieval the spread is 0.6-1.0, several times that between
  interpolations.
- **The radar is not ground truth either.** At the gauges it is worse than the other gauges in
  Gothenburg (1.47 vs 1.10) and Emilia-Romagna (2.26 vs 1.46), and worse than the best link maps
  there; only in New York is it the best point reference (0.91 vs 0.98 for the other PWS). Hence
  every comparison is made against the radar *and* against the gauges.

# Part 2 - Merging: the best map from links, gauges and radar together

## Design

`run.py merge` (`merging.py`, report in `results/merging/report.md`) asks the question
`pcpn_maps` asked of New York's extreme events (`reports/extremes` there) on all three
networks: which combination of sensors, merged how, gives the best hourly map? On the same 10
storms per network, every product is built on the network's grid:

- the **radar** (New York: MRMS Pass 2, already corrected by NOAA with gauges, and MRMS
  radar-only, which is not); **links** alone (every retrieval, midpoint and line IDW);
  **gauges** alone (IDW); **links + gauges** in one IDW;
- the **radar adjusted** with the gauges, with the links, or with both, by seven methods -
  mean-field bias, additive and multiplicative residual IDW (`core.maps.merge`, ported from
  `pcpn_maps`) and the four methods of the OpenSense package `mergeplg` 0.1.0: difference IDW
  (additive, multiplicative), difference block kriging, and block kriging with external drift
  (`core.maps.mergeplg_methods`). The latter reproduce `mergeplg`'s own `adjust()` exactly
  ([`tests/test_merging.py`](../../tests/test_merging.py), repository root), vectorised and evaluable at any cells, so a study with held-out
  gauges runs in minutes rather than days. The kriging variogram is fitted to each network's
  radar fields (range 53-56 km); `mergeplg`'s default (5 km) changes the scores by at most 0.07.

## Findings (merging)

Scores: the check gauges are split into 5 groups; each is held out in turn, every product that
uses gauges is rebuilt without it and scored at it, hour by hour, all products on the same
station-hours. Hourly NRMSE at held-out gauges (29 storms; New York's 2024-03-06 has one
working link and is left out):

| network | radar | links | gauges | radar + gauges | radar + links | all three |
|---|---|---|---|---|---|---|
| Gothenburg (municipal gauges) | 1.47 | 1.04 | 1.07 | 1.14 | 1.03 | **1.01** (`radar idw_add [rnn + gauges]`) |
| Emilia-Romagna (ARPAE gauges) | 2.21 | 1.83 (<= 5 km) | 1.58 | **1.27** (`radar ked [gauges]`) | 1.82 | 1.39 |
| New York City (PWS) | 0.93 | 1.13 | 0.89 | **0.88** (`radar only idw_mul [gauges]`) | 1.10 | 0.94 |

- **What the links add depends on what else is there.** In Gothenburg, whose radar is the
  weakest sensor, the RNN links carry the map: radar adjusted with the links alone (1.03) beats
  radar adjusted with 40 gauges (1.14), and adding the gauges gains only 0.02 more. In
  Emilia-Romagna, with 319 gauges and a radar that reads 51% high, the gauges fix the radar
  (2.21 to 1.27, KED) and adding the links makes it worse (1.39), also within 5 km of a link
  (1.18 vs 1.35). In New York the PWS are dense and the radar already gauge-corrected; there the
  links make every merged map worse (0.88 vs 0.94).
- **No merging method wins everywhere.** `mergeplg`'s additive difference IDW and kriging
  methods are best with links in Gothenburg; KED is best with gauges in Emilia-Romagna but the weakest in New
  York; in New York the IDW adjustments of both packages tie at 0.88-0.89. Mean-field bias - a
  single factor per hour - is never the best, except with links + PWS in New York. `mergeplg`'s
  multiplicative difference IDW has no bound on the gauge/radar ratio and blows up where the
  radar is near zero (NRMSE up to 428 in Emilia-Romagna); the clipped `pcpn_maps` version does not.
- **Retrieval still matters after merging.** With the radar and links, the RNN is the best
  retrieval in Gothenburg and New York and ties with the power laws in Emilia-Romagna. The
  dynamic baseline is the worst retrieval in every combination but one (New York, radar + links),
  in Gothenburg even with the radar and gauges (2.52 against 1.01 for the RNN).
- **The independent gauges agree:** at Gothenburg's SMHI gauge the links (0.79) and radar +
  links + gauges (0.80) beat the radar (1.58); at New York's ASOS station radar + gauges (0.34)
  beats the radar alone (0.57). Both are single stations (191 and 162 station-hours).

# Running it

## How to run

```bash
pip install -e ".[opensense,mrms,notebooks]"                 # from the repository root
python projects/multisensor_maps/src/run.py events           # the study events per network
python projects/multisensor_maps/src/run.py event --network openrainer --start "2021-09-26 06:00" --end "2021-09-26 19:00"
python projects/multisensor_maps/src/run.py study            # -> results/report.md
python projects/multisensor_maps/src/run.py merge            # -> results/merging/report.md
python projects/multisensor_maps/src/run.py merge --network openmrg --refresh   # one network, rescored
```

The full records must be downloaded first (`python -m core.opensense.fetch --dataset openmrg`,
`openrainer`, `openmesh`, `openmesh_pws`; OpenMRG's PWS come from `openmrg2_pws`). Radar is
cached per month under `dataset/open_datasets/<network>/processed/networks/`.

| notebook | |
|---|---|
| `notebooks/01_three_networks.ipynb` | the three networks, one storm each mapped from every sensor |
| `notebooks/02_results.ipynb` | the study's pooled scores, read from `results/` |

## Layout

```
multisensor_maps/
├── src/run.py
├── src/multisensor_maps/
│   ├── event.py       # run_event: QC'd links -> retrievals -> idw/line/gmz maps; gauges; radar; scores
│   ├── study.py       # the 10 largest events per network, pooled scores
│   ├── report.py      # results/report.md
│   ├── merging.py     # links + gauges + radar merged every way, scored at held-out gauges
│   ├── merging_report.py  # results/merging/report.md and figures
│   ├── plots.py, settings.py
├── notebooks/
└── results/           # report.md, pooled and per-event scores, figures
    └── merging/       # the merging study: report.md, rankings, figures
```

Intermediate files (not tracked) are under `dataset/open_datasets/_multisensor_maps/`: the
merging study caches every storm's inputs and station-hour scores in `merging/`, so after the
first run (about an hour) `run.py merge` rebuilds the report in a few minutes;
`--refresh` rescores. What each results file holds: [`results/README.md`](results/README.md).

## Related

- [`radar_adjustment`](../radar_adjustment/): the OpenSense radar-adjustment intercomparison
  reproduced over whole summers with mergeplg `main` (eight methods, three range-check
  settings, RADOLAN), with links and weather stations as adjusters. It complements Part 2 here, which
  scores ten storms per network with held-out gauges.
- [`opensense_pipeline`](../opensense_pipeline/): the first merging benchmark on OpenMRG and
  OpenRainER, with the mergeplg 0.1.0 and `main` comparison.
- [`cml_rnn`](../cml_rnn/): the RNN retrieval used here.
- [`tutorials/05_2d_rain_maps.ipynb`](../../tutorials/) and
  [`06_radar_adjustment.ipynb`](../../tutorials/): the interpolation and merging methods,
  step by step.

## References

1. Goldshtein, O., Messer, H., and Zinevich, A. (2009). Rain rate estimation using
   measurements from commercial telecommunications links. *IEEE Transactions on Signal
   Processing*, 57(4), 1616-1625. [doi:10.1109/TSP.2009.2012554](https://doi.org/10.1109/TSP.2009.2012554)
2. Overeem, A., Leijnse, H., and Uijlenhoet, R. (2016). Retrieval algorithm for rainfall
   mapping from microwave links in a cellular communication network. *Atmospheric Measurement
   Techniques*, 9, 2425-2444. [doi:10.5194/amt-9-2425-2016](https://doi.org/10.5194/amt-9-2425-2016)
3. Andersson, J. C. M., et al. (2022). OpenMRG. *Earth System Science Data*, 14, 5411-5426.
   [doi:10.5194/essd-14-5411-2022](https://doi.org/10.5194/essd-14-5411-2022)
4. Jacoby, D., et al. (2026). OpenMesh. *Earth System Science Data*, 18, 5817-5836.
   [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026)
5. Software: [mergeplg](https://github.com/OpenSenseAction/mergeplg),
   [poligrain](https://github.com/OpenSenseAction/poligrain),
   [pycomlink](https://github.com/pycomlink/pycomlink),
   [PyKrige](https://github.com/GeoStat-Framework/PyKrige).
