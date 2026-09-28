# multisensor_maps: links, gauges and radar on three networks

The comparison `nyc_rain_maps` makes for New York, made on all three open networks with
all three kinds of sensor - and with the rain map built every way that matters: which
retrieval turns each link's signal into rain, and which interpolation turns the links into a
field.

| network | links | point gauges | radar |
|---|---|---|---|
| Gothenburg, OpenMRG (Jun-Aug 2015) | 728 sublinks, TSL+RSL at 10 s | 30 Netatmo PWS, 10 municipal, 1 SMHI | SMHI C-band, 5 min |
| Emilia-Romagna, OpenRainER (2021-22) | 151 links x 2 channels, 1 min | 319 ARPAE gauges, 15 min | ARPAE composite, 15-min totals |
| New York City, OpenMesh (2023-24) | 103 sublinks, RSL at 1 min | 37 WU PWS, 4 ASOS | MRMS Pass 2, hourly |

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

## Run

```bash
pip install -e ".[opensense,mrms,notebooks]"                 # from the repository root
python projects/multisensor_maps/src/run.py events           # the study events per network
python projects/multisensor_maps/src/run.py event --network openrainer --start "2021-09-26 06:00" --end "2021-09-26 19:00"
python projects/multisensor_maps/src/run.py study            # -> results/report.md
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
│   ├── plots.py, settings.py
├── notebooks/
└── results/           # report.md, pooled and per-event scores, figures
```

## Findings

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
