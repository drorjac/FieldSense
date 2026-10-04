# nyc_rain_maps: links, PWS, radar and gauges over New York City

Can a community wireless network map rain over a city? This project turns the signal of NYC
Mesh microwave links (the OpenMesh dataset) into hourly rain maps, and compares them with
every other sensor over the city on the same grid and the same hours:

| sensor | role | data | code |
|---|---|---|---|
| **NYC Mesh links** (CML) | the sensor under test | OpenMesh, 103 sublinks, 1 min | `core.opensense.openmesh` |
| **MRMS radar** | reference map | NOAA/NSSL, gauge-corrected, 0.01°, hourly | `core.radar.mrms` |
| **PWS** | second, denser reference | Weather Underground, 37 stations, ~5 min | `core.opensense.openmesh` |
| **ASOS** | independent point check, never in a map | NWS airports: Central Park, LaGuardia, JFK, Newark | `core.asos` |

Across 52 storms of the OpenMesh record (October 2023 - June 2024) it asks which retrieval
method works, on which links, in rain, snow and mixed precipitation.

## Where it comes from

The retrieval methods are two student implementations of the same task: **implementation 1**
by Gabriela and **implementation 2** by Jeries. Both were unified in the `pcpn_maps`
repository (`drorjac/pcpn_maps`, private), which keeps their original code and checks that every
method reproduces their committed outputs. This project carries the unified methods, rebuilt on
FieldSense's `core/`. The port was checked against `pcpn_maps`: the regenerated study agrees
table for table to the last saved digit, and the link selection is identical.

`docs/METHODS.md` describes each method, where the two implementations differ, and what was
found and fixed in the original code.

## Data

- **OpenMesh links**: [doi:10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692) (CC BY 4.0) and
  **WU PWS**: [doi:10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286) (CC BY-NC 4.0);
  Jacoby et al. (2026), [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026).
- **MRMS** QPE from NOAA's [open-data archive](https://registry.opendata.aws/noaa-mrms-pds/)
  (client and conventions: `core/radar/MRMS.md`).
- **ASOS** one-minute and METAR reports from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/).

All accumulations are hour-ending in UTC. File locations: [`DATA.md`](../../../DATA.md).

## Start here

```bash
pip install -e ".[opensense,mrms,notebooks]"               # from the repository root
pip install -r projects/maps/nyc/requirements.txt
```

| notebook | what it shows |
|---|---|
| `notebooks/01_data.ipynb` | the four sensors on one storm: links, PWS, MRMS, ASOS |
| `notebooks/02_events_and_links.ipynb` | the event catalog (rain / snow / mix) and the automatic link selection |
| `notebooks/03_one_event.ipynb` | one storm end to end: five methods, maps vs radar, every sensor on one grid |
| `notebooks/04_results.ipynb` | the results over all events, read from `results/` |

The first run downloads the OpenMesh record from Zenodo (28 MB) and the MRMS and ASOS data it
needs into the shared store `~/data/cml/openmesh/` (see `DATA.md`); intermediate files go
under `dataset/open_datasets/OpenMesh_NYC/nyc_rain_maps/`. Nothing is tracked.

Command line (`python projects/maps/nyc/src/run.py --help`):

```bash
python projects/maps/nyc/src/run.py event   --start "2024-01-09 16:00" --end "2024-01-10 11:00"   # methods vs MRMS
python projects/maps/nyc/src/run.py compare --start "2024-01-09 16:00" --end "2024-01-10 11:00"   # every sensor
python projects/maps/nyc/src/run.py select        # link selection          -> results/link_selection/
python projects/maps/nyc/src/run.py study         # 10 catalog events       -> results/study/report.md
python projects/maps/nyc/src/run.py all-events    # all 52 events           -> results/all_events/README.md
python projects/maps/nyc/src/run.py validate      # radar audit, vs ASOS    -> results/validation/README.md
python projects/maps/nyc/src/run.py events        # rebuild the catalog (hours of MRMS downloads)
```

## Layout

```
nyc_rain_maps/
├── src/
│   ├── run.py                   # command line
│   └── nyc_rain_maps/
│       ├── (methods, QC, IDW, scores: core/cml/ and core/maps/)
│       ├── link_selection.py    # gauge-calibrated link selection
│       ├── events.py            # event detection and rain/snow/mix classification
│       ├── pipeline.py          # run_event: links -> rain -> maps -> scores vs MRMS
│       ├── compare.py           # CML, PWS, MRMS and ASOS on one grid, pairwise
│       ├── study.py             # events x methods x link sets -> report
│       ├── all_events.py        # every event of the record; rankings and error drivers
│       ├── validation.py        # MRMS fetching audit; every source vs ASOS
│       ├── plots.py, settings.py
├── notebooks/
├── events/all_detected_events.csv   # the event catalog (105 events)
├── results/                     # link_selection/, study/, all_events/, validation/ (generated)
├── docs/                        # METHODS.md, EVENTS.md
└── tests/
```

Shared pieces live in `core/`: the retrieval methods and link QC (`core/cml/`), IDW maps and
scores (`core/maps/`), the MRMS client (`core/radar/mrms/`, documented in
`core/radar/MRMS.md`), ASOS (`core/asos.py`), the OpenMesh full-record loader
(`core/opensense/openmesh.py`) and lat/lon grids (`core/geo.py`).

## How scoring works

MRMS and the PWS are not ground truth, and they fail in different ways: gauges measure rain
directly at a point but vary in quality; radar covers the city evenly but infers rain from
reflectivity aloft. A CML map that agrees with only one has not been validated.

- Every sensor becomes hourly accumulations (hour-ending, UTC) on the same MRMS-aligned 0.01° grid,
  interpolated with the same IDW (power 2, 10 km).
- Only cells and hours where both sides are valid are scored, and the sample size is reported.
- Scores are reported per event and pooled per precipitation type. Pooled scores use the
  cell-hours every full-coverage method has; a method with gaps is scored on its share and its
  coverage is shown, so it cannot look better by staying silent.
- Headline metric: NRMSE, the RMSE divided by the reference mean, so storms of different size
  compare. "Near" scores keep cells within 2 km of a link, where the network constrains the map.

## Findings

From `results/study/report.md` (10 catalog events) and `results/all_events/README.md` (52 events),
both regenerated by `run.py`:

- **Rain works; the gauge-radar gap is the yardstick.** Near the links the PWS map differs from
  MRMS by NRMSE 0.33; the best CML map by 0.57 (bias -9%). CML maps are within a factor of two
  of the disagreement between the two references.
- **The dynamic-baseline methods are the only unbiased ones.** Over the rain events of the record
  the wet/dry-gated methods under-estimate in 96-100% of events (median -49% to -61%); the
  dynamic baseline over- and under-estimates about equally often (median bias +1% and +7%). The two families fail on different storms: the gated
  methods on long stratiform rain, the dynamic baseline when heavy rain lasts longer than its
  200-minute window.
- **Mixed precipitation is over-read, snow is not measured.** In mixed events every method
  over-estimates (+25% to +164% on the selected links): wet snow and melting ice attenuate far more than the liquid-rain
  power law assumes. In snow the PWS read nothing (bias -100%) and no CML method is usable.
- **Choosing links matters more than adding them.** The gauge-calibrated selection recovers 7 of
  the 8 links the authors picked by hand and scores within 0.05 NRMSE of them in 14 of 15 cases;
  the ~25 links that pass automatic QC do worse in rain (NRMSE 0.83 against 0.60), because QC
  removes impossible links but keeps noisy ones.
- **MRMS is a sound reference in rain.** The data fetched here matches NOAA's own products value
  for value, and against the official ASOS gauges MRMS Pass 2 has bias within a few percent in
  rain (`results/validation/`).

## Related projects

- [`openmesh_nyc`](../../data/openmesh_nyc/) - the OpenMesh dataset itself: collection pipelines, the
  paper, and KOKX NEXRAD on rain and snow days.
- [`opensense_pipeline`](../archive_pipeline/) - the same question on OpenMRG and OpenRainER,
  and CML against KOKX NEXRAD over New York (`src/compare_radar_cml.py`).
- [`multisensor_maps`](../multisensor/) - New York side by side with Gothenburg and
  Emilia-Romagna, including the RNN retrieval and merged maps.
- [`radar_adjustment`](../radar_adjustment/) - MRMS radar-only adjusted with the PWS and links,
  scored at ASOS.

## Limitations

- MRMS is a radar estimate: the beam is ~50 km from KOKX, buildings block and clutter it, and
  snow QPE is much less reliable than rain QPE.
- Only two snow and three mixed events fall inside the OpenMesh record (2023-24 was a
  snow-poor winter); treat those rows as indicative.
- IDW from a handful of links cannot resolve convective cells smaller than the link spacing.
- Methods run with the parameters of the original implementations; nothing is tuned against
  the radar.

## References

1. Jacoby, D., et al. (2026). OpenMesh: wireless signal dataset for opportunistic urban
   weather sensing in New York City. *Earth System Science Data*, 18, 5817-5836.
   [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026)
2. Messer, H., Zinevich, A., and Alpert, P. (2006). Environmental monitoring by wireless
   communication networks. *Science*, 312(5774), 713.
   [doi:10.1126/science.1120034](https://doi.org/10.1126/science.1120034)
3. Overeem, A., Leijnse, H., and Uijlenhoet, R. (2016). Retrieval algorithm for rainfall
   mapping from microwave links in a cellular communication network. *Atmospheric Measurement
   Techniques*, 9, 2425-2444. [doi:10.5194/amt-9-2425-2016](https://doi.org/10.5194/amt-9-2425-2016)
