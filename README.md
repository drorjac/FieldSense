# FieldSense

Rainfall sensing with the networks that already exist.

Rain gauges are sparse, and weather radar measures rain indirectly, high above the ground.
Networks that are already everywhere can fill the gap: the microwave links between
mobile-phone towers (commercial microwave links, CMLs) lose signal in rain, and personal
weather stations (PWS) are dense and free. FieldSense turns the signal loss into rain rates
and rainfall maps, merges them with gauges and radar, forecasts the field a short time ahead,
and tests where machine learning helps and where physics is enough.

## Where to start

| if you want to | read |
|---|---|
| set up your copy and open a first dataset | [GETTING_STARTED.md](GETTING_STARTED.md) |
| learn the methods, step by step, on real data | [`tutorials/`](tutorials/): eleven executed notebooks, from the open datasets to multi-sensor nowcasting and corrected link maps |
| see what is in each open dataset | [`examples/`](examples/) |
| know which files are read, and where they live | [DATA.md](DATA.md), [`dataset/README.md`](dataset/README.md) |
| find a paper, dataset or package | [docs/references.md](docs/references.md) |

## The chain

The projects are grouped by stage of the chain. Each stage folder has a README with its
question, its subprojects and their headline numbers.

```
 open data ──► signal loss to rain rate ──► rainfall map, merged with radar and stations ──► forecast
 data/          retrieval/                   maps/                                             nowcasting/

 known truth for every stage: simulation/        physics found from data: physics_ml/
```

| stage | subproject | question | start here |
|---|---|---|---|
| [`data`](projects/data/) | [`openmesh_nyc`](projects/data/openmesh_nyc/) | the OpenMesh NYC dataset: links, PWS, ASOS, radar; collection pipelines and data paper | `notebooks/openmesh_data.ipynb` |
| [`retrieval`](projects/retrieval/) | [`openmrg`](projects/retrieval/openmrg/) | signal loss to rain rate with PyNNcml on OpenMRG: model-driven chain vs a two-step RNN; five map methods | `notebooks/model_driven_retrieval.ipynb` |
| | [`rnn_three_networks`](projects/retrieval/rnn_three_networks/) | PyNNcml's two-step RNN trained on three networks against radar and gauges, head to head with the power law | `notebooks/02_rnn_vs_power_law.ipynb` |
| [`maps`](projects/maps/) | [`multisensor`](projects/maps/multisensor/) | links, gauges and radar on one grid on three networks: retrieval (power law, RNN) x interpolation (IDW, line IDW, GMZ), and the three merged every way | `notebooks/01_three_networks.ipynb` |
| | [`radar_adjustment`](projects/maps/radar_adjustment/) | the OpenSense radar-adjustment intercomparison reproduced from the raw archives, and extended to weather stations, RADOLAN and New York | `notebooks/01_intercomparison.ipynb` |
| | [`nyc`](projects/maps/nyc/) | New York City: NYC Mesh link maps against MRMS radar, PWS and ASOS gauges over 52 storms of rain, snow and mix | `notebooks/01_data.ipynb` |
| | [`wet_area`](projects/maps/wet_area/) | maps that can be dry: a wet/dry mask from the links before interpolation, on simulated truth and 29 storms | `notebooks/01_wet_area.ipynb` |
| | [`link_weights`](projects/maps/link_weights/) | links weighted by their expected error, learned by length on half the storms, when mapped | `notebooks/01_link_weights.ipynb` |
| | [`learned_2d`](projects/maps/learned_2d/) | (proposed project, starter) learned link-to-map models: an OpenMRG dataset split by storm, IDW/OK/GMZ baselines with the proposal's metrics, a minimal U-Net, simulated truth | `notebooks/01_dataset.ipynb` |
| | [`netherlands`](projects/maps/netherlands/) | the whole Dutch network (Overeem et al. 2024) through RAINLINK, summer 2012: paths and IDW maps against KNMI's hourly gauges | `notebooks/01_netherlands.ipynb` |
| | [`archive_pipeline`](projects/maps/archive_pipeline/) | (frozen) the first end-to-end version: raw open data (OpenMRG, OpenRainER) to merged rainfall maps | `notebooks/02_end_to_end.ipynb` |
| [`nowcasting`](projects/nowcasting/) | [`pysteps`](projects/nowcasting/pysteps/) | the OpenSense pysteps nowcasting session as a study: radar, link and merged maps nowcast with every pysteps method | `notebooks/05_results.ipynb` |
| | [`multisensor`](projects/nowcasting/multisensor/) | links, radar and weather stations nowcast together, by pysteps and by neural networks, in Gothenburg and New York: what each sensor adds, and whether learning adds anything | `notebooks/04_results.ipynb` |
| | [`spatial_interpolation`](projects/spatial_interpolation/) | nowcasting 15-60 min ahead from CML maps: Transformer, GRU, POD-SINDy vs persistence (paper) | `notebooks/nowcasting.ipynb` |
| [`simulation`](projects/simulation/) | [`regimes`](projects/simulation/regimes/) | on simulated rain: does the error come from the sensors or from where the links are? | `notebooks/01_regimes_and_reconstruction.ipynb` |
| | [`testbed`](projects/simulation/testbed/) | a rain simulator seen by radar, links, gauges and PWS: every map, merging, motion and nowcast method scored against the truth, from street level to city scale | `notebooks/01_generators_and_motion.ipynb` |
| [`physics_ml`](projects/physics_ml/) | [`discovery`](projects/physics_ml/discovery/) | hybrid physics + neural retrieval; rediscovering the ITU-R rain law (PySR) and advection (SINDy) | `notebooks/hybrid_retrieval.ipynb` |
| | [`path_law_1d`](projects/physics_ml/path_law_1d/) | (proposed project, starter) the path law f(R, L) and a wet antenna with memory: simulated and OpenMRG data, scored baselines, where PySR and SINDy plug in | `notebooks/01_simulated_path_law.ipynb` |

Proposed projects that build on FieldSense, and what it gives them, are in
[docs/pre_projects/](docs/pre_projects/README.md).

## Findings so far

Each is computed in the project named, where the details and caveats are.

- **A trained network beats the power law, per link and in maps.** PyNNcml's two-step RNN,
  trained on three networks against the average of radar and gauges and given the power-law
  rate of the excess loss and what the neighbouring links see, has lower error and higher
  correlation than every power-law method on held-out weeks in Gothenburg, Emilia-Romagna and
  New York - against the radar alone and each gauge network alone (e.g. New York: correlation
  0.78 against 0.50 for the best power law). On the ten largest storms of each network its maps
  are the best link maps against the radar and at held-out gauges. *(retrieval/rnn_three_networks, maps/multisensor)*
- **Retrieval matters more than interpolation.** Across three networks, line IDW (virtual gauges
  along each path) is the best interpolation for power-law retrievals and GMZ adds nothing to it,
  but the spread between retrievals is several times the spread between interpolations.
  *(maps/multisensor)*
- **What links add to a merged map depends on the other sensors.** Merging radar, links and
  gauges every way (seven methods from `pcpn_maps` and `mergeplg`) on 29 storms, scored at
  held-out gauges: in Gothenburg, where the radar is weakest, radar + RNN links is best (NRMSE
  1.01 with the gauges, 1.03 without, radar 1.47); in Emilia-Romagna the 319 gauges fix a radar
  that reads 51% high (KED, 2.21 to 1.27) and the links add nothing; in New York the PWS
  and gauge-corrected radar are best (0.88) and links make merged maps worse. *(maps/multisensor)*
- **Merging helps only where the links are weak.** On the dense Swedish
  network the links alone beat radar and every merge against held-out gauges
  (RMSE 4.45 vs 6.16 mm/h); on the sparse Italian network merging only edges
  radar (7.90 vs 8.00), and all of the gain is within 5 km of a link.
  *(maps/archive_pipeline)*
- **The OpenSense radar-adjustment intercomparison reproduces exactly from the raw data,**
  but only with a newer `mergeplg` than the one its repository pins. Every adjustment beats
  the radar at independent gauges (hourly RMSE, Gothenburg 1.44 to 1.27 mm; Emilia-Romagna
  3.58 to 3.08, radar bias +107% to -7%), almost all of the gain is near the links, and
  without range checks multiplicative adjustment breaks down. Weather stations beat links
  as adjusters wherever they exist: in New York, MRMS radar-only adjusted with the PWS by
  RADOLAN (1.39) beats NOAA's gauge-corrected MRMS (1.59). *(maps/radar_adjustment)*
- **A link map cannot be nowcast on its own; a radar map merged with links can.** pysteps
  extrapolation of an IDW link map is no better than holding it still (CSI 0.16 vs 0.17 at
  60 min): the blobs are anchored to the links and their apparent motion is unrelated to the
  rain's. Merged into the radar, the links keep the radar's motion and nowcast as well as
  the radar. On the radar, S-PROG adds a little to plain extrapolation (CSI at 1 mm/h,
  60 min: 0.44 vs 0.40, persistence 0.27). *(nowcasting/pysteps)*
- **Links and weather stations improve the next-hour forecast at the ground; learning does
  not improve on pysteps.** Merged into the radar and moved along the radar's motion, they
  lower the next-hour error at Gothenburg's independent gauges from 1.20 to 1.02 mm
  (difference -0.19, 95% interval -0.36 to -0.03); New York points the same way but its three
  ASOS stations cannot resolve it. None of 15 U-Nets, with any combination of radar, links and
  stations as inputs, beats the radar's pysteps extrapolation on either network.
  *(nowcasting/multisensor)*
- **The wet-antenna correction sets the magnitude of retrieved rain,** and
  its default does not transfer between networks: the ratio to the OpenSense
  reference moves from 1.93 to 0.74 across plausible settings.
  *(maps/archive_pipeline)*
- **Radar is not always the reference.** Over Manhattan the links correlate
  +0.53 with PWS and about 0 with the KOKX radar, whose beam passes far above
  the city; in snow the links carry no precipitation signal.
  *(maps/archive_pipeline)*
- **Over New York, link maps sit within twice the gauge-radar gap.** Near the links the PWS map
  differs from MRMS radar by NRMSE 0.33 and the best link map by 0.57; the dynamic-baseline
  methods are the only unbiased ones. Mixed precipitation is over-read by every method, and
  snow is measured by neither links nor PWS. *(maps/nyc)*
- **A country-wide network maps hourly rain well at the gauges.** The whole Dutch network
  (~2800 paths, summer 2012) through RAINLINK, mapped by IDW and scored at KNMI's 31 covered
  gauges: hourly correlation 0.77, daily r² 0.72, bias -3%. Single paths are noisier, and a
  few shorter than 1 km read several times the gauge's rain. *(maps/netherlands)*
- **Geometry, not sensor physics, limits the map.** At 90 links the sensor
  chain changes the reconstruction error by about 1% or less; almost all of
  it comes from rain that falls between links. *(simulation/regimes)*
- **On simulated rain, no map resolves a street over five minutes; near the links, links halve
  the error.** In a simulated city at 100 m and 1 min, the best map of 5-min totals (radar
  adjusted with links and gauges by KED) has an error larger than the rain itself (NRMSE 1.12,
  radar alone 1.60), barely better at 800 m, 0.51 at 6.4 km and 0.38-0.46 over an hour. Within
  250 m of a link it is 0.85, half the radar's. On a 64 km domain the gauges carry the radar
  adjustment instead (0.44 vs 0.62 in 23 of 24 scenarios). VET recovers the true motion best,
  DARTS underestimates speed even for pure translation, and a motion network trained only on
  simulations nowcasts real OpenMRG radar as well as VET. *(simulation/testbed)*
- **Machine learning recovers part of the physics.** PySR finds the ITU-R
  exponent from real data (0.911 vs 0.913) but not the prefactor (83% high,
  the wet-antenna offset again); SINDy finds the advection velocity from
  exact fields but not through a CML network; the hybrid retrieval does not
  beat its own physics branch. *(physics_ml/discovery)*

## Data

Nothing is stored in the repository. [`examples/`](examples/) and the tutorials read the small
OpenSense example subsets
([OpenSenseAction/opensense_example_data](https://github.com/OpenSenseAction/opensense_example_data)),
which download on first use, about 60 MB in all:

```python
from core.opensense import example_data
data = example_data.load("openmrg", "8d")    # {"cml": ..., "radar": ..., "gauge_municipal": ..., ...}
```

The projects read the full published records, downloaded once into `~/data/cml/`
(`python -m core.opensense.fetch --dataset <name>`). [DATA.md](DATA.md) lists every file
and what reads it; [`dataset/README.md`](dataset/README.md) gives each dataset's source,
licence and citation.

## Layout

```
FieldSense/
├── GETTING_STARTED.md    # start here
├── tutorials/            # eleven executed notebooks: data, retrieval, training, radar, maps, merging, nowcasting, simulation, dry maps
├── examples/             # one notebook per OpenSense example dataset
├── projects/             # the projects above, by stage: projects/<stage>/<subproject>/ with src/, notebooks/, results/, README
├── core/                 # code more than one project imports (see core/README.md)
│   ├── opensense/            # data: fetch, example subsets, three networks on one grid, retrieval, PWS QC, the Dutch network
│   ├── cml/                  # link retrieval: power law (four variants), PyNNcml RNN
│   ├── maps/                 # IDW, line IDW, GMZ, merging (pcpn_maps and mergeplg methods), scores
│   ├── radar/                # KOKX NEXRAD and MRMS for New York
│   ├── simulation/           # rain generators, flows and evolution, radar/CML/gauge simulators, benchmark
│   ├── scientific_packages/  # PyNNcml compatibility and RNN training; PySINDy/PySR notes
│   ├── asos.py, geo.py       # NWS airport gauges; lat/lon domains and grids
│   ├── itu_p838.py           # ITU-R P.838-3 rain attenuation
│   └── viz_style.py          # shared palette and matplotlib defaults
├── docs/references.md    # every dataset, package and paper, with links
├── dataset/              # dataset documentation; data is downloaded, never committed
├── DATA.md               # every input file FieldSense reads, and where it lives
├── tests/                # pytest for core/ (synthetic data, no downloads)
└── pyproject.toml        # installs core/ as a package
```

Projects import shared code as `from core... import` and never from each other; code moves
into `core/` when a second project needs it. Notebooks stay short: each cell calls into the
project's `src/` or `core/`, and every number shown is computed.

## Installation

```bash
git clone https://github.com/<your-username>/FieldSense.git && cd FieldSense   # your fork
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[opensense,notebooks,dev]"
pip install -r projects/<project>/requirements.txt    # the project you work on
```

`projects/maps/radar_adjustment` and parts of `projects/maps/archive_pipeline` need the development
version of `mergeplg`, which replaces the released 0.1.0; they run in a second environment,
`.venv-mergeplg-main`, set up as described in
[`projects/maps/archive_pipeline/README.md`](projects/maps/archive_pipeline/README.md).

## Tests

```bash
python -m pytest        # core and every project's tests, offline, about 1.5 min
```

Tests that need the development `mergeplg` are skipped in `.venv` and run in
`.venv-mergeplg-main`.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT License, see [LICENSE](LICENSE).
