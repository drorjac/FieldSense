# core — shared code

Code that **more than one project imports**. Nothing else belongs here.

`core` is an installable package. `pip install -e .` (or `pip install -r
requirements.txt`) from the repo root makes `from core... import` work from any
script or notebook, with no `sys.path` edits.

```
core/
├── itu_p838.py              # ITU-R P.838-3 rain attenuation (k, alpha) tables
├── viz_style.py             # shared palette and matplotlib defaults
├── geo.py                   # lat/lon Domain and MRMS-aligned Grid; NYC and OpenMesh domains
├── asos.py                  # NWS ASOS reports from IEM: hourly rain, present weather (rain/snow/mix)
├── events.py                # precipitation events in an hourly radar record
├── cml/                     # link retrieval methods on a flat link set (from pcpn_maps)
│   ├── estimators.py            # DynamicBaseline, ConstantBaselineSTD, PycomlinkRSD, NearbyLinks, ...
│   ├── rnn.py                   # PyNNcml's two-step RNN as a method: features, HourlyRNN
│   ├── power_law.py, baseline.py, preprocess.py, link_qc.py
├── maps/                    # link and gauge values to fields on a lat/lon grid
│   ├── idw.py                   # midpoint IDW; rates to hour-ending totals
│   ├── gmz.py                   # line IDW and GMZ (Goldshtein-Messer-Zinevich) from path averages
│   └── scores.py                # NRMSE, bias, corr, POD/FAR/CSI; maps and links vs radar
├── simulation/              # synthetic rain fields + CML network sampling
│   ├── rain_fields.py           # stratiform / convective / frontal models, statistics
│   ├── moving_fields.py         # the same fields in time: translation, growth, evolution
│   ├── cml_network.py           # topology, forward model, impairments, retrieval
│   └── reconstruct.py           # IDW variants, scoring, error decomposition
├── opensense/               # OpenSense data: pull, normalize, retrieve, score
│   ├── fetch.py                 # Zenodo full records, resumable + verified
│   ├── example_data.py          # curated OpenSense example subsets, normalized on load
│   ├── openmesh.py              # the full OpenMesh record (links + PWS) as link sets
│   ├── networks.py              # OpenMRG, OpenRainER, OpenMesh: links, point gauges, hourly radar on one grid
│   ├── conventions.py           # units, polarization, projected geometry across sources
│   ├── retrieval.py             # CML attenuation -> rain rate chain (arrays or xarray)
│   ├── wet_dry.py               # radar, nearby-link and CNN wet/dry masks (poligrain, pycomlink)
│   ├── pws_qc.py                # PWS quality control: pypwsqc FZ/HI/SO + a rate check
│   ├── quality.py               # receiver-floor (outage) detection
│   ├── plots.py                 # standard figures (network, retrieval steps, hexbins, maps)
│   └── evaluation.py            # poligrain matching of lines/points/grids + metrics
├── radar/
│   ├── nexrad.py                # KOKX NEXRAD for the OpenMesh NYC days
│   ├── mrms/                    # NOAA MRMS: fetch, decode (ecCodes), crop, cache; rainfall maps
│   └── MRMS.md                  # products, sources, processing, validation
└── scientific_packages/
    ├── PYSINDY.md  PYSR.md  PYNNcml.md   # reference notes
    ├── pynncml_compat.py                 # PyNNcml 0.3.7 workarounds: GMZ bugs, NumPy 2, local OpenMRG
    └── pynncml_rnn.py                    # two-step RNN: loss, windowing, train (with val), predict
```

| Module | Came from | Used by |
|---|---|---|
| `itu_p838` | - | rainfall_field_sim, opensense_pipeline, physics_ml |
| `viz_style` | rainfall_field_sim | rainfall_field_sim, opensense_pipeline, physics_ml |
| `simulation.*` | rainfall_field_sim | rainfall_field_sim, opensense_pipeline, physics_ml, spatial_interpolation |
| `opensense.*` | opensense_pipeline | opensense_pipeline, cml_retrieval, physics_ml, openmesh_nyc (notebook) |
| `radar.nexrad` | openmesh_nyc | openmesh_nyc (notebook), opensense_pipeline |
| `radar.mrms`, `asos`, `geo`, `opensense.openmesh` | pcpn_maps (see `projects/nyc_rain_maps`) | nyc_rain_maps, multisensor_maps, cml_rnn |
| `cml.*`, `maps.idw`, `maps.scores`, `events` | pcpn_maps / nyc_rain_maps | nyc_rain_maps, multisensor_maps, cml_rnn |
| `opensense.networks`, `maps.gmz`, `cml.rnn` | - | multisensor_maps, cml_rnn |
| `scientific_packages.pynncml_compat`, `pynncml_rnn` | cml_retrieval | cml_retrieval, spatial_interpolation |

The command-line tools run as modules from the repo root:

```bash
python -m core.opensense.fetch --list
python -m core.opensense.example_data --dataset openmrg --subset 8d
python -m core.radar.nexrad --classify
```

## `opensense/`

The path from a published CML dataset to a scored rain-rate estimate. Each
module does one stage, and every stage takes and returns OpenSense-1.0
`xarray` objects, so they compose:

```python
from core.opensense import example_data, retrieval as rt, wet_dry, evaluation as ev

data = example_data.load("openmrg", "8d", time=slice("2015-07-28", "2015-07-28"))
cml = data["cml"]                                   # km, GHz, projected endpoints
rain = rt.combine_sublinks(rt.retrieve_dataset(cml)).R
ev.rainfall_metrics(ev.radar_along_links(data["radar"].R, cml),
                    ev.aggregate(rain, "5min"))
```

| module | stage | built on |
|---|---|---|
| `fetch` | full Zenodo records, md5-verified, resumable | `requests` |
| `example_data` | curated subsets; `time=` and `components=` select before reading | ported from `poligrain.example_data` |
| `openmesh` | the full OpenMesh record: `sublinks_table`, `load_links(start, end, links)` as `rsl(link, time)`, `load_pws`; downloads through `fetch` | `netCDF4` |
| `conventions` | m/km, MHz/GHz, polarization spellings, `project_cml`, `project_grid` | `poligrain.spatial` |
| `retrieval` | `retrieve_dataset`, `retrieve_improved`, `combine_sublinks`, and each step as a function | ITU-R P.838-3, `pycomlink` wet-antenna models |
| `wet_dry` | `from_radar`, `nearby_links` (Overeem 2016), `cnn` (Polz 2020), `fill_undecided` | `poligrain`, `pycomlink`, PyTorch |
| `pws_qc` | `flag` (faulty zeros, high influx, station outlier, rate without rain), `summary`, `usable` | `pypwsqc`, `poligrain` |
| `quality` | `censored_at_floor`: receiver outages, where loss is only a lower bound | - |
| `plots` | one function per standard figure, so notebooks stay a sequence of calls | `poligrain.plot_map`, `plot_metadata`, `validation` |
| `evaluation` | `radar_along_links`, `closest_gauges`, `grid_at_points`, `rainfall_metrics`, `skill_table`, `aggregate` (start- or end-stamped bins) | `poligrain.spatial`, `poligrain.validation` |

The wrappers exist because calling poligrain directly has four silent traps
in this setting, each covered by a test in `tests/`: `get_closest_points_to_line`
reads `length` in coordinate units (metres, not the km the files carry),
`GridAtLines`/`GridAtPoints` require lon/lat even in projected mode, the
metadata plots expect metres and MHz and divide by 1000 themselves, and
flattening two DataArrays with different dimension order before scoring
pairs the wrong values. See `projects/opensense_pipeline/README.md` for what
the retrieval variants achieve.

**PWS quality control pays off in station selection.** On the Amsterdam PWS
set (June-August 2016, 45 stations within 2 km of a gauge, hourly, gauges
start-stamped), raw PWS correlate r = 0.02 with the gauges at 14.8x their
total, because a few counter resets dominate; removing flagged steps gives
r = 0.18, and keeping only the 20 stations `pws_qc.usable` accepts gives
r = 0.74 at 0.85x.

`wet_dry.cnn` runs the Polz et al. (2020) network (pinned, BSD-3) with
pycomlink's loader but its own windowing: pycomlink 0.6.0's `cnn_wd` returns
all-NaN under NumPy 2.4 and builds every window as a Python list. The
predictions match pycomlink's to 4e-8.

`pycomlink` is imported lazily, only by the non-default wet-antenna models and
the nearby-link mask.

## `opensense/networks.py`, `cml/`, `maps/`

One interface to three city networks, and the methods that run on it.

```python
from core.opensense.networks import NETWORKS, radar_along_links, points_near_links
net = NETWORKS["openrainer"]                          # also "openmrg", "openmesh"
links = net.links("2021-09-26", "2021-09-27")         # rsl(link, time) at 1 min, tsl where recorded
radar = net.radar_hourly("2021-09-26", "2021-09-27")  # mm, hour-ending, on net.grid
gauges = net.points_hourly("2021-09-26", "2021-09-27")

from core.cml.estimators import study_estimator
from core.maps.idw import accumulate, idw_map
from core.maps.gmz import gmz_map
rain = study_estimator("nearby", links.time[0], links.time[-1]).estimate(links)["rain"]
hourly = accumulate(rain, "1h")
field_idw, field_gmz = idw_map(hourly, net.grid), gmz_map(hourly, net.grid)
```

`networks` checked each source's time-label convention by lagging it against the links
(module docstring). `cml.rnn.HourlyRNN` runs a model trained in `projects/cml_rnn` on any
link set of these networks.

## `radar/`, `asos.py`, `geo.py`

Reference observations over New York City, and the grid they share.

| module | what | built on |
|---|---|---|
| `radar.nexrad` | KOKX NEXRAD reflectivity frames for chosen days, Z-R by rain or snow | IEM archive |
| `radar.mrms` | MRMS products (QPE 1 h Pass 1/2, radar-only, 24 h, rate, PrecipFlag, RQI): download from NOAA AWS with IEM as fallback, decode, crop to a `Domain`, cache per day; `hourly_rainfall`, `event_accumulation`, `rain_rate`, `to_grid`, `sample_points`, `path_average` | `eccodes` (`pip install -e ".[mrms]"`) |
| `asos` | METAR from IEM for NYC, LGA, JFK, EWR: `hourly_precip`, `hourly_ptype` (rain / snow / mix / freezing), 1-min precipitation | `requests` |
| `geo` | `Domain` (lat/lon box) and `Grid` (cell centres on MRMS's 0.01° lattice), `NYC`, `OPENMESH`, `haversine_m` | - |

Downloads are cached under `dataset/open_datasets/MRMS/` and `ASOS/` and not tracked.
`radar/MRMS.md` documents the products, the processing and how the data was validated.

```python
from core.geo import NYC
from core.radar.mrms import hourly_rainfall
from core.asos import fetch_asos, hourly_precip

qpe = hourly_rainfall("2024-01-09 16:00", "2024-01-10 11:00", NYC)       # (time, lat, lon), mm
gauges = hourly_precip(fetch_asos("2024-01-09 16:00", "2024-01-10 11:00"))
```

## `simulation/`

Synthetic rain whose truth is known exactly, and a CML network to measure it.
`rain_fields` builds three regimes (stratiform, convective cells, a frontal
band) at comparable mean rain; `cml_network` samples them with a realistic
link topology and sensor chain; `reconstruct` maps them back.
`moving_fields` adds time:

```python
from core.simulation import moving_fields as mf
from core.simulation.rain_fields import Grid, ConvectiveField

seq = mf.sequence(ConvectiveField(), Grid(n=128, dx_km=0.25), n_steps=13,
                  dt_min=5, evolve_tau_min=60)       # moves at the model's own velocity
seq.frames                                           # (13, 128, 128) mm/h
seq.lagrangian_persistence(t=4, h=3)                 # frame 4 moved 15 min on: the motion-only oracle
seq.predictability(3)                                # how much of +15 min that oracle explains
```

Frozen sequences are pure translation (exact, by a Fourier phase ramp);
evolving ones mix in fresh realizations of the same regime with e-folding
time `evolve_tau_min` and keep wet area and intensities fixed. The frontal
band is not periodic and is moved on a padded domain, so nothing wraps.
`physics_ml` recovers the velocity from these sequences with SINDy,
`spatial_interpolation` benchmarks its forecasters on them, and
`rainfall_field_sim` shows how fast each regime stops being predictable.

## `itu_p838.py`

Specific rain attenuation, `gamma = k * R**alpha`, with ITU-R P.838-3 Table 5
for both polarizations, 1–1000 GHz, interpolated the way the recommendation
prescribes (log-log in `k`, semi-log in `alpha`).

Imported by `projects/rainfall_field_sim/`, `projects/opensense_pipeline/` and
`projects/physics_ml/` — which used to carry its own copy of Table 5 with a
different interpolation, and now re-exports from here.

Anything new should import the table from here rather than paste it.

```python
from core.itu_p838 import get_k_alpha, specific_attenuation
k, alpha = get_k_alpha(23.0, "vertical")
```

Run it directly (`python -m core.itu_p838`) to print a coefficient table.
