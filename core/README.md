# core — shared code

Code that **more than one project imports**. Nothing else belongs here.

`core` is an installable package. `pip install -e .` (or `pip install -r
requirements.txt`) from the repo root makes `from core... import` work from any
script or notebook, with no `sys.path` edits.

```
core/
├── itu_p838.py              # ITU-R P.838-3 rain attenuation (k, alpha) tables
├── viz_style.py             # shared palette and matplotlib defaults
├── simulation/              # synthetic rain fields + CML network sampling
│   ├── rain_fields.py           # stratiform / convective / frontal models, advection
│   ├── cml_network.py           # topology, forward model, impairments, retrieval
│   └── reconstruct.py           # IDW variants, scoring, error decomposition
├── opensense/               # OpenSense data: pull, normalize, retrieve, score
│   ├── fetch.py                 # Zenodo full records, resumable + verified
│   ├── example_data.py          # curated OpenSense example subsets, normalized on load
│   ├── conventions.py           # units, polarization, projected geometry across sources
│   ├── retrieval.py             # CML attenuation -> rain rate chain (arrays or xarray)
│   ├── wet_dry.py               # radar and nearby-link wet/dry masks (poligrain, pycomlink)
│   ├── quality.py               # receiver-floor (outage) detection
│   └── plots.py                 # standard figures (network, retrieval steps, hexbins, maps)
│   └── evaluation.py            # poligrain matching of lines/points/grids + metrics
├── radar/
│   └── nexrad.py                # KOKX NEXRAD for the OpenMesh NYC days
└── scientific_packages/
    ├── PYSINDY.md  PYSR.md  PYNNcml.md   # reference notes
    └── pynncml_wrapper.py                # thin PyNNcml wrapper
```

| Module | Came from | Used by |
|---|---|---|
| `itu_p838` | - | rainfall_field_sim, opensense_pipeline, physics_ml |
| `viz_style` | rainfall_field_sim | rainfall_field_sim, opensense_pipeline, physics_ml |
| `simulation.*` | rainfall_field_sim | rainfall_field_sim, opensense_pipeline, physics_ml |
| `opensense.*` | opensense_pipeline | opensense_pipeline, cml_retrieval, physics_ml, openmesh_nyc (notebook) |
| `radar.nexrad` | openmesh_nyc | openmesh_nyc (notebook), opensense_pipeline |

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
| `conventions` | m/km, MHz/GHz, polarization spellings, `project_cml`, `project_grid` | `poligrain.spatial` |
| `retrieval` | `retrieve_dataset`, `retrieve_improved`, `combine_sublinks`, and each step as a function | ITU-R P.838-3, `pycomlink` wet-antenna models |
| `wet_dry` | `from_radar`, `nearby_links` (Overeem 2016), `fill_undecided` | `poligrain`, `pycomlink` |
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

`pycomlink` is imported lazily, only by the non-default wet-antenna models and
the nearby-link mask.

## `itu_p838.py`

Specific rain attenuation, `gamma = k * R**alpha`, with ITU-R P.838-3 Table 5
for both polarizations, 1–1000 GHz, interpolated the way the recommendation
prescribes (log-log in `k`, semi-log in `alpha`).

Imported by `projects/rainfall_field_sim/`, `projects/opensense_pipeline/` and
`projects/physics_ml/` — which used to carry its own copy of Table 5 with a
different interpolation, and now re-exports from here.

One copy remains outside this module: the table is pasted inline in
`projects/physics_ml/notebooks/Simulation_MBML.ipynb`. That is a working
experiment rather than library code, so it is left as it is; anything new
should import from here.

```python
from core.itu_p838 import get_k_alpha, specific_attenuation
k, alpha = get_k_alpha(23.0, "vertical")
```

Run it directly (`python -m core.itu_p838`) to print a coefficient table.

## What is *not* here any more

`core/examples/` used to hold six notebooks spanning three unrelated topics,
and `core/utils.py`, `core/plotting.py` and `core/data_loaders.py` were empty
files advertised as shared utilities. The notebooks moved to the projects that
own them — `projects/physics_ml/notebooks/` and
`projects/cml_retrieval/notebooks/` — and the empty modules were removed. Add
them back when there is something to put in them.
