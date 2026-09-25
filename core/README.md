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
├── opensense/               # OpenSense data: download, conventions, retrieval
│   ├── fetch.py                 # Zenodo full records, resumable + verified
│   ├── example_data.py          # curated OpenSense example subsets
│   ├── conventions.py           # unit / polarization normalization across sources
│   └── retrieval.py             # CML attenuation -> rain rate chain
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
