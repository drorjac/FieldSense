# core — shared code

Code that **more than one project imports**. Nothing else belongs here.

```
core/
├── itu_p838.py              # ITU-R P.838-3 rain attenuation (k, alpha) tables
└── scientific_packages/
    ├── PYSINDY.md  PYSR.md  PYNNcml.md   # reference notes
    └── pynncml_wrapper.py                # thin PyNNcml wrapper
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

Run it directly (`python core/itu_p838.py`) to print a coefficient table.

## What is *not* here any more

`core/examples/` used to hold six notebooks spanning three unrelated topics,
and `core/utils.py`, `core/plotting.py` and `core/data_loaders.py` were empty
files advertised as shared utilities. The notebooks moved to the projects that
own them — `projects/physics_ml/notebooks/` and
`projects/cml_retrieval/notebooks/` — and the empty modules were removed. Add
them back when there is something to put in them.
