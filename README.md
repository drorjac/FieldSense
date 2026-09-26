# FieldSense

Rainfall sensing with the networks that already exist. Commercial microwave
links (CMLs) lose signal in rain; FieldSense turns that loss into rain rates
and rainfall maps, fuses it with gauges, personal weather stations and radar,
and uses physics-informed machine learning to retrieve, reconstruct and
nowcast the field.

## Projects

Each project is self-contained: its own `src/`, notebooks, results and
README. Start with the entry point listed.

| project | question | start here |
|---|---|---|
| [`opensense_pipeline`](projects/opensense_pipeline/) | open CML data (OpenMRG, OpenRainER) to merged rainfall maps, scored against radar and gauges | `notebooks/02_end_to_end.ipynb` |
| [`cml_retrieval`](projects/cml_retrieval/) | PyNNcml retrieval on OpenMRG: model-driven chain, two-step RNN, five map methods | `notebooks/model_driven_retrieval.ipynb` |
| [`physics_ml`](projects/physics_ml/) | hybrid physics + neural retrieval; rediscovering ITU-R P.838 (PySR) and advection (SINDy) | `notebooks/hybrid_retrieval.ipynb` |
| [`rainfall_field_sim`](projects/rainfall_field_sim/) | three rain regimes, a simulated CML network: what limits the reconstruction, sensors or geometry? | `python src/run_demo.py` |
| [`spatial_interpolation`](projects/spatial_interpolation/) | nowcasting from CML-derived maps: Transformer, GRU, POD-SINDy vs persistence | `notebooks/nowcasting.ipynb` |
| [`openmesh_nyc`](projects/openmesh_nyc/) | the OpenMesh NYC dataset: links, PWS, ASOS, radar; paper | `notebooks/openmesh_data.ipynb` |
| [`mphysics`](projects/mphysics/) | physics-ML methods on classical problems (n-body, PINNs); no rain data | `notebooks/nbody_discovery.ipynb` |
| [`estimation_after_detection`](projects/estimation_after_detection/) | placeholder | - |

Notebooks are short: each cell is a call into the project's `src/` or into
`core/`, and every number shown is computed. The originals they replaced are
kept in each project's `notebooks/archive/`.

## Layout

```
FieldSense/
├── core/                 # code more than one project imports (see core/README.md)
│   ├── opensense/            # OpenSense data: fetch, conventions, retrieval, wet/dry, evaluation
│   ├── simulation/           # synthetic rain fields, moving fields, CML network, reconstruction
│   ├── scientific_packages/  # PyNNcml compatibility and RNN training; PySINDy/PySR notes
│   ├── radar/                # KOKX NEXRAD
│   ├── itu_p838.py           # ITU-R P.838-3 rain attenuation
│   └── viz_style.py          # shared palette and matplotlib defaults
├── projects/             # the research projects above
├── tests/                # pytest for core/ (synthetic data, no downloads)
├── dataset/open_datasets/  # OpenMRG, OpenRainER, OpenMesh NYC, CML Netherlands (data not in git)
└── pyproject.toml        # installs core/ as a package
```

Projects import shared code as `from core... import` and never from each
other; code moves into `core/` when a second project needs it.

## Installation

```bash
git clone git@github.com:drorjac/FieldSense.git && cd FieldSense
python -m venv .venv && source .venv/bin/activate
pip install -e ".[opensense,notebooks,dev]"
pip install -r projects/<project>/requirements.txt    # the project you work on
python -m core.opensense.fetch --list                 # open datasets, downloaded on request
```

## Tests

```bash
python -m pytest        # core and every project's tests, offline, about 30 s
```

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT License, see [LICENSE](LICENSE).
