# FieldSense

Environmental sensing (ES) using physics-informed AI tools with wireless communication networks.

## Overview

This project explores multi-sensor data fusion for physical field sensing, including spatio-temporal field characterization and integration of wireless link sensors into sensing environments.

## Structure

```
FieldSense/
├── core/                 # Genuinely shared code (protected), installed as a package
│   ├── itu_p838.py           # ITU-R P.838-3 rain attenuation
│   ├── viz_style.py          # shared palette and matplotlib defaults
│   ├── simulation/           # synthetic rain fields + CML network sampling
│   ├── opensense/            # OpenSense data: fetch, conventions, retrieval
│   ├── radar/                # KOKX NEXRAD
│   └── scientific_packages/  # PySINDy / PySR / PyNNcml reference notes + wrapper
├── tests/                # pytest suite for core/ (synthetic data, no downloads)
├── dataset/              # Shared open datasets (protected)
│   └── open_datasets/        # OpenMRG, OpenRainER, OpenMesh NYC, CML Netherlands
├── projects/             # Research projects — each self-contained, notebooks included
│   ├── cml_retrieval/          # PyNNcml retrieval tutorials on OpenMRG
│   ├── estimation_after_detection/
│   ├── mphysics/               # Physics-ML on classical problems (no FieldSense data)
│   ├── openmesh_nyc/           # OpenMesh NYC: paper, fetch pipelines, notebooks
│   ├── opensense_pipeline/     # Open CML data -> merged rainfall maps (OpenSense)
│   ├── physics_ml/             # Hybrid retrieval + equation discovery on CML data
│   ├── rainfall_field_sim/     # Synthetic rain fields + CML sampling/retrieval
│   └── spatial_interpolation/  # Rainfall nowcasting from CML networks
├── pyproject.toml        # makes core/ importable: pip install -e .
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

Every project owns its own notebooks, source and results. `core/` holds only
code that more than one project imports (see [core/README.md](core/README.md));
`dataset/` holds the published data itself, not the analysis of it. Projects
import shared code as `from core... import`, and never from each other.

## Installation

```bash
git clone https://github.com/USERNAME/FieldSense.git
cd FieldSense
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # = pip install -e ".[notebooks]"
pip install -r projects/<project>/requirements.txt   # the project you work on
pip install -e ".[opensense,dev]" && python -m pytest  # core.opensense + its tests
```

New to the OpenSense side? `projects/opensense_pipeline/notebooks/02_end_to_end.ipynb`
takes one day of real CML data from download to a merged rainfall map.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

MIT License — see [LICENSE](LICENSE) for details.