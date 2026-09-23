# FieldSense

Environmental sensing (ES) using physics-informed AI tools with wireless communication networks.

## Overview

This project explores multi-sensor data fusion for physical field sensing, including spatio-temporal field characterization and integration of wireless link sensors into sensing environments.

## Structure

```
FieldSense/
├── core/                 # Genuinely shared code (protected)
│   ├── itu_p838.py           # ITU-R P.838-3 rain attenuation, used by several projects
│   └── scientific_packages/  # PySINDy / PySR / PyNNcml reference notes + wrapper
├── dataset/              # Shared open datasets (protected)
│   └── open_datasets/        # OpenMRG, OpenRainER, OpenMesh NYC, CML Netherlands
├── projects/             # Research projects — each self-contained, notebooks included
│   ├── cml_retrieval/          # PyNNcml retrieval tutorials on OpenMRG
│   ├── estimation_after_detection/
│   ├── openmesh_nyc/           # OpenMesh NYC: paper, fetch pipelines, notebooks
│   ├── opensense_pipeline/     # Open CML data -> merged rainfall maps (OpenSense)
│   ├── physics_ml/             # Hybrid physics + NN rain retrieval, method tutorials
│   ├── rainfall_field_sim/     # Synthetic rain fields + CML sampling/retrieval
│   └── spatial_interpolation/  # Rainfall nowcasting from CML networks
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

Every project owns its own notebooks, source and results. `core/` holds only
code that more than one project imports; `dataset/` holds the published data
itself, not the analysis of it.

## Installation

```bash
git clone https://github.com/USERNAME/FieldSense.git
cd FieldSense
pip install -r requirements.txt
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

MIT License — see [LICENSE](LICENSE) for details.