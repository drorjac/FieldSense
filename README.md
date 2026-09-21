# FieldSense

Environmental sensing (ES) using physics-informed AI tools with wireless communication networks.

## Overview

This project explores multi-sensor data fusion for physical field sensing, including spatio-temporal field characterization and integration of wireless link sensors into sensing environments.

## Structure

```
FieldSense/
├── core/                 # Shared utilities (protected)
├── dataset/              # Shared data (protected)
├── projects/             # Research projects
│   ├── estimation_after_detection/
│   ├── openmesh_nyc_paper/   # OpenMesh NYC dataset paper
│   ├── physics_ml/           # Hybrid physics + NN rain retrieval (CMLs)
│   └── spatial_interpolation/  # Rainfall nowcasting from CML networks
├── requirements.txt
└── CONTRIBUTING.md
```

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