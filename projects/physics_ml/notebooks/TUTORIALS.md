# Method tutorials

Tutorial notebooks for the physics-informed machine-learning tools this
project builds on. They are reference material, not the project's own
experiment — that is `Simulation_MBML.ipynb` in this folder.

## Notebooks

| Notebook | Description | Tools |
|----------|-------------|-------|
| `01_sindy_basics.ipynb` | SINDy on Lorenz system | PySINDy |
| `02_pysr_basics.ipynb` | Symbolic regression basics | PySR |
| `pinn_vs_nn_comparison.ipynb` | PINN vs a plain network on N-body gravity | PyTorch |
| `03_nbody_full_pipeline.ipynb` | Full pipeline: simulation → data → discovery | PySINDy, PySR |

## Getting Started

1. Activate the environment:
```bash
source .venv/bin/activate   # from the repository root
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Start with `01_sindy_basics.ipynb` or `02_pysr_basics.ipynb` for tool basics, then move to `03_nbody_full_pipeline.ipynb` for a complete workflow.

## Learning Path

```
Basics                          Full Pipeline
┌─────────────────┐            ┌─────────────────────────┐
│ 01_sindy_basics │──┐         │ 03_nbody_full_pipeline  │
└─────────────────┘  │         │                         │
                     ├────────▶│ • Simulation            │
┌─────────────────┐  │         │ • Visualization         │
│ 02_pysr_basics  │──┘         │ • Data extraction       │
└─────────────────┘            │ • SINDy + PySR learning │
                               └─────────────────────────┘
```

## See Also

- [PySINDy Resources](../docs/PYSINDY.md)
- [PySR Resources](../docs/PYSR.md)
