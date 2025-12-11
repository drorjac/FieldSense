# Examples

This folder contains tutorial notebooks demonstrating physics-informed machine learning tools.

## Notebooks

| Notebook | Description | Tools |
|----------|-------------|-------|
| `01_sindy_basics.ipynb` | SINDy on Lorenz system | PySINDy |
| `02_pysr_basics.ipynb` | Symbolic regression basics | PySR |
| `05_nbody_full_pipeline.ipynb` | Full pipeline: simulation → data → discovery | PySINDy, PySR |

## Getting Started

1. Activate the environment:
```bash
conda activate fieldsense
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Start with `01_sindy_basics.ipynb` or `02_pysr_basics.ipynb` for tool basics, then move to `05_nbody_full_pipeline.ipynb` for a complete workflow.

## Learning Path

```
Basics                          Full Pipeline
┌─────────────────┐            ┌─────────────────────────┐
│ 01_sindy_basics │──┐         │ 05_nbody_full_pipeline  │
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
