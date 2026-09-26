# Method tutorials

Tutorial notebooks for the physics-informed machine-learning tools this
project builds on. They are reference material, not the project's own
experiment — that is `hybrid_retrieval.ipynb` in this folder.

## Notebooks

| Notebook | Description | Tools |
|----------|-------------|-------|
| `01_sindy_basics.ipynb` | SINDy on Lorenz system | PySINDy |
| `02_pysr_basics.ipynb` | Symbolic regression basics | PySR |

## Getting Started

1. Activate the environment:
```bash
source .venv/bin/activate   # from the repository root
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Start with `01_sindy_basics.ipynb` or `02_pysr_basics.ipynb` for the tool basics, then read `../src/discover_advection.py` and `../src/discover_itu.py`, which run those tools on CML data.

## Learning path

```
01_sindy_basics   ──►  ../src/discover_advection.py   (SINDy on rain fields)
02_pysr_basics    ──►  ../src/discover_itu.py         (PySR on CML attenuation)
```

Each tutorial teaches a method on a textbook problem; each script runs that
method on a FieldSense quantity where the right answer is known.

---

The N-body pipeline and the PINN-vs-network comparison moved to
`projects/mphysics/` — they are classical-physics demonstrations with no
FieldSense counterpart. The two tutorials that remain teach methods this
project runs on real data: see `src/discover_itu.py` (PySR) and
`src/discover_advection.py` (SINDy).
