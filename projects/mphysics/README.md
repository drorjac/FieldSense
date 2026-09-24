# mphysics — physics-ML methods, not FieldSense data

A sandbox for physics-informed machine learning on **classical physics
problems**. Nothing here touches rainfall, microwave links, or any FieldSense
measurement. It is kept separate for exactly that reason.

```
mphysics/
├── notebooks/
│   ├── 03_nbody_full_pipeline.ipynb   # simulate N-body -> discover its equations
│   └── pinn_vs_nn_comparison.ipynb    # PINN vs a plain network on gravity
├── src/
│   └── gravity/                       # the PINN implementation
│       ├── pinn_main.py   pinn_model.py
│       └── pinn_learning.py pinn_utils.py
└── README.md
```

## Why this is its own project

These notebooks lived in `core/examples/`, then briefly in
`projects/physics_ml/`. Neither was right. `core/` is for code several
projects import, and these import nowhere. `physics_ml/` is FieldSense's
own physics-ML work — hybrid CML retrieval, and recovering ITU-R and
advection coefficients from measured data — and mixing a gravity PINN into
that makes the project look like a tutorial collection rather than a
research track.

The methods are still worth having. Physics-informed losses and
equation discovery on a problem with an exact analytic answer are a
reasonable way to learn what those techniques do before pointing them at
data where you cannot check the result.

## What FieldSense actually uses, and where

| method | used for | where |
|---|---|---|
| symbolic regression (PySR) | recovering ITU-R `k`, `alpha` from CML attenuation | `projects/physics_ml/src/discover_itu.py` |
| sparse regression (SINDy) | recovering rain-field advection; nowcasting dynamics | `projects/physics_ml/src/discover_advection.py`, `projects/spatial_interpolation/` |
| physics + NN hybrid | CML rain retrieval with a learnable ITU branch | `projects/physics_ml/src/hybrid_nn.py` |

The tutorials for the first two stay next to that work. The N-body and
gravity material has no such counterpart, which is what it is doing here.

## Running

```bash
pip install -r requirements.txt
jupyter lab projects/mphysics/notebooks/
```
