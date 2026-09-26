# mphysics — physics-ML methods, not FieldSense data

A sandbox for physics-informed machine learning on **classical physics
problems**. Nothing here touches rainfall, microwave links, or any FieldSense
measurement. It is kept separate for exactly that reason.

```
mphysics/
├── notebooks/
│   ├── nbody_discovery.ipynb          # simulate, measure, rediscover gravity  (concise)
│   ├── pinn_gravity.ipynb             # PINN vs plain network, two residuals    (concise)
│   ├── 03_nbody_full_pipeline.ipynb   # the original, kept
│   └── pinn_vs_nn_comparison.ipynb    # the original, kept
├── src/
│   ├── nbody.py                       # simulator, initial conditions, measurements, PySR/SINDy
│   ├── plots.py                       # every figure the notebooks draw
│   └── gravity/
│       ├── pinn.py                    # model, losses, training, sweep
│       ├── pinn_main.py               # script (via the pinn_model shim)
│       └── pinn_model.py  pinn_utils.py
├── tests/test_mphysics.py
└── README.md
```

## What the concise notebooks show

**Newton's law, rediscovered.** PySR on (r, |a|) finds `1/r²` as the
simplest accurate law. SINDy finds it only with the right library: with
degree-2 polynomials, the original notebook's choice, it returns nonsense and
its rollout diverges within one orbit, because 1/r³ cannot be written in that
library. With `x/r³` and `y/r³` in the library it returns
`v̇x = −1.0009 x/r³`, i.e. G(M + m) = 1.001, and its rollout follows the orbit
to 0.01 over several periods.

**A PINN's physics term, and what it contains.** The original comparison
used `dh/dt = v0 − g t`, which gives the network this throw's true v0: more
physics weight is then always better. With `d²h/dt² = −g`, which knows only
gravity, there is an optimum near λ = 1 and the fit fails again at λ = 100,
when the physics swamps the data that has to fix v0 and h0.

**Fixed along the way.** The original figure-8 initial conditions had vy
0.4662 instead of 0.4324, which is not the Chenciner–Montgomery orbit and
breaks up into chaotic close encounters. The simulator now sets an absolute
tolerance as well as a relative one, which cut energy drift from ~1e-5 to
~5e-9. `pinn_learning.py` was a self-contained copy of `pinn_main.py` and is
removed.

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
