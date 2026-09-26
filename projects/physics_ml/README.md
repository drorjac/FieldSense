# Physics-Informed Machine Learning

This project focuses on integrating domain knowledge, primarily from physics and Partial Differential Equations (PDEs), with machine learning algorithms to enhance environmental sensing and field reconstruction.

## Overview

The `physics_ml` module explores the intersection of physics-based modeling and machine learning, with a particular emphasis on:

- **Physics-Informed Neural Networks (PINNs)**: Neural networks that incorporate physical laws and constraints directly into the learning process
- **PDE-Based Modeling**: Leveraging partial differential equations to model spatio-temporal phenomena
- **Domain Knowledge Integration**: Combining physical principles with data-driven approaches for improved accuracy and generalization
- **Machine Learning Algorithms**: Introduction and implementation of various ML techniques tailored for physics-informed applications

## Key Concepts

### Physics-Informed Approaches
- Incorporation of physical laws (conservation laws, boundary conditions, etc.) as soft constraints in neural networks
- PDE-constrained optimization
- Hybrid models that combine physics-based and data-driven components

### Machine Learning Algorithms
- Neural networks with physics-informed loss functions
- Deep learning architectures for spatio-temporal data
- Transfer learning from physics-based models
- Uncertainty quantification in physics-informed models

## Applications

This module is particularly relevant for:
- Environmental field sensing and reconstruction
- Spatio-temporal evolution modeling
- Multi-sensor data fusion with physical constraints
- Weather and climate modeling
- Fluid dynamics and transport phenomena

## Contents

The current codebase implements a **hybrid rain-retrieval model** for Commercial
Microwave Links (CMLs): a model-based branch built on the ITU-R P.838-3 power law
is fused with a data-driven neural branch, so the physics carries the retrieval
where it is valid and the network absorbs what the power law cannot explain.

| File | Role |
|------|------|
| `src/rain_simulator.py` | Synthetic rain-attenuation generator. ITU-R P.838-3 (k, α) tables, AR(1) temporal correlation, configurable noise. |
| `src/hybrid_nn.py` | `PhysicsBranch` (learnable log-space k, α power-law inversion), the data-driven branch, and the dynamic fusion module. |
| `src/training_utils.py` | Training loops, device selection (CUDA → MPS → CPU), curve plotting, result analysis. |
| `src/data_analysis.py` | Dataset generation and pre-training inspection of the synthetic data. |
| `src/main_experiment.py` | Entry point. `ExperimentConfig` sweeps frequencies and noise levels end to end. |
| `src/mixing.py` | Six ways to combine a physics and a data-driven estimate, swept over frequency and noise. |
| `src/hybrid_training.py` | Staged vs joint training of the hybrid model, tracking gate, loss and learned (k, α). |
| `src/plots.py` | The figures `hybrid_retrieval.ipynb` draws. |
| `notebooks/hybrid_retrieval.ipynb` | **Start here.** Both experiments in ~20 lines of calls, every number computed. |
| `notebooks/archive/Simulation_MBML.ipynb` | The original working notebook, kept as the record (see below). |
| `results/training_curves.png` | Training curves from the last recorded run. |

## Structure

```
physics_ml/
├── src/
│   ├── main_experiment.py   # Entry point — experiment configuration and sweep
│   ├── hybrid_nn.py         # Physics + data-driven branches and fusion
│   ├── rain_simulator.py    # ITU-R P.838-3 synthetic attenuation generator
│   ├── data_analysis.py     # Dataset generation and inspection
│   └── training_utils.py    # Training loops, device selection, plotting
│   ├── mixing.py            # six ways to combine physics and data estimates
│   ├── hybrid_training.py   # staged vs joint training of the hybrid model
│   ├── plots.py             # figures for hybrid_retrieval.ipynb
│   ├── discover_itu.py      # PySR: recover ITU-R k, alpha from CML attenuation
│   └── discover_advection.py # SINDy: recover rain-field advection
├── notebooks/
│   ├── hybrid_retrieval.ipynb         # the hybrid experiments, concise
│   ├── TUTORIALS.md                   # guide to the two method tutorials
│   ├── 01_sindy_basics.ipynb          # SINDy on the Lorenz system
│   ├── 02_pysr_basics.ipynb           # symbolic regression basics
│   └── archive/Simulation_MBML.ipynb  # the original working notebook, kept
├── tests/test_physics_ml.py  # mixing + staged/joint training
├── results/
│   └── training_curves.png  # Outputs and figures
├── requirements.txt         # Project-specific dependencies
└── README.md                # This file
```

## Why each method is here

This project holds two things: a hybrid CML retrieval, and two experiments
that run equation discovery **on FieldSense's own quantities**. Every method
present has a job:

| method | question it answers | script | tutorial |
|---|---|---|---|
| **PySR** (symbolic regression) | does the ITU-R power law fall out of measured attenuation, and with what coefficients? | `src/discover_itu.py` | `notebooks/02_pysr_basics.ipynb` |
| **SINDy** (sparse regression) | does a rain field's advection fall out of the field's own evolution? | `src/discover_advection.py` | `notebooks/01_sindy_basics.ipynb` |
| **physics + NN hybrid** | can a learnable ITU branch and a GRU branch be fused per sample? | `src/hybrid_nn.py`, `src/hybrid_training.py` | `notebooks/hybrid_retrieval.ipynb` |

**Why SINDy specifically.** A rain field crossing a region is a dynamical
system, and field estimation is where that matters: a nowcast has to
propagate the field forward, which means knowing how it moves.
`projects/spatial_interpolation` benchmarks POD-SINDy against a Transformer
and a Mamba-style SSM for exactly this. The Lorenz notebook teaches the
method; `discover_advection.py` runs it on rainfall with a velocity known
exactly, so the answer can be checked.

The N-body and PINN-gravity material has no such counterpart and moved to
`projects/mphysics/`. See `core/scientific_packages/` for PySINDy and PySR
reference notes.

## The hybrid, measured

`notebooks/hybrid_retrieval.ipynb` answers two questions on synthetic links,
all numbers computed in the notebook.

**Combining two estimates depends on frequency.** At 5 GHz a 1 km link
attenuates so weakly that inverting the power law turns noise into an MSE of
~3,000 (mm/h)²; every learned combination is ~40x better. At 60-70 GHz physics
alone is within ~5% of the best, and a learned gate on attenuation edges it.

**The hybrid physics + GRU model does not beat its own physics branch.** At
38 GHz, with the original notebook's training budget:

| noise | method | fused RMSE | physics branch | neural branch | gate |
|---|---|---|---|---|---|
| 0.05 dB | staged | 4.00 | **0.49** | 4.04 | 0.13 |
| 0.5 dB | staged | 4.12 | **1.26** | 4.16 | 0.13 |
| 2.0 dB | staged | 4.68 | **3.74** | 4.73 | 0.23 |

(RMSE in mm/h; joint training is worse than staged at every level.) The gate
leans on the neural branch although physics is 1.3-8x more accurate. A likely
cause is the input: `prepare_sequences` builds each GRU "sequence" as random
noise around one attenuation value, so the network has no temporal
information to learn from.

### About `archive/Simulation_MBML.ipynb`

Kept unchanged as the working record, but it is not a reliable source of
results: it defines the hybrid model three times, calls a
`ThreeStageTrainer` that is defined nowhere in it (reconstructed as
`hybrid_training.train(..., "staged")`), and its summary figures (cells 17,
18, 22) plot numbers typed into the cells rather than computed. Those show
the hybrid ahead; the code above does not reproduce that.

## Rediscovering ITU-R P.838-3 with symbolic regression

Every retrieval here *inverts* `gamma = k * R^alpha`. `src/discover_itu.py`
asks the opposite question: given attenuation and rain rate, does PySR find
the power law, and does it recover the tabulated coefficients?

```bash
python projects/physics_ml/src/discover_itu.py --all --band 30
```

Three experiments in deliberate order, all in the 30 GHz band where OpenMRG
has 172 links (ITU: k = 0.229, alpha = 0.913):

| experiment | rain rate from | recovered k | recovered alpha | alpha error |
|---|---|---|---|---|
| synthetic | ITU + noise, truth known | 0.229 | 0.913 | **0.000** |
| reference | OpenSense's published retrieval | 0.419 | 0.911 | **0.002** |
| gauge | municipal rain gauges | 0.488 | 0.478 | **0.435** |

**The exponent is recoverable from real data; the prefactor is not.** Against
the reference retrieval PySR lands on alpha = 0.911 against ITU's 0.913 — but
k comes out 83% high. The fitted curve is *parallel* to ITU and displaced
upward, which is the signature of a constant added to the attenuation rather
than an error in the power law.

That offset is almost certainly unmodelled wet-antenna attenuation, and the
size is telling: +83% here, against the ~1.86x over-read that
`opensense_pipeline/src/validate_retrieval.py` measures for the same chain
against the same reference. Two independent methods, the same bias, and both
locate it in the prefactor.

**The gauge experiment fails, for a statistical reason worth naming.** Alpha
collapses to 0.478 and the naive log-log fit to 0.312. This is regression
dilution: the predictor is a point gauge, the response is a path average over
kilometres, and when the predictor carries that much independent noise the
fitted slope is biased toward zero. It is not evidence against the power law
— it is evidence that gauge-link pairs are the wrong data to fit one with.

Two notes on method. PySR needs `^` in its operator set or the search cannot
reach a power law at all and spends its budget on polynomial approximations
that fit acceptably and explain nothing. And `model_selection="best"` tends
to pick a straight line at CML frequencies, because alpha sits within ~0.15
of 1 across 15–40 GHz; the script therefore reduces *every* Pareto candidate
to the `k * R^alpha` it is equivalent to, rather than trusting the single
"best" expression.

![Recovering ITU-R P.838-3](results/discover_itu.png)

## Recovering field dynamics with SINDy

`src/discover_advection.py` is the temporal counterpart to the ITU experiment.
A rain field advects, so to first order

```
dR/dt = -u dR/dx - v dR/dy
```

and fitting a library of spatial derivatives against the time derivative
should return the wind in its coefficients. The fields come from
`projects/rainfall_field_sim` translated at a velocity we choose, so the
answer is known exactly.

```bash
python projects/physics_ml/src/discover_advection.py --all
```

True velocity u = 14.0, v = 5.0 km/h:

| observation | SINDy u | SINDy v | active terms |
|---|---|---|---|
| exact fields | **13.68** | **4.87** | 2 — the two correct ones |
| + 5% noise | 13.77 | 2.64 | 3 — one spurious |
| through 90 CML paths, IDW back to a grid | 4.14 | 0.00 | 2 |

**Clean recovery works, and the sparsity is real** — the library offers
second derivatives and quadratic terms that the true dynamics do not use, and
on exact fields SINDy selects neither.

**Noise costs the weaker component first.** At 5% multiplicative noise `u`
survives but `v` halves, which is what you would expect: v = 5 km/h carries
less signal than u = 14, so the same absolute derivative error eats a larger
fraction of it.

**Through a CML network it fails completely.** Sampling the field along 90
link paths and interpolating back with IDW leaves u = 4.1 and v = 0. This is
consistent with what the rest of the repository finds about IDW —
`rainfall_field_sim` measures it destroying intermittency and 83–87% of peak
intensity. Advection lives in the *gradients* of the field, and those are
exactly what the interpolation smooths away. A nowcast fitted on IDW-derived
fields is fitting the reconstruction, not the weather.

### A methodological trap worth knowing

Naive finite differences bias the recovered velocity **high by 20%** on exact,
noise-free data — 16.8 km/h for a true 14.0. A central difference computes
`sin(k dx)/dx` instead of `k`, so it under-reads high-wavenumber content, and
it under-reads the *spatial* derivative more than the temporal one, because
the field moves only a fraction of a cell per timestep. Their ratio is the
velocity, so the errors do not cancel. Spectral spatial derivatives are exact
for a periodic field and give 13.6.

The time derivative stays a finite difference: the sequence is not periodic in
time — the field translates out of one edge and into the other — so a spectral
time derivative picks up Gibbs error and reads 12.8 instead.

![SINDy advection recovery](results/discover_advection.png)

## Getting Started

```bash
# from the repository root
pip install -r requirements.txt
pip install -r projects/physics_ml/requirements.txt

# the modules import each other by bare name, so run from src/
cd projects/physics_ml/src
python main_experiment.py
```

Experiment parameters (frequencies, noise levels, sample count, epochs, batch
size, learning rate, hidden size, link length, baseline attenuation) live in
`ExperimentConfig` at the top of `main_experiment.py`.

## Dependencies

PyTorch for the neural branches, NumPy/SciPy for the simulator, scikit-learn for
splitting and baselines, and Matplotlib/Seaborn for figures. See
`requirements.txt`; it extends the root `requirements.txt` rather than replacing it.

## References

- ITU-R P.838-3: Specific attenuation model for rain for use in prediction methods
- Physics-Informed Neural Networks (PINNs) literature — see `projects/mphysics/src/gravity/` and `projects/mphysics/notebooks/pinn_gravity.ipynb` for a worked PINN example
- PDE-constrained optimization
- Domain-informed machine learning
