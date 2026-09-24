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
| `notebooks/Simulation_MBML.ipynb` | Exploratory model-based / ML simulation notebook. |
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
│   ├── discover_itu.py      # PySR: recover k, alpha from CML data
│   └── gravity/             # worked PINN example (N-body gravity)
│       ├── pinn_main.py     pinn_model.py
│       └── pinn_learning.py pinn_utils.py
├── notebooks/
│   ├── Simulation_MBML.ipynb          # the CML rain-retrieval experiment
│   ├── TUTORIALS.md                   # guide to the method tutorials below
│   ├── 01_sindy_basics.ipynb          # SINDy on the Lorenz system
│   ├── 02_pysr_basics.ipynb           # symbolic regression basics
│   ├── 03_nbody_full_pipeline.ipynb   # simulation -> data -> discovery
│   └── pinn_vs_nn_comparison.ipynb    # PINN against a plain network
├── results/
│   └── training_curves.png  # Outputs and figures
├── requirements.txt         # Project-specific dependencies
└── README.md                # This file
```

`Simulation_MBML.ipynb` is the project's own experiment. The numbered
notebooks and the PINN comparison are **method tutorials** for the
physics-informed tools this project builds on — they moved here from
`core/examples/`, which was holding notebooks from three unrelated topics.
See `core/scientific_packages/` for the PySINDy and PySR reference notes.

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
- Physics-Informed Neural Networks (PINNs) literature — see `src/gravity/` and `notebooks/pinn_vs_nn_comparison.ipynb` for a worked PINN example
- PDE-constrained optimization
- Domain-informed machine learning
