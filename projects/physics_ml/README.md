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
├── notebooks/
│   └── Simulation_MBML.ipynb
├── results/
│   └── training_curves.png  # Outputs and figures
├── requirements.txt         # Project-specific dependencies
└── README.md                # This file
```

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
- Physics-Informed Neural Networks (PINNs) literature — see `core/examples/gravity/` for a worked PINN example
- PDE-constrained optimization
- Domain-informed machine learning
