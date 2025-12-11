# PySINDy Resources

**Sparse Identification of Nonlinear Dynamics**

PySINDy discovers governing equations from time-series data using sparse regression.

## Overview

SINDy assumes dynamics can be written as:

$$\dot{x} = f(x) = \Theta(x) \xi$$

Where:
- $\Theta(x)$ is a library of candidate functions (polynomials, trig, etc.)
- $\xi$ is a sparse vector of coefficients

The algorithm finds the sparsest $\xi$ that fits the data.

## Installation

```bash
pip install pysindy
```

## Official Resources

| Resource | Link |
|----------|------|
| GitHub | https://github.com/dynamicslab/pysindy |
| Documentation | https://pysindy.readthedocs.io |
| Paper | https://arxiv.org/abs/2004.08424 |

## Video Tutorials

| Title | Link | Duration |
|-------|------|----------|
| SINDy Introduction (Steve Brunton) | https://www.youtube.com/watch?v=NxAn0oglMVw | 15 min |
| SINDy Algorithm Explained | https://www.youtube.com/watch?v=NxAn0oglMVw&t=697s | - |
| Data-Driven Discovery (Full Playlist) | https://www.youtube.com/playlist?list=PLMrJAkhIeNNRHP5UA-PIt660E7OEaOLnB | - |

## Example Repositories

| Repository | Description |
|------------|-------------|
| [luckystarufo/pySINDy](https://github.com/luckystarufo/pySINDy) | Alternative implementation with examples |
| [dynamicslab/pysindy](https://github.com/dynamicslab/pysindy) | Official repository |
| [Examples folder](https://github.com/dynamicslab/pysindy/tree/master/examples) | Official examples |

## Quick Example

```python
import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp

# Generate Lorenz data
def lorenz(t, x, sigma=10, rho=28, beta=8/3):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2]
    ]

t = np.linspace(0, 10, 1000)
sol = solve_ivp(lorenz, (0, 10), [-8, 7, 27], t_eval=t)
X = sol.y.T

# Fit SINDy
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1),
    feature_library=ps.PolynomialLibrary(degree=2)
)
model.fit(X, t=t, feature_names=['x', 'y', 'z'])
model.print()
```

Output:
```
(x)' = -10.000 x + 10.000 y
(y)' = 28.000 x + -1.000 y + -1.000 x z
(z)' = -2.667 z + 1.000 x y
```

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `threshold` | Sparsity cutoff | 0.01 - 0.5 |
| `degree` | Polynomial degree | 2 - 5 |
| `include_bias` | Include constant term | True/False |

## Feature Libraries

```python
# Polynomial
ps.PolynomialLibrary(degree=3)

# Fourier
ps.FourierLibrary(n_frequencies=3)

# Custom
ps.CustomLibrary(library_functions=[
    lambda x: np.sin(x),
    lambda x: 1/x,
])

# Combined
ps.GeneralizedLibrary([
    ps.PolynomialLibrary(degree=2),
    ps.FourierLibrary(n_frequencies=2),
])
```

## Tips

1. **Clean data**: SINDy is sensitive to noise. Smooth data if needed.
2. **Enough samples**: Need sufficient temporal resolution.
3. **Right library**: Include functions you expect in the dynamics.
4. **Tune threshold**: Start high, decrease if missing terms.

## Related Methods

- **SINDy-PI**: Physics-informed SINDy
- **SINDy-c**: SINDy with control inputs
- **Ensemble SINDy**: Bootstrap for uncertainty
- **Weak SINDy**: Integral formulation for noisy data
