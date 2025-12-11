# PySR Resources

**Symbolic Regression with Genetic Programming**

PySR discovers mathematical expressions that fit data using evolutionary algorithms.

## Overview

PySR searches for symbolic expressions that minimize:

$$\mathcal{L} = \text{Error}(f, \text{data}) + \lambda \cdot \text{Complexity}(f)$$

It returns a **Pareto front** of equations trading off accuracy vs. simplicity.

## Installation

```bash
pip install pysr
```

Note: PySR uses Julia backend. First run may take time to compile.

## Official Resources

| Resource | Link |
|----------|------|
| GitHub | https://github.com/MilesCranmer/PySR |
| Documentation | https://astroautomata.com/PySR |
| Paper | https://arxiv.org/abs/2305.01582 |

## Video Tutorials

| Title | Link | Description |
|-------|------|-------------|
| PySR Introduction | https://www.youtube.com/watch?v=q4PVVF3JSMI | Official intro |
| Symbolic Regression Explained | https://www.youtube.com/watch?v=Ef0BjLEv6Ts | Concepts |
| AI Feynman + Symbolic Regression | https://www.youtube.com/watch?v=lBRSSR6fJA8 | Physics applications |

## Example Repositories

| Repository | Description |
|------------|-------------|
| [MilesCranmer/PySR](https://github.com/MilesCranmer/PySR) | Official repository |
| [Examples](https://github.com/MilesCranmer/PySR/tree/master/examples) | Official examples |
| [Benchmarks](https://github.com/cavalab/srbench) | Symbolic regression benchmarks |

## Quick Example

```python
import numpy as np
from pysr import PySRRegressor

# Generate data: y = x^2 + 2*sin(x)
np.random.seed(42)
X = np.random.uniform(-5, 5, (100, 1))
y = X[:, 0]**2 + 2 * np.sin(X[:, 0])

# Fit PySR
model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "square"],
    maxsize=15,
)
model.fit(X, y)

# Results
print(model)  # Pareto front
print("Best:", model.sympy())  # Best equation
```

Output:
```
Best: x**2 + 2*sin(x)
```

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `niterations` | Evolution iterations | 40 - 200 |
| `populations` | Number of populations | 15 - 30 |
| `population_size` | Size of each population | 33 - 100 |
| `maxsize` | Max equation complexity | 15 - 30 |
| `binary_operators` | Two-input ops | `["+", "-", "*", "/"]` |
| `unary_operators` | One-input ops | `["sin", "square", "exp"]` |

## Available Operators

**Binary:**
```python
["+", "-", "*", "/", "^", "max", "min", "mod"]
```

**Unary:**
```python
["sin", "cos", "tan", "exp", "log", "sqrt", "square", "cube",
 "abs", "sign", "floor", "ceil", "round", "sinh", "cosh", "tanh",
 "asin", "acos", "atan", "inv"]
```

## Tips

1. **Start simple**: Begin with basic operators, add more if needed.
2. **Enough iterations**: Complex expressions need more iterations.
3. **Multiple runs**: Results can vary; run multiple times.
4. **Domain knowledge**: Include operators you expect (e.g., `inv` for 1/x).
5. **Normalize data**: Scale inputs/outputs for better convergence.

## Advanced Features

### Multi-output

```python
model = PySRRegressor(...)
model.fit(X, y_matrix)  # y_matrix shape: (n_samples, n_outputs)
```

### Constraints

```python
model = PySRRegressor(
    nested_constraints={"sin": {"sin": 0}},  # No sin(sin(x))
    complexity_of_operators={"/": 2},  # Division costs more
)
```

### Custom Loss

```python
model = PySRRegressor(
    loss="loss(prediction, target) = (prediction - target)^2",
)
```

## Comparison with SINDy

| Aspect | PySR | SINDy |
|--------|------|-------|
| Approach | Evolutionary search | Sparse regression |
| Output | Algebraic expressions | ODEs (dx/dt = ...) |
| Best for | Static relationships | Dynamical systems |
| Speed | Slower | Faster |
| Flexibility | High (any expression) | Limited to library |

## Use Cases

- **Physics**: Discover force laws, constitutive relations
- **Finance**: Find pricing formulas
- **Biology**: Model growth curves
- **Engineering**: Derive empirical equations

## Related Tools

- **gplearn**: Scikit-learn compatible genetic programming
- **Eureqa**: Commercial symbolic regression (discontinued)
- **AI Feynman**: Physics-focused symbolic regression
- **DSR**: Deep symbolic regression
