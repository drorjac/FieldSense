"""
A body thrown upward, fitted from noisy heights: plain network vs PINN.

One module for what ``notebooks/archive/pinn_vs_nn_comparison.ipynb``, ``pinn_main.py`` and
``pinn_learning.py`` each wrote out separately.

The physics can enter the loss two ways, and the difference matters:

``velocity``      ``dh/dt = v0 - g t``, the residual the original notebook
                  used. It contains the true initial velocity *and* g, so
                  the network is told the answer's derivative; only h0 is
                  left to the data.
``acceleration``  ``d2h/dt2 = -g``. Knows gravity and nothing about this
                  particular throw, which is what "physics-informed"
                  should mean; v0 and h0 come from the data.

    from gravity.pinn import Throw, sweep
    throw = Throw(noise=1.5)
    table, runs = sweep(throw, lambdas=(0, 0.1, 1, 10, 100), residual="acceleration")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class Throw:
    """The problem: h(t) = h0 + v0 t - g t^2 / 2, sampled with noise."""

    g: float = 9.8
    h0: float = 1.0
    v0: float = 10.0
    t_max: float = 2.0
    n_data: int = 20
    n_phys: int = 50
    noise: float = 1.5
    seed: int = 42

    def height(self, t):
        return self.h0 + self.v0 * t - 0.5 * self.g * t ** 2

    def tensors(self):
        """Noisy (t, h) training data and physics collocation points."""
        rng = np.random.default_rng(self.seed)
        t = np.linspace(0.0, self.t_max, self.n_data)
        h = self.height(t) + self.noise * rng.standard_normal(self.n_data)
        colloc = np.linspace(0.0, self.t_max, self.n_phys)
        as_t = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 1)
        return as_t(t), as_t(h), as_t(colloc)


class MLP(nn.Module):
    """Scalar t -> scalar h, tanh activations (smooth, so derivatives exist)."""

    def __init__(self, n_hidden: int = 32, depth: int = 3):
        super().__init__()
        layers, width = [], 1
        for _ in range(depth):
            layers += [nn.Linear(width, n_hidden), nn.Tanh()]
            width = n_hidden
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, t):
        return self.net(t)


# kept for pinn_main.py: the two-hidden-layer network it was written for
def PINN(n_hidden: int) -> MLP:  # noqa: N802
    return MLP(n_hidden, depth=2)


def derivative(y, x):
    """dy/dx through autograd, keeping the graph for higher derivatives."""
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def data_loss(model, t, h):
    return torch.mean((model(t) - h) ** 2)


def physics_loss(model, t, v0, g, residual: str = "velocity"):
    """Mean squared residual of the chosen form of the equation of motion."""
    t = t.clone().requires_grad_(True)
    dh = derivative(model(t), t)
    if residual == "velocity":
        return torch.mean((dh - (v0 - g * t)) ** 2)
    if residual == "acceleration":
        return torch.mean((derivative(dh, t) + g) ** 2)
    raise ValueError(f"residual must be 'velocity' or 'acceleration', got {residual!r}")


def initial_condition_loss(model, h0, device=None):
    t0 = torch.zeros(1, 1, device=device)
    return (model(t0) - h0).pow(2).mean()


def train(throw: Throw, lambda_ode: float, residual: str = "velocity",
          lambda_ic: float = 1.0, epochs: int = 3000, lr: float = 0.01,
          n_hidden: int = 32) -> tuple:
    """Train one network; ``lambda_ode = 0`` is the plain network (data only).

    The initial-condition term goes with the physics: a plain network gets
    neither. Returns (model, history of per-epoch losses).
    """
    torch.manual_seed(throw.seed)
    t, h, colloc = throw.tensors()
    model = MLP(n_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"total": [], "data": [], "physics": [], "ic": []}
    use_physics = lambda_ode > 0
    for _ in range(epochs):
        opt.zero_grad()
        l_data = data_loss(model, t, h)
        l_phys = physics_loss(model, colloc, throw.v0, throw.g, residual) if use_physics \
            else torch.zeros(())
        l_ic = initial_condition_loss(model, throw.h0) if use_physics else torch.zeros(())
        loss = l_data + lambda_ode * l_phys + (lambda_ic if use_physics else 0.0) * l_ic
        loss.backward()
        opt.step()
        for k, v in (("total", loss), ("data", l_data), ("physics", l_phys), ("ic", l_ic)):
            history[k].append(float(v.detach()))
    return model, history


def rmse(model, throw: Throw, n: int = 200) -> float:
    """Error against the true trajectory on a dense grid, not the noisy data."""
    t = np.linspace(0.0, throw.t_max, n, dtype=np.float32).reshape(-1, 1)
    with torch.no_grad():
        pred = model(torch.from_numpy(t)).numpy()
    return float(np.sqrt(np.mean((pred - throw.height(t)) ** 2)))


def sweep(throw: Throw, lambdas=(0.0, 0.1, 1.0, 10.0, 100.0),
          residual: str = "velocity", **train_kw):
    """Train one network per physics weight; returns (table, runs)."""
    import pandas as pd

    rows, runs = [], {}
    for lam in lambdas:
        model, history = train(throw, lam, residual, **train_kw)
        runs[lam] = {"model": model, "history": history}
        rows.append({"lambda_ode": lam, "residual": residual if lam > 0 else "none",
                     "rmse_vs_truth": rmse(model, throw),
                     "final_data_loss": history["data"][-1],
                     "noise_variance": throw.noise ** 2})
    return pd.DataFrame(rows), runs
