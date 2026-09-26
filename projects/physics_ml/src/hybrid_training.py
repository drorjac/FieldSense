"""
Training the hybrid physics + GRU model: in stages, or jointly from the start.

``hybrid_nn.HybridRainModel`` fuses a learnable ITU-R branch and a GRU branch
through a gate. ``notebooks/Simulation_MBML.ipynb`` compared two ways to
train it, but the staged trainer it calls (``ThreeStageTrainer``) is not
defined anywhere in the notebook, so that comparison cannot be re-run from
it. This module is the comparison, reconstructed from how the notebook used
it:

staged   1. both branches on their own losses, gate frozen
         2. branches frozen, gate only
         3. everything jointly, at a lower rate
joint    everything jointly from the start

Every epoch records train/validation loss, mean gate weight (1 = physics,
0 = neural) and the physics branch's current (k, alpha), so one can see
whether the learnable power law drifts away from ITU-R.

    from hybrid_training import TrainingConfig, dataset, compare
    data = dataset(freq_ghz=38, noise_db=0.5)
    histories = compare(data, TrainingConfig())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from hybrid_nn import create_hybrid_model, prepare_sequences
from rain_simulator import RainAttenuationGenerator, RainParams

from core.itu_p838 import get_k_alpha


@dataclass(frozen=True)
class TrainingConfig:
    """Budget and optimizer settings; defaults follow the notebook's runs."""

    n_train: int = 300
    n_val: int = 100
    batch_size: int = 16
    hidden_size: int = 32
    link_length_km: float = 1.5
    baseline_db: float = 2.0
    stage_epochs: tuple = (8, 5, 7)         # staged: branches, gate, joint
    stage_lr: tuple = (1e-3, 5e-4, 1e-4)
    joint_epochs: int = 20                  # = sum(stage_epochs), same budget
    joint_lr: float = 1e-4
    seed: int = 42


def dataset(freq_ghz: float, noise_db: float, n_samples: int = 10_000,
            cfg: TrainingConfig = TrainingConfig(), pol: str = "vertical") -> dict:
    """Synthetic (R, A) for one link, split train / validation / test."""
    k, alpha = get_k_alpha(freq_ghz, pol)
    gen = RainAttenuationGenerator(seed=cfg.seed)
    out = gen.generate_data(n_samples=n_samples, rain_params=RainParams(k, alpha),
                            link_length_km=cfg.link_length_km, sigma_db=noise_db,
                            rain_sampler=lambda n: gen._rain_gamma(n, 1.5, 4.0),
                            baseline_atten_db=cfg.baseline_db, test_size=0.3)
    r_tr, r_te, a_tr, a_te = out[:4]
    r_train, r_val, a_train, a_val = train_test_split(r_tr, a_tr, test_size=0.25,
                                                      random_state=cfg.seed)
    return {"freq_ghz": freq_ghz, "noise_db": noise_db, "k": k, "alpha": alpha,
            "train": (r_train, a_train), "val": (r_val, a_val), "test": (r_te, a_te)}


def _tensors(split, n, seed):
    np.random.seed(seed)                    # prepare_sequences draws noise
    r, a = split
    return prepare_sequences(np.asarray(r)[:n], np.asarray(a)[:n])


def _batches(x, y, size):
    for i in range(0, len(x), size):
        yield x[i:i + size], y[i:i + size]


def _epoch(model, x, y, cfg, optimizer=None, loss_fn=None):
    """One pass; trains when an optimizer is given. Returns (loss, gate)."""
    training = optimizer is not None
    model.train(training)
    losses, gates = [], []
    with torch.set_grad_enabled(training):
        for bx, by in _batches(x, y, cfg.batch_size):
            out = model(bx, link_length=cfg.link_length_km, baseline=cfg.baseline_db)
            loss = loss_fn(out, by) if loss_fn else model.compute_loss(out, by)["total"]
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            gates.append(out["gate_weight"].detach().mean().item())
    return float(np.mean(losses)), float(np.mean(gates))


def _record(history, model, stage, train_loss, val_loss, gate):
    k, alpha = model.physics.coefficients
    for key, v in (("stage", stage), ("train_loss", train_loss), ("val_loss", val_loss),
                   ("gate", gate), ("k", k), ("alpha", alpha)):
        history[key].append(v)


def _branch_loss(out, y):
    return F.mse_loss(out["rain_physics"], y) + F.mse_loss(out["rain_neural"], y)


def _set_trainable(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def train(model, data: dict, cfg: TrainingConfig, method: str = "joint") -> dict:
    """Train ``model`` on ``data`` by ``method`` ("staged" or "joint")."""
    x_tr, y_tr = _tensors(data["train"], cfg.n_train, cfg.seed)
    x_va, y_va = _tensors(data["val"], cfg.n_val, cfg.seed + 1)
    history = {k: [] for k in ("stage", "train_loss", "val_loss", "gate", "k", "alpha")}

    if method == "joint":
        plan = [("joint", cfg.joint_epochs, cfg.joint_lr, model.parameters(), None)]
    elif method == "staged":
        branches = list(model.physics.parameters()) + list(model.neural.parameters())
        plan = [("branches", cfg.stage_epochs[0], cfg.stage_lr[0], branches, _branch_loss),
                ("gate", cfg.stage_epochs[1], cfg.stage_lr[1], model.fusion.parameters(), None),
                ("joint", cfg.stage_epochs[2], cfg.stage_lr[2], model.parameters(), None)]
    else:
        raise ValueError(f"method must be 'staged' or 'joint', got {method!r}")

    for stage, epochs, lr, params, loss_fn in plan:
        _set_trainable(model, stage == "joint")
        if stage == "branches":
            _set_trainable(model.physics, True)
            _set_trainable(model.neural, True)
        elif stage == "gate":
            _set_trainable(model.fusion, True)
        optimizer = torch.optim.Adam([p for p in params if p.requires_grad], lr=lr)
        for _ in range(epochs):
            tr_loss, gate = _epoch(model, x_tr, y_tr, cfg, optimizer, loss_fn)
            va_loss, _ = _epoch(model, x_va, y_va, cfg)
            _record(history, model, stage, tr_loss, va_loss, gate)
    _set_trainable(model, True)
    return history


def evaluate(model, data: dict, cfg: TrainingConfig) -> dict:
    """Test RMSE of the fused output and of each branch alone, and mean gate."""
    x, y = _tensors(data["test"], len(data["test"][0]), cfg.seed + 2)
    model.eval()
    with torch.no_grad():
        out = model(x, link_length=cfg.link_length_km, baseline=cfg.baseline_db)
    rmse = {name: float(torch.sqrt(F.mse_loss(out[key], y)))
            for name, key in (("fused", "rain_rate"), ("physics", "rain_physics"),
                              ("neural", "rain_neural"))}
    return {**rmse, "gate": float(out["gate_weight"].mean())}


def compare(data: dict, cfg: TrainingConfig = TrainingConfig()) -> dict:
    """Staged vs joint on the same data from the same initialization."""
    results = {}
    for method in ("staged", "joint"):
        torch.manual_seed(cfg.seed)
        model = create_hybrid_model(freq_ghz=data["freq_ghz"], pol="vertical",
                                    hidden_size=cfg.hidden_size)
        history = train(model, data, cfg, method)
        results[method] = {"history": history, "test": evaluate(model, data, cfg)}
    return results


def compare_over_noise(freq_ghz: float, noise_levels, cfg: TrainingConfig = TrainingConfig()):
    """``compare`` at several noise levels; returns (table, {noise: results}).

    The table has test RMSE of the fused output and of each branch alone,
    the mean gate weight, and the learned (k, alpha) next to ITU-R's.
    """
    import pandas as pd

    rows, runs = [], {}
    for sigma in noise_levels:
        data = dataset(freq_ghz, sigma, cfg=cfg)
        runs[sigma] = compare(data, cfg)
        for method, r in runs[sigma].items():
            rows.append({"noise_db": sigma, "method": method, **r["test"],
                         "k": r["history"]["k"][-1], "alpha": r["history"]["alpha"][-1],
                         "k_itu": data["k"], "alpha_itu": data["alpha"]})
    return pd.DataFrame(rows).set_index(["noise_db", "method"]), runs
