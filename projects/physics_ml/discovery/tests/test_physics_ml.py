"""Mixing strategies and hybrid training, on tiny synthetic inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

import hybrid_training as ht  # noqa: E402
import mixing  # noqa: E402


def test_physics_inverse_is_exact_without_noise():
    link = mixing.Link.itu(38, "vertical", length_km=2.0)
    rain = np.array([0.5, 5.0, 40.0])
    att = link.k * rain ** link.alpha * link.length_km
    np.testing.assert_allclose(mixing.physics_inverse(att, link), rain, rtol=1e-9)


def test_itu_table_matches_the_original_horizontal_coefficients():
    link = mixing.Link.itu(23)                   # the notebook's hardcoded (0.1286, 1.0214)
    assert (link.k, link.alpha) == pytest.approx((0.1286, 1.0214), rel=2e-3)


def test_scalar_and_gate_weights_recover_the_right_branch():
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 30, 200)
    good, bad = y + rng.normal(0, 0.01, 200), rng.uniform(1, 30, 200)
    assert mixing.scalar_weight(good, bad, y) == pytest.approx(1.0, abs=0.02)
    assert mixing.scalar_weight(bad, good, y) == pytest.approx(0.0, abs=0.02)
    gate = mixing.gate_weight(y, good, bad, y)
    assert np.mean(gate(y)) > 0.9


def test_run_once_scores_every_strategy():
    out = mixing.run_once(mixing.Link.itu(60), noise_db=0.1, n=80,
                          rng=np.random.default_rng(1))
    assert set(mixing.STRATEGIES) <= set(out)
    assert all(np.isfinite(out[s]) and out[s] >= 0 for s in mixing.STRATEGIES)


def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _changed(before, module):
    return any(not torch.equal(b, p) for b, p in zip(before, module.parameters()))


def test_staged_training_freezes_what_each_stage_should():
    cfg = ht.TrainingConfig(n_train=48, n_val=16, stage_epochs=(1, 1, 0), joint_epochs=1)
    data = ht.dataset(38, 0.5, n_samples=400, cfg=cfg)
    torch.manual_seed(0)
    model = ht.create_hybrid_model(freq_ghz=38, pol="vertical", hidden_size=8)

    # stage 1 alone: branches move, gate does not
    fusion0, physics0 = _snapshot(model.fusion), _snapshot(model.physics)
    ht.train(model, data, cfg.__class__(**{**cfg.__dict__, "stage_epochs": (1, 0, 0)}), "staged")
    assert _changed(physics0, model.physics) and not _changed(fusion0, model.fusion)

    # stage 2 alone: gate moves, branches do not
    fusion1, physics1, neural1 = (_snapshot(m) for m in (model.fusion, model.physics, model.neural))
    ht.train(model, data, cfg.__class__(**{**cfg.__dict__, "stage_epochs": (0, 1, 0)}), "staged")
    assert _changed(fusion1, model.fusion)
    assert not _changed(physics1, model.physics) and not _changed(neural1, model.neural)
    assert all(p.requires_grad for p in model.parameters())      # restored


def test_compare_trains_both_methods_from_one_initialization():
    cfg = ht.TrainingConfig(n_train=32, n_val=16, stage_epochs=(1, 1, 1), joint_epochs=3)
    out = ht.compare(ht.dataset(38, 0.5, n_samples=300, cfg=cfg), cfg)
    assert set(out) == {"staged", "joint"}
    for r in out.values():
        assert len(r["history"]["val_loss"]) == 3
        assert {"fused", "physics", "neural", "gate"} <= set(r["test"])
    with pytest.raises(ValueError):
        ht.train(None, ht.dataset(38, 0.5, n_samples=300, cfg=cfg), cfg, "sideways")
