"""
The forecasters against moving rain whose future is known.

On radar data nobody knows how much of the next hour was predictable. On a
field from ``core.simulation.moving_fields`` we do: it moves at a set
velocity and evolves at a set rate, so two references bound every model -

``persistence``   the last frame, unmoved - the floor a forecaster must beat
``oracle``        the last frame moved at the *true* velocity - the ceiling
                  for anything that forecasts by motion, below 1 only
                  because the field evolves

A model trained on these sequences that cannot beat persistence has not
learned advection; one that approaches the oracle has. The same models and
training code as the real pipeline (``forecasting``), on a small grid, so
it runs in about a minute on a CPU.

    from synthetic import moving_benchmark
    table, seq = moving_benchmark(evolve_tau_min=120)
"""

from __future__ import annotations

import evaluation as ev
import forecasting as fc
import numpy as np
import pandas as pd

from core.simulation import moving_fields as mf
from core.simulation.rain_fields import Grid, StratiformField


def moving_benchmark(model=None, n: int = 32, dx_km: float = 2.0, dt_min: float = 15.0,
                     velocity_kmh=(20.0, 8.0), evolve_tau_min: float | None = 120.0,
                     growth_per_h: float = 0.0, n_frames=(400, 100, 160),
                     lookback: int = 8, horizons=(1, 2, 3, 4), epochs: int = 20,
                     seed: int = 0) -> tuple[pd.DataFrame, mf.MovingSequence]:
    """Score persistence, the oracle, Transformer, GRU and POD-SINDy by horizon.

    ``n_frames`` is (train, val, test) of one continuous sequence. Returns the
    score table (one row per model x horizon, with the evolution's own
    predictability bound) and the sequence.
    """
    model = model or StratiformField()
    seq = mf.sequence(model, Grid(n=n, dx_km=dx_km), sum(n_frames), dt_min, velocity_kmh,
                      growth_per_h, evolve_tau_min, seed=seed)
    a, b = n_frames[0], n_frames[0] + n_frames[1]
    splits = {"train": seq.frames[:a], "val": seq.frames[a:b], "test": seq.frames[b:]}
    test = splits["test"]
    count = len(test) - lookback - max(horizons) + 1

    preds = {"persistence": fc.persistence(test, lookback, horizons)}
    preds["oracle (true motion)"] = {
        h: np.stack([seq.lagrangian_persistence(b + i + lookback - 1, h)
                     for i in range(count)]) for h in horizons}
    shape = (n, n)
    for name, factory, hp in (
            ("transformer", lambda k: fc.RainTransformer(shape, 64, 4, 2, k),
             {"lr": 1e-3, "batch_size": 32}),
            ("gru", lambda k: fc.GRUForecaster(shape, 128, 1, k), {"lr": 1e-3, "batch_size": 32})):
        grid = fc.train_grid(factory, splits, lookback, horizons, hp, epochs, seed)
        preds[f"{name} (multi)"] = grid["multi"]
    preds["pod-sindy"] = fc.PODSindy(8, dt=dt_min / 60, seed=seed).fit(
        splits["train"]).predict(test, lookback, horizons)

    rows = []
    for name, by_h in preds.items():
        for h in horizons:
            truth = ev.truth_frames(test, lookback, h, len(by_h[h]))
            rows.append({"model": name, "horizon_min": int(h * dt_min),
                         **ev.metrics(truth, by_h[h]),
                         "predictability": seq.predictability(h)})
    return pd.DataFrame(rows), seq
