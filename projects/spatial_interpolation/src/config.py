"""
One configuration for the nowcasting pipeline.

Everything ``advanced_models_colab_v2.ipynb`` set as module-level constants,
in one frozen dataclass, plus where artifacts are cached.

``faithful=True`` reproduces the notebook's choices exactly. ``faithful=False``
applies four corrections, so their effect can be measured on the same data:

====================  =================================  ==============================
step                  faithful (as in the notebook)      corrected
====================  =================================  ==============================
IDW distance          raw degrees (1 deg lon = 0.53      kilometres (local equirect.)
                      deg lat at 57.7 N)
GMZ interpolation     pynncml 0.3.7 as shipped (corner   patched (core.scientific_
                      reads the wrong cell)              packages.pynncml_compat)
GMZ -> radar grid     linear map of the 0..1 grid onto   pynncml's own normalization
                      the bounding box of link midpoints inverted exactly (UTM)
pySTEPS alignment     extrapolates from the frame the    extrapolates from the last
                      target is compared against         input frame, like every model
====================  =================================  ==============================

    cfg = NowcastConfig()                        # the paper's setup
    cfg = NowcastConfig.smoke()                  # 1 week, a few epochs, CPU
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from core import data_paths as dp

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class NowcastConfig:
    # time
    # (first day, last day), inclusive
    train: tuple = ("2015-06-01", "2015-07-20")
    val: tuple = ("2015-07-21", "2015-08-05")
    test: tuple = ("2015-08-06", "2015-08-31")
    lookback: int = 8                          # 15-min frames of history
    horizons: tuple = (1, 2, 3, 4)             # steps of 15 min
    wet_threshold: float = 0.1                 # mm/h

    # space: evaluation crop over the gauge-rich area
    lat_range: tuple = (57.55, 57.85)
    lon_range: tuple = (11.70, 12.20)

    # maps
    idw_power: float = 2.0
    gmz_roi: float = 3.0
    gmz_points_per_link: int = 2

    # models
    est_epochs: int = 200                      # "Model-2" CML estimation RNN
    est_hparams: dict = field(default_factory=lambda: dict(
        rnn_n_features=256, n_layers=2, lr=1e-4, weight_decay=1e-4, batch_size=16))
    forecast_epochs: int = 80                  # Transformer / GRU forecasters
    transformer: dict = field(default_factory=lambda: dict(
        d_model=128, n_heads=4, n_layers=2, lr=1e-3, batch_size=16))
    gru: dict = field(default_factory=lambda: dict(
        hidden_dim=256, n_layers=1, lr=1e-3, batch_size=16))
    sindy_modes: int = 8

    faithful: bool = True
    cache_dir: Path = REPO_ROOT / "dataset/open_datasets/OpenMRG_Sweden/processed/nowcast"
    radar_nc: Path = dp.data_path(dp.OPENMRG_RADAR)   # ~/data/cml: see core/data_paths.py
    seed: int = 42

    def period(self, which: str) -> slice:
        """A split, or "full", as a slice of dates."""
        if which == "full":
            return slice(self.train[0], self.test[1])
        return slice(*getattr(self, which))

    @property
    def full(self) -> slice:
        return self.period("full")

    @property
    def horizons_min(self) -> tuple:
        return tuple(15 * h for h in self.horizons)

    @property
    def tag(self) -> str:
        """Cache folder: one per period (inputs depend on nothing else)."""
        return f"{self.full.start}_{self.full.stop}".replace("-", "")

    # the settings each pipeline stage depends on; a stage's cache file is
    # keyed by a hash of these, so changing any of them recomputes it
    STAGE_KEYS = {
        "inputs": ("train", "val", "test", "lat_range", "lon_range"),
        "cml_maps": ("faithful", "idw_power", "gmz_roi", "gmz_points_per_link"),
        "estimation": ("faithful", "idw_power", "gmz_roi", "gmz_points_per_link",
                       "est_epochs", "est_hparams", "seed"),
        "forecasts": ("faithful", "idw_power", "gmz_roi", "gmz_points_per_link",
                      "est_epochs", "est_hparams", "lookback", "horizons",
                      "forecast_epochs", "transformer", "gru", "sindy_modes", "seed"),
        "pysteps": ("train", "val", "test", "lat_range", "lon_range", "faithful",
                    "lookback", "horizons"),
    }

    def stage_key(self, stage: str) -> str:
        import hashlib
        import json
        fields = {k: getattr(self, k) for k in self.STAGE_KEYS.get(stage, ())}
        blob = json.dumps({**fields, "period": self.tag}, sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:10]

    def path(self, name: str) -> Path:
        p = Path(self.cache_dir) / self.tag
        p.mkdir(parents=True, exist_ok=True)
        return p / name

    def with_(self, **changes) -> "NowcastConfig":
        return replace(self, **changes)

    @classmethod
    def smoke(cls, **overrides) -> "NowcastConfig":
        """Three weeks and a few epochs: checks the wiring on a CPU in minutes.

        Its scores say nothing about the models - too little data, too
        little training - only that every stage runs end to end.
        """
        # every split needs rain, or the estimation network learns "always
        # dry" and its best validation epoch is the untrained one: training
        # holds ~48 mm (incl. Torslanda, 28 Jul), validation ~11, test ~9
        base = dict(train=("2015-07-23", "2015-07-31"),
                    val=("2015-08-01", "2015-08-06"),
                    test=("2015-08-14", "2015-08-16"),
                    est_epochs=20, forecast_epochs=5, sindy_modes=4,
                    est_hparams=dict(rnn_n_features=64, n_layers=1, lr=1e-3,
                                     weight_decay=1e-4, batch_size=16),
                    transformer=dict(d_model=32, n_heads=2, n_layers=1, lr=1e-3, batch_size=16),
                    gru=dict(hidden_dim=64, n_layers=1, lr=1e-3, batch_size=16))
        return cls(**{**base, **overrides})

    def split_index(self, times: pd.DatetimeIndex, which: str) -> tuple:
        """(lo, hi) indices of a split in a 15-min time axis, whole days."""
        sl = self.period(which)
        lo = pd.Timestamp(sl.start)
        hi = pd.Timestamp(sl.stop) + pd.Timedelta(days=1)
        idx = ((times >= lo) & (times < hi)).nonzero()[0]
        return (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, 0)
