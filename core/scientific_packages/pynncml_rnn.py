"""
Training and evaluating PyNNcml's two-step RNN (Habi & Messer 2020) on OpenMRG.

Shared by ``projects/retrieval/openmrg`` (the data-driven retrieval notebook) and
``projects/spatial_interpolation`` (the "Model-2" estimation maps).

The upstream PyNNcml data-driven tutorial writes the sliding-window loop out three times - for loss balancing,
training and validation - and defines the loss inline. Here each exists once:

``openmrg_dataset``   PyNNcml's OpenMRG loader on the repository's archive,
                      with the ``pynncml_compat`` workarounds applied
``RegressionLoss``    the rain-weighted MSE of the paper
``windows``           the sliding window over a batch, in one place
``build_model``       the two-step network with normalization from the data
``train``             loss balancing, then the epoch loop; returns a history
``predict``           validation-set estimates, detections and references
``detection_scores``  accuracy, F1 and confusion matrix, reference first

    from core.scientific_packages.pynncml_rnn import (TrainConfig, openmrg_dataset,
                                                      build_model, train, predict)
    cfg = TrainConfig(n_epochs=20)
    train_loader, val_loader, stats = openmrg_dataset(slice("2015-06-01", "2015-06-10"), cfg)
    model = build_model(cfg, train_loader)
    history = train(model, train_loader, cfg, stats["exp_gamma"])
    result = predict(model, val_loader, cfg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters, defaulting to the upstream tutorial's."""

    batch_size: int = 16
    window_size: int = 32
    rnn_n_features: int = 128
    metadata_n_features: int = 32
    n_layers: int = 2
    rnn_type: str = "GRU"            # or "LSTM"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    n_epochs: int = 200
    wet_threshold: float = 0.1       # mm/h, for the detection target
    clip_grad: float | None = None   # max grad norm; spatial_interpolation used 1.0
    balance_batches: int | None = None  # batches used to set lambda (None = all)
    val_fraction: float = 0.2
    seed: int = 0                    # the tutorial's split was unseeded

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def openmrg_dataset(time_slice: slice, cfg: TrainConfig):
    """Train/validation loaders plus the rain-rate statistics the loss needs.

    Returns ``(train_loader, val_loader, stats)``; ``stats`` holds the gauge
    rain rates and the scale of an exponential fitted to them
    (``exp_gamma``), which ``RegressionLoss`` uses to weight wet samples.
    """
    import pynncml as pnc
    import scipy.stats

    from core.scientific_packages import pynncml_compat

    pynncml_compat.apply()
    dataset = pynncml_compat.quietly(
        pnc.datasets.loader_open_mrg_dataset,
        data_path=pynncml_compat.openmrg_data_path(), time_slice=time_slice)
    split = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [1 - cfg.val_fraction, cfg.val_fraction], generator=split)

    rain = np.stack([p.data_array for p in dataset.point_set]).ravel()
    exp_fit = scipy.stats.expon.fit(rain)
    stats = {"rain": rain, "exp_fit": exp_fit, "exp_gamma": exp_fit[1],
             "wet_fraction": float((rain > 0).mean()), "dataset": dataset}
    return (torch.utils.data.DataLoader(train_set, cfg.batch_size),
            torch.utils.data.DataLoader(val_set, cfg.batch_size), stats)


# --------------------------------------------------------------------------
# model and loss
# --------------------------------------------------------------------------
class RegressionLoss(torch.nn.Module):
    """Rain-weighted MSE: ``sum_s mean_b (1 - g_s exp(-g r)) (r - r_hat)^2``.

    Most samples are dry, so plain MSE learns to predict zero. The weight
    grows from ``1 - gamma_s`` at ``r = 0`` towards 1 for heavy rain, with
    ``gamma`` the scale of the exponential fitted to the gauge record.
    """

    def __init__(self, gamma: float, gamma_s: float = 0.9):
        super().__init__()
        self.gamma = gamma
        self.gamma_s = gamma_s

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1 - self.gamma_s * torch.exp(-self.gamma * target)
        return torch.sum(torch.mean(weight * (target - estimate) ** 2, dim=0))


def build_model(cfg: TrainConfig, train_loader):
    """The two-step (estimation + detection) RNN, normalized on the training set.

    Weights are initialized under ``cfg.seed``, so two runs of the same
    config train the same network.
    """
    import pynncml as pnc

    torch.manual_seed(cfg.seed)
    return pnc.scm.rain_estimation.two_step_network(
        n_layers=cfg.n_layers,
        rnn_type=getattr(pnc.neural_networks.RNNType, cfg.rnn_type),  # DNNType before 0.3.7
        normalization_cfg=pnc.training_helpers.compute_data_normalization(train_loader),
        rnn_input_size=180,                      # 90 RSL + 90 TSL features
        rnn_n_features=cfg.rnn_n_features,
        metadata_input_size=2,
        metadata_n_features=cfg.metadata_n_features,
        pretrained=False,
    ).to(cfg.device)


def windows(model, batch, cfg: TrainConfig):
    """Run the model over a batch window by window, carrying the RNN state.

    Yields ``(rain_hat, wet_prob, rain_ref)`` per window. The state is
    detached between windows (truncated back-propagation through time), as
    in the upstream tutorial.
    """
    rain_rate, rsl, tsl, metadata = batch
    device = cfg.device
    state = model.init_state(batch_size=rsl.shape[0])
    metadata = metadata.to(device)
    w = cfg.window_size
    for step in range(math.floor(rain_rate.shape[1] / w)):
        sl = slice(step * w, (step + 1) * w)
        signal = torch.cat([rsl[:, sl, :], tsl[:, sl, :]], dim=-1).to(device)
        out, state = model(signal, metadata, state.detach())
        yield out[:, :, 0], out[:, :, 1], rain_rate[:, sl, 0].float().to(device)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def _losses(model, batch, cfg, est_loss, det_loss):
    for rain_hat, wet_prob, ref in windows(model, batch, cfg):
        yield est_loss(rain_hat, ref), det_loss(wet_prob, (ref > cfg.wet_threshold).float())


def balance_weight(model, loader, cfg: TrainConfig, est_loss, det_loss) -> float:
    """Weight on the estimation loss that makes both losses equal at start.

    Over the whole loader, or its first ``cfg.balance_batches`` batches (the
    spatial_interpolation pipeline used one).
    """
    total_est = total_det = 0.0
    with torch.no_grad():
        for b, batch in enumerate(loader):
            if cfg.balance_batches is not None and b >= cfg.balance_batches:
                break
            for le, ld in _losses(model, batch, cfg, est_loss, det_loss):
                total_est += float(le)
                total_det += float(ld)
    return total_det / max(total_est, 1e-9)


def train(model, loader, cfg: TrainConfig, exp_gamma: float,
          progress: bool = True, val_loader=None) -> dict:
    """Train with RAdam on ``lambda * L_est + L_det``; returns per-epoch means.

    With ``val_loader`` the validation loss is tracked each epoch and the
    model is left at its best-validation state (``history["best_epoch"]``).
    Non-finite batch losses are skipped rather than stepped on.
    """
    torch.manual_seed(cfg.seed)
    est_loss, det_loss = RegressionLoss(exp_gamma), torch.nn.BCELoss()
    lam = balance_weight(model, loader, cfg, est_loss, det_loss)
    opt = torch.optim.RAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"loss": [], "loss_est": [], "loss_det": [], "val_loss": [],
               "lambda": lam, "best_epoch": None}
    best_val, best_state = float("inf"), None
    epochs = range(cfg.n_epochs)
    if progress:
        from tqdm.auto import tqdm
        epochs = tqdm(epochs, desc="epochs")
    for epoch in epochs:
        model.train()
        sums = np.zeros(3)
        n = 0
        for batch in loader:
            for le, ld in _losses(model, batch, cfg, est_loss, det_loss):
                opt.zero_grad()
                loss = lam * le + ld
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                opt.step()
                sums += (loss.item(), le.item(), ld.item())
                n += 1
        for key, value in zip(("loss", "loss_est", "loss_det"), sums / max(n, 1)):
            history[key].append(float(value))
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                v = [float(lam * le + ld) for batch in val_loader
                     for le, ld in _losses(model, batch, cfg, est_loss, det_loss)]
            v = float(np.mean(v)) if v else float("inf")
            history["val_loss"].append(v)
            if v < best_val:
                best_val, history["best_epoch"] = v, epoch
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def predict(model, loader, cfg: TrainConfig) -> dict:
    """Validation estimates: ``ref``, ``rain_hat`` and ``detection``.

    Each is (links, time). Estimated rain is zeroed where the detector says
    dry, as the two-step design intends.
    """
    model.eval()
    ref, hat, det = [], [], []
    with torch.no_grad():
        for batch in loader:
            parts = [(r_hat * torch.round(p), torch.round(p), r)
                     for r_hat, p, r in windows(model, batch, cfg)]
            hat.append(torch.cat([h for h, _, _ in parts], dim=1).cpu().numpy())
            det.append(torch.cat([d for _, d, _ in parts], dim=1).cpu().numpy())
            ref.append(torch.cat([r for _, _, r in parts], dim=1).cpu().numpy())
    return {"ref": np.concatenate(ref), "rain_hat": np.concatenate(hat),
            "detection": np.concatenate(det)}


def detection_scores(ref: np.ndarray, detection: np.ndarray,
                     wet_threshold: float = 0.1) -> dict:
    """Accuracy, F1 and the confusion matrix, with the gauge as reference.

    (The upstream tutorial passed the arguments to ``confusion_matrix`` the
    other way round, which transposes the matrix; F1 is symmetric and
    unaffected.)
    """
    from sklearn import metrics

    truth = (np.ravel(ref) > wet_threshold).astype(int)
    pred = np.round(np.ravel(detection)).astype(int)
    return {"accuracy": float((truth == pred).mean()),
            "f1": float(metrics.f1_score(truth, pred)),
            "confusion": metrics.confusion_matrix(truth, pred, labels=[0, 1])}


def estimation_scores(ref: np.ndarray, rain_hat: np.ndarray) -> dict:
    """Bias and RMSE of the rain-rate estimate over all validation samples."""
    delta = np.ravel(rain_hat) - np.ravel(ref)
    return {"bias": float(delta.mean()), "rmse": float(np.sqrt((delta ** 2).mean()))}
