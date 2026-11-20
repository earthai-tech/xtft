"""Training and prediction loops for PyTorch forecasters.

A deliberately small trainer: Adam, gradient clipping, early
stopping on the validation pinball loss with best-weight restore,
and a fixed seed. It is model-agnostic as long as the model takes
``(x_static, x_dynamic, x_future)`` and returns quantiles, so the
same loop trains XTFT and its ablations.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, RandomSampler

from ...compat.torch import resolve_device
from .probabilistic import pinball_loss, validate_quantiles

__all__ = ["TrainConfig", "fit", "predict"]

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Optimisation settings.

    Parameters
    ----------
    max_epochs : int, default=100
        Upper bound on epochs.
    batch_size : int, default=64
        Windows per step.
    lr : float, default=1e-3
        Adam learning rate.
    weight_decay : float, default=0.0
        Adam weight decay.
    patience : int, default=10
        Epochs without validation improvement before stopping.
    min_delta : float, default=0.0
        Smallest decrease of the validation loss that counts as an
        improvement.
    grad_clip : float or None, default=1.0
        Max global gradient norm; ``None`` disables clipping.
    steps_per_epoch : int or None, default=None
        If set, each epoch draws this many random batches (with
        replacement) instead of a full pass; useful for long
        panels with millions of windows.
    num_workers : int, default=0
        DataLoader workers.
    seed : int, default=0
        Seed of parameter initialisation is the caller's; this one
        seeds batch order and dropout.
    device : str, default="auto"
        ``"auto"``, ``"cpu"`` or ``"cuda[:i]"``.
    """

    max_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    min_delta: float = 0.0
    grad_clip: float | None = 1.0
    steps_per_epoch: int | None = None
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"


def _inputs(batch: dict[str, Tensor], device):
    """Map a collated window batch to model inputs."""
    xs = batch["static"].to(device)
    xf = batch["future"].to(device)
    return (
        xs if xs.shape[-1] else None,
        batch["past"].to(device),
        xf if xf.shape[-1] else None,
    )


def _as_4d(pred: Tensor, y: Tensor) -> Tensor:
    """Reshape predictions to ``(B, H, O, K)``."""
    if pred.dim() == 3 and y.dim() == 3:
        if y.shape[-1] == 1 and pred.shape[-1] != 1:
            return pred.unsqueeze(-2)  # (B, H, K) quantiles, O = 1
        return pred.unsqueeze(-1)  # (B, H, O) point forecast
    return pred


def _loader(ds, cfg: TrainConfig, shuffle: bool, gen=None):
    if shuffle and cfg.steps_per_epoch:
        sampler = RandomSampler(
            ds,
            replacement=True,
            num_samples=cfg.steps_per_epoch * cfg.batch_size,
            generator=gen,
        )
        return DataLoader(
            ds,
            cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
        )
    return DataLoader(
        ds,
        cfg.batch_size,
        shuffle=shuffle,
        generator=gen,
        num_workers=cfg.num_workers,
    )


def _evaluate(model, loader, quantiles, device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            y = batch["y"].to(device)
            pred = _as_4d(model(*_inputs(batch, device)), y)
            n = int(torch.isfinite(y).sum())
            if n:
                total += float(pinball_loss(y, pred, quantiles)) * n
                count += n
    return total / count if count else math.nan


def fit(
    model: nn.Module,
    train_ds,
    val_ds,
    quantiles: Sequence[float],
    config: TrainConfig | None = None,
) -> dict:
    r"""Train `model` with early stopping on validation pinball loss.

    Parameters
    ----------
    model : torch.nn.Module
        Forecaster taking ``(x_static, x_dynamic, x_future)``.
    train_ds, val_ds : Dataset
        Window datasets (e.g. :class:`xtft.data.WindowDataset`);
        `val_ds` may be ``None`` to train for `max_epochs`.
    quantiles : sequence of float
        Levels of the model output.
    config : TrainConfig or None
        Settings; defaults if ``None``.

    Returns
    -------
    dict
        ``train_loss`` and ``val_loss`` per epoch, ``best_epoch``,
        ``best_val_loss``, ``epochs``, ``seconds`` and ``config``.
        The model is left on the training device with the best
        validation weights loaded.

    Raises
    ------
    ValueError
        If the training set is empty or the loss becomes
        non-finite.

    Notes
    -----
    The objective is the mean pinball loss of
    :func:`~xtft.nn.torch.probabilistic.pinball_loss` over observed
    targets. Gradients are clipped to global norm `grad_clip`.

    Examples
    --------
    >>> import numpy as np, pandas as pd, torch
    >>> from xtft.data import PanelData, PanelSpec, WindowDataset
    >>> from xtft.nn.torch.models import XTFT
    >>> from xtft.nn.torch.trainer import TrainConfig, fit
    >>> df = pd.DataFrame({"unique_id": "a", "ds": np.arange(200),
    ...                    "y": np.sin(np.arange(200) / 3)})
    >>> panel = PanelData.from_frame(df, PanelSpec())
    >>> idx = panel.origins(np.array([0]), None, 16, 4, 1)
    >>> ds = WindowDataset(panel, idx, 16, 4)
    >>> _ = torch.manual_seed(0)
    >>> model = XTFT(0, 1, 0, 4, quantiles=[0.5], hidden_units=8,
    ...              num_heads=2)
    >>> hist = fit(model, ds, ds, [0.5], TrainConfig(max_epochs=2))
    >>> hist["epochs"]
    2
    """
    cfg = config or TrainConfig()
    levels = validate_quantiles(quantiles)
    if len(train_ds) == 0:
        raise ValueError("Empty training set.")
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model.to(device)
    opt_kw = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    try:
        opt = torch.optim.Adam(
            model.parameters(), fused=True, **opt_kw
        )
    except (RuntimeError, TypeError):
        opt = torch.optim.Adam(model.parameters(), **opt_kw)
    train_loader = _loader(train_ds, cfg, shuffle=True, gen=gen)
    val_loader = (
        _loader(val_ds, cfg, shuffle=False)
        if val_ds is not None and len(val_ds)
        else None
    )
    history = {"train_loss": [], "val_loss": []}
    best, best_epoch, best_state, waited = math.inf, -1, None, 0
    start = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        model.train()
        running, steps = 0.0, 0
        for batch in train_loader:
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = _as_4d(model(*_inputs(batch, device)), y)
            loss = pinball_loss(y, pred, levels)
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite training loss at epoch {epoch}."
                )
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
            opt.step()
            running += float(loss)
            steps += 1
        history["train_loss"].append(running / max(steps, 1))
        if val_loader is None:
            history["val_loss"].append(math.nan)
            continue
        val = _evaluate(model, val_loader, levels, device)
        history["val_loss"].append(val)
        logger.info(
            "epoch %d train %.5f val %.5f",
            epoch,
            history["train_loss"][-1],
            val,
        )
        if val < best - cfg.min_delta:
            best, best_epoch, waited = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history.update(
        best_epoch=best_epoch,
        best_val_loss=best if best_state is not None else math.nan,
        epochs=len(history["train_loss"]),
        seconds=time.perf_counter() - start,
        config=asdict(cfg),
    )
    return history


def predict(
    model: nn.Module,
    ds,
    batch_size: int = 256,
    device: str = "auto",
) -> np.ndarray:
    """Forecast every window of `ds`.

    Parameters
    ----------
    model : torch.nn.Module
        Trained forecaster.
    ds : Dataset
        Window dataset.
    batch_size : int, default=256
        Windows per forward pass.
    device : str, default="auto"
        Device for inference.

    Returns
    -------
    numpy.ndarray
        ``(N, H, O, K)`` for quantile models, ``(N, H, O, 1)`` for
        point models, in the units of the dataset (scaled).
    """
    dev = resolve_device(device)
    model.to(dev).eval()
    loader = DataLoader(ds, batch_size, shuffle=False)
    out = []
    with torch.inference_mode():
        for batch in loader:
            y = batch["y"]
            pred = _as_4d(model(*_inputs(batch, dev)), y.to(dev))
            out.append(pred.cpu().numpy())
    if not out:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(out)
