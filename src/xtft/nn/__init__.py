"""Forecasting models and their training loop, at two levels.

Everything public in the neural-network part of xtft is importable
from here::

    from xtft.nn import XTFT, TrainConfig, fit, predict

Names are resolved lazily, on first access, so ``import xtft.nn``
loads no deep-learning framework and the subpackages never import
each other at module level (no circular imports).

Models are looked up from the implementation that matches
:func:`xtft.backends.get_backend`:

* ``"torch"`` (default): :mod:`xtft.nn.torch`, the reference
  implementation used in the benchmarks;
* ``"tensorflow"``: the Keras implementation in ``xtft.nn.tf``.

``TFT`` is provided by the TensorFlow backend only. Benchmarks use
an independent PyTorch TFT (neuralforecast) as the baseline, so the
comparison does not rest on our own re-implementation of the
competitor. The Keras ``SuperXTFT`` is not part of the public API;
it stays importable from ``xtft.nn.tf.hybrid``.

The training loop and the probabilistic building blocks
(``TrainConfig``, ``fit``, ``predict``, ``QuantileLoss``,
``pinball_loss``, ``NonCrossingQuantileHead``) belong to the
PyTorch implementation and resolve to it whatever the backend.

Examples
--------
>>> import xtft.nn
>>> sorted(xtft.nn.__all__)  # doctest: +NORMALIZE_WHITESPACE
['NonCrossingQuantileHead', 'QuantileLoss', 'TFT', 'TrainConfig',
 'XTFT', 'fit', 'pinball_loss', 'predict']
"""

from __future__ import annotations

import importlib

from ..backends import get_backend

# Models whose implementation depends on the backend.
_MODELS = ("TFT", "XTFT")

# PyTorch training utilities, the same for every backend:
# name -> (module, attribute).
_TORCH = {
    "TrainConfig": ("xtft.nn.torch.trainer", "TrainConfig"),
    "fit": ("xtft.nn.torch.trainer", "fit"),
    "predict": ("xtft.nn.torch.trainer", "predict"),
    "QuantileLoss": ("xtft.nn.torch.probabilistic", "QuantileLoss"),
    "pinball_loss": ("xtft.nn.torch.probabilistic", "pinball_loss"),
    "NonCrossingQuantileHead": (
        "xtft.nn.torch.probabilistic",
        "NonCrossingQuantileHead",
    ),
}

__all__ = sorted([*_MODELS, *_TORCH])

# (module, attribute) of each public model per backend; None marks
# a model that the backend deliberately does not provide.
_REGISTRY = {
    "tensorflow": {
        "XTFT": ("xtft.nn.tf.hybrid._xtft", "XTFT"),
        "TFT": ("xtft.nn.tf.transformers._adj_tft", "TFT"),
    },
    "torch": {
        "XTFT": ("xtft.nn.torch.models.xtft", "XTFT"),
        "TFT": None,
    },
}

_UNAVAILABLE = {
    ("torch", "TFT"): (
        "TFT is not implemented for the 'torch' backend: use "
        "neuralforecast.models.TFT as the PyTorch baseline, or "
        "xtft.backends.set_backend('tensorflow') for the Keras TFT."
    ),
}


def _resolve_model(name: str):
    backend = get_backend()
    entry = _REGISTRY[backend][name]
    if entry is None:
        raise NotImplementedError(_UNAVAILABLE[(backend, name)])
    module_name, attr = entry
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only a missing xtft module means "not implemented"; a
        # missing third-party package must surface unchanged.
        missing = exc.name or ""
        if not (
            missing.startswith("xtft.")
            and module_name.startswith(missing)
        ):
            raise
        raise NotImplementedError(
            f"{name} is not yet implemented for the {backend!r} "
            "backend."
        ) from exc
    return getattr(module, attr)


def __getattr__(name: str):
    if name in _MODELS:
        # Not cached: the backend may change between two accesses.
        return _resolve_model(name)
    if name in _TORCH:
        module_name, attr = _TORCH[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value  # cache: independent of the backend
        return value
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    return sorted(list(globals()) + __all__)
