"""Forecasting models, resolved for the active backend.

Model classes are looked up lazily, on attribute access, from the
implementation that matches :func:`xtft.backends.get_backend`:

* ``"torch"`` (default): :mod:`xtft.nn.torch`, the reference
  implementation used in the benchmarks;
* ``"tensorflow"``: :mod:`xtft.nn.tf`, the Keras implementation
  ported from fusionlab-learn.

``TFT`` is provided by the TensorFlow backend only. Benchmarks use
an independent PyTorch TFT (neuralforecast) as the baseline, so the
comparison does not rest on our own re-implementation of the
competitor. The Keras ``SuperXTFT`` is no longer part of the
public API; it stays importable from :mod:`xtft.nn.tf.hybrid`.

Examples
--------
>>> import xtft.nn
>>> sorted(xtft.nn.__all__)
['TFT', 'XTFT']
"""

from __future__ import annotations

import importlib

from ..backends import get_backend

__all__ = ["TFT", "XTFT"]

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


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
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


def __dir__():
    return sorted(list(globals()) + __all__)
