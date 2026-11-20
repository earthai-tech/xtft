"""Forecasting models, resolved for the active backend.

Model classes are looked up lazily, on attribute access, from the
implementation that matches :func:`xtft.backends.get_backend`:

* ``"torch"`` (default): :mod:`xtft.nn.torch`, the reference
  implementation used in the benchmarks;
* ``"tensorflow"``: :mod:`xtft.nn.tf`, the Keras implementation
  ported from fusionlab-learn.

Examples
--------
>>> import xtft.nn
>>> sorted(xtft.nn.__all__)
['SuperXTFT', 'TFT', 'XTFT']
"""

from __future__ import annotations

import importlib

from ..backends import get_backend

__all__ = ["SuperXTFT", "TFT", "XTFT"]

# (module, attribute) of each public model, per backend.
_REGISTRY = {
    "tensorflow": {
        "XTFT": ("xtft.nn.tf.hybrid._xtft", "XTFT"),
        "SuperXTFT": ("xtft.nn.tf.hybrid._super_xtft", "SuperXTFT"),
        "TFT": ("xtft.nn.tf.transformers._adj_tft", "TFT"),
    },
    "torch": {
        "XTFT": ("xtft.nn.torch.models.xtft", "XTFT"),
        "SuperXTFT": ("xtft.nn.torch.models.xtft", "SuperXTFT"),
        "TFT": ("xtft.nn.torch.models.tft", "TFT"),
    },
}


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    backend = get_backend()
    module_name, attr = _REGISTRY[backend][name]
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
            "backend. Use xtft.backends.set_backend('tensorflow') "
            "for the Keras implementation."
        ) from exc
    return getattr(module, attr)


def __dir__():
    return sorted(list(globals()) + __all__)
