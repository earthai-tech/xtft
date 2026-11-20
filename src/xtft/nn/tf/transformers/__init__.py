"""Keras Temporal Fusion Transformer (ported from fusionlab-learn)."""

from .. import KERAS_BACKEND

if KERAS_BACKEND:
    from ._adj_tft import TFT

    __all__ = ["TFT"]
else:
    __all__ = []
