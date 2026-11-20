"""PyTorch implementation of XTFT (reference implementation).

Planned layout, filled during the method redesign (see PLAN.md):

``layers/``
    Variable selection, gated residual networks, anti-aliased
    multi-resolution encoder, cross-scale attention, learnable
    windows.
``models/``
    ``tft.py`` (TFT baseline) and ``xtft.py`` (XTFT, SuperXTFT).
``probabilistic/``
    Non-crossing quantile head, conformal calibration, losses.

Models are resolved through :mod:`xtft.nn`; until a module exists,
the lookup raises :class:`NotImplementedError`.
"""
