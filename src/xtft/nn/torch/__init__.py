"""PyTorch implementation of XTFT (reference implementation).

``layers/``
    Gated residual networks, variable selection, anti-aliased
    multi-resolution encoder, local and cross-scale attention.
``models/``
    :class:`~xtft.nn.torch.models.XTFT`.
``probabilistic/``
    Non-crossing quantile head and pinball (quantile) loss.
``trainer``
    :func:`~xtft.nn.torch.trainer.fit` and
    :func:`~xtft.nn.torch.trainer.predict`.

Conformal calibration is framework-agnostic and lives in
:mod:`xtft.evaluation`. Models are resolved through :mod:`xtft.nn`.
"""
