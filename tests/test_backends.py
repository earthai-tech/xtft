"""Tests for the deep-learning backend switch."""

import importlib.util

import pytest

import xtft
from xtft.backends import _framework as fw
from xtft.backends import (
    available_backends,
    get_backend,
    is_available,
    set_backend,
    use_backend,
)


@pytest.fixture(autouse=True)
def _reset_backend(monkeypatch):
    """Isolate each test from the selection state and env."""
    monkeypatch.delenv(fw.ENV_VAR, raising=False)
    monkeypatch.setitem(fw._state, "backend", None)


def test_default_is_torch():
    assert get_backend() == "torch"


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("torch", "torch"),
        ("PyTorch", "torch"),
        ("pt", "torch"),
        ("tensorflow", "tensorflow"),
        ("TF", "tensorflow"),
        ("keras", "tensorflow"),
    ],
)
def test_env_var_aliases(monkeypatch, alias, expected):
    monkeypatch.setenv(fw.ENV_VAR, alias)
    assert get_backend() == expected


def test_explicit_choice_beats_env(monkeypatch):
    monkeypatch.setenv(fw.ENV_VAR, "tensorflow")
    set_backend("torch", strict=False)
    assert get_backend() == "torch"


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        set_backend("jax")


def test_non_string_raises():
    with pytest.raises(TypeError):
        set_backend(3)


def test_strict_refuses_missing_framework(monkeypatch):
    monkeypatch.setattr(fw, "is_available", lambda name: False)
    with pytest.raises(ImportError, match="not installed"):
        set_backend("torch")
    assert fw._state["backend"] is None


def test_use_backend_restores_previous():
    set_backend("torch", strict=False)
    with use_backend("tf") as name:
        assert name == "tensorflow"
        assert get_backend() == "tensorflow"
    assert get_backend() == "torch"


def test_availability_matches_find_spec():
    for name, module in (
        ("torch", "torch"),
        ("tensorflow", "tensorflow"),
    ):
        found = importlib.util.find_spec(module) is not None
        assert is_available(name) is found
        assert (name in available_backends()) is found


def test_top_level_reexports():
    assert xtft.get_backend is get_backend
    assert xtft.set_backend is set_backend


def test_torch_models_not_implemented_yet():
    import xtft.nn

    with use_backend("torch"):
        with pytest.raises(NotImplementedError, match="torch"):
            xtft.nn.XTFT  # noqa: B018


def test_unknown_model_attribute():
    import xtft.nn

    with pytest.raises(AttributeError):
        xtft.nn.NotAModel  # noqa: B018
