"""Tests for xtft.data (panel, splits, windows)."""

import numpy as np
import pandas as pd
import pytest

from xtft.data import (
    PanelData,
    PanelSpec,
    WindowDataset,
    add_calendar_features,
    rolling_origins,
)


def make_frame(n_series=2, length=50, gap=False):
    frames = []
    for i in range(n_series):
        t = np.arange(length)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": t,
                    "y": 100.0 * i + t,
                    "temp": np.cos(t),
                    "cal": np.sin(t),
                    "size": float(i + 1),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    if gap:
        df = df.drop(index=5)
    return df


SPEC = PanelSpec(static=("size",), past=("temp",), known=("cal",))


def test_rolling_origins_aligned_to_end():
    o = rolling_origins(100, first=50, horizon=10, stride=7)
    assert o[-1] == 89  # last origin leaves exactly H targets
    assert np.all(np.diff(o) == 7)
    assert o[0] >= 50
    assert rolling_origins(10, first=5, horizon=10, stride=1).size == 0


def test_spec_validation():
    with pytest.raises(ValueError):
        PanelSpec(target=())
    with pytest.raises(ValueError):
        PanelSpec(past=("y",))
    assert PanelSpec(known="cal").known == ("cal",)


def test_from_frame_shapes_and_errors():
    panel = PanelData.from_frame(make_frame(), SPEC)
    assert panel.n_series == 2
    assert panel.target[0].shape == (50, 1)
    assert panel.static.shape == (2, 1)
    with pytest.raises(KeyError):
        PanelData.from_frame(make_frame(), PanelSpec(past=("nope",)))
    with pytest.raises(ValueError, match="irregular"):
        PanelData.from_frame(make_frame(gap=True), SPEC)
    bad = make_frame()
    bad.loc[3, "size"] = 9.0
    with pytest.raises(ValueError, match="static"):
        PanelData.from_frame(bad, SPEC)


def test_from_frame_does_not_alias_input():
    df = make_frame()
    panel = PanelData.from_frame(df, SPEC)
    panel.fit_scaling(np.array([30, 30]))
    assert df["y"].iloc[0] == 0.0


def test_scaling_uses_train_only_and_inverts():
    panel = PanelData.from_frame(make_frame(), SPEC)
    raw = panel.target[1].copy()
    panel.fit_scaling(np.array([30, 30]))
    train = panel.target[1][:30, 0]
    assert train.mean() == pytest.approx(0.0, abs=1e-5)
    assert train.std() == pytest.approx(1.0, abs=1e-5)
    # The test period is outside the train range: mean is not 0.
    assert panel.target[1][30:, 0].mean() > 1.0
    back = panel.inverse_target(
        panel.target[1][None, :, :], np.array([1])
    )
    assert np.allclose(back[0], raw, atol=1e-3)
    with pytest.raises(RuntimeError):
        panel.fit_scaling(np.array([30, 30]))


def test_windows_content_and_no_leakage():
    panel = PanelData.from_frame(make_frame(), SPEC)
    idx = panel.origins(np.array([0, 0]), np.array([40, 40]), 8, 5, 1)
    ds = WindowDataset(panel, idx, input_size=8, horizon=5)
    # Train targets must end before the cut at 40.
    assert (idx[:, 1] + 5).max() == 39
    i, t0 = idx[3]
    item = ds[3]
    assert item["past"].shape == (8, 2)
    assert item["future"].shape == (13, 1)
    assert item["y"][:, 0].tolist() == [
        100.0 * i + t for t in range(t0 + 1, t0 + 6)
    ]
    assert item["past"][-1, 0] == 100.0 * i + t0
    with pytest.raises(ValueError):
        WindowDataset(panel, np.array([[0, 48]]), 8, 5)


def test_windows_fill_missing_past_keep_missing_targets():
    df = make_frame(n_series=1)
    df.loc[df["ds"] == 20, "y"] = np.nan
    df.loc[df["ds"] == 30, "y"] = np.nan
    panel = PanelData.from_frame(df, SPEC)
    ds = WindowDataset(panel, np.array([[0, 25]]), 10, 6)
    item = ds[0]
    assert np.all(np.isfinite(item["past"]))
    assert np.isnan(item["y"][4, 0])  # step 30


def test_calendar_features():
    df = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=48, freq="h")}
    )
    out, names = add_calendar_features(df, features=["hour"])
    assert names == ("hour_sin", "hour_cos")
    radius = out["hour_sin"] ** 2 + out["hour_cos"] ** 2
    assert np.allclose(radius, 1.0)
    assert out["hour_sin"].iloc[0] == pytest.approx(
        out["hour_sin"].iloc[24]
    )
    with pytest.raises(ValueError):
        add_calendar_features(df, features=["fortnight"])
