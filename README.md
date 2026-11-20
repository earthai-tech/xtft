<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/xtft-logo-dark.svg">
    <img src="docs/_static/logo/xtft-logo.svg" alt="XTFT: Cross-Scale Temporal Fusion Transformer" width="560"/>
  </picture>
  <br/>
  <em>Multi-horizon probabilistic forecasting of many related series, with history read at several time scales and predictive intervals you can trust.</em>
  <br/><br/>
  <img alt="Status: pre-alpha" src="https://img.shields.io/badge/status-pre--alpha-orange"/>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0.dev0-blueviolet"/>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue"/></a>
  <br/>
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A5%203.10-blue?logo=python&logoColor=white"/>
  <a href="https://pytorch.org"><img alt="PyTorch backend" src="https://img.shields.io/badge/PyTorch-reference%20backend-EE4C2C?logo=pytorch&logoColor=white"/></a>
  <a href="https://www.tensorflow.org"><img alt="TensorFlow backend" src="https://img.shields.io/badge/TensorFlow-legacy%20backend-FF6F00?logo=tensorflow&logoColor=white"/></a>
  <img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey"/>
  <a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-D7FF64?logo=ruff&logoColor=black"/></a>
  <br/><br/>

[Why XTFT?](#-why-xtft) ·
[Installation](#-installation) ·
[Quick start](#-quick-start) ·
[Calibrated intervals](#-calibrated-intervals) ·
[Benchmarks](#-benchmarks) ·
[Project](#-project)

</div>

---

## 🧭 Why XTFT?

Many forecasting problems mix slow and fast dynamics: a daily cycle
on top of a weekly one, a seasonal level drifting over years, a
subsidence trend modulated by pumping. Reading the history at one
resolution forces a model to learn every scale from the same
sequence. And a forecast used for decisions needs intervals whose
coverage is what it claims.

XTFT keeps the gated, variable-selecting design of the Temporal Fusion
Transformer and changes how the history is read and how uncertainty
is reported:

| | What XTFT does |
|---|---|
| 🎚️ **Anti-aliased multi-resolution encoder** | Each coarse view of the history is low-pass filtered (windowed sinc, cut-off at the new Nyquist frequency) before decimation, on a grid anchored at the forecast origin, so no scale aliases and the newest observation is never dropped. |
| 🔀 **Cross-scale attention** | Causal attention with a learned soft window inside each resolution, then attention from every forecast step across all resolutions, with a per-step scale attribution. |
| 📐 **Calibrated quantiles** | A monotone quantile head (quantiles never cross) and split-conformal calibration per horizon step, applied identically to any model. |
| 🛰️ **Shift-robust intervals** | Adaptive conformal calibration re-estimates the interval margins causally as new observations arrive, keeping long-run coverage near nominal when the error distribution drifts after the calibration period. |
| 🛤️ **Whole-path bands** | Conformal bands that contain the entire forecast path, not just each step, with a shift-adaptive version that updates causally as paths are observed. |
| 🔍 **Interpretability** | Static, past and future variable weights, scale attributions and learned attention windows come out of the forward pass. |
| 🌊 **Distribution shift** | Optional reversible instance normalisation for series whose level drifts outside the training range. |

## 🚀 Installation

From source (development):

```powershell
git clone https://github.com/earthai-tech/xtft.git
cd xtft
pip install -e ".[torch,dev]"
```

Optional extras: `tf` (Keras backend), `bench` (benchmark baselines:
neuralforecast, statsforecast, datasetsforecast, LightGBM, XGBoost).

## ⚡ Quick start

```python
import torch
from xtft.nn import XTFT  # PyTorch by default
from xtft.nn.torch.probabilistic import pinball_loss

model = XTFT(
    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,
    forecast_horizon=3, quantiles=[0.1, 0.5, 0.9],
)
q = model(x_static, x_dynamic, x_future)        # (batch, 3, 3)
loss = pinball_loss(y, q, model.quantiles)      # NaN targets skipped
q, info = model(x_static, x_dynamic, x_future, return_details=True)
info["scale_attribution"]                       # (batch, 3, n_scales)
```

Inputs: `x_static` is `(batch, n_static)`, `x_dynamic` is
`(batch, past_steps, n_dynamic)` and `x_future` is
`(batch, past_steps + horizon, n_future)` (or `(batch, horizon,
n_future)`). Static or known inputs may be absent (dimension 0).

Panels in long format (`unique_id`, `ds`, target and covariates) go
through `xtft.data`: chronological splits, train-only scaling and
rolling forecast windows; `xtft.nn.torch.trainer.fit` trains with
early stopping.

### GPU

Training and inference run on CUDA (and Apple MPS) without code
changes: `TrainConfig(device="auto")` picks the GPU when one is
visible, batches are copied asynchronously from pinned memory, and
`TrainConfig(amp="bf16")` or `amp="fp16"` (CUDA) trains in mixed
precision with float32 parameters and loss. To check a GPU machine
(a workstation, a cluster node, or a free Colab or Kaggle notebook):

```powershell
pip install -e ".[torch,dev]"
python tools/gpu_check.py
```

It runs the device tests, compares GPU and CPU outputs, and times a
training step in float32, bf16 and fp16.

### Backends

| Backend | Selected by | Provides |
|---|---|---|
| `torch` (default) | `xtft.set_backend("torch")` or `XTFT_BACKEND=torch` | `XTFT` (reference implementation) |
| `tensorflow` | `xtft.set_backend("tensorflow")` or `XTFT_BACKEND=tf` | `XTFT`, `TFT` (Keras models from fusionlab-learn) |

The two `XTFT` classes share the input contract, not the
architecture: the Keras one is the original fusionlab-learn model,
kept for reproducibility.

## 📐 Calibrated intervals

```python
from xtft.evaluation import (
    ConformalQuantileCalibrator, evaluate_quantile_forecast)

cal = ConformalQuantileCalibrator(quantiles).fit(y_val, q_val)
q_test_cal = cal.transform(q_test)   # CQR, one margin per step
evaluate_quantile_forecast(y_test, q_test_cal, quantiles)
# -> pinball, crps_q, mae, rmse, coverage_80, winkler_80, ...
```

When the series may drift after the calibration period, calibrate
along the forecast origins instead; each margin uses only targets
observed before its origin:

```python
from xtft.evaluation import AdaptiveConformalCalibrator

aci = AdaptiveConformalCalibrator(quantiles, gamma=0.005)
q_test_aci = aci.fit(y_val, q_val).transform(
    y_test, q_test, times=origin_times)
```

Per-step intervals say little about the whole path: an 80% band
for every hour of tomorrow can contain all 24 hours on only a small
fraction of days. `SimultaneousConformalBand` covers whole paths:

```python
from xtft.evaluation import SimultaneousConformalBand, path_coverage

band = SimultaneousConformalBand(quantiles, coverage=0.8).fit(y_val, q_val)
lower, upper = band.transform_online(y_test, q_test, times=origin_times)
path_coverage(y_test, lower, upper)   # fraction of fully covered paths
```

`xtft.evaluation` works on NumPy arrays from any model: pinball loss,
CRPS, coverage and width of central intervals, Winkler score, MASE
and MSIS, calibration curves, Diebold-Mariano tests and
Friedman/Nemenyi rankings. Baselines are scored and calibrated by
the same code as XTFT.

## 📊 Benchmarks

```powershell
python benchmarks/run.py --dataset ETTh1 --model xtft --horizon 96
python benchmarks/run.py --dataset ETTh1 --model nf:PatchTST --horizon 96
python benchmarks/aggregate.py results --metric crps_q --reference xtft
```

* **Datasets**: long-horizon ETTh1/h2/m1/m2, Electricity, Traffic,
  Exchange, ILI and Weather; day-ahead electricity prices of five
  markets (`EPF-NP`, `EPF-PJM`, `EPF-BE`, `EPF-FR`, `EPF-DE`, open
  benchmark of Lago et al., 2021) with load and generation
  forecasts as known-future covariates. Downloaded on first use
  into `data/raw`.
* **Gap study**: `--missing-rate`, `--missing-mode random|blocks`
  remove target values; models train on the gapped series and are
  scored on the clean one.
* **Models**: `xtft`, any neuralforecast model as `nf:<Name>`
  (TFT, PatchTST, NHITS, DLinear, TiDE, NBEATSx, ...), any
  statsforecast model as `sf:<Name>` (SeasonalNaive, AutoETS, ...).
  Every model receives the covariates it supports.
* **Protocol**: chronological train/validation/test periods, the
  same rolling origins, input length and optimisation budget for
  every learned model, conformal calibration fitted on validation
  forecasts, metrics before and after calibration, several seeds.

On a SLURM cluster, `benchmarks/make_tasks.py` writes a task list and
`jobs/bench_array.sbatch` runs it as a job array (see
[jobs/README.md](jobs/README.md)).

## 🧩 Project

```
src/xtft/
  nn/           models for the active backend
    torch/      PyTorch implementation (reference) and trainer
    tf/         Keras implementation from fusionlab-learn
  data/         panel data, chronological splits, rolling windows
  evaluation/   probabilistic metrics, conformal calibration, tests
  backends/     framework switch (torch / tensorflow)
  compat/       version shims: numpy, pandas, scipy, sklearn, tf, torch
  api/ core/ utils/ metrics/ decorators.py exceptions.py
benchmarks/     dataset registry, model adapters, runner, aggregation
experiments/    ablations, subsidence case study
jobs/           SLURM scripts (site settings in jobs/site.env)
data/           local datasets (not versioned)
docs/           documentation and logo
tests/          pytest suite (with doctests of the public modules)
```

Run the tests with `pytest`; the code follows the numpydoc
convention and is checked with `ruff`.

## 📖 Citing

A paper describing XTFT is in preparation; citation details will be
added here.

## 🙏 Provenance

Most support code and the Keras models come from
[fusionlab-learn](https://github.com/earthai-tech/fusionlab-learn)
(BSD-3-Clause); see [PORTING.md](PORTING.md).

## ⚖️ Licence

BSD-3-Clause. See [LICENSE](LICENSE).
