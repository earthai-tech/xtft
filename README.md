<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/xtft-logo-dark.svg">
    <img src="docs/_static/logo/xtft-logo.svg" alt="XTFT: covariate-aware probabilistic forecasting with calibrated, shift-robust uncertainty" width="600"/>
  </picture>
  <br/>
  <em>Covariate-aware probabilistic forecasting that stays robust to shift and gaps, with predictive intervals and whole-path bands you can trust.</em>
  <br/><br/>
  <a href="https://github.com/earthai-tech/xtft/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/earthai-tech/xtft/actions/workflows/tests.yml/badge.svg?branch=dev"/></a>
  <a href="https://github.com/earthai-tech/xtft/actions/workflows/lint.yml"><img alt="Lint" src="https://github.com/earthai-tech/xtft/actions/workflows/lint.yml/badge.svg?branch=dev"/></a>
  <a href="https://github.com/earthai-tech/xtft/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/earthai-tech/xtft/actions/workflows/docs.yml/badge.svg?branch=dev"/></a>
  <a href="https://codecov.io/gh/earthai-tech/xtft"><img alt="Coverage" src="https://codecov.io/gh/earthai-tech/xtft/branch/dev/graph/badge.svg"/></a>
  <br/>
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

Forecasts that feed decisions rarely come from clean, stationary,
univariate series. Prices depend on load and weather forecasts,
sensors drop out, levels drift after the training period, and a
planner asks whether a threshold will be crossed *anywhere* in the
next day or year, not at one step. XTFT is built for that setting:
it keeps the gated, variable-selecting design of the Temporal
Fusion Transformer, adds robustness to shift and gaps, and comes
with a calibration layer whose coverage claims hold up on held-out
data.

| | What XTFT does |
|---|---|
| 🧩 **Covariate-aware fusion** | Static, past-observed and known-future covariates are selected and fused per step; many strong recent forecasters (PatchTST, DLinear, iTransformer) cannot use covariates at all. |
| 🌊 **Robust to level shift** | Reversible instance normalisation, mask-aware, so series whose level drifts outside the training range stay forecastable. |
| 🕳️ **Native missing data** | Pass gaps as `NaN`: masks, time since the last observation and a learned decay replace ad-hoc imputation. |
| 📐 **Calibrated quantiles** | A monotone quantile head (quantiles never cross) and split-conformal calibration per horizon step, applied identically to any model. |
| 🛰️ **Shift-robust intervals** | Adaptive conformal calibration re-estimates the interval margins causally as new observations arrive, keeping long-run coverage near nominal when the error distribution drifts after the calibration period. |
| 🛤️ **Whole-path bands** | Conformal bands that contain the entire forecast path, not just each step, with a shift-adaptive version that updates causally as paths are observed. |
| 🔄 **Safe updating** | In deployment, an anchored update fine-tunes from the trained weights, early-stops on the latest weeks and is kept only if it helps; repeated naive fine-tuning drifts. |
| 🔍 **Interpretability** | Static, past and future variable weights come out of the forward pass. |
| 🎚️ **Optional multi-resolution encoder** | Anti-aliased, origin-aligned coarse views of the history with cross-scale attention (`scales=(1, 2, 4)`); in our ablations it neither helped nor hurt, and `scales=(1,)` gives a smaller, equally accurate model. |

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
info["past_weights"]                            # (batch, steps, n_vars)
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
  forecasts as known-future covariates; M5 sales at the store x
  department level (`M5-StoreDept`, 70 series) with prices, SNAP
  days, events and static store/department/state attributes.
  Downloaded on first use into `data/raw`.
* **Gap study**: `--missing-rate`, `--missing-mode random|blocks`
  remove target values; models train on the gapped series and are
  scored on the clean one.
* **Models**: `xtft`, any neuralforecast model as `nf:<Name>`
  (TFT, PatchTST, NHITS, DLinear, TiDE, NBEATSx, ...), any
  statsforecast model as `sf:<Name>` (SeasonalNaive, AutoETS, ...),
  a global LightGBM quantile model (`lgb`) and LEAR, the
  recalibrated LASSO reference of the electricity price benchmark
  (`lear`). Every model receives the covariates it supports.
* **Updating**: `--model-kw '{"refit_every": 672, "refit_mode":
  "finetune_es"}'` updates XTFT every four weeks during the test
  period (`finetune`, `finetune_es` or `retrain`); LEAR takes
  `recalibrate_every` (days, 0 = once).
* **Protocol**: chronological train/validation/test periods, the
  same rolling origins, input length and optimisation budget for
  every learned model, conformal calibration fitted on validation
  forecasts, metrics before and after calibration, several seeds.
  `benchmarks/recalibrate.py` re-scores saved forecasts with other
  calibration settings or whole-path bands without refitting.

### Headline results

CRPS after calibration, mean over three seeds (lower is better);
significance from Diebold-Mariano tests at 5% on every task.

| Benchmark | XTFT | Best other model | Finding |
|---|---|---|---|
| EPF, 5 markets (H = 24) | 0.153 | LEAR daily 0.145; TFT 0.155 | best neural/tree model, never significantly worse than one |
| EPF, monthly anchored update | 0.146 | LEAR daily 0.145 | within 1% of a model recalibrated every day |
| M5 store x dept. (H = 28) | 0.417 | TFT 0.425 | significantly better than all nine baselines |
| ETT, 16 tasks (univariate) | rank 2.6 | PatchTST rank 1.5 | specialised univariate models lead without covariates |

Adaptive conformal calibration brings the per-step 80% coverage of
every model on EPF within 1.3 points of nominal on average (18.8
points before calibration), and adaptive whole-path bands cover
77-83% of days at 80%, where per-step intervals cover at most 58%.

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
experiments/    ablation notes
jobs/           SLURM scripts (site settings in jobs/site.env)
data/           local datasets (not versioned)
docs/           documentation and logo
tests/          pytest suite (with doctests of the public modules)
```

Run the tests with `pytest` (coverage: `pip install -e
".[torch,test]"`, then `pytest --cov=xtft`); the code follows the
numpydoc convention and is checked with `ruff`. Build the
documentation with `pip install -e ".[torch,docs]"` and
`sphinx-build -b html docs docs/_build/html`.

### Continuous integration and releases

GitHub Actions run on every push to `dev` and `master` and on pull
requests:

| Workflow | What it does |
|---|---|
| `tests.yml` | pytest with coverage on Python 3.10-3.12 (Linux) and 3.12 (Windows, macOS), CPU PyTorch; Keras backend tests on `master`, weekly and on demand; coverage to Codecov |
| `lint.yml` | `ruff check` and `ruff format --check` |
| `docs.yml` | Sphinx build with warnings as errors; deployed to GitHub Pages from `master` and version tags |
| `release.yml` | on a `vX.Y.Z` tag: build, check the tag against `pyproject.toml`, test the wheel, publish (final versions to PyPI, pre-releases to TestPyPI) and create the GitHub release |

To release: set `version` in `pyproject.toml`, merge into `master`,
then `git tag vX.Y.Z && git push origin vX.Y.Z`.

## 📖 Citing

A paper describing XTFT is in preparation; citation details will be
added here.

## 🙏 Provenance

Most support code and the Keras models come from
[fusionlab-learn](https://github.com/earthai-tech/fusionlab-learn)
(BSD-3-Clause); see [PORTING.md](PORTING.md).

## ⚖️ Licence

BSD-3-Clause. See [LICENSE](LICENSE).
