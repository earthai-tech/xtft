<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/xtft-logo-dark.svg">
    <img src="docs/_static/logo/xtft-logo.svg" alt="XTFT: covariate-aware probabilistic forecasting" width="560"/>
  </picture>
  <br/><br/>
  <a href="https://github.com/earthai-tech/xtft/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/earthai-tech/xtft/actions/workflows/tests.yml/badge.svg?branch=dev"/></a>
  <a href="https://github.com/earthai-tech/xtft/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/earthai-tech/xtft/actions/workflows/docs.yml/badge.svg?branch=dev"/></a>
  <a href="https://codecov.io/gh/earthai-tech/xtft"><img alt="Coverage" src="https://codecov.io/gh/earthai-tech/xtft/branch/dev/graph/badge.svg"/></a>
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A5%203.10-blue?logo=python&logoColor=white"/>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue"/></a>
</div>

**XTFT** forecasts many related time series at once, with quantiles
you can trust. It uses static attributes, past observations and
covariates known in advance; it stays stable when the level of a
series drifts or inputs go missing; and its calibration layer turns
the quantiles of any model into intervals, and whole-path bands,
with the coverage they claim.

## Features

| | |
|---|---|
| 🧩 **Covariates** | Static, past and known-future inputs, selected and fused at every step; variable weights come out of the forward pass. |
| 🌊 **Level shift** | Reversible instance normalisation of each window. |
| 🕳️ **Missing data** | Pass gaps as `NaN`; masks and a learned decay handle them inside the model. |
| 📐 **Quantiles** | A monotone head, so quantiles never cross. |
| 🛰️ **Calibration** | Split-conformal and adaptive conformal intervals for any model's forecasts. |
| 🛤️ **Whole-path bands** | Bands that contain the entire forecast path, not just each step. |

## Installation

```bash
pip install -e ".[torch]"      # from a clone of this repository
```

Requires Python 3.10 or later. Extras: `tf` (Keras backend), `bench`
(benchmark baselines), `test`, `docs`.

## Quick start

```python
from xtft.nn import XTFT  # PyTorch backend by default

model = XTFT(
    static_input_dim=4,
    dynamic_input_dim=5,
    future_input_dim=2,
    forecast_horizon=24,
    quantiles=[0.1, 0.5, 0.9],
    instance_norm=True,
)
q = model(x_static, x_dynamic, x_future)  # (batch, 24, 3)
```

Shapes: `x_static` `(batch, n_static)`, `x_dynamic`
`(batch, past_steps, n_dynamic)` with `NaN` for missing values,
`x_future` `(batch, past_steps + horizon, n_future)`. Panels in long
format go through `xtft.data` (chronological splits, scaling,
rolling windows), and `xtft.nn.torch.trainer.fit` trains with early
stopping on CPU or GPU (`TrainConfig(device="auto", amp="bf16")`).

## Calibration

`xtft.evaluation` works on the quantile forecasts of any model:

```python
from xtft.evaluation import (
    AdaptiveConformalCalibrator,
    SimultaneousConformalBand,
    path_coverage,
)

aci = AdaptiveConformalCalibrator(quantiles).fit(y_val, q_val)
q_test = aci.transform(y_test, q_test, times=origin_times)

band = SimultaneousConformalBand(quantiles, coverage=0.8)
lower, upper = band.fit(y_val, q_val).transform_online(
    y_test, q_test, times=origin_times
)
path_coverage(y_test, lower, upper)  # share of fully covered paths
```

It also provides pinball loss, CRPS, coverage, interval scores,
Diebold-Mariano tests and Friedman/Nemenyi rankings.

## More

* [Documentation](https://earthai-tech.github.io/xtft) (API
  reference and guides).
* [benchmarks/](benchmarks/README.md): reproduce the comparisons
  with other forecasters.
* [CONTRIBUTING.md](CONTRIBUTING.md): development setup, tests and
  releases.

## Citing

A paper describing XTFT is in preparation; citation details will be
added here.

## Licence

BSD-3-Clause, see [LICENSE](LICENSE). Support code and the Keras
models come from
[fusionlab-learn](https://github.com/earthai-tech/fusionlab-learn);
see [PORTING.md](PORTING.md).
