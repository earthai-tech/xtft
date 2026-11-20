# xtft

Cross-scale Temporal Fusion Transformer (XTFT) for probabilistic
forecasting of many related series with static, past and
known-future covariates.

> Status: pre-alpha. The PyTorch XTFT (reference implementation)
> is in place; benchmarks are next. See [PLAN.md](PLAN.md).

## The model

XTFT keeps the gated, variable-selecting design of the Temporal
Fusion Transformer and changes how the history is encoded:

* **anti-aliased multi-resolution encoder**: each coarse view of the
  history is low-pass filtered (windowed sinc, cutoff at the new
  Nyquist frequency) and decimated on a grid anchored at the
  forecast origin, so the newest observation is never dropped;
* **hierarchical attention**: causal attention with a learned soft
  window inside each resolution, then attention from every forecast
  step across all resolutions;
* **non-crossing quantiles**: a monotone quantile head, so
  predictive intervals are always valid;
* **interpretability**: static, past and future variable weights,
  per-step scale attribution and the learned windows.

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

## Backends

| Backend | Selected by | Provides |
|---|---|---|
| `torch` (default) | `xtft.set_backend("torch")` or `XTFT_BACKEND=torch` | `XTFT` (redesigned) |
| `tensorflow` | `xtft.set_backend("tensorflow")` or `XTFT_BACKEND=tf` | `XTFT`, `TFT` (fusionlab-learn, legacy) |

The two `XTFT` classes share the input contract but not the
architecture: the Keras one is the original fusionlab-learn model.
The PyTorch TFT baseline in the benchmarks is neuralforecast's
independent implementation. The legacy Keras `SuperXTFT` remains in
`xtft.nn.tf.hybrid` but is not part of the public API.

## Install (development)

```powershell
pip install -e ".[torch,dev]"
```

Add `tf` to the extras for the Keras backend and `bench` for the
benchmark baselines.

## Calibrated intervals

```python
from xtft.evaluation import (
    ConformalQuantileCalibrator, evaluate_quantile_forecast)

cal = ConformalQuantileCalibrator(quantiles).fit(y_val, q_val)
q_test_cal = cal.transform(q_test)   # CQR, one margin per step
evaluate_quantile_forecast(y_test, q_test_cal, quantiles)
# -> pinball, crps_q, mae, rmse, coverage_80, winkler_80, ...
```

Both work on NumPy arrays of any model, so baselines are scored
and calibrated with the same code.

## Benchmarks

```powershell
python benchmarks/run.py --dataset ETTh1 --model xtft --horizon 96
python benchmarks/run.py --dataset ETTh1 --model nf:PatchTST --horizon 96
python benchmarks/aggregate.py results --metric crps_q
```

Models: `xtft`, any neuralforecast model as `nf:<Name>`, any
statsforecast model as `sf:<Name>`. On the HPC, generate a task
list with `benchmarks/make_tasks.py` and submit
`jobs/bench_array.sbatch` as a SLURM array.

## Layout

```
src/xtft/
  backends/     framework switch (torch / tensorflow); array/ backends
  compat/       version shims: numpy, pandas, scipy, sklearn, tf, torch
  nn/           models resolved for the active backend
    torch/      PyTorch implementation (reference) and trainer
    tf/         Keras implementation ported from fusionlab-learn
  data/         panel data, chronological splits, rolling windows
  evaluation/   probabilistic metrics, conformal calibration
  api/ core/ utils/ metrics/ decorators.py exceptions.py
benchmarks/     dataset registry, model adapters, runner, aggregation
experiments/    ablations, Nansha/Zhongshan case study
jobs/           SLURM scripts for the HPC
paper/          manuscript and editable figure folders
```

## Provenance

Most support code and the Keras models come from
[fusionlab-learn](https://github.com/earthai-tech/fusionlab-learn)
(BSD-3-Clause); see [PORTING.md](PORTING.md).

## License

BSD-3-Clause.
