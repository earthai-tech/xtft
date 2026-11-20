# xtft

Cross-scale Temporal Fusion Transformer (XTFT) for probabilistic
forecasting of many related series with static, past and
known-future covariates.

> Status: pre-alpha. The PyTorch implementation is being written;
> the TensorFlow/Keras implementation ported from fusionlab-learn
> works today. See [PLAN.md](PLAN.md).

## Backends

xtft has two implementations behind one API:

| Backend | Selected by | Status |
|---|---|---|
| `torch` (default) | `xtft.set_backend("torch")` or `XTFT_BACKEND=torch` | in development |
| `tensorflow` | `xtft.set_backend("tensorflow")` or `XTFT_BACKEND=tf` | ported from fusionlab-learn |

```python
import xtft

xtft.set_backend("tensorflow")
from xtft.nn import XTFT

model = XTFT(
    static_input_dim=4, dynamic_input_dim=5, future_input_dim=2,
    forecast_horizon=3, quantiles=[0.1, 0.5, 0.9],
)
y = model([x_static, x_dynamic, x_future])  # (batch, 3, 3)
```

Inputs: `x_static` is `(batch, n_static)`, `x_dynamic` is
`(batch, past_steps, n_dynamic)` and `x_future` is
`(batch, past_steps + horizon, n_future)`.

## Install (development)

```powershell
pip install -e ".[torch,dev]"
```

Add `tf` to the extras for the Keras backend and `bench` for the
benchmark baselines.

## Layout

```
src/xtft/
  backends/     framework switch (torch / tensorflow); array/ backends
  compat/       version shims: numpy, pandas, scipy, sklearn, tf, torch
  nn/           models resolved for the active backend
    torch/      PyTorch implementation (reference)
    tf/         Keras implementation ported from fusionlab-learn
  api/ core/ utils/ metrics/ decorators.py exceptions.py
benchmarks/     configs, runners, baseline adapters, statistics
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
