# XTFT rebuild plan

Status: 2026-11-20. Origin: TPAMI-2025-04-1034 rejected on
2026-07-30 (four reviewers). Old manuscript and reviews:
`E:\_REPOSITORIES\TPAMI`.

## Decisions taken

| Topic | Decision |
|---|---|
| Framework | PyTorch is the reference implementation and the one benchmarked; the Keras code from fusionlab-learn is kept under `xtft.nn.tf` and selected with `set_backend("tensorflow")` |
| Name | XTFT stays; "X" means *cross-scale*, not "Extreme" |
| Target venue | International Journal of Forecasting (first), Engineering Applications of AI (alternative); software paper (JMLR MLOSS or JOSS) for the library |
| Role of Nansha | Application case study (with Zhongshan), not the main evidence: 8 annual steps (2015-2022) cannot support a multiscale-temporal claim |
| SuperXTFT | Dropped from the paper and the public API (2026-11-20): its extra per-input VSN/GRN is now part of XTFT itself; one model, one contribution. Legacy Keras class kept in `xtft.nn.tf.hybrid` |
| TFT baseline | neuralforecast's independent PyTorch TFT, not our own re-implementation, so the baseline cannot be suspected of being weakened |
| Memory attention, anomaly scoring | Not in the PyTorch XTFT; legacy Keras only |

## Reviewer issues and how each is answered

| Issue | Answer |
|---|---|
| Bag of existing components; "Extreme" unjustified | One thesis (cross-scale fusion with calibrated quantiles); four principled changes below |
| One dataset, only TFT as baseline, no ablation | 8-10 public datasets, about 15 baselines, full ablations |
| q0.1-q0.9 called a 90% interval (it is 80%); coverage only | CRPS, pinball, Winkler/interval score, PICP and PINAW at the correct nominal level, reliability diagrams, MSIS |
| Possible random 70/30 leakage, no test set | Chronological train/val/test, rolling origin, spatial blocks for Nansha |
| Stride downsampling aliasing; stride choice | Learnable low-pass filter before decimation; strides from periodicity; LSTM vs TCN vs patch encoders |
| "Hierarchical" attention is a sum | Local (within scale) then global (across scale) attention |
| Fixed window W | Soft, differentiable learned window |
| Complexity formulas wrong | Corrected asymptotics plus measured params, FLOPs, time, memory |
| Interpretability not shown | Variable-selection weights, per-scale attribution, attention analysis, expert check on Nansha |
| Capacity confound | Parameter-matched TFT and XTFT |

## Method changes (PyTorch implementation)

1. Anti-aliased multi-resolution encoder (done:
   `layers/multiresolution.py`; windowed-sinc low-pass, causal,
   origin-aligned; learnable taps).
2. Cross-scale hierarchical attention with a learned window (done:
   `layers/attention.py`; local causal attention with per-head
   learned window, then cross-scale attention with scale
   attribution).
3. Non-crossing quantile head (done) plus conformalized quantile
   regression (done: `xtft.evaluation.ConformalQuantileCalibrator`,
   per-horizon CQR with monotone rearrangement; model-agnostic, so
   baselines are calibrated identically).
4. Measurable interpretability (done in the forward pass:
   variable weights, scale attribution, learned windows; analysis
   and plots to do).

Flaws of the original XTFT found while porting, worth one
paragraph in the paper: `x[:, ::s]` aliases and, being anchored
at step 0, drops the newest observation unless `s` divides
`T - 1`; the "dynamic time window" is a fixed slice of the last
`W` steps; the Keras XTFT had no variable selection, and future
covariates only entered the cross-attention embedding.

5. Reversible instance normalisation (`instance_norm=True`, done
   2026-11-20; on by default in the benchmark adapter, as the
   neuralforecast baselines use `scaler_type="standard"`). Needed
   on ETTh1, whose test period sits at -1.34 train standard
   deviations with a third of the train spread: without it the
   XTFT test MAE was 0.468, with it 0.186 (smoke budget). It costs
   a little on stationary data (Synthetic CRPS 0.257 -> 0.272), so
   a variant that feeds the window mean and log-scale back as
   static inputs (keeping the level information) is worth an
   ablation.

Still to do on the model: scale suggestion from the periodogram,
alternative encoders (TCN, patch) for the ablation,
`torch.compile` and GPU timing.

## Benchmarks (availability and licences to verify)

* Long-horizon: ETTh1/h2/m1/m2, Weather, Electricity, Traffic,
  Exchange, ILI.
* Covariate-rich and probabilistic: Electricity and Traffic in the
  TFT setting, EPF electricity prices, GEFCom2014, M5 or Favorita
  (Kaggle terms: author downloads).
* Application: Nansha and Zhongshan subsidence (processed with
  `D:\projects\geoprior-v3`; Stage-1 data in
  `E:\nature\nature\results\nansha_GeoPriorSubsNet_stage1`).

Baselines: naive and seasonal naive; ETS, ARIMA, Theta; LightGBM
and XGBoost on lags; DeepAR, N-BEATS, N-HiTS, TFT, Informer,
Autoformer, FEDformer, PatchTST, iTransformer, Crossformer, TimesNet,
DLinear, TiDE, TSMixer; optional zero-shot Chronos and TimesFM.

Protocol (implemented in `benchmarks/run.py` and `xtft.data`):

* chronological train / validation / test periods with each
  dataset's conventional lengths;
* models fitted on train, early stopping on validation;
* forecasts at shared rolling origins, aligned to the series end
  as neuralforecast and statsforecast do; windows straddling a
  period boundary are dropped;
* CQR fitted on validation forecasts, applied to test forecasts;
  both raw and calibrated test metrics are reported;
* metrics on train-standardised units plus MASE and MSIS
  (seasonal scale), computed by `xtft.evaluation` for every model;
* 11 quantile levels (0.05, 0.1, ..., 0.9, 0.95), so the CRPS
  proxy and the 90/80/60/40/20% intervals are all available;
* five seeds for learned models (statistical models are
  deterministic: one run), Diebold-Mariano tests and
  Friedman/Nemenyi critical-difference diagrams (to add).

Harness: `benchmarks/datasets.py` (registry; `Synthetic` for smoke
tests), `benchmarks/models.py` (adapters `xtft`, `nf:<Model>`,
`sf:<Model>`), `benchmarks/make_tasks.py` (task list),
`jobs/bench_array.sbatch` (SLURM array, 8 cores per task, at most
20 concurrent), `benchmarks/aggregate.py` (tables and ranks).
Results go to `/SCRATCH/kouakloMI/xtft/results/<array job id>`.

Data checks (2026-11-20): `datasetsforecast` 1.1.0 gives ETTh1/h2
the 15-minute split sizes (11520) and labels ECL 15-minute; the
loader uses the 12/4/4-month ETTh split (8640/2880/2880) and infers
the frequency from the time stamps (ECL is hourly). ETTh1 and ETTh2
there are the univariate oil-temperature series (1 series); ETTm
have 7.

Smoke run (job 106507, 2026-11-20; 250 steps or 5 x 50 epochs,
H=96 on ETTh1, H=24 on Synthetic): harness end to end for xtft,
nf:TFT/PatchTST/NHITS/DLinear and sf:SeasonalNaive/AutoETS. Points
for the full runs:

* validation-to-test shift on ETTh1 makes CQR unreliable there
  (AutoETS coverage 0.94 -> 0.35 after CQR); with stride H there
  are only 30 calibration windows. Report raw and CQR metrics,
  consider a smaller validation stride, and discuss the
  exchangeability assumption;
* neuralforecast TFT is slow on CPU (833 s vs 42 s for XTFT on
  Synthetic at default sizes; not parameter-matched) - budget its
  array tasks with longer time limits.

Equal tuning budget: same input size, horizon, max epochs or
steps and early-stopping patience for all learned models;
per-model hyperparameters stay at their library defaults unless a
search with the same budget is run for every model.

## Work phases

1. Repository bootstrap and port from fusionlab-learn (done).
2. PyTorch XTFT with the redesign (done, 2026-11-20; tests in
   `tests/torch`); speed against the Keras XTFT in
   `benchmarks/speed/compare_xtft.py`. Remaining items listed
   under "Method changes".
3. Probabilistic metrics, conformal calibration, panel data
   handling and trainer (done, 2026-11-20).
4. Benchmark harness, baseline adapters, SLURM job templates
   (done for long-horizon datasets; covariate-rich datasets and
   statistical tests to add).
5. Runs, ablations, statistics.
6. Nansha/Zhongshan case study.
7. Manuscript and figures (editable figure folders).
8. Public GitHub repository `earthai-tech/xtft`, documentation,
   PyPI release, software paper.
