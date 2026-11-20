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
   regression (to do).
4. Measurable interpretability (done in the forward pass:
   variable weights, scale attribution, learned windows; analysis
   and plots to do).

Flaws of the original XTFT found while porting, worth one
paragraph in the paper: `x[:, ::s]` aliases and, being anchored
at step 0, drops the newest observation unless `s` divides
`T - 1`; the "dynamic time window" is a fixed slice of the last
`W` steps; the Keras XTFT had no variable selection, and future
covariates only entered the cross-attention embedding.

Still to do on the model: scale suggestion from the periodogram,
alternative encoders (TCN, patch) for the ablation, conformal
calibration, `torch.compile` and GPU timing.

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

Protocol: five seeds, equal tuning budget, Diebold-Mariano tests,
Friedman/Nemenyi critical-difference diagrams. Heavy runs on the
HPC CPU nodes (env `sci` or a project env under
`/SCRATCH/kouakloMI/envs/xtft`), never on the login node.

## Work phases

1. Repository bootstrap and port from fusionlab-learn (done).
2. PyTorch XTFT with the redesign (done, 2026-11-20; tests in
   `tests/torch`); speed against the Keras XTFT in
   `benchmarks/speed/compare_xtft.py`. Remaining items listed
   under "Method changes".
3. Probabilistic metrics module and data loaders with checksums.
4. Benchmark harness, baseline adapters, SLURM job templates.
5. Runs, ablations, statistics.
6. Nansha/Zhongshan case study.
7. Manuscript and figures (editable figure folders).
8. Public GitHub repository `earthai-tech/xtft`, documentation,
   PyPI release, software paper.
