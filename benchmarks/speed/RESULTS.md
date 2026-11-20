# Speed: PyTorch XTFT vs Keras (fusionlab-learn) XTFT

Job 106505, 2026-11-20, node wombele180 (partition
`B720-ibe-compute_28_64`, 28 cores, exclusive), env
`xtft` (torch 2.5.1+cpu, TensorFlow 2.20).
Code: working tree on top of commit `c39393d` (the PyTorch XTFT
before its first commit). Raw data: `results/speed_t*_106505.json`.

Setup: batch 64, horizon 24, 5 static, 8 past, 4 known variables,
width 64, 4 heads, scales (1, 2, 4), quantiles (0.1, 0.5, 0.9).
Median of 10 runs after 3 warm-up runs. "Train" is one
optimisation step (forward, backward, Adam update; fused Adam for
PyTorch, `train_on_batch` for Keras). "Infer" is one forward pass
(`torch.inference_mode` for PyTorch; eager call for Keras).

| Threads | T | Params torch / keras | Train torch / keras (ms) | Speed-up | Infer torch / keras (ms) | Speed-up |
|---|---|---|---|---|---|---|
| 8 | 16 | 672,968 / 760,552 | 66.6 / 297.8 | 4.5x | 16.6 / 272.6 | 16.4x |
| 8 | 64 | | 118.6 / 626.5 | 5.3x | 28.6 / 651.5 | 22.8x |
| 8 | 168 | | 261.4 / 1418.3 | 5.4x | 58.7 / 1380.2 | 23.5x |
| 28 | 16 | | 81.6 / 304.9 | 3.7x | 17.3 / 267.7 | 15.5x |
| 28 | 64 | | 118.4 / 659.5 | 5.6x | 28.9 / 617.6 | 21.4x |
| 28 | 168 | | 226.3 / 1424.6 | 6.3x | 52.4 / 1376.1 | 26.3x |

## Reading these numbers

* The two models are different architectures with the same input
  contract: this compares the redesigned XTFT with the original
  one (memory attention, dynamic window, unfiltered strides,
  independent quantile heads), not two implementations of one
  network. Both are configured at the same width.
* PyTorch time grows sub-linearly with `T` (x3.9 from T=16 to
  T=168 at 8 threads); Keras grows x4.8.
* 28 threads bring little over 8 at this batch size: the model is
  latency-bound, so many small jobs (seeds, datasets) per node
  will use the cluster better than one large job.
* The Keras eager inference time is close to its compiled training
  step, so part of its inference cost is Python overhead; a
  `tf.function`-wrapped predict would be faster. Report this
  caveat if the inference ratio is quoted.
* A local run (Windows laptop, 4 threads, batch 16, T=16) gave
  58 ms vs 45 ms per training step in favour of Keras: at very
  small batches the compiled Keras graph wins on per-op overhead.
  `torch.compile` is the next thing to try there.
