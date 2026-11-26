"""Write the task list of the CAMELS-US flood forecasting study.

Seven-day streamflow forecasts from one year of history for the
CAMELS-US basins: the main setting without future weather
(``CAMELS-US``) and the oracle-weather scenario
(``CAMELS-US-Weather``). One ``run.py`` argument line per task, for
``jobs/bench_array.sbatch``.

Run: python jobs/camels_tasks.py --out benchmarks/tasks_camels.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

COMMON = (
    "--horizon 7 --input-size 365 --max-epochs 20 "
    "--steps-per-epoch 100 --max-steps 2000 --patience 5 "
    "--batch-size 128"
)
SEEDS = (0, 1, 2)
# bench_array.sbatch splits each line on whitespace without shell
# quoting, so JSON arguments must contain no spaces and no quotes
# around them.
TFT64 = '--model-kw {"hidden_size":64} --tag w64'
# (dataset, model, extra arguments, seeds)
TASKS = [
    ("CAMELS-US", "xtft", "", SEEDS),
    ("CAMELS-US", "nf:TFT", TFT64, SEEDS),
    ("CAMELS-US", "nf:LSTM", "", SEEDS),
    ("CAMELS-US", "lgb", "", SEEDS),
    ("CAMELS-US", "xtft", "--no-covariates --tag nocov", SEEDS),
    ("CAMELS-US", "sf:SeasonalNaive", "", (0,)),
    ("CAMELS-US", "sf:Naive", "", (0,)),
    ("CAMELS-US-Weather", "xtft", "", SEEDS),
    ("CAMELS-US-Weather", "nf:TFT", TFT64, SEEDS),
    ("CAMELS-US-Weather", "nf:LSTM", "", SEEDS),
    ("CAMELS-US-Weather", "lgb", "", SEEDS),
]


def main(argv=None) -> None:
    """Write one argument line per (dataset, model, seed)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--out", type=Path, default=Path("benchmarks/tasks_camels.txt")
    )
    args = ap.parse_args(argv)
    lines = [
        f"--dataset {d} --model {m} --seed {s} {COMMON} {x}".strip()
        for d, m, x, seeds in TASKS
        for s in seeds
    ]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
