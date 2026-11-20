"""Write the benchmark task list, one ``run.py`` argument line each.

The list is the Cartesian product of datasets, models, horizons and
seeds; ``jobs/bench_array.sbatch`` runs line ``SLURM_ARRAY_TASK_ID``.

Example
-------
python benchmarks/make_tasks.py --datasets ETTh1 ETTm1 \
    --models xtft nf:TFT nf:PatchTST sf:SeasonalNaive \
    --horizons 96 192 --seeds 0 1 2 --out benchmarks/tasks.txt
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

# Local statistical models give identical forecasts for every seed.
_DETERMINISTIC = ("sf:",)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--input-size", type=int, default=96)
    ap.add_argument(
        "--extra", default="", help="arguments appended to every line"
    )
    ap.add_argument(
        "--out", type=Path, default=Path("benchmarks/tasks.txt")
    )
    args = ap.parse_args(argv)
    lines = []
    for d, m, h in itertools.product(
        args.datasets, args.models, args.horizons
    ):
        seeds = (
            args.seeds[:1]
            if m.startswith(_DETERMINISTIC)
            else (args.seeds)
        )
        for s in seeds:
            lines.append(
                f"--dataset {d} --model {m} --horizon {h} "
                f"--input-size {args.input_size} --seed {s} "
                f"{args.extra}".strip()
            )
    args.out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
