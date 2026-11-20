"""Collect benchmark JSON files into tables.

Writes ``runs.csv`` (one row per run) and ``summary.csv`` (mean and
standard deviation over seeds per dataset, horizon, model and
evaluation split), and prints the test table of the chosen metric
with each model's average rank.

Example
-------
python benchmarks/aggregate.py results/ --metric crps_q
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            res = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "metrics" not in res:
            continue
        for split, metrics in res["metrics"].items():
            rows.append(
                {
                    "dataset": res["dataset"],
                    "model": res["model"],
                    "horizon": res["horizon"],
                    "seed": res["seed"],
                    "split": split,
                    "params": res.get("model_info", {}).get("params"),
                    "fit_s": res["seconds"]["fit"],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--metric", default="crps_q")
    ap.add_argument("--split", default="test_cqr")
    args = ap.parse_args(argv)
    runs = collect(args.root)
    if runs.empty:
        raise SystemExit(f"No result files under {args.root}.")
    runs.to_csv(args.root / "runs.csv", index=False)
    keys = ["dataset", "horizon", "model", "split"]
    numeric = runs.select_dtypes("number").columns.drop(
        ["horizon", "seed"], errors="ignore"
    )
    summary = runs.groupby(keys)[list(numeric)].agg(["mean", "std"])
    summary["n_seeds"] = runs.groupby(keys)["seed"].nunique()
    summary.to_csv(args.root / "summary.csv")

    test = runs[runs["split"] == args.split]
    table = test.pivot_table(
        index=["dataset", "horizon"],
        columns="model",
        values=args.metric,
        aggfunc="mean",
    )
    ranks = table.rank(axis=1).mean().sort_values()
    pd.set_option("display.width", 160)
    print(f"{args.metric} ({args.split}, mean over seeds)")
    print(table.round(4).to_string())
    print("\naverage rank (1 = best):")
    print(ranks.round(2).to_string())


if __name__ == "__main__":
    main()
