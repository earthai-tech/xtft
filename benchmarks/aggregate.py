"""Collect benchmark JSON files into tables.

Writes ``runs.csv`` (one row per run) and ``summary.csv`` (mean and
standard deviation over seeds per dataset, horizon, model and
evaluation split), and prints the test table of the chosen metric
with each model's average rank, the Friedman test and the Nemenyi
critical difference over the (dataset, horizon) tasks. With
``--reference``, it also writes ``dm.csv``: Diebold-Mariano tests
of the reference model against every other model on each task,
from the per-window losses that ``run.py`` saves.

Example
-------
python benchmarks/aggregate.py results/ --metric crps_q \
    --reference xtft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xtft.evaluation import diebold_mariano, friedman_nemenyi


def _label(res: dict) -> str:
    """Model name with its variant tag and gap setting, if any."""
    label = res["model"]
    if res.get("tag"):
        label += f"/{res['tag']}"
    miss = res.get("missing") or {}
    if miss.get("rate"):
        label += f"[{miss['mode']} {miss['rate']:g}]"
    return label


def _runs(roots: list[Path], need_losses: bool = False) -> dict:
    """Newest result file per run, keyed by run identity.

    A run is (dataset, horizon, label, seed). When a run was repeated
    (e.g. a rerun in a later array job), the most recently written
    file wins, so seeds are never counted twice. With `need_losses`,
    only runs with a ``_losses.npz`` file are considered.
    """
    found = {}
    for path in (p for r in roots for p in r.rglob("*.json")):
        if (
            need_losses
            and not path.with_name(
                path.name[: -len(".json")] + "_losses.npz"
            ).exists()
        ):
            continue
        try:
            res = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "metrics" not in res:
            continue
        key = (
            res["dataset"],
            res["horizon"],
            _label(res),
            res["seed"],
        )
        mtime = path.stat().st_mtime
        if key not in found or mtime > found[key][0]:
            found[key] = (mtime, path, res)
    return {k: (p, r) for k, (_, p, r) in found.items()}


def collect(roots: list[Path]) -> pd.DataFrame:
    rows = []
    for (_, _, label, _), (_, res) in sorted(_runs(roots).items()):
        for split, metrics in res["metrics"].items():
            rows.append(
                {
                    "dataset": res["dataset"],
                    "model": label,
                    "horizon": res["horizon"],
                    "seed": res["seed"],
                    "split": split,
                    "params": res.get("model_info", {}).get("params"),
                    "fit_s": res["seconds"]["fit"],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _origin_losses(path: Path, key: str) -> pd.Series:
    """Per-origin loss (mean over series) from a losses file."""
    data = np.load(path)
    frame = pd.DataFrame(
        {"origin": data["test_index"][:, 1], "loss": data[key]}
    )
    return frame.groupby("origin")["loss"].mean()


def dm_table(
    roots: list[Path], reference: str, key: str
) -> pd.DataFrame:
    """DM tests of `reference` against every model, per task.

    Losses are averaged over seeds, then over the series sharing an
    origin; origins are spaced by the run's stride, so the overlap
    passed to the test is ceil(horizon / stride).
    """
    rows = []
    files = {}
    runs = _runs(roots, need_losses=True)
    for (dataset, horizon, label, _), (path, res) in runs.items():
        # Not with_suffix: tags such as "_missb0.3" contain dots.
        loss_path = path.with_name(
            path.name[: -len(".json")] + "_losses.npz"
        )
        files.setdefault((dataset, horizon, label), []).append(
            (loss_path, res)
        )
    tasks = {(d, h) for d, h, _ in files}
    for d, h in sorted(tasks):
        if (d, h, reference) not in files:
            continue

        def mean_loss(model, d=d, h=h):
            parts = [
                _origin_losses(p, key) for p, _ in files[(d, h, model)]
            ]
            return pd.concat(parts, axis=1).mean(axis=1)

        ref = mean_loss(reference)
        stride = files[(d, h, reference)][0][1].get("stride", h)
        overlap = max(1, -(-h // stride))
        for model in sorted(
            {m for dd, hh, m in files if (dd, hh) == (d, h)}
        ):
            if model == reference:
                continue
            other = mean_loss(model)
            both = pd.concat([ref, other], axis=1, join="inner")
            if len(both) <= overlap + 2:
                continue
            res = diebold_mariano(
                both.iloc[:, 0], both.iloc[:, 1], horizon=overlap
            )
            rows.append(
                {
                    "dataset": d,
                    "horizon": h,
                    "reference": reference,
                    "model": model,
                    "loss": key,
                    "n_origins": res.n,
                    "mean_diff": res.mean_difference,
                    "dm": res.statistic,
                    "pvalue": res.pvalue,
                    "n_seeds_ref": len(files[(d, h, reference)]),
                    "n_seeds_model": len(files[(d, h, model)]),
                }
            )
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "roots",
        type=Path,
        nargs="+",
        help="result folders (e.g. one per array job)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="folder for the CSV files (default: first root)",
    )
    ap.add_argument("--metric", default="crps_q")
    ap.add_argument("--split", default="test_cqr")
    ap.add_argument(
        "--reference",
        default=None,
        help="model tested against all others (Diebold-Mariano)",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    out = args.out or args.roots[0]
    out.mkdir(parents=True, exist_ok=True)
    runs = collect(args.roots)
    if runs.empty:
        raise SystemExit(f"No result files under {args.roots}.")
    runs.to_csv(out / "runs.csv", index=False)
    keys = ["dataset", "horizon", "model", "split"]
    numeric = runs.select_dtypes("number").columns.drop(
        ["horizon", "seed"], errors="ignore"
    )
    summary = runs.groupby(keys)[list(numeric)].agg(["mean", "std"])
    summary[("n_seeds", "")] = runs.groupby(keys)["seed"].nunique()
    summary.to_csv(out / "summary.csv")

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

    complete = table.dropna(axis=0)
    if complete.shape[0] >= 2 and complete.shape[1] >= 2:
        fn = friedman_nemenyi(complete.to_numpy(), alpha=args.alpha)
        print(
            f"\nFriedman over {fn.n_tasks} complete tasks: "
            f"chi2 = {fn.statistic:.2f}, p = {fn.pvalue:.3g}; "
            f"Nemenyi CD ({args.alpha:g}) = "
            f"{fn.critical_difference:.2f}"
        )
        print(
            pd.Series(fn.average_ranks, index=complete.columns)
            .sort_values()
            .round(2)
            .to_string()
        )

    if args.reference:
        key = {"test_cqr": "crps_cqr", "test_aci": "crps_aci"}.get(
            args.split, "crps"
        )
        dm = dm_table(args.roots, args.reference, key)
        if dm.empty:
            print("\nNo per-window losses for the DM tests.")
            return
        dm.to_csv(out / "dm.csv", index=False)
        sig = dm["pvalue"] < args.alpha
        dm["win"] = sig & (dm["mean_diff"] < 0)
        dm["loss"] = sig & (dm["mean_diff"] > 0)
        summary = dm.groupby("model")[["win", "loss"]].sum()
        summary["tasks"] = dm.groupby("model").size()
        print(
            f"\nDiebold-Mariano, {args.reference} against each model "
            f"({key}, p < {args.alpha:g}): tasks where "
            f"{args.reference} is significantly better (win) or "
            "worse (loss)"
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
