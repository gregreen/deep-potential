from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class MetricRow:
    step: int
    epoch: int
    loss: float
    residual_mean: float
    residual_std: float
    residual_p99_abs: float


def _resolve_run_dir(run: str) -> Path:
    p = Path(run)
    if p.exists():
        return p
    p2 = Path("runs/plummer") / run
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Run directory not found: {run}")


def _load_metrics(metrics_path: Path) -> List[MetricRow]:
    rows: List[MetricRow] = []
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(float(row["step"]))
                epoch = int(float(row["epoch"]))
                loss = float(row["loss"])
                residual_mean = float(row["residual_mean"])
                v1 = float(row["residual_std"])
                v2 = float(row["residual_p99_abs"])
            except Exception:
                continue
            if not (0.0 < loss < 1.0e6):
                continue
            # Some logs swapped these two columns; normalize here.
            residual_std = min(v1, v2)
            residual_p99_abs = max(v1, v2)
            rows.append(
                MetricRow(
                    step=step,
                    epoch=epoch,
                    loss=loss,
                    residual_mean=residual_mean,
                    residual_std=residual_std,
                    residual_p99_abs=residual_p99_abs,
                )
            )
    return rows


def _format_row(title: str, r: MetricRow) -> str:
    return (
        f"| {title} | {r.step} | {r.epoch} | {r.loss:.6f} | "
        f"{r.residual_mean:+.6f} | {r.residual_std:.6f} | {r.residual_p99_abs:.6f} |"
    )


def _print_metrics_summary(rows: List[MetricRow]) -> None:
    first = rows[0]
    best = min(rows, key=lambda x: x.loss)
    last = rows[-1]

    print("## Training Metrics")
    print("| point | step | epoch | loss | residual_mean | residual_std | residual_p99_abs |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print(_format_row("first", first))
    print(_format_row("best", best))
    print(_format_row("last", last))

    max_epoch = last.epoch
    split_epoch = max_epoch // 2
    p1 = [r for r in rows if r.epoch <= split_epoch]
    p2 = [r for r in rows if r.epoch > split_epoch]
    if p1 and p2:
        b1 = min(p1, key=lambda x: x.loss)
        b2 = min(p2, key=lambda x: x.loss)
        print()
        print("## Phase Best")
        print("| phase | step | epoch | loss | residual_std | residual_p99_abs |")
        print("|---|---:|---:|---:|---:|---:|")
        print(f"| [0, {split_epoch}] | {b1.step} | {b1.epoch} | {b1.loss:.6f} | {b1.residual_std:.6f} | {b1.residual_p99_abs:.6f} |")
        print(f"| [{split_epoch + 1}, {max_epoch}] | {b2.step} | {b2.epoch} | {b2.loss:.6f} | {b2.residual_std:.6f} | {b2.residual_p99_abs:.6f} |")


def _print_eval_summary(run_dir: Path) -> None:
    eval_stats = run_dir / "eval" / "eval_stats.json"
    if not eval_stats.exists():
        print()
        print("## Eval Stats")
        print("No eval file found at `eval/eval_stats.json`.")
        return

    stats = json.loads(eval_stats.read_text())
    print()
    print("## Eval Stats")
    print("| residual_mean | residual_std | residual_p99_abs | residual_p999_abs | residual_max_abs |")
    print("|---:|---:|---:|---:|---:|")
    print(
        "| "
        f"{stats['residual_mean']:+.6f} | {stats['residual_std']:.6f} | "
        f"{stats['residual_p99_abs']:.6f} | {stats['residual_p999_abs']:.6f} | {stats['residual_max_abs']:.6f} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a PHI run's training and eval metrics.")
    parser.add_argument("run_dir", type=str, help="Run directory path or short name under runs/plummer.")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    rows = _load_metrics(metrics_path)
    if not rows:
        raise RuntimeError(f"No valid metric rows found in: {metrics_path}")

    print(f"# PHI Run Summary: `{run_dir}`")
    print(f"- rows: {len(rows)}")
    print(f"- max_step: {rows[-1].step}")
    print(f"- max_epoch: {rows[-1].epoch}")
    print()

    _print_metrics_summary(rows)
    _print_eval_summary(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
