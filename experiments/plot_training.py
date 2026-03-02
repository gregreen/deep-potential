from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}

        fieldnames = list(reader.fieldnames)
        cols: dict[str, list[float]] = {k: [] for k in fieldnames}

        def _maybe_float(x: str | None) -> float:
            if x is None:
                return float("nan")
            x = x.strip()
            if x == "":
                return float("nan")
            try:
                return float(x)
            except ValueError:
                return float("nan")

        for row in reader:
            step = _maybe_float(row.get("step"))
            # Skip repeated header lines or malformed rows.
            if not np.isfinite(step):
                continue

            for k in fieldnames:
                cols[k].append(_maybe_float(row.get(k)))

    return {k: np.asarray(v, dtype=np.float32) for k, v in cols.items() if len(v) > 0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.csv.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _read_csv(metrics_path)

    import matplotlib.pyplot as plt

    step = data.get("step")
    if step is None:
        raise ValueError("metrics.csv missing 'step' column")

    # Loss curve
    if "loss" in data:
        plt.figure(dpi=args.dpi)
        plt.plot(step, data["loss"], lw=1.5)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "loss.png")
        plt.close()

    # DF score stats
    if "score_p50" in data or "score_p99" in data or "score_max_abs" in data:
        plt.figure(dpi=args.dpi)
        if "score_p50" in data:
            plt.plot(step, data["score_p50"], label="score| p50", lw=1.2)
        if "score_p99" in data:
            plt.plot(step, data["score_p99"], label="score| p99", lw=1.2)
        if "score_max_abs" in data:
            plt.plot(step, data["score_max_abs"], label="score| max", lw=1.2)
        plt.xlabel("step")
        plt.ylabel("|score| stats")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "df_score_stats.png")
        plt.close()

    # Phi residual stats
    if "residual_mean" in data or "residual_std" in data or "residual_p99_abs" in data:
        plt.figure(dpi=args.dpi)
        if "residual_mean" in data:
            plt.plot(step, data["residual_mean"], label="mean", lw=1.2)
        if "residual_std" in data:
            plt.plot(step, data["residual_std"], label="std", lw=1.2)
        if "residual_p99_abs" in data:
            plt.plot(step, data["residual_p99_abs"], label="p99(|r|)", lw=1.2)
        plt.xlabel("step")
        plt.ylabel("residual stats")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "phi_residual_stats.png")
        plt.close()

    print(f"Wrote training plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
