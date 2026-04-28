from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class EvalStats:
    residual_mean: float
    residual_std: float
    residual_p99_abs: float
    residual_p999_abs: float
    residual_max_abs: float


@dataclass
class RadialStats:
    ar_mae: float
    ar_p95: float
    ar_max: float
    phi_mae: float
    phi_max: float
    phi_region: dict[str, float]


def _resolve_run_dir(run: str) -> Path:
    p = Path(run)
    if p.exists():
        return p
    p2 = Path("runs/plummer") / run
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Run directory not found: {run}")


def _load_eval_stats(run_dir: Path) -> EvalStats:
    p = run_dir / "eval" / "eval_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing eval stats: {p}")
    s = json.loads(p.read_text())
    return EvalStats(
        residual_mean=float(s["residual_mean"]),
        residual_std=float(s["residual_std"]),
        residual_p99_abs=float(s["residual_p99_abs"]),
        residual_p999_abs=float(s["residual_p999_abs"]),
        residual_max_abs=float(s["residual_max_abs"]),
    )


def _load_radial_stats(run_dir: Path) -> RadialStats:
    p = run_dir / "eval" / "radial_curves_plummer.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing radial curves: {p}")
    d = np.load(p)
    r = d["r"]
    ar_err = np.abs(d["ar_learned"] - d["ar_true"])
    phi_err = np.abs(d["phi_learned_shift"] - d["phi_true"])

    regions = {
        "r[0.001,0.1)": float(phi_err[(r >= 0.001) & (r < 0.1)].mean()),
        "r[0.1,1)": float(phi_err[(r >= 0.1) & (r < 1.0)].mean()),
        "r[1,3)": float(phi_err[(r >= 1.0) & (r < 3.0)].mean()),
        "r[3,10]": float(phi_err[(r >= 3.0) & (r <= 10.0)].mean()),
    }

    return RadialStats(
        ar_mae=float(ar_err.mean()),
        ar_p95=float(np.percentile(ar_err, 95)),
        ar_max=float(ar_err.max()),
        phi_mae=float(phi_err.mean()),
        phi_max=float(phi_err.max()),
        phi_region=regions,
    )


def _print_eval_table(run_dirs: List[Path], names: List[str], evals: List[EvalStats]) -> None:
    print("## Eval Stats")
    print("| run | residual_mean | residual_std | residual_p99_abs | residual_p999_abs | residual_max_abs |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, e in zip(names, evals):
        print(
            f"| {name} | {e.residual_mean:+.6f} | {e.residual_std:.6f} | "
            f"{e.residual_p99_abs:.6f} | {e.residual_p999_abs:.6f} | {e.residual_max_abs:.6f} |"
        )


def _print_radial_table(names: List[str], rstats: List[RadialStats]) -> None:
    print()
    print("## Radial Errors")
    print("| run | ar_mae | ar_p95 | ar_max | phi_mae | phi_max |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, rs in zip(names, rstats):
        print(
            f"| {name} | {rs.ar_mae:.6e} | {rs.ar_p95:.6e} | {rs.ar_max:.6e} | "
            f"{rs.phi_mae:.6e} | {rs.phi_max:.6e} |"
        )

    print()
    print("## Per-region phi_mae")
    print("| run | r[0.001,0.1) | r[0.1,1) | r[1,3) | r[3,10] |")
    print("|---|---:|---:|---:|---:|")
    for name, rs in zip(names, rstats):
        print(
            f"| {name} | {rs.phi_region['r[0.001,0.1)']:.6f} | {rs.phi_region['r[0.1,1)']:.6f} | "
            f"{rs.phi_region['r[1,3)']:.6f} | {rs.phi_region['r[3,10]']:.6f} |"
        )


def _print_delta(names: List[str], evals: List[EvalStats], rstats: List[RadialStats], base_idx: int) -> None:
    base_name = names[base_idx]
    be = evals[base_idx]
    br = rstats[base_idx]

    print()
    print(f"## Delta vs {base_name}")
    print("| run | d_std | d_p99 | d_p999 | d_max | d_ar_mae | d_phi_mae | d_phi_tail(r[3,10]) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, name in enumerate(names):
        if i == base_idx:
            continue
        e = evals[i]
        r = rstats[i]
        print(
            f"| {name} | {e.residual_std - be.residual_std:+.6f} | "
            f"{e.residual_p99_abs - be.residual_p99_abs:+.6f} | "
            f"{e.residual_p999_abs - be.residual_p999_abs:+.6f} | "
            f"{e.residual_max_abs - be.residual_max_abs:+.6f} | "
            f"{r.ar_mae - br.ar_mae:+.6e} | {r.phi_mae - br.phi_mae:+.6e} | "
            f"{r.phi_region['r[3,10]'] - br.phi_region['r[3,10]']:+.6f} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PHI eval results across multiple run directories.")
    parser.add_argument("runs", nargs="+", help="Run directory paths or short names under runs/plummer.")
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="Base run for delta table. Default: first run in the list.",
    )
    args = parser.parse_args()

    run_dirs = [_resolve_run_dir(r) for r in args.runs]
    names = [p.name for p in run_dirs]

    evals = [_load_eval_stats(p) for p in run_dirs]
    rstats = [_load_radial_stats(p) for p in run_dirs]

    base_idx = 0
    if args.base is not None:
        base_resolved = _resolve_run_dir(args.base)
        for i, p in enumerate(run_dirs):
            if p.resolve() == base_resolved.resolve():
                base_idx = i
                break
        else:
            raise ValueError(f"--base run not found in input list: {args.base}")

    print("# PHI Run Comparison")
    print()
    _print_eval_table(run_dirs, names, evals)
    _print_radial_table(names, rstats)
    _print_delta(names, evals, rstats, base_idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
