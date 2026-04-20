"""Generate Plummer-sphere phase-space data with optional train/test split.

Usage
-----
    python experiments/gendata_plummer.py \
        --total-n 524288 --test-frac 0.1 --max-dist 10.0 \
        --train-out data/plummer_train.h5 \
        --test-out data/plummer_test.h5

This is a pure-CPU script (no JAX/GPU required).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Allow running from repo root without PYTHONPATH
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from plummer.plummer_gendata import sample_df  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_h5(data: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("eta", data=data, compression="lzf", chunks=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Plummer-sphere phase-space data (train + optional test split)."
    )
    parser.add_argument("--total-n", type=int, required=True, help="Total number of samples to generate.")
    parser.add_argument("--test-frac", type=float, default=0.0, help="Fraction of data for test set (0 = no split).")
    parser.add_argument("--max-dist", type=float, default=10.0, help="Maximum radial distance for sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--train-out", type=str, required=True, help="Output path for training data (.h5).")
    parser.add_argument("--test-out", type=str, default=None, help="Output path for test data (.h5). Required if --test-frac > 0.")
    args = parser.parse_args()

    if args.test_frac > 0 and args.test_out is None:
        parser.error("--test-out is required when --test-frac > 0")

    # Generate
    print(f"Generating {args.total_n} Plummer samples (max_dist={args.max_dist}) ...")
    eta_all = np.asarray(sample_df(args.total_n, max_dist=args.max_dist), dtype=np.float32)
    print(f"  Generated {eta_all.shape[0]} samples, shape={eta_all.shape}")

    # Split
    if args.test_frac > 0:
        n_test = int(round(eta_all.shape[0] * args.test_frac))
        n_test = max(n_test, 1)

        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(eta_all.shape[0])
        eta_test = eta_all[perm[:n_test]]
        eta_train = eta_all[perm[n_test:]]

        train_out = Path(args.train_out)
        test_out = Path(args.test_out)

        _save_h5(eta_train, train_out)
        _save_h5(eta_test, test_out)

        print(f"  Train: {eta_train.shape[0]} samples -> {train_out}")
        print(f"  Test:  {eta_test.shape[0]} samples -> {test_out}")
    else:
        train_out = Path(args.train_out)
        _save_h5(eta_all, train_out)
        print(f"  All: {eta_all.shape[0]} samples -> {train_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
