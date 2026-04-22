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
    parser.add_argument("--test-frac", type=float, default=0.0, help="Fraction of data for test set (0 = no split). Ignored if --test-n is set.")
    parser.add_argument("--test-n", type=int, default=None, help="Exact number of test samples (overrides --test-frac).")
    parser.add_argument("--max-dist", type=float, default=10.0, help="Maximum radial distance for sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--train-out", type=str, required=True, help="Output path for training data (.h5).")
    parser.add_argument("--test-out", type=str, default=None, help="Output path for test data (.h5). Required if --test-frac > 0.")
    args = parser.parse_args()

    has_split = args.test_n is not None or args.test_frac > 0
    if has_split and args.test_out is None:
        parser.error("--test-out is required when --test-frac > 0 or --test-n is set")

    # Seed the global RNG so that sample_df (which uses np.random internally) is
    # reproducible.  A separate RNG is used later for the train/test shuffle.
    np.random.seed(args.seed)

    # Generate – oversample in a loop to guarantee exactly total_n points after
    # the max_dist filter inside sample_df.
    target = args.total_n
    chunks: list[np.ndarray] = []
    n_collected = 0
    print(f"Generating {target} Plummer samples (max_dist={args.max_dist}, seed={args.seed}) ...")
    while n_collected < target:
        n_request = int(1.2 * (target - n_collected)) + 1024
        chunk = np.asarray(sample_df(n_request, max_dist=args.max_dist), dtype=np.float32)
        chunks.append(chunk)
        n_collected += chunk.shape[0]
        print(f"  ... drew {chunk.shape[0]} samples (total so far: {n_collected})")
    eta_all = np.concatenate(chunks, axis=0)[:target]
    assert eta_all.shape[0] == target, f"Expected {target}, got {eta_all.shape[0]}"
    print(f"  Final dataset: {eta_all.shape[0]} samples, shape={eta_all.shape}")

    # Split
    if has_split:
        if args.test_n is not None:
            n_test = args.test_n
        else:
            n_test = int(round(eta_all.shape[0] * args.test_frac))
        n_test = max(min(n_test, eta_all.shape[0] - 1), 1)

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
