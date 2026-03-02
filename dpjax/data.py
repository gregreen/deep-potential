from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np


def load_eta_h5(path: str | Path, dataset: str = "eta") -> np.ndarray:
    """Load `eta` from an HDF5 file.

    Expected shape: (N, 6)
    Expected order: [x, y, z, vx, vy, vz]
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset {dataset!r} not found in {str(path)!r}.")
        eta = np.asarray(f[dataset])

    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError(f"Expected eta shape (N, 6), got {eta.shape}.")

    return eta.astype(np.float32, copy=False)


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray  # (6,)
    std: np.ndarray  # (6,)

    def transform(self, eta: np.ndarray) -> np.ndarray:
        eta = eta.astype(np.float32, copy=False)
        return (eta - self.mean) / self.std

    def inverse(self, eta_std: np.ndarray) -> np.ndarray:
        eta_std = eta_std.astype(np.float32, copy=False)
        return eta_std * self.std + self.mean

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load_npz(path: str | Path) -> "Normalizer":
        path = Path(path)
        with np.load(path) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
            std = np.asarray(data["std"], dtype=np.float32)
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError(f"Invalid normalizer shapes: mean={mean.shape}, std={std.shape}")
        return Normalizer(mean=mean, std=std)


def fit_normalizer(eta: np.ndarray, eps: float = 1.0e-6) -> Normalizer:
    eta = eta.astype(np.float32, copy=False)
    mean = np.mean(eta, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(eta, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return Normalizer(mean=mean, std=std)


def iter_batches(
    eta: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    *,
    shuffle: bool = True,
    drop_remainder: bool = True,
    max_batches: Optional[int] = None,
) -> Iterator[np.ndarray]:
    n = eta.shape[0]
    if shuffle:
        idx = rng.permutation(n)
        eta = eta[idx]

    n_full = n // batch_size
    n_batches = n_full if drop_remainder else int(np.ceil(n / batch_size))
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    for i in range(n_batches):
        lo = i * batch_size
        hi = lo + batch_size
        if hi > n:
            if drop_remainder:
                break
            hi = n
        yield eta[lo:hi]
