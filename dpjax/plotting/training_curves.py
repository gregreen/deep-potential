from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np


def read_metrics(run_dir: str | Path) -> dict[str, np.ndarray]:
    """Read ``metrics.csv`` from a training run directory.

    Returns a dict mapping column names to 1-D numpy arrays.
    """
    path = Path(run_dir) / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"No metrics.csv in {run_dir}")
    with path.open() as f:
        reader = csv.DictReader(f)
        header_keys = reader.fieldnames
        # Skip duplicate header rows (can happen when training is resumed/re-run)
        rows = [r for r in reader if r.get(header_keys[0]) != header_keys[0]]
    if not rows:
        raise ValueError(f"metrics.csv in {run_dir} is empty")
    return {k: np.array([float(r[k]) for r in rows]) for k in header_keys}


def plot_df_training(
    run_dir: str | Path,
    save_dir: Optional[str | Path] = None,
    *,
    dpi: int = 100,
):
    """Plot DF training curves (NLL loss + score p99) from ``metrics.csv``.

    Parameters
    ----------
    run_dir : path to the DF training run directory
    save_dir : if given, save the figure there instead of showing
    """
    import matplotlib.pyplot as plt

    m = read_metrics(run_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(m["step"], m["loss"], lw=1.2, label="train")
    if "val_loss" in m:
        ax1.plot(m["step"], m["val_loss"], lw=1.2, ls="--", label="val")
    ax1.set_xlabel("step")
    ax1.set_ylabel("NLL")
    ax1.set_title("DF Loss")
    ax1.grid(True, alpha=0.2)
    if "val_loss" in m:
        ax1.legend()

    ax2.plot(m["step"], m["score_p99"], lw=1.2, color="tab:orange", label="train")
    if "val_score_p99" in m:
        ax2.plot(m["step"], m["val_score_p99"], lw=1.2, ls="--", color="tab:red", label="val")
    ax2.set_xlabel("step")
    ax2.set_ylabel("|score| p99")
    ax2.set_title("Score p99")
    ax2.grid(True, alpha=0.2)
    if "val_score_p99" in m:
        ax2.legend()

    fig.suptitle("DF Training", fontsize=13)
    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "df_training_curves.png", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_phi_training(
    run_dir: str | Path,
    save_dir: Optional[str | Path] = None,
    *,
    dpi: int = 100,
):
    """Plot Phi training curves (loss + residual std/p99) from ``metrics.csv``.

    Parameters
    ----------
    run_dir : path to the Phi training run directory
    save_dir : if given, save the figure there instead of showing
    """
    import matplotlib.pyplot as plt

    m = read_metrics(run_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(m["step"], m["loss"], lw=1.2)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Phi Loss")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(m["step"], m["residual_std"], lw=1.2, color="tab:orange")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("std")
    axes[1].set_title("Residual Std")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(m["step"], m["residual_p99_abs"], lw=1.2, color="tab:red")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("|r| p99")
    axes[2].set_title("Residual p99")
    axes[2].grid(True, alpha=0.2)

    fig.suptitle("Phi Training", fontsize=13)
    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "phi_training_curves.png", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

    return fig
