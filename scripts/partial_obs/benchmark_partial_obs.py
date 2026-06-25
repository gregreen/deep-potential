#!/usr/bin/env python
"""Benchmark partial-observation Deep Potential against Plummer ground truth.

Compares:
    - Φ(r) to analytical Plummer Φ(r) = -1/√(1+r²)
    - ρ(r) to analytical ρ(r) = 3/(4π)(1+r²)^(-5/2)
    - CBE residual distribution
    - Entropy evolution
    - Symmetry axis convergence (cylindrical only)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import jax
import jax.numpy as jnp
import equinox as eqx

# Allow importing from parent scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from partial_obs.dim_spec import DimSpec
from partial_obs.mock_data import load_partial_obs_data
from partial_obs.pobs_model import ObservedDensityFlow
from partial_obs.punk_models import make_punk_model
from partial_obs.symmetry_potential import ProjectedPotential
from partial_obs.joint_training import (
    assemble_full_6d_gradient,
    compute_cbe_residual,
)


def benchmark(
    run_dir: Path,
    fig_dir: Path,
    n_eval: int = 5000,
):
    """Run benchmark comparisons and generate plots.

    Args:
        run_dir: Directory containing models and data.
        fig_dir: Directory for output figures.
        n_eval: Number of evaluation points for CBE residual.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_dir = run_dir / "data"
    joint_dir = run_dir / "models" / "joint"
    config_path = joint_dir / "config.json"

    # Load config
    import json
    with open(config_path, "r") as f:
        config = json.load(f)

    dim_spec = DimSpec.from_string(config["dim_spec"])
    symmetry = config.get("symmetry", "spherical")

    # Load data
    print("Loading data...")
    data = load_partial_obs_data(data_dir)
    r_true = data["r"]
    phi_true = data["phi_true"]
    rho_true = data["rho_true"]
    eta_obs = data["eta_obs"]

    # Load models
    print("Loading models...")
    pobs_model = ObservedDensityFlow.load(run_dir / "models" / "pobs" / "pobs_model.eqx")

    phi_model = ProjectedPotential(
        key=jax.random.key(0), symmetry_type=symmetry,
        width_size=config.get("phi_width", 64),
        depth=config.get("phi_depth", 3),
    )
    phi_model = eqx.tree_deserialise_leaves(
        joint_dir / "phi_final.eqx", like=phi_model
    )

    punk_model = make_punk_model(
        model_type=config.get("punk_type", "gaussian"),
        key=jax.random.key(0),
        unk_dim=dim_spec.unk_dim,
        obs_dim=dim_spec.obs_dim,
        width_size=config.get("punk_width", 64),
        depth=config.get("punk_depth", 3),
        n_layers=config.get("punk_n_layers", 3),
    )
    punk_model = eqx.tree_deserialise_leaves(
        joint_dir / "punk_final.eqx", like=punk_model
    )

    # Load loss history
    with open(joint_dir / "loss_history.json", "r") as f:
        loss_history = json.load(f)

    # =====================================================================
    # Figure 1: Φ(r) and ρ(r) comparison
    # =====================================================================
    print("Computing Φ(r) and ρ(r) profiles...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Evaluate Φ on a subsample of positions
    idx = np.random.default_rng(42).choice(
        len(r_true), min(n_eval, len(r_true)), replace=False
    )
    r_eval = r_true[idx]
    x_eval = data["eta_full"][idx][:, :3]

    # Model Φ
    phi_model_val = np.array(jax.vmap(phi_model.__call__)(jnp.array(x_eval)))
    # Model ρ via -∇²Φ
    _, d2phi = phi_model.calc_phi_derivatives(jnp.array(x_eval))
    rho_model_val = np.array(d2phi) / (4 * np.pi)  # Poisson: ρ = ∇²Φ/(4πG), G=1

    # Φ comparison
    sort_idx = np.argsort(r_eval)
    ax1.plot(r_eval[sort_idx], phi_true[idx][sort_idx], "k-", lw=2, label="True")
    ax1.plot(r_eval[sort_idx], phi_model_val[sort_idx], "r--", lw=1.5, label="Model")
    ax1.set_xlabel("r")
    ax1.set_ylabel(r"$\Phi$")
    ax1.set_title("Gravitational Potential")
    ax1.legend()

    # ρ comparison
    ax2.plot(r_eval[sort_idx], rho_true[idx][sort_idx], "k-", lw=2, label="True")
    ax2.plot(r_eval[sort_idx], rho_model_val[sort_idx], "r--", lw=1.5, label="Model")
    ax2.set_xlabel("r")
    ax2.set_ylabel(r"$\rho$")
    ax2.set_title("Mass Density")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "potential_comparison.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'potential_comparison.pdf'}")

    # =====================================================================
    # Figure 2: CBE residual distribution
    # =====================================================================
    print("Computing CBE residuals...")
    key = jax.random.key(123)
    n_cbe = min(n_eval, len(eta_obs))

    # Pre-compute ∇ ln p_obs
    idx_cbe = np.random.default_rng(7).choice(len(eta_obs), n_cbe, replace=False)
    eta_obs_cbe = jnp.array(eta_obs[idx_cbe])
    ln_pobs_cbe, grad_obs_ln_pobs_cbe = pobs_model.log_prob_and_obs_grad(eta_obs_cbe)

    # Sample η_unk and compute gradients
    key, sample_key = jax.random.split(key)
    eta_unk_cbe = punk_model.sample(sample_key, eta_obs_cbe)
    ln_punk_cbe, grad_unk_ln_punk, grad_obs_ln_punk = punk_model.log_prob_and_grad(
        eta_unk_cbe, eta_obs_cbe
    )

    grad_x, grad_v = assemble_full_6d_gradient(
        dim_spec,
        jnp.array(grad_obs_ln_pobs_cbe),
        grad_unk_ln_punk,
        grad_obs_ln_punk,
    )

    eta_full_cbe = dim_spec.combine_eta(eta_obs_cbe, eta_unk_cbe)
    v_full = eta_full_cbe[..., 3:6]
    x_full = eta_full_cbe[..., :3]

    dphi_dq, _ = phi_model.calc_phi_derivatives(x_full)
    df_dt = np.array(compute_cbe_residual(grad_x, grad_v, v_full, dphi_dq))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df_dt, bins=80, density=True, alpha=0.7, color="steelblue",
            edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\partial \ln f / \partial t$")
    ax.set_ylabel("Density")
    ax.set_title(f"CBE Residual Distribution (|mean|={np.abs(df_dt).mean():.4f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "cbe_residual.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'cbe_residual.pdf'}")

    # =====================================================================
    # Figure 3: Training curves
    # =====================================================================
    print("Plotting training curves...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    epochs = np.arange(len(loss_history["train_loss"]))

    axes[0, 0].plot(epochs, loss_history["train_loss"], label="Train", alpha=0.8)
    axes[0, 0].plot(epochs, loss_history["val_loss"], label="Val", alpha=0.8)
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].set_title("Total Loss")

    axes[0, 1].plot(epochs, loss_history["train_cbe"], label="Train", alpha=0.8)
    axes[0, 1].plot(epochs, loss_history["val_cbe"], label="Val", alpha=0.8)
    axes[0, 1].set_ylabel("CBE Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].set_title("CBE Loss")

    axes[1, 0].plot(epochs, loss_history["train_entropy"], label="Train", alpha=0.8)
    axes[1, 0].plot(epochs, loss_history["val_entropy"], label="Val", alpha=0.8)
    axes[1, 0].set_ylabel("Entropy Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].set_title("Entropy Regularization")

    axes[1, 1].plot(epochs, loss_history["train_df_dt"], label="Train", alpha=0.8)
    axes[1, 1].plot(epochs, loss_history["val_df_dt"], label="Val", alpha=0.8)
    axes[1, 1].set_ylabel(r"$\langle |\partial \ln f / \partial t| \rangle$")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].set_title("Mean |CBE Residual|")

    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'training_curves.pdf'}")

    # =====================================================================
    # Figure 4: Symmetry axis convergence (cylindrical only)
    # =====================================================================
    if symmetry == "cylindrical" and "axis_x" in loss_history:
        print("Plotting symmetry axis convergence...")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, loss_history["axis_x"], label=r"$n_x$", alpha=0.7)
        ax.plot(epochs, loss_history["axis_y"], label=r"$n_y$", alpha=0.7)
        ax.plot(epochs, loss_history["axis_z"], label=r"$n_z$", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Axis Component")
        ax.set_title("Symmetry Axis Direction")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "axis_convergence.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig_dir / 'axis_convergence.pdf'}")

    # =====================================================================
    # Summary statistics
    # =====================================================================
    print("\n--- Benchmark Summary ---")
    phi_mae = np.mean(np.abs(phi_model_val - phi_true[idx]))
    rho_mae = np.mean(np.abs(rho_model_val - rho_true[idx]))
    print(f"  Φ MAE: {phi_mae:.4f}")
    print(f"  ρ MAE: {rho_mae:.4f}")
    print(f"  ⟨|∂ ln f / ∂t|⟩: {np.abs(df_dt).mean():.4f}")
    print(f"  Final train loss: {loss_history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {loss_history['val_loss'][-1]:.4f}")

    return {
        "phi_mae": float(phi_mae),
        "rho_mae": float(rho_mae),
        "cbe_mean_abs": float(np.abs(df_dt).mean()),
    }


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Benchmark partial-obs Deep Potential against Plummer ground truth.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, default="runs/partial_obs",
                        help="Run directory containing models and data.")
    parser.add_argument("--fig-dir", type=str, default=None,
                        help="Output directory for figures.")
    parser.add_argument("--n-eval", type=int, default=5000,
                        help="Number of evaluation points.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.fig_dir is None:
        fig_dir = run_dir / "figures"
    else:
        fig_dir = Path(args.fig_dir)

    results = benchmark(run_dir, fig_dir, n_eval=args.n_eval)


if __name__ == "__main__":
    main()
