#!/usr/bin/env python
"""Benchmark partial-observation Deep Potential against Plummer ground truth.

Generates:
    Fig 1: Φ(r) and ρ(r) comparison (binned, with Φ zero-point correction)
    Fig 2: ρ(r) binned radial profile with error bars
    Fig 3: Φ(r) residual (model - true, after offset removal)
    Fig 4: p_unk conditional validation (sampled vs true unobserved dims)
    Fig 5: CBE residual distribution
    Fig 6: Training curves
    Fig 7: Symmetry axis convergence (cylindrical only)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx

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


# =========================================================================
# Helpers
# =========================================================================

def _phi_zero_point_offset(phi_model_val, phi_true, r):
    """Compute the offset such that Φ_model - offset matches Φ_true.

    Uses the median difference, which is robust to outliers.
    The potential is only defined up to an additive constant.
    """
    return float(np.median(phi_model_val - phi_true))


def _binned_stats(r, values, n_bins=25):
    """Bin values by radius, returning bin centers, means, and stds."""
    r = np.asarray(r)
    values = np.asarray(values)
    bins = np.geomspace(max(r.min(), 1e-3), r.max() * 1.01, n_bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.sum() > 5:
            means[i] = np.mean(values[mask])
            stds[i] = np.std(values[mask])
    return centers, means, stds


# =========================================================================
# Main benchmark function
# =========================================================================

def benchmark(run_dir: Path, fig_dir: Path, n_eval: int = 5000):
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_dir = run_dir / "data"
    joint_dir = run_dir / "models" / "joint"

    # Load config
    import json
    with open(joint_dir / "config.json", "r") as f:
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
    eta_full = data["eta_full"]

    # Load models
    print("Loading models...")
    pobs_model = ObservedDensityFlow.load(
        run_dir / "models" / "pobs" / "pobs_model.eqx")

    phi_model = ProjectedPotential(
        key=jax.random.key(0), symmetry_type=symmetry,
        width_size=config.get("phi_width", 64),
        depth=config.get("phi_depth", 3),
    )
    phi_model = eqx.tree_deserialise_leaves(
        joint_dir / "phi_final.eqx", like=phi_model)

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
        joint_dir / "punk_final.eqx", like=punk_model)

    # Load loss history
    with open(joint_dir / "loss_history.json", "r") as f:
        loss_history = json.load(f)

    # Evaluate Φ and ρ on evaluation subset
    print("Evaluating Φ and ρ...")
    idx = np.random.default_rng(42).choice(
        len(r_true), min(n_eval, len(r_true)), replace=False)
    r_eval = r_true[idx]
    x_eval = eta_full[idx][:, :3]

    phi_model_val = np.array(jax.vmap(phi_model.__call__)(jnp.array(x_eval)))
    _, d2phi = phi_model.calc_phi_derivatives(jnp.array(x_eval))
    rho_model_val = np.array(d2phi) / (4 * np.pi)

    # ---- Φ zero-point offset ----
    offset = _phi_zero_point_offset(phi_model_val, phi_true[idx], r_eval)
    phi_model_corrected = phi_model_val - offset
    print(f"  Φ zero-point offset: {offset:.4f}")

    # ---- Binned statistics ----
    r_bin, phi_true_bin, phi_true_std = _binned_stats(r_eval, phi_true[idx])
    _, phi_model_bin, phi_model_std = _binned_stats(r_eval, phi_model_corrected)
    _, rho_true_bin, rho_true_std = _binned_stats(r_eval, rho_true[idx])
    _, rho_model_bin, rho_model_std = _binned_stats(r_eval, rho_model_val)
    # Φ residual
    phi_residual = phi_model_corrected - phi_true[idx]
    _, phi_res_bin, phi_res_std = _binned_stats(r_eval, phi_residual)

    # =====================================================================
    # Figure 1: Φ(r) and ρ(r) comparison (binned overlay on scatter)
    # =====================================================================
    print("Fig 1: Φ and ρ comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    sort_idx = np.argsort(r_eval)

    # Φ
    ax1.plot(r_eval[sort_idx], phi_true[idx][sort_idx], "k-", lw=2, label="True")
    ax1.plot(r_eval[sort_idx], phi_model_corrected[sort_idx], "r--", lw=1.5,
             label="Model (offset removed)")
    ax1.fill_between(r_bin, phi_true_bin - phi_true_std,
                     phi_true_bin + phi_true_std, alpha=0.15, color="k")
    ax1.fill_between(r_bin, phi_model_bin - phi_model_std,
                     phi_model_bin + phi_model_std, alpha=0.15, color="r")
    ax1.set_xlabel("r")
    ax1.set_ylabel(r"$\Phi$")
    ax1.set_title("Gravitational Potential (zero-point corrected)")
    ax1.legend(fontsize=8)

    # ρ
    ax2.plot(r_eval[sort_idx], rho_true[idx][sort_idx], "k-", lw=2, label="True")
    ax2.plot(r_eval[sort_idx], rho_model_val[sort_idx], "r--", lw=1.5, label="Model")
    ax2.fill_between(r_bin, rho_true_bin - rho_true_std,
                     rho_true_bin + rho_true_std, alpha=0.15, color="k")
    ax2.fill_between(r_bin, rho_model_bin - rho_model_std,
                     rho_model_bin + rho_model_std, alpha=0.15, color="r")
    ax2.set_xlabel("r")
    ax2.set_ylabel(r"$\rho$")
    ax2.set_title("Mass Density")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "potential_comparison.pdf", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # Figure 2: Binned radial profile of ρ (clean view)
    # =====================================================================
    print("Fig 2: Binned ρ profile...")
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.errorbar(r_bin, rho_true_bin, yerr=rho_true_std,
                fmt="ko-", lw=2, markersize=4, capsize=2, label="True")
    ax.errorbar(r_bin, rho_model_bin, yerr=rho_model_std,
                fmt="r^--", lw=1.5, markersize=4, capsize=2, label="Model")
    ax.set_xscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("Binned Radial Density Profile")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "rho_profile.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # Figure 3: Φ residual (model - true after offset) vs r
    # =====================================================================
    print("Fig 3: Φ residual...")
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.errorbar(r_bin, phi_res_bin, yerr=phi_res_std,
                fmt="o-", color="steelblue", lw=1.5, markersize=4, capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\Phi_\mathrm{model} - \Phi_\mathrm{true}$ (offset removed)")
    ax.set_title(f"Φ Residual (MAE = {np.nanmean(np.abs(phi_res_bin)):.4f})")

    fig.tight_layout()
    fig.savefig(fig_dir / "phi_residual.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # Figure 4: p_unk conditional validation
    # =====================================================================
    print("Fig 4: p_unk conditional validation...")
    key = jax.random.key(456)

    # Select particles in a narrow r-bin
    r_mask = (r_eval >= 1.5) & (r_eval <= 2.5)
    if r_mask.sum() < 100:
        # Fallback: use widest bin with at least 100 particles
        r_mid = np.median(r_eval)
        r_mask = (r_eval >= r_mid * 0.7) & (r_eval <= r_mid * 1.3)
    n_mask = r_mask.sum()
    print(f"  Using {n_mask} particles in r-bin for p_unk validation")

    obs_eval = jnp.array(eta_obs[idx][r_mask])
    unk_true = data["eta_unk"][idx][r_mask]

    # Sample from learned p_unk
    key, sample_key = jax.random.split(key)
    unk_sampled = np.array(punk_model.sample(sample_key, obs_eval))

    # Plot histograms for each unobserved dimension
    unk_labels = dim_spec.unk_labels()
    fig, axes = plt.subplots(1, dim_spec.unk_dim, figsize=(4 * dim_spec.unk_dim, 3.5))
    if dim_spec.unk_dim == 1:
        axes = [axes]

    for d in range(dim_spec.unk_dim):
        ax = axes[d]
        ax.hist(unk_true[:, d], bins=40, density=True, alpha=0.5,
                color="k", label="True")
        ax.hist(unk_sampled[:, d], bins=40, density=True, alpha=0.5,
                color="r", label="Model")
        ax.set_xlabel(unk_labels[d])
        ax.set_ylabel("Density")
        if d == 0:
            ax.legend(fontsize=7)
        ax.set_title(f"p_unk: {unk_labels[d]} (r-bin)")

    fig.suptitle("p_unk Conditional Validation", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "punk_validation.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # Figure 5: CBE residual distribution
    # =====================================================================
    print("Fig 5: CBE residual...")
    key = jax.random.key(123)
    n_cbe = min(n_eval, len(eta_obs))
    idx_cbe = np.random.default_rng(7).choice(
        len(eta_obs), n_cbe, replace=False)
    eta_obs_cbe = jnp.array(eta_obs[idx_cbe])
    ln_pobs_cbe, grad_obs_ln_pobs_cbe = pobs_model.log_prob_and_obs_grad(
        eta_obs_cbe)

    key, sample_key = jax.random.split(key)
    eta_unk_cbe = punk_model.sample(sample_key, eta_obs_cbe)
    _, grad_unk_ln_punk, grad_obs_ln_punk = punk_model.log_prob_and_grad(
        eta_unk_cbe, eta_obs_cbe)

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
    ax.set_title(f"CBE Residual (|mean|={np.abs(df_dt).mean():.4f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "cbe_residual.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # Figure 6: Training curves
    # =====================================================================
    print("Fig 6: Training curves...")
    epochs_arr = np.arange(len(loss_history["train_loss"]))
    if len(epochs_arr) == 0:
        print("  No training history found, skipping.")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].plot(epochs_arr, loss_history["train_loss"], label="Train",
                         alpha=0.8)
        axes[0, 0].plot(epochs_arr, loss_history["val_loss"], label="Val", alpha=0.8)
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].legend()
        axes[0, 0].set_title("Total Loss")

        axes[0, 1].plot(epochs_arr, loss_history["train_cbe"], label="Train",
                         alpha=0.8)
        axes[0, 1].plot(epochs_arr, loss_history["val_cbe"], label="Val", alpha=0.8)
        axes[0, 1].set_ylabel("CBE Loss")
        axes[0, 1].legend()
        axes[0, 1].set_title("CBE Loss")

        axes[1, 0].plot(epochs_arr, loss_history["train_entropy"], label="Train",
                         alpha=0.8)
        axes[1, 0].plot(epochs_arr, loss_history["val_entropy"], label="Val",
                         alpha=0.8)
        axes[1, 0].set_ylabel("Entropy Loss")
        axes[1, 0].legend()
        axes[1, 0].set_title("Entropy Regularization")

        axes[1, 1].plot(epochs_arr, loss_history["train_df_dt"], label="Train",
                         alpha=0.8)
        axes[1, 1].plot(epochs_arr, loss_history["val_df_dt"], label="Val", alpha=0.8)
        axes[1, 1].set_ylabel(r"$\langle |\partial \ln f / \partial t| \rangle$")
        axes[1, 1].legend()
        axes[1, 1].set_title("Mean |CBE Residual|")

        for ax in axes.flat:
            ax.set_xlabel("Epoch")

        fig.tight_layout()
        fig.savefig(fig_dir / "training_curves.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # =====================================================================
    # Figure 7: Symmetry axis convergence (cylindrical only)
    # =====================================================================
    if symmetry == "cylindrical" and "axis_x" in loss_history:
        print("Fig 7: Symmetry axis convergence...")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs_arr, loss_history["axis_x"], label=r"$n_x$", alpha=0.7)
        ax.plot(epochs_arr, loss_history["axis_y"], label=r"$n_y$", alpha=0.7)
        ax.plot(epochs_arr, loss_history["axis_z"], label=r"$n_z$", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Axis Component")
        ax.set_title("Symmetry Axis Direction")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "axis_convergence.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n--- Benchmark Summary ---")
    phi_mae = np.nanmean(np.abs(phi_res_bin))
    rho_bin_ok = ~np.isnan(rho_true_bin)
    rho_mae = np.mean(np.abs(
        rho_model_bin[rho_bin_ok] - rho_true_bin[rho_bin_ok]))
    cbe_mean_abs = float(np.abs(df_dt).mean())
    print(f"  Φ MAE (binned, offset removed): {phi_mae:.4f}")
    print(f"  ρ MAE (binned): {rho_mae:.4f}")
    print(f"  ⟨|∂ ln f / ∂t|⟩: {cbe_mean_abs:.4f}")
    if len(epochs_arr) > 0:
        print(f"  Final train loss: {loss_history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {loss_history['val_loss'][-1]:.4f}")

    return {"phi_mae": phi_mae, "rho_mae": rho_mae, "cbe_mean_abs": cbe_mean_abs}


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Benchmark partial-obs Deep Potential against Plummer.",
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
    fig_dir = Path(args.fig_dir) if args.fig_dir else run_dir / "figures"

    benchmark(run_dir, fig_dir, n_eval=args.n_eval)


if __name__ == "__main__":
    main()
