#!/usr/bin/env python
"""Joint training loop for Φ and p_unk with partial observations.

Trains the potential Φ and the conditional unobserved density p_unk
simultaneously, while p_obs (learned from data) is held fixed.

Follows the existing Deep Potential pre-sample pattern:
    1. Pre-sample η_obs and pre-compute ∇ ln p_obs (one-time cost).
    2. Shuffle η_obs each epoch, sample η_unk from p_unk on-the-fly.
    3. Combine precomputed + on-the-fly gradients for full 6D CBE.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx
import optax
import numpy as np
from tqdm import trange

# Allow importing from parent scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import potential as pot  # for build_trainable_filter pattern, ResMLP
from partial_obs.dim_spec import DimSpec
from partial_obs.pobs_model import ObservedDensityFlow
from partial_obs.punk_models import (
    GaussianConditionalDensity,
    GaussianMixtureConditionalDensity,
    DiscreteFlowConditionalDensity,
    ConditionalFlowDensity,
)
from partial_obs.symmetry_potential import ProjectedPotential


# Type alias for the pluggable p_unk models
PunkModel = (
    GaussianConditionalDensity
    | GaussianMixtureConditionalDensity
    | DiscreteFlowConditionalDensity
    | ConditionalFlowDensity
)


# =============================================================================
# Pre-computation: prepare fixed η_obs pool with ∇ ln p_obs
# =============================================================================

def prepare_training_data(
    pobs_model: ObservedDensityFlow,
    eta_obs_pool: Array,
    batch_size: int = 1024,
) -> dict:
    """Pre-compute ln p_obs and ∇_{η_obs} ln p_obs for the entire η_obs pool.

    This is done once before joint training since p_obs is frozen.

    Args:
        pobs_model: Trained, frozen observed density model.
        eta_obs_pool: (N, obs_dim) pool of observed phase-space points.
        batch_size: Batch size for gradient computation.

    Returns:
        Dictionary with:
            eta_obs: (N, obs_dim)
            ln_pobs: (N,)
            grad_obs_ln_pobs: (N, obs_dim)
    """
    n = eta_obs_pool.shape[0]
    all_ln_pobs = []
    all_grad_obs = []

    print(f"Pre-computing ∇ ln p_obs for {n:,} points...")
    for i in trange(0, n, batch_size):
        batch = eta_obs_pool[i:i + batch_size]
        ln_p, grad_obs = pobs_model.log_prob_and_obs_grad(batch)
        all_ln_pobs.append(np.array(ln_p))
        all_grad_obs.append(np.array(grad_obs))

    return {
        "eta_obs": np.array(eta_obs_pool),
        "ln_pobs": np.concatenate(all_ln_pobs, axis=0).astype(np.float32),
        "grad_obs_ln_pobs": np.concatenate(all_grad_obs, axis=0).astype(np.float32),
    }


def split_precomputed(
    precomputed: dict, val_frac: float = 0.25
) -> Tuple[dict, dict]:
    """Split precomputed data into training and validation sets.

    Args:
        precomputed: Output of prepare_training_data.
        val_frac: Fraction for validation.

    Returns:
        (train_data, val_data) dictionaries.
    """
    n = precomputed["eta_obs"].shape[0]
    n_val = int(n * val_frac)
    n_train = n - n_val

    train_data = {
        k: v[n_val:] for k, v in precomputed.items()
    }
    val_data = {
        k: v[:n_val] for k, v in precomputed.items()
    }
    return train_data, val_data


def save_precomputed(precomputed: dict, path: Path):
    """Save precomputed ∇ ln p_obs data to HDF5.

    Args:
        precomputed: Output of prepare_training_data.
        path: File path (e.g., joint_dir / "precomputed.h5").
    """
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key in ["eta_obs", "ln_pobs", "grad_obs_ln_pobs"]:
            f.create_dataset(key, data=precomputed[key], compression="lzf",
                             chunks=True)
    print(f"  Saved precomputed data to {path}")


def load_precomputed(path: Path) -> dict:
    """Load precomputed ∇ ln p_obs data from HDF5.

    Args:
        path: File path to precomputed.h5.

    Returns:
        Dictionary with eta_obs, ln_pobs, grad_obs_ln_pobs.
    """
    import h5py
    precomputed = {}
    with h5py.File(path, "r") as f:
        for key in ["eta_obs", "ln_pobs", "grad_obs_ln_pobs"]:
            precomputed[key] = f[key][:].astype("f4")
    print(f"  Loaded precomputed data from {path} "
          f"({precomputed['eta_obs'].shape[0]:,} points)")
    return precomputed


# =============================================================================
# CBE residual computation
# =============================================================================

def assemble_full_6d_gradient(
    dim_spec: DimSpec,
    precomputed_grad_obs_ln_pobs: Array,
    grad_unk_ln_punk: Array,
    grad_obs_ln_punk: Array,
) -> Tuple[Array, Array]:
    """Assemble full 6D gradient of ln f and split into position/velocity parts.

    ∇_{η_obs} ln f = precomputed ∇ ln p_obs + ∇_{η_obs} ln p_unk (on-the-fly)
    ∇_{η_unk} ln f = ∇_{η_unk} ln p_unk (p_obs contributes 0 here)

    Args:
        dim_spec: Dimension specification.
        precomputed_grad_obs_ln_pobs: (batch, obs_dim)
        grad_unk_ln_punk: (batch, unk_dim) from p_unk.log_prob_and_grad
        grad_obs_ln_punk: (batch, obs_dim) from p_unk.log_prob_and_grad

    Returns:
        grad_x_lnf: (batch, 3) gradient w.r.t. position
        grad_v_lnf: (batch, 3) gradient w.r.t. velocity
    """
    # Combine observed-part gradient
    grad_obs_total = precomputed_grad_obs_ln_pobs + grad_obs_ln_punk

    # Scatter both parts to 6D
    grad_6d = (
        dim_spec.scatter_obs_gradient_to_6d(grad_obs_total)
        + dim_spec.scatter_unk_gradient_to_6d(grad_unk_ln_punk)
    )

    # Split into position (0:3) and velocity (3:6)
    grad_x_lnf = grad_6d[..., :3]
    grad_v_lnf = grad_6d[..., 3:6]

    return grad_x_lnf, grad_v_lnf


def compute_cbe_residual(
    grad_x_lnf: Array,
    grad_v_lnf: Array,
    v_full: Array,
    dphi_dq: Array,
) -> Array:
    """Compute the CBE residual ∂ ln f / ∂t.

    ∂ ln f / ∂t = Σ_i v_i · (∇_x ln f)_i - Σ_i (∇_x Φ)_i · (∇_v ln f)_i

    Args:
        grad_x_lnf: (batch, 3) ∇_x ln f
        grad_v_lnf: (batch, 3) ∇_v ln f
        v_full: (batch, 3) full 3D velocities
        dphi_dq: (batch, 3) ∇_x Φ

    Returns:
        df_dt: (batch,) CBE residual per sample
    """
    df_dt = jnp.sum(v_full * grad_x_lnf, axis=-1) - jnp.sum(
        dphi_dq * grad_v_lnf, axis=-1
    )
    return df_dt


# =============================================================================
# Loss function
# =============================================================================

def partial_obs_loss_fn(
    phi_model: ProjectedPotential,
    punk_model: PunkModel,
    dim_spec: DimSpec,
    key: PRNGKeyArray,
    eta_obs_batch: Array,
    precomputed_ln_pobs: Array,
    precomputed_grad_obs_ln_pobs: Array,
    loss_params: dict,
) -> Tuple[Array, Tuple[Array, Array, Array, Array]]:
    """Compute the joint loss for Φ and p_unk.

    Loss = ⟨arcsinh(α·|df_dt|)⟩                       (CBE stationarity)
         + λ·⟨arcsinh(β·max(-∇²Φ, 0))⟩                (Poisson penalty)
         + γ·⟨ln p_obs + ln p_unk⟩                      (Entropy regularization)
         + l2_phi·||W_Φ||² + l2_punk·||W_punk||²       (L2 regularization)

    Args:
        phi_model: Potential model.
        punk_model: Conditional unobserved density model.
        dim_spec: Dimension specification.
        key: JAX random key (for sampling η_unk).
        eta_obs_batch: (batch, obs_dim) observed points.
        precomputed_ln_pobs: (batch,) precomputed ln p_obs.
        precomputed_grad_obs_ln_pobs: (batch, obs_dim).
        loss_params: Dictionary with hyperparameters.

    Returns:
        (total_loss, (cbe_loss, poisson_loss, entropy_loss, df_dt_mean))
    """
    # Unpack loss parameters with defaults
    alpha = loss_params.get("alpha", 1.0)
    beta = loss_params.get("beta", 1.0)
    lambda_poisson = loss_params.get("lambda_poisson", 1.0)
    gamma_entropy = loss_params.get("gamma_entropy", 0.1)
    l2_phi = loss_params.get("l2_phi", 0.0)
    l2_punk = loss_params.get("l2_punk", 0.0)

    # 1. Sample η_unk from p_unk(· | η_obs)
    eta_unk_batch = punk_model.sample(key, eta_obs_batch)

    # 2. Compute ln p_unk and its gradients
    ln_punk, grad_unk_ln_punk, grad_obs_ln_punk = punk_model.log_prob_and_grad(
        eta_unk_batch, eta_obs_batch
    )

    # 3. Assemble full 6D gradient of ln f
    grad_x_lnf, grad_v_lnf = assemble_full_6d_gradient(
        dim_spec,
        precomputed_grad_obs_ln_pobs,
        grad_unk_ln_punk,
        grad_obs_ln_punk,
    )

    # 4. Reconstruct full 6D eta for velocity
    eta_full = dim_spec.combine_eta(eta_obs_batch, eta_unk_batch)
    v_full = eta_full[..., 3:6]  # (batch, 3) velocities
    x_full = eta_full[..., :3]   # (batch, 3) positions

    # 5. Compute potential derivatives
    dphi_dq, d2phi_dq2 = phi_model.calc_phi_derivatives(x_full)

    # 6. CBE residual
    df_dt = compute_cbe_residual(grad_x_lnf, grad_v_lnf, v_full, dphi_dq)

    # 7. Loss components
    cbe_loss = jnp.mean(jnp.arcsinh(alpha * jnp.abs(df_dt)) / alpha)

    poisson_loss = jnp.mean(
        jnp.arcsinh(beta * jnp.clip(-d2phi_dq2, a_min=0.0)) / beta
    )

    entropy_loss = gamma_entropy * jnp.mean(precomputed_ln_pobs + ln_punk)

    # 8. L2 regularization
    def _l2_penalty(model):
        weights = [
            leaf.weight
            for leaf in jax.tree_util.tree_leaves(
                model, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
            )
            if isinstance(leaf, eqx.nn.Linear)
        ]
        if not weights:
            return jnp.array(0.0)
        total_sq = jnp.sum(jnp.array([jnp.sum(w**2) for w in weights]))
        total_count = jnp.sum(jnp.array([w.size for w in weights]))
        return total_sq / (total_count + 1e-8)

    l2_loss = l2_phi * _l2_penalty(phi_model) + l2_punk * _l2_penalty(punk_model)

    total_loss = cbe_loss + lambda_poisson * poisson_loss + entropy_loss + l2_loss

    return total_loss, (cbe_loss, poisson_loss, entropy_loss,
                         jnp.mean(jnp.abs(df_dt)))


# =============================================================================
# Training step
# =============================================================================

@eqx.filter_jit
def _train_step(
    phi_params,
    punk_params,
    phi_static,
    punk_static,
    dim_spec,
    loss_params,
    optimizer,
    opt_state,
    key,
    eta_obs_batch,
    precomputed_ln_pobs,
    precomputed_grad_obs,
    val_loss,
):
    """Single joint training step (jitted)."""
    def _loss(phi_p, punk_p, k, eobs, lnp, gobs):
        phi_m = eqx.combine(phi_p, phi_static)
        punk_m = eqx.combine(punk_p, punk_static)
        total_loss, aux = partial_obs_loss_fn(
            phi_m, punk_m, dim_spec, k,
            eobs, lnp, gobs, loss_params,
        )
        return total_loss, aux

    (loss, aux), grads = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(
        phi_params, punk_params, key,
        eta_obs_batch, precomputed_ln_pobs, precomputed_grad_obs,
    )

    combined_grads = {"phi": grads[0], "punk": grads[1]}
    combined_params = {"phi": phi_params, "punk": punk_params}

    updates, opt_state = optimizer.update(
        combined_grads, opt_state, combined_params,
        value=val_loss,
    )
    combined_params = optax.apply_updates(combined_params, updates)

    return (combined_params["phi"], combined_params["punk"],
            opt_state, loss, aux)


# =============================================================================
# Build trainable filter for eqx.partition
# =============================================================================

def build_partial_obs_filter(
    phi_model: ProjectedPotential,
    punk_model: PunkModel,
) -> Tuple[dict, dict, dict, dict]:
    """Partition phi_model and punk_model into (params, static) pairs.

    Returns:
        (phi_params, phi_static, punk_params, punk_static)
    """
    # Non-trainable names to filter out
    non_trainable_names = {"scale", "pos_mean", "pos_std", "pos_encoder",
                           "r_scale", "R_scale", "z_scale"}

    def _filter_fn(path, leaf):
        from jax.tree_util import GetAttrKey
        final_key = path[-1] if path else None
        if isinstance(final_key, GetAttrKey):
            if final_key.name in non_trainable_names:
                return False
        return isinstance(leaf, jax.Array)

    phi_filter_spec = jax.tree_util.tree_map_with_path(_filter_fn, phi_model)
    punk_filter_spec = jax.tree_util.tree_map_with_path(_filter_fn, punk_model)

    phi_params, phi_static = eqx.partition(phi_model, phi_filter_spec)
    punk_params, punk_static = eqx.partition(punk_model, punk_filter_spec)

    return phi_params, phi_static, punk_params, punk_static


# =============================================================================
# Main training loop
# =============================================================================

def train_partial_obs(
    key: PRNGKeyArray,
    phi_model: ProjectedPotential,
    punk_model: PunkModel,
    pobs_model: ObservedDensityFlow,
    dim_spec: DimSpec,
    precomputed_train: dict,
    precomputed_val: dict,
    optimizer: optax.GradientTransformation,
    n_epochs: int,
    batch_size: int,
    loss_params: dict,
    checkpoint_frequency_epochs: int = 50,
    checkpoint_dir: Optional[Path] = None,
    loss_history: Optional[dict] = None,
    timeout_hours: Optional[float] = None,
) -> Tuple[ProjectedPotential, PunkModel, dict]:
    """Jointly train Φ and p_unk with partial observations.

    Args:
        key: JAX random key.
        phi_model: Potential model (ProjectedPotential).
        punk_model: Conditional unobserved density model.
        pobs_model: Frozen observed density model.
        dim_spec: Dimension specification.
        precomputed_train: Training data from prepare_training_data.
        precomputed_val: Validation data.
        optimizer: Optax optimizer (manages both phi and punk params).
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        loss_params: Dictionary of loss hyperparameters.
        checkpoint_frequency_epochs: How often to save checkpoints.
        checkpoint_dir: Directory for checkpoint files.
        loss_history: Existing loss history to resume from.
        timeout_hours: Maximum training time in hours.

    Returns:
        (phi_model, punk_model, loss_history)
    """
    if loss_history is None:
        loss_history = {
            "train_loss": [], "val_loss": [],
            "train_cbe": [], "val_cbe": [],
            "train_entropy": [], "val_entropy": [],
            "train_df_dt": [], "val_df_dt": [],
            "lr": [],
        }
        if isinstance(phi_model, ProjectedPotential) and \
           phi_model.symmetry_type == "cylindrical":
            loss_history["axis_x"] = []
            loss_history["axis_y"] = []
            loss_history["axis_z"] = []

    n_train = precomputed_train["eta_obs"].shape[0]
    n_val = precomputed_val["eta_obs"].shape[0]
    steps_per_epoch = n_train // batch_size
    val_batch_size = min(n_val, batch_size)

    # Partition models into (params, static)
    phi_params, phi_static, punk_params, punk_static = \
        build_partial_obs_filter(phi_model, punk_model)

    # Initialize optimizer state with combined params
    opt_state = optimizer.init({"phi": phi_params, "punk": punk_params})

    print("Starting joint training of Φ + p_unk...")
    print(f"  Φ params: {phi_model.count_parameters():,}")
    print(f"  p_unk params: {punk_model.count_parameters():,}")
    print(f"  p_unk type: {type(punk_model).__name__}")
    print(f"  Batch size: {batch_size}, Steps/epoch: {steps_per_epoch}")
    print(f"  Epochs: {n_epochs}, Training samples: {n_train:,}")

    start_epoch = len(loss_history["lr"])

    for epoch in (pbar := trange(start_epoch, n_epochs)):
        key, shuffle_key = jax.random.split(key)

        # Shuffle training data
        perm = jax.random.permutation(shuffle_key, n_train)
        # Convert precomputed arrays to jax for indexing
        eta_obs_all = jnp.array(precomputed_train["eta_obs"])
        ln_pobs_all = jnp.array(precomputed_train["ln_pobs"])
        grad_obs_all = jnp.array(precomputed_train["grad_obs_ln_pobs"])

        epoch_losses = []
        epoch_cbe = []
        epoch_entropy = []
        epoch_df_dt = []

        for step in range(steps_per_epoch):
            key, step_key = jax.random.split(key)

            idx = perm[step * batch_size:(step + 1) * batch_size]
            if len(idx) != batch_size:
                continue

            eta_obs_batch = eta_obs_all[idx]
            ln_pobs_batch = ln_pobs_all[idx]
            grad_obs_batch = grad_obs_all[idx]

            phi_params, punk_params, opt_state, loss, aux = _train_step(
                phi_params, punk_params,
                phi_static, punk_static,
                dim_spec, loss_params, optimizer, opt_state,
                step_key, eta_obs_batch, ln_pobs_batch, grad_obs_batch,
                val_loss=jnp.array(0.0),
            )

            epoch_losses.append(float(loss))
            epoch_cbe.append(float(aux[0]))
            epoch_entropy.append(float(aux[2]))
            epoch_df_dt.append(float(aux[3]))

        # ---- Validation ----
        val_eta_obs = jnp.array(precomputed_val["eta_obs"])
        val_ln_pobs = jnp.array(precomputed_val["ln_pobs"])
        val_grad_obs = jnp.array(precomputed_val["grad_obs_ln_pobs"])

        n_val_steps = max(1, n_val // val_batch_size)
        val_losses = []
        val_cbe = []
        val_entropy = []
        val_df_dt = []

        phi_model_eval = eqx.combine(phi_params, phi_static)
        punk_model_eval = eqx.combine(punk_params, punk_static)

        for vi in range(n_val_steps):
            key, val_key = jax.random.split(key)
            v_start = vi * val_batch_size
            v_end = min(v_start + val_batch_size, n_val)

            v_loss, v_aux = partial_obs_loss_fn(
                phi_model_eval, punk_model_eval, dim_spec, val_key,
                val_eta_obs[v_start:v_end],
                val_ln_pobs[v_start:v_end],
                val_grad_obs[v_start:v_end],
                loss_params,
            )
            val_losses.append(float(v_loss))
            val_cbe.append(float(v_aux[0]))
            val_entropy.append(float(v_aux[2]))
            val_df_dt.append(float(v_aux[3]))

        # ---- Logging ----
        train_loss = np.mean(epoch_losses)
        val_loss = np.mean(val_losses)

        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
        loss_history["train_cbe"].append(np.mean(epoch_cbe))
        loss_history["val_cbe"].append(np.mean(val_cbe))
        loss_history["train_entropy"].append(np.mean(epoch_entropy))
        loss_history["val_entropy"].append(np.mean(val_entropy))
        loss_history["train_df_dt"].append(np.mean(epoch_df_dt))
        loss_history["val_df_dt"].append(np.mean(val_df_dt))
        loss_history["lr"].append(
            float(optax.tree_get(opt_state, "scale"))
            if hasattr(opt_state, "scale") else 0.0
        )

        # Log symmetry axis
        if isinstance(phi_model_eval, ProjectedPotential) and \
           phi_model_eval.symmetry_type == "cylindrical":
            axis = np.array(phi_model_eval.axis.get_axis())
            loss_history["axis_x"].append(float(axis[0]))
            loss_history["axis_y"].append(float(axis[1]))
            loss_history["axis_z"].append(float(axis[2]))

        pbar.set_description(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"CBE: {loss_history['train_cbe'][-1]:.4f} | "
            f"Ent: {loss_history['train_entropy'][-1]:.4f}"
        )

        # ---- Checkpointing ----
        if checkpoint_frequency_epochs > 0 and \
           (epoch + 1) % checkpoint_frequency_epochs == 0 and \
           checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            phi_final = eqx.combine(phi_params, phi_static)
            punk_final = eqx.combine(punk_params, punk_static)

            eqx.tree_serialise_leaves(
                checkpoint_dir / f"phi_epoch{epoch+1}.eqx", phi_final
            )
            eqx.tree_serialise_leaves(
                checkpoint_dir / f"punk_epoch{epoch+1}.eqx", punk_final
            )
            import json
            with open(checkpoint_dir / f"loss_epoch{epoch+1}.json", "w") as f:
                json.dump(loss_history, f, indent=2)

        # Timeout check
        if timeout_hours is not None:
            elapsed = pbar.format_dict["elapsed"]
            if elapsed > 30 and epoch > 2:
                rate = pbar.format_dict["rate"]
                if rate and rate > 0:
                    remaining = (n_epochs - epoch) / rate
                    if remaining > timeout_hours * 3600:
                        print("Timeout limit reached, stopping.")
                        break

    # Final model assembly
    phi_final = eqx.combine(phi_params, phi_static)
    punk_final = eqx.combine(punk_params, punk_static)

    return phi_final, punk_final, loss_history
