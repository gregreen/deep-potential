#!/usr/bin/env python
"""Observed density model p_obs(η_obs) — a frozen normalizing flow.

Wraps an unconditional NormalizingFlow trained on the observed dimensions.
Provides log-probability evaluation and gradient computation needed for the
full 6D CBE assembly (gradients are zero-padded into unobserved slots).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx

# Allow importing from parent scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flow_ot_flow_matching as fom
from partial_obs.dim_spec import DimSpec


class ObservedDensityFlow(eqx.Module):
    """Wrapper around an unconditional NormalizingFlow for p_obs(η_obs).

    This model is trained once on observed data and then frozen during
    the joint Φ + p_unk training.

    Attributes:
        flow: The underlying unconditional NormalizingFlow on η_obs.
        dim_spec: Which dimensions are observed.
    """

    flow: fom.NormalizingFlow
    dim_spec: DimSpec = eqx.field(static=True)
    vf_type: str = eqx.field(static=True)
    vf_width: int = eqx.field(static=True)
    vf_depth: int = eqx.field(static=True)
    vf_time_emb: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        dim_spec: DimSpec,
        data_mean: Array,
        data_std: Array,
        vector_field_params: dict,
    ):
        """Initialize the observed density flow."""
        self.dim_spec = dim_spec
        self.vf_type = vector_field_params.get("type", "FourierTimeResMLP")
        self.vf_width = vector_field_params.get("width", 32)
        self.vf_depth = vector_field_params.get("depth", 3)
        self.vf_time_emb = vector_field_params.get("time_embedding_dim", 32)

        if data_mean.shape[0] != dim_spec.obs_dim:
            raise ValueError(
                f"data_mean has dim {data_mean.shape[0]} but obs_dim is "
                f"{dim_spec.obs_dim}"
            )

        self.flow = fom.NormalizingFlow(
            key=key,
            data_mean=data_mean,
            data_std=data_std,
            vector_field_params=vector_field_params,
            model_dir=None,  # saved via this wrapper, not the inner flow
        )

    # ---- Core density interface ----

    def log_prob(self, eta_obs: Array) -> Array:
        """Log-probability ln p_obs(η_obs).

        Args:
            eta_obs: Array of shape (..., obs_dim).

        Returns:
            Log-probability per sample, shape (...,).
        """
        return self.flow.log_prob(eta_obs)

    @eqx.filter_jit
    def sample(self, key: PRNGKeyArray, num_samples: int) -> Array:
        """Sample η_obs ~ p_obs.

        Args:
            key: JAX random key.
            num_samples: Number of samples to draw.

        Returns:
            Array of shape (num_samples, obs_dim).
        """
        return self.flow.sample(key, (num_samples,))

    def log_prob_and_obs_grad(self, eta_obs: Array) -> tuple[Array, Array]:
        """Compute ln p_obs and its gradient w.r.t. observed dimensions.

        Args:
            eta_obs: Array of shape (..., obs_dim).

        Returns:
            (ln_pobs, grad_obs_ln_pobs) where:
                ln_pobs: shape (...,)
                grad_obs_ln_pobs: shape (..., obs_dim)
        """
        fn = eqx.filter_value_and_grad(self.log_prob)
        ln_pobs, grad_obs = jax.vmap(fn)(eta_obs)
        return ln_pobs, grad_obs

    # ---- 6D gradient utilities ----

    def obs_grad_to_6d(self, grad_obs: Array) -> Array:
        """Map an obs-dim gradient to a full 6D gradient (zeros in unk slots).

        Args:
            grad_obs: Array of shape (..., obs_dim).

        Returns:
            Array of shape (..., 6).
        """
        return self.dim_spec.scatter_obs_gradient_to_6d(grad_obs)

    # ---- I/O ----

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(
            x.size
            for x in jax.tree_util.tree_leaves(eqx.filter(self.flow, eqx.is_array))
        )

    def save(self, path: Path):
        """Save model with full architecture metadata for exact reconstruction."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, self)
        meta = {
            "obs_dim": self.dim_spec.obs_dim,
            "obs_indices": list(self.dim_spec.obs_indices),
            "vf_type": self.vf_type,
            "vf_width": self.vf_width,
            "vf_depth": self.vf_depth,
            "vf_time_emb": self.vf_time_emb,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def from_flow(
        cls,
        dim_spec: DimSpec,
        trained_flow: fom.NormalizingFlow,
        vf_type: str = "FourierTimeResMLP",
        vf_width: int = 64,
        vf_depth: int = 3,
        vf_time_emb: int = 32,
    ) -> "ObservedDensityFlow":
        """Wrap an already-trained NormalizingFlow.

        Stores architecture params so the model can be reloaded correctly.
        """
        obs_dim = dim_spec.obs_dim
        dummy = cls(
            key=jax.random.key(0), dim_spec=dim_spec,
            data_mean=jnp.zeros(obs_dim), data_std=jnp.ones(obs_dim),
            vector_field_params={
                "type": vf_type, "input_dim": obs_dim,
                "width": vf_width, "depth": vf_depth, "cond_dim": 0,
                "base_dist_dim": obs_dim, "time_embedding_dim": vf_time_emb,
            },
        )
        wrapped = eqx.tree_at(lambda m: m.flow, dummy, trained_flow)
        return wrapped

    @classmethod
    def load(cls, path: Path) -> "ObservedDensityFlow":
        """Load model from disk. Uses metadata JSON for exact architecture."""
        import json
        path = Path(path)
        json_path = path.with_suffix(".json")
        with open(json_path, "r") as f:
            meta = json.load(f)
        obs_dim = meta["obs_dim"]
        dim_spec = DimSpec(meta["obs_indices"])
        dummy = cls(
            key=jax.random.key(0), dim_spec=dim_spec,
            data_mean=jnp.zeros(obs_dim), data_std=jnp.ones(obs_dim),
            vector_field_params={
                "type": meta.get("vf_type", "FourierTimeResMLP"),
                "input_dim": obs_dim,
                "width": meta.get("vf_width", 64),
                "depth": meta.get("vf_depth", 3),
                "cond_dim": 0,
                "base_dist_dim": obs_dim,
                "time_embedding_dim": meta.get("vf_time_emb", 32),
            },
        )
        return eqx.tree_deserialise_leaves(path, like=dummy)

    def __repr__(self) -> str:
        return (
            f"ObservedDensityFlow(obs_dim={self.dim_spec.obs_dim}, "
            f"dims={self.dim_spec.obs_labels()}, "
            f"params={self.count_parameters():,})"
        )
