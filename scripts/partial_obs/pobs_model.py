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

    def __init__(
        self,
        key: PRNGKeyArray,
        dim_spec: DimSpec,
        data_mean: Array,
        data_std: Array,
        vector_field_params: dict,
    ):
        """Initialize the observed density flow.

        Args:
            key: JAX random key.
            dim_spec: Dimension specification.
            data_mean: Mean of observed training data, shape (obs_dim,).
            data_std: Std of observed training data, shape (obs_dim,).
            vector_field_params: Parameters for the vector field neural network.
        """
        self.dim_spec = dim_spec

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
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, self)

    @classmethod
    def from_flow(
        cls,
        dim_spec: DimSpec,
        trained_flow: fom.NormalizingFlow,
    ) -> "ObservedDensityFlow":
        """Wrap an already-trained NormalizingFlow.

        Args:
            dim_spec: Dimension specification.
            trained_flow: Trained unconditional NormalizingFlow on η_obs.

        Returns:
            ObservedDensityFlow wrapping the trained flow.
        """
        # Create minimal instance and replace the flow
        dummy = cls(
            key=jax.random.key(0),
            dim_spec=dim_spec,
            data_mean=jnp.zeros(dim_spec.obs_dim),
            data_std=jnp.ones(dim_spec.obs_dim),
            vector_field_params={"input_dim": dim_spec.obs_dim, "width": 32,
                                 "depth": 2, "cond_dim": 0,
                                 "base_dist_dim": dim_spec.obs_dim},
        )
        # Replace the flow while keeping dim_spec
        wrapped = eqx.tree_at(lambda m: m.flow, dummy, trained_flow)
        return wrapped

    @classmethod
    def load(cls, path: Path) -> "ObservedDensityFlow":
        """Load a model from disk.

        Needs a minimal shell model for deserialization. We create one with
        dummy parameters and then overwrite via tree_deserialise_leaves.
        """
        # Create a dummy with minimal params
        dummy = cls(
            key=jax.random.key(0),
            dim_spec=DimSpec([0]),  # will be overwritten
            data_mean=jnp.zeros(1),
            data_std=jnp.ones(1),
            vector_field_params={"input_dim": 1, "width": 32, "depth": 2,
                                 "cond_dim": 0, "base_dist_dim": 1},
        )
        return eqx.tree_deserialise_leaves(Path(path), like=dummy)

    def __repr__(self) -> str:
        return (
            f"ObservedDensityFlow(obs_dim={self.dim_spec.obs_dim}, "
            f"dims={self.dim_spec.obs_labels()}, "
            f"params={self.count_parameters():,})"
        )
