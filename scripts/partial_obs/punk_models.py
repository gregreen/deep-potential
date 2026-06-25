#!/usr/bin/env python
"""Pluggable conditional density models for p_unk(η_unk | η_obs).

All models implement a common interface (duck-typed):
    log_prob(eta_unk, eta_obs) -> Array          # ln p_unk (batched)
    sample(key, eta_obs) -> Array                # sample η_unk (batched)
    log_prob_and_grad(eta_unk, eta_obs) -> (ln_p, grad_unk, grad_obs)
    count_parameters() -> int

All models are properly normalized (Z ≡ 1). Supported types:
    "gaussian"           — Diagonal Gaussian with ResNet-predicted μ, σ
    "gaussian_mixture"   — K-component Gaussian mixture
    "discrete_flow"      — RealNVP-style coupling layers (no ODE)
    "flow"               — Continuous-time conditional normalizing flow

Architecture: internal methods operate on single samples. Public methods
(log_prob, sample, log_prob_and_grad) vmap over the batch dimension.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import potential as pot
import flow_ot_flow_matching_conditional as fomc
from partial_obs.dim_spec import DimSpec


# =============================================================================
# 1. Gaussian Conditional Density
# =============================================================================

class GaussianConditionalDensity(eqx.Module):
    """Diagonal Gaussian p_unk(η_unk | η_obs) = N(μ(η_obs), diag(σ²(η_obs)))."""

    cond_net: pot.ResMLP
    unk_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, unk_dim: int, obs_dim: int,
                 width_size: int = 64, depth: int = 3):
        self.unk_dim = unk_dim
        self.obs_dim = obs_dim
        self.cond_net = pot.ResMLP(
            in_size=obs_dim, out_size=unk_dim * 2,
            width_size=width_size, depth=depth, key=key,
        )

    def _log_prob_one(self, eta_unk: Array, eta_obs: Array) -> Array:
        out = self.cond_net(eta_obs)
        mu = out[:self.unk_dim]
        log_sigma = out[self.unk_dim:]
        sigma = jnp.exp(log_sigma)
        diff = (eta_unk - mu) / sigma
        return -0.5 * jnp.sum(diff**2 + 2 * log_sigma + jnp.log(2 * jnp.pi))

    def _sample_one(self, key: PRNGKeyArray, eta_obs: Array) -> Array:
        out = self.cond_net(eta_obs)
        mu = out[:self.unk_dim]
        log_sigma = out[self.unk_dim:]
        sigma = jnp.exp(log_sigma)
        eps = jax.random.normal(key, shape=mu.shape)
        return mu + sigma * eps

    def log_prob(self, eta_unk: Array, eta_obs: Array) -> Array:
        return jax.vmap(self._log_prob_one)(eta_unk, eta_obs)

    def sample(self, key: PRNGKeyArray, eta_obs: Array) -> Array:
        keys = jax.random.split(key, eta_obs.shape[0])
        return jax.vmap(self._sample_one)(keys, eta_obs)

    def log_prob_and_grad(self, eta_unk: Array, eta_obs: Array
                          ) -> tuple[Array, Array, Array]:
        fn = jax.vmap(jax.value_and_grad(self._log_prob_one, argnums=(0, 1)))
        ln_p, (grad_unk, grad_obs) = fn(eta_unk, eta_obs)
        return ln_p, grad_unk, grad_obs

    def count_parameters(self) -> int:
        return sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(self, eqx.is_array)))


# =============================================================================
# 2. Gaussian Mixture Conditional Density
# =============================================================================

class GaussianMixtureConditionalDensity(eqx.Module):
    """K-component Gaussian mixture: p_unk = Σ_k π_k(η_obs) N(μ_k, diag(σ_k²))."""

    cond_net: pot.ResMLP
    n_components: int = eqx.field(static=True)
    unk_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, unk_dim: int, obs_dim: int,
                 n_components: int = 4, width_size: int = 64, depth: int = 3):
        self.n_components = n_components
        self.unk_dim = unk_dim
        self.obs_dim = obs_dim
        out_size = n_components + n_components * unk_dim * 2
        self.cond_net = pot.ResMLP(
            in_size=obs_dim, out_size=out_size,
            width_size=width_size, depth=depth, key=key,
        )

    def _parse(self, out: Array) -> tuple[Array, Array, Array]:
        K, D = self.n_components, self.unk_dim
        logits = out[:K]
        mus = out[K:K + K * D].reshape(K, D)
        log_sigmas = out[K + K * D:].reshape(K, D)
        return logits, mus, log_sigmas

    def _log_prob_one(self, eta_unk: Array, eta_obs: Array) -> Array:
        out = self.cond_net(eta_obs)
        logits, mus, log_sigmas = self._parse(out)
        diff = (eta_unk - mus) / jnp.exp(log_sigmas)
        comp_lp = -0.5 * jnp.sum(diff**2 + 2 * log_sigmas + jnp.log(2 * jnp.pi),
                                 axis=-1)
        log_weights = jax.nn.log_softmax(logits)
        return jax.scipy.special.logsumexp(log_weights + comp_lp)

    def _sample_one(self, key: PRNGKeyArray, eta_obs: Array) -> Array:
        out = self.cond_net(eta_obs)
        logits, mus, log_sigmas = self._parse(out)
        K, D = self.n_components, self.unk_dim
        key_comp, key_eps = jax.random.split(key)
        comp = jax.random.categorical(key_comp, logits)
        mu = mus[comp]
        sigma = jnp.exp(log_sigmas[comp])
        eps = jax.random.normal(key_eps, shape=(D,))
        return mu + sigma * eps

    def log_prob(self, eta_unk: Array, eta_obs: Array) -> Array:
        return jax.vmap(self._log_prob_one)(eta_unk, eta_obs)

    def sample(self, key: PRNGKeyArray, eta_obs: Array) -> Array:
        keys = jax.random.split(key, eta_obs.shape[0])
        return jax.vmap(self._sample_one)(keys, eta_obs)

    def log_prob_and_grad(self, eta_unk: Array, eta_obs: Array
                          ) -> tuple[Array, Array, Array]:
        fn = jax.vmap(jax.value_and_grad(self._log_prob_one, argnums=(0, 1)))
        ln_p, (grad_unk, grad_obs) = fn(eta_unk, eta_obs)
        return ln_p, grad_unk, grad_obs

    def count_parameters(self) -> int:
        return sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(self, eqx.is_array)))


# =============================================================================
# 3. Discrete Flow Conditional Density (RealNVP-style)
# =============================================================================

class _AffineCouplingLayer(eqx.Module):
    """One affine coupling layer conditioned on η_obs (single-sample)."""

    scale_net: pot.ResMLP
    shift_net: pot.ResMLP
    mask: Array

    def __init__(self, key: PRNGKeyArray, dim: int, cond_dim: int,
                 width_size: int = 32, depth: int = 2,
                 mask: Optional[Array] = None):
        # Use consistent mask size: ceil(dim/2) elements are True
        n_masked = dim - dim // 2
        if mask is not None:
            self.mask = mask
        else:
            self.mask = jnp.array(
                [True] * n_masked + [False] * (dim - n_masked)
            )
        key_s, key_t = jax.random.split(key)
        in_size = n_masked + cond_dim
        self.scale_net = pot.ResMLP(
            in_size=in_size, out_size=dim,
            width_size=width_size, depth=depth, key=key_s)
        self.shift_net = pot.ResMLP(
            in_size=in_size, out_size=dim,
            width_size=width_size, depth=depth, key=key_t)

    def forward(self, x: Array, cond: Array) -> tuple[Array, Array]:
        x_masked = x * self.mask
        net_input = jnp.concatenate(
            [x_masked[self.mask.astype(bool)], cond])
        scale = self.scale_net(net_input)
        shift = self.shift_net(net_input)
        inv_mask = 1.0 - self.mask
        y = x_masked + inv_mask * (x * jnp.exp(scale) + shift)
        log_det = jnp.sum(scale * inv_mask)
        return y, log_det

    def inverse(self, y: Array, cond: Array) -> tuple[Array, Array]:
        y_masked = y * self.mask
        net_input = jnp.concatenate(
            [y_masked[self.mask.astype(bool)], cond])
        scale = self.scale_net(net_input)
        shift = self.shift_net(net_input)
        inv_mask = 1.0 - self.mask
        x = y_masked + inv_mask * ((y - shift) * jnp.exp(-scale))
        log_det = jnp.sum(-scale * inv_mask)
        return x, log_det


class DiscreteFlowConditionalDensity(eqx.Module):
    """Discrete normalizing flow with coupling layers. No ODE integration."""

    layers: list[_AffineCouplingLayer]
    unk_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, unk_dim: int, obs_dim: int,
                 n_layers: int = 3, width_size: int = 32, depth: int = 2):
        self.unk_dim = unk_dim
        self.obs_dim = obs_dim
        keys = jax.random.split(key, n_layers)
        self.layers = []
        n_masked = unk_dim - unk_dim // 2
        for i in range(n_layers):
            # Rotate which dimensions are masked
            start = i % unk_dim
            mask_indices = [(start + j) % unk_dim for j in range(n_masked)]
            mask = jnp.zeros(unk_dim, dtype=bool)
            mask = mask.at[jnp.array(mask_indices)].set(True)
            self.layers.append(_AffineCouplingLayer(
                key=keys[i], dim=unk_dim, cond_dim=obs_dim,
                width_size=width_size, depth=depth, mask=mask))

    def _forward_one(self, x: Array, cond: Array) -> tuple[Array, Array]:
        total_ld = jnp.array(0.0)
        for layer in self.layers:
            x, ld = layer.forward(x, cond)
            total_ld = total_ld + ld
        return x, total_ld

    def _inverse_one(self, y: Array, cond: Array) -> tuple[Array, Array]:
        total_ld = jnp.array(0.0)
        for layer in reversed(self.layers):
            y, ld = layer.inverse(y, cond)
            total_ld = total_ld + ld
        return y, total_ld

    def _log_prob_one(self, eta_unk: Array, eta_obs: Array) -> Array:
        z, log_det = self._inverse_one(eta_unk, eta_obs)
        base_lp = -0.5 * jnp.sum(z**2 + jnp.log(2 * jnp.pi))
        return base_lp + log_det

    def _sample_one(self, key: PRNGKeyArray, eta_obs: Array) -> Array:
        z = jax.random.normal(key, shape=(self.unk_dim,))
        y, _ = self._forward_one(z, eta_obs)
        return y

    def log_prob(self, eta_unk: Array, eta_obs: Array) -> Array:
        return jax.vmap(self._log_prob_one)(eta_unk, eta_obs)

    def sample(self, key: PRNGKeyArray, eta_obs: Array) -> Array:
        keys = jax.random.split(key, eta_obs.shape[0])
        return jax.vmap(self._sample_one)(keys, eta_obs)

    def log_prob_and_grad(self, eta_unk: Array, eta_obs: Array
                          ) -> tuple[Array, Array, Array]:
        fn = jax.vmap(jax.value_and_grad(self._log_prob_one, argnums=(0, 1)))
        ln_p, (grad_unk, grad_obs) = fn(eta_unk, eta_obs)
        return ln_p, grad_unk, grad_obs

    def count_parameters(self) -> int:
        return sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(self, eqx.is_array)))


# =============================================================================
# 4. Continuous-Time Conditional Flow (wrapper)
# =============================================================================

class ConditionalFlowDensity(eqx.Module):
    """Wraps a continuous-time conditional NormalizingFlow from flowjax."""

    flow: fomc.NormalizingFlow
    unk_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, unk_dim: int, obs_dim: int,
                 unk_data_mean: Array, unk_data_std: Array,
                 obs_data_mean: Array, obs_data_std: Array,
                 vector_field_params: dict):
        self.unk_dim = unk_dim
        self.obs_dim = obs_dim
        vf_params = vector_field_params.copy()
        vf_params["cond_dim"] = obs_dim
        vf_params["base_dist_dim"] = unk_dim
        self.flow = fomc.NormalizingFlow(
            key=key, dim=unk_dim,
            data_mean=unk_data_mean, data_std=unk_data_std,
            vector_field_params=vf_params,
            cond_dim=obs_dim,
            cond_mean=obs_data_mean, cond_std=obs_data_std)

    def log_prob(self, eta_unk: Array, eta_obs: Array) -> Array:
        return self.flow.log_prob(eta_unk, condition=eta_obs)

    def sample(self, key: PRNGKeyArray, eta_obs: Array) -> Array:
        n = eta_obs.shape[0] if eta_obs.ndim > 0 else 1
        return self.flow.sample(key, n, condition=eta_obs)

    def log_prob_and_grad(self, eta_unk: Array, eta_obs: Array
                          ) -> tuple[Array, Array, Array]:
        def _lp(u, o):
            return self.flow.log_prob(u, condition=o)
        fn = jax.vmap(jax.value_and_grad(_lp, argnums=(0, 1)))
        ln_p, (grad_unk, grad_obs) = fn(eta_unk, eta_obs)
        return ln_p, grad_unk, grad_obs

    def count_parameters(self) -> int:
        return sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(self, eqx.is_array)))


# =============================================================================
# Factory
# =============================================================================

def make_punk_model(model_type: str, key: PRNGKeyArray, unk_dim: int,
                    obs_dim: int, **model_params):
    """Create a conditional density model for p_unk."""
    model_type = model_type.lower()
    if model_type == "gaussian":
        return GaussianConditionalDensity(
            key=key, unk_dim=unk_dim, obs_dim=obs_dim,
            width_size=model_params.get("width_size", 64),
            depth=model_params.get("depth", 3))
    elif model_type == "gaussian_mixture":
        return GaussianMixtureConditionalDensity(
            key=key, unk_dim=unk_dim, obs_dim=obs_dim,
            n_components=model_params.get("n_components", 4),
            width_size=model_params.get("width_size", 64),
            depth=model_params.get("depth", 3))
    elif model_type == "discrete_flow":
        return DiscreteFlowConditionalDensity(
            key=key, unk_dim=unk_dim, obs_dim=obs_dim,
            n_layers=model_params.get("n_layers", 3),
            width_size=model_params.get("width_size", 32),
            depth=model_params.get("depth", 2))
    elif model_type == "flow":
        for req in ["unk_data_mean", "unk_data_std",
                    "obs_data_mean", "obs_data_std"]:
            if req not in model_params:
                raise ValueError(f"model_type='flow' requires '{req}'")
        return ConditionalFlowDensity(
            key=key, unk_dim=unk_dim, obs_dim=obs_dim,
            unk_data_mean=model_params["unk_data_mean"],
            unk_data_std=model_params["unk_data_std"],
            obs_data_mean=model_params["obs_data_mean"],
            obs_data_std=model_params["obs_data_std"],
            vector_field_params=model_params.get("vector_field_params", {}))
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid: gaussian, gaussian_mixture, discrete_flow, flow.")
