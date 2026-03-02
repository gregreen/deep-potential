from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class PotentialConfig:
    hidden_sizes: tuple[int, ...] = (256, 256, 256)


class PotentialMLP(nn.Module):
    cfg: PotentialConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 3)
        h = x
        for i, width in enumerate(self.cfg.hidden_sizes):
            h = nn.Dense(width, name=f"dense_{i}")(h)
            h = nn.tanh(h)
        out = nn.Dense(1, name="dense_out")(h)
        return jnp.squeeze(out, axis=-1)


def phi_apply(model: PotentialMLP, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    return model.apply({"params": params}, x)


def grad_phi_apply(model: PotentialMLP, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Compute ∇_x Φ(x) for x with shape (N, 3)."""

    def phi_single(xi: jnp.ndarray) -> jnp.ndarray:
        return model.apply({"params": params}, xi)

    return jax.vmap(jax.grad(phi_single))(x)
