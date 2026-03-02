from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


def _standard_normal_log_prob(z: jnp.ndarray) -> jnp.ndarray:
    dim = z.shape[-1]
    return -0.5 * (jnp.sum(z**2, axis=-1) + dim * jnp.log(2.0 * jnp.pi))


class _MLP(nn.Module):
    hidden_sizes: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, width in enumerate(self.hidden_sizes):
            x = nn.Dense(width, name=f"dense_{i}")(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.out_dim,
            name="dense_out",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)
        return x


class AffineCoupling(nn.Module):
    mask: jnp.ndarray  # (dim,) with 1 for identity dims
    hidden_sizes: Sequence[int]
    s_max: float = 2.0

    def setup(self) -> None:
        dim = int(self.mask.shape[0])
        self.nn = _MLP(hidden_sizes=self.hidden_sizes, out_dim=2 * dim)

    def _st(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        masked = x * self.mask
        out = self.nn(masked)
        t, raw_s = jnp.split(out, 2, axis=-1)
        s = self.s_max * jnp.tanh(raw_s)
        inv_mask = 1.0 - self.mask
        t = t * inv_mask
        s = s * inv_mask
        return s, t

    def __call__(self, x: jnp.ndarray, *, inverse: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        s, t = self._st(x)
        inv_mask = 1.0 - self.mask

        if inverse:
            y = x * self.mask + ((x - t) * jnp.exp(-s)) * inv_mask
            log_det = -jnp.sum(s, axis=-1)
        else:
            y = x * self.mask + (x * jnp.exp(s) + t) * inv_mask
            log_det = jnp.sum(s, axis=-1)

        return y, log_det


@dataclass(frozen=True)
class RealNVPConfig:
    dim: int = 6
    n_coupling: int = 10
    hidden_sizes: tuple[int, ...] = (128, 128)
    s_max: float = 2.0


class RealNVP(nn.Module):
    cfg: RealNVPConfig

    def setup(self) -> None:
        if self.cfg.dim != 6:
            raise ValueError("This implementation currently assumes dim=6 (x,y,z,vx,vy,vz).")

        mask0 = jnp.array([1, 1, 1, 0, 0, 0], dtype=jnp.float32)
        mask1 = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.float32)
        masks = [mask0 if (i % 2 == 0) else mask1 for i in range(self.cfg.n_coupling)]

        self.couplings = [
            AffineCoupling(mask=m, hidden_sizes=self.cfg.hidden_sizes, s_max=self.cfg.s_max, name=f"coupling_{i}")
            for i, m in enumerate(masks)
        ]

    @staticmethod
    def _permute(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.flip(x, axis=-1)

    def forward(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = z
        log_det = jnp.zeros(z.shape[:-1], dtype=z.dtype)
        for i, c in enumerate(self.couplings):
            x, ld = c(x, inverse=False)
            log_det = log_det + ld
            x = self._permute(x)
        return x, log_det

    def inverse(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        z = x
        log_det = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        for c in reversed(self.couplings):
            z = self._permute(z)
            z, ld = c(z, inverse=True)
            log_det = log_det + ld
        return z, log_det

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        z, log_det_inv = self.inverse(x)
        return _standard_normal_log_prob(z) + log_det_inv

    def sample(self, rng: jax.Array, n: int) -> jnp.ndarray:
        z = jax.random.normal(rng, shape=(n, self.cfg.dim), dtype=jnp.float32)
        x, _ = self.forward(z)
        return x


def log_prob_apply(flow: RealNVP, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    return flow.apply({"params": params}, x, method=RealNVP.log_prob)


def score_apply(flow: RealNVP, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Compute score = ∇_x log_prob(x) for a batch x with shape (N, dim)."""

    def lp_single(xi: jnp.ndarray) -> jnp.ndarray:
        return flow.apply({"params": params}, xi, method=RealNVP.log_prob)

    return jax.vmap(jax.grad(lp_single))(x)
