from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml
from flax import linen as nn

from dpjax.utils.ckpt import create_manager, restore_latest


@dataclass(frozen=True)
class PotentialConfig:
    hidden_sizes: tuple[int, ...] = (512, 512, 512, 512)


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


def laplacian_phi_apply(
    model: PotentialMLP,
    params: dict,
    x: jnp.ndarray,
    *,
    std_x: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute ∇²Φ(x) for x with shape (N, 3).

    If ``std_x`` is provided, the Laplacian is converted from standardized
    coordinates to physical coordinates using chain-rule scaling.
    """

    def phi_single(xi: jnp.ndarray) -> jnp.ndarray:
        return model.apply({"params": params}, xi)

    hess = jax.vmap(jax.hessian(phi_single))(x)
    diag = jnp.stack([hess[:, 0, 0], hess[:, 1, 1], hess[:, 2, 2]], axis=-1)

    if std_x is None:
        return jnp.sum(diag, axis=-1)

    std_x = jnp.asarray(std_x, dtype=x.dtype)
    return jnp.sum(diag / (std_x[None, :] ** 2), axis=-1)


def load_phi(phi_run_dir: str | Path) -> tuple[PotentialMLP, dict, dict[str, Any]]:
    """Load a trained Phi model from *phi_run_dir*.

    Returns ``(model, params, full_config_dict)``.
    """

    phi_run_dir = Path(phi_run_dir)
    cfg_path = phi_run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text())
    pot_cfg = cfg.get("potential", {})
    model = PotentialMLP(
        PotentialConfig(
            hidden_sizes=tuple(
                int(x) for x in pot_cfg.get("hidden_sizes", [512, 512, 512, 512])
            )
        )
    )

    ckpt_mgr = create_manager(phi_run_dir / "ckpt")
    restored = restore_latest(ckpt_mgr)
    params = restored["params"]

    return model, params, cfg
