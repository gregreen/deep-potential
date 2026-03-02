from __future__ import annotations

import jax.numpy as jnp

from dpjax.data import Normalizer


def residual_A(
    eta_std: jnp.ndarray,
    score_std: jnp.ndarray,
    grad_phi_std: jnp.ndarray,
    normalizer: Normalizer,
) -> jnp.ndarray:
    """Collisionless Boltzmann residual (form A) in physical units.

    r(eta) = v · ∇_x log f(eta) - ∇_x Φ(x) · ∇_v log f(eta)

    Inputs are in standardized coordinates; this function applies the
    chain-rule rescalings defined in plan.md.

    Shapes:
      eta_std: (N, 6)
      score_std: (N, 6) = ∇_{eta_std} log f
      grad_phi_std: (N, 3) = ∇_{x_std} Φ
    """

    mean = jnp.asarray(normalizer.mean)
    std = jnp.asarray(normalizer.std)

    std_x = std[:3]
    std_v = std[3:]
    mean_v = mean[3:]

    v_std = eta_std[:, 3:]
    v_phys = v_std * std_v + mean_v

    score_x_std = score_std[:, :3]
    score_v_std = score_std[:, 3:]

    score_x_phys = score_x_std / std_x
    score_v_phys = score_v_std / std_v

    grad_phi_phys = grad_phi_std / std_x

    term1 = jnp.sum(v_phys * score_x_phys, axis=-1)
    term2 = jnp.sum(grad_phi_phys * score_v_phys, axis=-1)
    return term1 - term2


def loss_cbe_A(
    eta_std: jnp.ndarray,
    score_std: jnp.ndarray,
    grad_phi_std: jnp.ndarray,
    normalizer: Normalizer,
) -> jnp.ndarray:
    r = residual_A(eta_std, score_std, grad_phi_std, normalizer)
    return jnp.mean(r**2)
