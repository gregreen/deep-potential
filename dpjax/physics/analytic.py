from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def plummer_phi(r: np.ndarray) -> np.ndarray:
    """Analytic Plummer potential: Φ(r) = -(1 + r^2)^(-1/2)."""
    return -(1.0 + r**2) ** (-0.5)


def plummer_ar(r: np.ndarray) -> np.ndarray:
    """Signed radial acceleration: a_r = -dΦ/dr = -r * (1 + r^2)^(-3/2)."""
    return -r * (1.0 + r**2) ** (-1.5)


# ---------------------------------------------------------------------------
# JAX-based Plummer DF score functions (used for flow diagnostics)
# ---------------------------------------------------------------------------

def plummer_score_std_batch(
    eta_std_batch: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
) -> jnp.ndarray:
    """Analytic ∇_{η_std} log f for the Plummer DF, vectorised over a batch.

    Parameters
    ----------
    eta_std_batch : (N, 6) standardised phase-space coordinates
    mean, std : (6,) normalizer statistics (physical → standardised)

    Returns
    -------
    (N, 6) score field in standardised coordinates
    """
    mean = jnp.asarray(mean, dtype=jnp.float32)
    std = jnp.asarray(std, dtype=jnp.float32)

    def _logf_single(eta_std_single: jnp.ndarray) -> jnp.ndarray:
        eta = eta_std_single * std + mean
        x = eta[:3]
        v = eta[3:]
        r2 = jnp.sum(x**2)
        v2 = jnp.sum(v**2)
        phi = -(1.0 + r2) ** (-0.5)
        E = 0.5 * v2 + phi
        f = jnp.clip(-E, 1.0e-12, jnp.inf) ** 3.5
        return jnp.log(f + 1.0e-30)

    return jax.vmap(jax.grad(_logf_single))(eta_std_batch)


def plummer_score_phys_batch(
    eta_phys_batch: jnp.ndarray,
) -> jnp.ndarray:
    """Analytic ∇_{η_phys} log f for the Plummer DF, vectorised over a batch.

    Parameters
    ----------
    eta_phys_batch : (N, 6) physical phase-space coordinates

    Returns
    -------
    (N, 6) score field in physical coordinates
    """

    def _logf_phys_single(eta_single: jnp.ndarray) -> jnp.ndarray:
        x = eta_single[:3]
        v = eta_single[3:]
        r2 = jnp.sum(x**2)
        v2 = jnp.sum(v**2)
        phi = -(1.0 + r2) ** (-0.5)
        E = 0.5 * v2 + phi
        A = 24.0 * jnp.sqrt(2.0) / (7.0 * jnp.pi**3)
        f = A * jnp.clip(-E, 1.0e-12, jnp.inf) ** 3.5
        return jnp.log(f + 1.0e-30)

    return jax.vmap(jax.grad(_logf_phys_single))(eta_phys_batch)


def plummer_rv_ideal_grid(
    r_lim: Tuple[float, float] = (0.0, 5.0),
    v_lim: Tuple[float, float] = (0.0, 1.5),
    bins: Tuple[int, int] = (50, 50),
) -> dict:
    """Compute the ideal Plummer (r, v) density grid for diagnostic plots.

    Returns
    -------
    dict with keys: ``r``, ``v``, ``r_edges``, ``v_edges``, ``n_ideal``
    """
    r_edges = np.linspace(r_lim[0], r_lim[1], bins[0] + 1)
    v_edges = np.linspace(v_lim[0], v_lim[1], bins[1] + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    v = 0.5 * (v_edges[:-1] + v_edges[1:])
    rr, vv = np.meshgrid(r, v)

    psi = 1.0 / np.sqrt(1.0 + rr**2)
    E = psi - vv**2 / 2.0
    df = np.clip(E, 0.0, np.inf) ** (7.0 / 2.0)
    A = 24.0 * np.sqrt(2.0) / (7.0 * np.pi**3)
    n_ideal = A * (4.0 * np.pi) ** 2 * rr**2 * vv**2 * df

    return {
        "r": r,
        "v": v,
        "r_edges": r_edges,
        "v_edges": v_edges,
        "n_ideal": n_ideal,
    }
