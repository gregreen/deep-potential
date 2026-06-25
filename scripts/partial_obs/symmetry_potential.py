#!/usr/bin/env python
"""Symmetry-constrained potential models with trainable axis projection.

Provides:
    SymmetryAxis        — Trainable axis direction (no gimbal lock)
    SphericalPotential  — Φ(r) with 1D ResMLP
    CylindricalPotential— Φ(R, z) with 2D ResMLP
    ProjectedPotential  — Composes SymmetryAxis + symmetry potential
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx

# Allow importing from parent scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import potential as pot  # for ResMLP and calc_phi_derivatives


# =============================================================================
# SymmetryAxis: Trainable 3D axis direction without gimbal lock
# =============================================================================

class SymmetryAxis(eqx.Module):
    """Trainable symmetry axis direction for cylindrical potentials.

    Stores 3 unconstrained scalars. The actual axis direction is the normalized
    unit vector. This avoids gimbal lock (no angle parameterization) and has
    well-behaved gradients everywhere.

    The rotation matrix R maps the intrinsic z-axis to the symmetry axis n̂.
    Since only the axis direction matters for Φ(R,z), azimuthal orientation
    around the axis is arbitrary.

    Attributes:
        raw_vector: (3,) unconstrained trainable parameters.
    """

    raw_vector: Array

    def __init__(self, initial_axis: Array = jnp.array([0.0, 0.0, 1.0])):
        """Initialize with an approximate axis direction.

        Args:
            initial_axis: (3,) initial direction. Will be stored as-is
                          (normalization happens at call time).
        """
        self.raw_vector = jnp.asarray(initial_axis, dtype=jnp.float32)

    def get_axis(self) -> Array:
        """Return the normalized unit vector n̂.

        Returns:
            (3,) unit vector.
        """
        norm = jnp.linalg.norm(self.raw_vector) + 1e-8
        return self.raw_vector / norm

    def get_rotation_matrix(self) -> Array:
        """Construct a 3×3 rotation matrix R that maps ẑ → n̂.

        Uses the Rodrigues-like construction: pick an orthonormal basis
        (e₁, e₂, n̂) via cross products. The matrix columns are (e₁, e₂, n̂),
        which maps intrinsic (x̂, ŷ, ẑ) to the observer frame.

        Returns:
            (3, 3) rotation matrix.
        """
        n = self.get_axis()  # (3,)

        # Pick a reference vector not parallel to n
        z_hat = jnp.array([0.0, 0.0, 1.0])
        x_hat = jnp.array([1.0, 0.0, 0.0])

        # Use x̂ unless n is parallel to x̂, then use ẑ
        ref = jnp.where(
            jnp.abs(jnp.dot(n, x_hat)) > 0.99,
            z_hat,
            x_hat,
        )

        # Build orthonormal basis via Gram-Schmidt
        e1 = ref - jnp.dot(ref, n) * n
        e1 = e1 / (jnp.linalg.norm(e1) + 1e-8)
        e2 = jnp.cross(n, e1)

        # R = [e1, e2, n] as columns (maps intrinsic → observer)
        R = jnp.stack([e1, e2, n], axis=1)  # (3, 3)
        return R

    def rotate_to_intrinsic(self, x: Array) -> Array:
        """Rotate observer-frame positions to intrinsic frame.

        Applies Rᵀ: observer → intrinsic (where symmetry axis is z).

        Args:
            x: (..., 3) positions in observer frame.

        Returns:
            (..., 3) positions in intrinsic frame.
        """
        R = self.get_rotation_matrix()  # (3, 3)
        return x @ R  # R^T applied as x @ R (since R columns are basis vectors)

    def count_parameters(self) -> int:
        return 3  # raw_vector


# =============================================================================
# SphericalPotential: Φ(r)
# =============================================================================

class SphericalPotential(eqx.Module):
    """Spherically symmetric potential Φ(r) where r = |x|.

    A 1D ResMLP maps r → Φ. The gradient ∇Φ and Laplacian ∇²Φ are computed
    via autodiff through the full 3D input.
    """

    net: pot.ResMLP
    r_scale: Array  # non-trainable

    def __init__(
        self,
        key: PRNGKeyArray,
        width_size: int = 64,
        depth: int = 3,
        r_scale: float = 1.0,
    ):
        """Initialize spherical potential.

        Args:
            key: JAX random key.
            width_size: Hidden width of the ResMLP.
            depth: Number of residual blocks.
            r_scale: Scale factor for r normalization.
        """
        self.net = pot.ResMLP(
            in_size=1, out_size=1,
            width_size=width_size, depth=depth, key=key,
        )
        self.r_scale = jnp.array(r_scale)

    def __call__(self, x: Array) -> Array:
        """Evaluate Φ at a SINGLE position x.

        Args:
            x: (3,) position vector.

        Returns:
            Scalar potential value.
        """
        r = jnp.sqrt(jnp.sum(x**2))
        r_scaled = r / self.r_scale
        return self.net(r_scaled[None]).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(
            x.size
            for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        )


# =============================================================================
# CylindricalPotential: Φ(R, z)
# =============================================================================

class CylindricalPotential(eqx.Module):
    """Cylindrically symmetric potential Φ(R, z) in the intrinsic frame.

    The intrinsic z-axis is the symmetry axis (mapped from observer frame
    by SymmetryAxis). A 2D ResMLP maps (R, z) → Φ.
    """

    net: pot.ResMLP
    R_scale: Array
    z_scale: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        width_size: int = 64,
        depth: int = 3,
        R_scale: float = 1.0,
        z_scale: float = 1.0,
    ):
        self.net = pot.ResMLP(
            in_size=2, out_size=1,
            width_size=width_size, depth=depth, key=key,
        )
        self.R_scale = jnp.array(R_scale)
        self.z_scale = jnp.array(z_scale)

    def __call__(self, x: Array) -> Array:
        """Evaluate Φ at a SINGLE position in the intrinsic frame.

        Args:
            x: (3,) position in intrinsic frame (z is symmetry axis).

        Returns:
            Scalar potential value.
        """
        R = jnp.sqrt(x[0]**2 + x[1]**2)
        z = x[2]
        inp = jnp.array([R / self.R_scale, z / self.z_scale])
        return self.net(inp).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(
            x.size
            for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        )


# =============================================================================
# ProjectedPotential: rotation + symmetry
# =============================================================================

class ProjectedPotential(eqx.Module):
    """Combines a symmetry potential with a trainable axis projection.

    For spherical symmetry, no axis is needed (rotation is identity).
    For cylindrical symmetry, SymmetryAxis rotates observer → intrinsic frame
    before evaluating the cylindrical potential.

    Gradients ∇Φ and ∇²Φ flow through the rotation into both the symmetry
    potential and the axis parameters.
    """

    symmetry_type: str = eqx.field(static=True)
    symmetry_potential: eqx.Module  # SphericalPotential or CylindricalPotential
    axis: SymmetryAxis  # used only for cylindrical

    def __init__(
        self,
        key: PRNGKeyArray,
        symmetry_type: str,
        width_size: int = 64,
        depth: int = 3,
        initial_axis: Array = jnp.array([0.0, 0.0, 1.0]),
        r_scale: float = 1.0,
        R_scale: float = 1.0,
        z_scale: float = 1.0,
    ):
        """Initialize projected potential.

        Args:
            key: JAX random key.
            symmetry_type: "spherical" or "cylindrical".
            width_size: Hidden width of the symmetry MLP.
            depth: Number of residual blocks in the symmetry MLP.
            initial_axis: Initial guess for symmetry axis (cylindrical only).
            r_scale: Scale for radial coordinate normalization.
            R_scale: Scale for cylindrical R normalization.
            z_scale: Scale for cylindrical z normalization.
        """
        self.symmetry_type = symmetry_type

        if symmetry_type == "spherical":
            self.symmetry_potential = SphericalPotential(
                key=key, width_size=width_size, depth=depth,
                r_scale=r_scale,
            )
            self.axis = SymmetryAxis(initial_axis=initial_axis)
        elif symmetry_type == "cylindrical":
            self.symmetry_potential = CylindricalPotential(
                key=key, width_size=width_size, depth=depth,
                R_scale=R_scale, z_scale=z_scale,
            )
            self.axis = SymmetryAxis(initial_axis=initial_axis)
        else:
            raise ValueError(
                f"Unknown symmetry_type '{symmetry_type}'. "
                f"Valid: spherical, cylindrical."
            )

    def __call__(self, x: Array) -> Array:
        """Evaluate Φ at observer-frame position x.

        Args:
            x: (..., 3) positions in observer frame.

        Returns:
            (...,) potential values.
        """
        if self.symmetry_type == "spherical":
            return self.symmetry_potential(x)
        else:
            x_int = self.axis.rotate_to_intrinsic(x)
            return self.symmetry_potential(x_int)

    def calc_phi_derivatives(self, q: Array) -> Tuple[Array, Array]:
        """Calculate ∇Φ and ∇²Φ at positions q.

        Args:
            q: (..., 3) positions.

        Returns:
            (dphi_dq, d2phi_dq2) each of shape (..., 3) and (...,).
        """
        grad_fn = jax.vmap(jax.grad(self.__call__))
        dphi_dq = grad_fn(q)  # (..., 3)

        hess_fn = jax.vmap(jax.hessian(self.__call__))
        hessian = hess_fn(q)  # (..., 3, 3)
        d2phi_dq2 = jnp.trace(hessian, axis1=-2, axis2=-1)  # (...,)

        return dphi_dq, d2phi_dq2

    def count_parameters(self) -> int:
        return self.symmetry_potential.count_parameters() + self.axis.count_parameters()
