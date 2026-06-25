#!/usr/bin/env python
"""Dimension specification for partial-observation phase-space configurations.

Defines which of the 6 Cartesian phase-space dimensions are observed vs. unobserved.
Index convention: 0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import jax.numpy as jnp
from jaxtyping import Array


# Mapping from string labels to indices and vice versa
_LABEL_TO_INDEX: dict[str, int] = {
    "x": 0, "y": 1, "z": 2,
    "vx": 3, "vy": 4, "vz": 5,
}
_INDEX_TO_LABEL: list[str] = ["x", "y", "z", "vx", "vy", "vz"]
_TOTAL_DIM: int = 6


class DimSpec:
    """Specifies which phase-space dimensions are observed vs. unobserved.

    The 6D Cartesian phase space is indexed as:
      0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz

    Attributes:
        obs_indices: Sorted list of observed dimension indices.
        unk_indices: Sorted list of unobserved dimension indices.
        obs_dim: Number of observed dimensions.
        unk_dim: Number of unobserved dimensions.
    """

    obs_indices: tuple[int, ...]
    unk_indices: tuple[int, ...]

    def __init__(self, obs_indices: List[int]):
        """Construct from a list of observed dimension indices.

        Args:
            obs_indices: List of integer indices (0-5) that are observed.
                         Unobserved dimensions are inferred as the complement.

        Raises:
            ValueError: If indices are not unique or out of range.
        """
        if len(set(obs_indices)) != len(obs_indices):
            raise ValueError("Duplicate indices in obs_indices")
        for i in obs_indices:
            if i < 0 or i >= _TOTAL_DIM:
                raise ValueError(f"Index {i} out of range [0, {_TOTAL_DIM})")

        obs_set = set(obs_indices)
        unk = tuple(sorted(set(range(_TOTAL_DIM)) - obs_set))

        self.obs_indices = tuple(sorted(obs_indices))
        self.unk_indices = unk

    # ---- Convenience constructors ----

    @classmethod
    def from_string(cls, spec: str) -> "DimSpec":
        """Parse a string specification like "xyvz" or "xyzvxvyvz".

        Supported tokens: "x", "y", "z", "vx", "vy", "vz".
        Tokens are parsed greedily: "vx" is tried before "v" + "x".

        Example:
            DimSpec.from_string("xyvz")  -> obs=[0,1,5], unk=[2,3,4]
            DimSpec.from_string("xyz")   -> obs=[0,1,2], unk=[3,4,5]
        """
        indices = []
        remaining = spec.strip().lower()
        # Try longer tokens first ("vx", "vy", "vz") then single chars
        tokens = ["vx", "vy", "vz", "x", "y", "z"]
        while remaining:
            matched = False
            for tok in tokens:
                if remaining.startswith(tok):
                    indices.append(_LABEL_TO_INDEX[tok])
                    remaining = remaining[len(tok):]
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Unrecognized token in spec string '{spec}' "
                    f"at position '{remaining}'. "
                    f"Valid tokens: x, y, z, vx, vy, vz."
                )
        return cls(indices)

    @classmethod
    def from_mask(cls, mask: List[bool]) -> "DimSpec":
        """Construct from a boolean mask of length 6."""
        if len(mask) != _TOTAL_DIM:
            raise ValueError(f"Mask must have length {_TOTAL_DIM}")
        indices = [i for i, m in enumerate(mask) if m]
        return cls(indices)

    @classmethod
    def full_6d(cls) -> "DimSpec":
        """All 6 dimensions observed."""
        return cls(list(range(_TOTAL_DIM)))

    # ---- Properties ----

    @property
    def obs_dim(self) -> int:
        return len(self.obs_indices)

    @property
    def unk_dim(self) -> int:
        return len(self.unk_indices)

    @property
    def spatial_obs_indices(self) -> tuple[int, ...]:
        """Observed indices that are spatial (0,1,2)."""
        return tuple(i for i in self.obs_indices if i < 3)

    @property
    def velocity_obs_indices(self) -> tuple[int, ...]:
        """Observed indices that are velocity (3,4,5)."""
        return tuple(i for i in self.obs_indices if i >= 3)

    @property
    def spatial_unk_indices(self) -> tuple[int, ...]:
        """Unobserved indices that are spatial (0,1,2)."""
        return tuple(i for i in self.unk_indices if i < 3)

    @property
    def velocity_unk_indices(self) -> tuple[int, ...]:
        """Unobserved indices that are velocity (3,4,5)."""
        return tuple(i for i in self.unk_indices if i >= 3)

    # ---- Data manipulation ----

    def split_eta(self, eta: Array) -> Tuple[Array, Array]:
        """Split a full 6D phase-space array into (obs, unk) parts.

        Args:
            eta: Array of shape (..., 6) with full phase-space coordinates.

        Returns:
            (eta_obs, eta_unk) each of shape (..., obs_dim) and (..., unk_dim).
        """
        eta_obs = eta[..., list(self.obs_indices)]
        eta_unk = eta[..., list(self.unk_indices)]
        return eta_obs, eta_unk

    def combine_eta(self, eta_obs: Array, eta_unk: Array) -> Array:
        """Combine observed and unobserved parts into full 6D.

        Args:
            eta_obs: Array of shape (..., obs_dim).
            eta_unk: Array of shape (..., unk_dim).

        Returns:
            Full 6D array of shape (..., 6).
        """
        # Build full array by placing each component in its correct slot
        full = jnp.zeros(eta_obs.shape[:-1] + (6,), dtype=eta_obs.dtype)
        for idx, i in enumerate(self.obs_indices):
            full = full.at[..., i].set(eta_obs[..., idx])
        for idx, i in enumerate(self.unk_indices):
            full = full.at[..., i].set(eta_unk[..., idx])
        return full

    def scatter_obs_gradient_to_6d(self, grad_obs: Array) -> Array:
        """Map an obs-dim gradient to a full 6D gradient (zeros in unk slots).

        Args:
            grad_obs: Array of shape (..., obs_dim).

        Returns:
            Array of shape (..., 6) with zeros in unobserved positions.
        """
        full = jnp.zeros(grad_obs.shape[:-1] + (6,), dtype=grad_obs.dtype)
        for idx, i in enumerate(self.obs_indices):
            full = full.at[..., i].set(grad_obs[..., idx])
        return full

    def scatter_unk_gradient_to_6d(self, grad_unk: Array) -> Array:
        """Map an unk-dim gradient to a full 6D gradient (zeros in obs slots).

        Args:
            grad_unk: Array of shape (..., unk_dim).

        Returns:
            Array of shape (..., 6) with zeros in observed positions.
        """
        full = jnp.zeros(grad_unk.shape[:-1] + (6,), dtype=grad_unk.dtype)
        for idx, i in enumerate(self.unk_indices):
            full = full.at[..., i].set(grad_unk[..., idx])
        return full

    def partition_full_gradient(self, grad_6d: Array) -> Tuple[Array, Array]:
        """Split a full 6D gradient into (obs, unk) parts.

        Args:
            grad_6d: Array of shape (..., 6).

        Returns:
            (grad_obs, grad_unk) of shapes (..., obs_dim) and (..., unk_dim).
        """
        grad_obs = grad_6d[..., list(self.obs_indices)]
        grad_unk = grad_6d[..., list(self.unk_indices)]
        return grad_obs, grad_unk

    # ---- Display ----

    def obs_labels(self) -> list[str]:
        """Human-readable labels for observed dimensions."""
        return [_INDEX_TO_LABEL[i] for i in self.obs_indices]

    def unk_labels(self) -> list[str]:
        """Human-readable labels for unobserved dimensions."""
        return [_INDEX_TO_LABEL[i] for i in self.unk_indices]

    def __repr__(self) -> str:
        obs_str = ",".join(self.obs_labels())
        unk_str = ",".join(self.unk_labels())
        return f"DimSpec(obs=[{obs_str}], unk=[{unk_str}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimSpec):
            return NotImplemented
        return (self.obs_indices == other.obs_indices
                and self.unk_indices == other.unk_indices)

    def __hash__(self) -> int:
        return hash((self.obs_indices, self.unk_indices))
