#!/usr/bin/env python
"""Mock data generator for partial-observation Deep Potential.

Generates 6D Plummer sphere samples, partitions them into observed and
unobserved dimensions according to a DimSpec, and saves the results along
with analytical ground truth for validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import h5py

# Allow importing from parent scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plummer.plummer_sphere as ps
from partial_obs.dim_spec import DimSpec


def generate_plummer_partial_obs(
    n_samples: int,
    dim_spec: DimSpec,
    r_max: float = 8.0,
    seed: int = 0,
) -> dict:
    """Generate partial-observation mock data from a Plummer sphere.

    Args:
        n_samples: Number of 6D phase-space samples to generate.
        dim_spec: Which dimensions are observed.
        r_max: Maximum radius for Plummer sampling (sharp cut-off).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - eta_obs: (n_samples, obs_dim) observed phase-space coords
            - eta_unk: (n_samples, unk_dim) unobserved phase-space coords
            - eta_full: (n_samples, 6) full 6D coords (for validation)
            - phi_true: (n_samples,) analytical potential Φ(r)
            - rho_true: (n_samples,) analytical density ρ(r)
            - r: (n_samples,) radial distance
    """
    rng = np.random.default_rng(seed)

    # Sample from unit Plummer sphere
    plummer = ps.UnitPlummerSphere(r_max=r_max)
    eta_full = plummer.sample_df(n_samples, rng=rng)

    # Compute ground truth
    x = eta_full[:, :3]
    r = np.sqrt(np.sum(x**2, axis=1))
    phi_true = np.array(plummer.phi(r))
    rho_true = np.array(plummer.rho(r))

    # Split into observed and unobserved
    eta_obs, eta_unk = dim_spec.split_eta(eta_full)

    return {
        "eta_obs": eta_obs.astype(np.float32),
        "eta_unk": eta_unk.astype(np.float32),
        "eta_full": eta_full.astype(np.float32),
        "phi_true": phi_true.astype(np.float32),
        "rho_true": rho_true.astype(np.float32),
        "r": r.astype(np.float32),
    }


def save_partial_obs_data(data: dict, output_dir: Path, dim_spec: DimSpec):
    """Save partial-observation data to HDF5 files.

    Produces:
        output_dir/obs_data.h5    — observed dimensions (training input)
        output_dir/unk_data.h5    — unobserved dimensions (validation)
        output_dir/ground_truth.h5 — analytical Φ, ρ, r

    Args:
        data: Dictionary from generate_plummer_partial_obs.
        output_dir: Directory to save files in.
        dim_spec: Dimension specification for metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Observed data (training input for p_obs)
    with h5py.File(output_dir / "obs_data.h5", "w") as f:
        dset = f.create_dataset("eta", data=data["eta_obs"], compression="lzf",
                                chunks=True)
        dset.attrs["dim_spec_str"] = "".join(dim_spec.obs_labels())
        dset.attrs["obs_indices"] = list(dim_spec.obs_indices)
        dset.attrs["unk_indices"] = list(dim_spec.unk_indices)
        dset.attrs["n_samples"] = data["eta_obs"].shape[0]
        dset.attrs["obs_dim"] = dim_spec.obs_dim
        dset.attrs["unk_dim"] = dim_spec.unk_dim
        # Also save uniform weights for compatibility with existing data loading
        f.create_dataset("weights", data=np.ones(data["eta_obs"].shape[0],
                                                  dtype=np.float32),
                         compression="lzf", chunks=True)

    # Unobserved data (validation)
    with h5py.File(output_dir / "unk_data.h5", "w") as f:
        f.create_dataset("eta", data=data["eta_unk"], compression="lzf",
                         chunks=True)
        f.create_dataset("eta_full", data=data["eta_full"], compression="lzf",
                         chunks=True)
        f.attrs["dim_spec_str"] = "".join(dim_spec.unk_labels())

    # Ground truth
    with h5py.File(output_dir / "ground_truth.h5", "w") as f:
        f.create_dataset("phi", data=data["phi_true"], compression="lzf",
                         chunks=True)
        f.create_dataset("rho", data=data["rho_true"], compression="lzf",
                         chunks=True)
        f.create_dataset("r", data=data["r"], compression="lzf", chunks=True)
        f.create_dataset("eta_full", data=data["eta_full"], compression="lzf",
                         chunks=True)

    # Full pairs (obs + unk together) for p_unk pre-training
    with h5py.File(output_dir / "full_pairs.h5", "w") as f:
        f.create_dataset("eta_obs", data=data["eta_obs"], compression="lzf",
                         chunks=True)
        f.create_dataset("eta_unk", data=data["eta_unk"], compression="lzf",
                         chunks=True)
        f.attrs["dim_spec_str"] = "".join(dim_spec.obs_labels())

    print(f"Saved partial-obs data to {output_dir}:")
    print(f"  obs_data.h5:   {data['eta_obs'].shape} (observed)")
    print(f"  unk_data.h5:   {data['eta_unk'].shape} (unobserved)")
    print(f"  ground_truth.h5: analytical Φ, ρ, r")
    print(f"  full_pairs.h5: paired obs+unk for convenience")


def load_partial_obs_data(data_dir: Path) -> dict:
    """Load previously generated partial-observation data.

    Args:
        data_dir: Directory containing the HDF5 files.

    Returns:
        Dictionary with the same structure as generate_plummer_partial_obs output.
    """
    data = {}
    with h5py.File(data_dir / "obs_data.h5", "r") as f:
        data["eta_obs"] = f["eta"][:]
    with h5py.File(data_dir / "unk_data.h5", "r") as f:
        data["eta_unk"] = f["eta"][:]
        data["eta_full"] = f["eta_full"][:]
    with h5py.File(data_dir / "ground_truth.h5", "r") as f:
        data["phi_true"] = f["phi"][:]
        data["rho_true"] = f["rho"][:]
        data["r"] = f["r"][:]
        if "eta_full" in f:
            data["eta_full"] = f["eta_full"][:]
    return data


def main():
    """CLI entry point for generating mock Plummer partial-obs data."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Generate partial-observation mock data from a Plummer sphere.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-samples", type=int, default=100_000,
        help="Number of 6D phase-space samples."
    )
    parser.add_argument(
        "--dim-spec", type=str, default="xyvz",
        help="Observed dimensions, e.g. 'xyvz' for x,y,v_z observed."
    )
    parser.add_argument(
        "--r-max", type=float, default=8.0,
        help="Maximum radius for Plummer sampling."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/partial_obs",
        help="Directory to save output files."
    )
    args = parser.parse_args()

    dim_spec = DimSpec.from_string(args.dim_spec)
    print(f"Generating mock data with {dim_spec}")
    print(f"  {args.n_samples:,} samples, r_max={args.r_max}, seed={args.seed}")

    data = generate_plummer_partial_obs(
        n_samples=args.n_samples,
        dim_spec=dim_spec,
        r_max=args.r_max,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    save_partial_obs_data(data, output_dir, dim_spec)
    print("Done.")


if __name__ == "__main__":
    main()
