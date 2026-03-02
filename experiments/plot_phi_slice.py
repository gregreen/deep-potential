from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dpjax.data import Normalizer
from dpjax.models.potential import PotentialConfig, PotentialMLP
from dpjax.utils.ckpt import create_manager, restore_latest


def _load_normalizer(df_run_dir: Path) -> Normalizer:
    p = df_run_dir / "normalizer.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return Normalizer.load_npz(p)


def _load_phi(phi_run_dir: Path) -> tuple[PotentialMLP, dict]:
    cfg_path = phi_run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text())
    pot_cfg = cfg.get("potential", {})
    model = PotentialMLP(PotentialConfig(hidden_sizes=tuple(int(x) for x in pot_cfg.get("hidden_sizes", [256, 256, 256]))))

    ckpt_mgr = create_manager(phi_run_dir / "ckpt")
    restored = restore_latest(ckpt_mgr)
    params = restored["params"]

    return model, params


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phi/rho/acc slices (TF-free) for JAX potential.")
    parser.add_argument("--df-run-dir", type=str, required=True)
    parser.add_argument("--phi-run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--rmax", type=float, default=5.0)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2048)

    args = parser.parse_args()

    df_run_dir = Path(args.df_run_dir)
    phi_run_dir = Path(args.phi_run_dir)

    normalizer = _load_normalizer(df_run_dir)
    phi_model, phi_params = _load_phi(phi_run_dir)

    out_dir = Path(args.out_dir) if args.out_dir else (phi_run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)
    std_x = np.asarray(normalizer.std[:3], dtype=np.float32)

    # Grid in physical coords
    rmax = float(args.rmax)
    grid = int(args.grid)
    xs = np.linspace(-rmax, rmax, grid, dtype=np.float32)
    ys = np.linspace(-rmax, rmax, grid, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    xyz = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(args.z), dtype=np.float32)], axis=-1)
    xyz_std = (xyz - mean_x[None, :]) / std_x[None, :]

    def phi_single(xi: jnp.ndarray) -> jnp.ndarray:
        return phi_model.apply({"params": phi_params}, xi)

    grad_fn = jax.grad(phi_single)
    hess_fn = jax.hessian(phi_single)

    @jax.jit
    def eval_batch(x_std_b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        phi_b = jax.vmap(phi_single)(x_std_b)
        grad_std_b = jax.vmap(grad_fn)(x_std_b)  # dPhi/dx_std
        hess_std_b = jax.vmap(hess_fn)(x_std_b)  # d2Phi/dx_std^2

        # Convert derivatives to physical units
        grad_phys_b = grad_std_b / jnp.asarray(std_x)[None, :]

        # Laplacian in physical coords: sum_i d2Phi/dx_i^2
        diag_std = jnp.stack([hess_std_b[:, 0, 0], hess_std_b[:, 1, 1], hess_std_b[:, 2, 2]], axis=-1)
        lap_phys_b = jnp.sum(diag_std / (jnp.asarray(std_x) ** 2)[None, :], axis=-1)

        acc_mag_b = jnp.linalg.norm(-grad_phys_b, axis=-1)
        rho_b = lap_phys_b / (4.0 * jnp.pi)

        return phi_b, acc_mag_b, rho_b

    # Batched evaluation
    batch = int(args.batch)
    n = xyz_std.shape[0]
    phi_all: list[np.ndarray] = []
    acc_all: list[np.ndarray] = []
    rho_all: list[np.ndarray] = []

    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        phi_b, acc_b, rho_b = eval_batch(jnp.asarray(xyz_std[sl]))
        phi_all.append(np.asarray(phi_b, dtype=np.float32))
        acc_all.append(np.asarray(acc_b, dtype=np.float32))
        rho_all.append(np.asarray(rho_b, dtype=np.float32))

    phi_img = np.concatenate(phi_all).reshape(X.shape)
    acc_img = np.concatenate(acc_all).reshape(X.shape)
    rho_img = np.concatenate(rho_all).reshape(X.shape)

    # Plot
    import matplotlib.pyplot as plt
    from matplotlib import colors

    # Phi (mean-subtracted)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=150)
    phi0 = phi_img - np.nanmean(phi_img)
    vmin, vmax = np.nanpercentile(phi0, [1, 99])
    if vmin * vmax < 0:
        divnorm = colors.TwoSlopeNorm(vcenter=0.0, vmin=float(vmin), vmax=float(vmax))
        im = ax.imshow(phi0, extent=[-rmax, rmax, -rmax, rmax], origin="lower", cmap="seismic", norm=divnorm)
    else:
        im = ax.imshow(phi0, extent=[-rmax, rmax, -rmax, rmax], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\Phi(x,y)$ (mean-subtracted)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_slice_xy.png")
    plt.close(fig)

    # rho (log)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=150)
    rho_pos = np.clip(rho_img, 1e-12, np.inf)
    im = ax.imshow(
        rho_pos,
        extent=[-rmax, rmax, -rmax, rmax],
        origin="lower",
        cmap="magma",
        norm=colors.LogNorm(vmin=np.nanpercentile(rho_pos, 5), vmax=np.nanpercentile(rho_pos, 99)),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\rho(x,y)=\nabla^2\Phi/(4\pi)$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_slice_xy.png")
    plt.close(fig)

    # |a|
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=150)
    im = ax.imshow(
        acc_img,
        extent=[-rmax, rmax, -rmax, rmax],
        origin="lower",
        cmap="cubehelix",
        norm=colors.LogNorm(vmin=np.nanpercentile(acc_img, 5), vmax=np.nanpercentile(acc_img, 99)),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$|a(x,y)|$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "accmag_slice_xy.png")
    plt.close(fig)

    np.savez(out_dir / "phi_slice_xy.npz", x=xs, y=ys, phi=phi_img, rho=rho_img, acc_mag=acc_img)

    print(f"Wrote slice plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
