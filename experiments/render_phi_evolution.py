from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dpjax.flows.api import load_df
from dpjax.models.potential import grad_phi_apply, load_phi, phi_apply
from dpjax.physics.analytic import plummer_ar, plummer_phi
from dpjax.utils.ckpt import create_manager, restore_step


def _sorted_steps(ckpt_dir: Path) -> list[int]:
    steps: list[int] = []
    for child in ckpt_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(int(child.name))
    return sorted(steps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Phi evolution frames from checkpoints.")
    parser.add_argument("--df-run-dir", type=str, required=True)
    parser.add_argument("--phi-run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument("--r-min", type=float, default=1.0e-3)
    parser.add_argument("--r-max", type=float, default=10.0)
    parser.add_argument("--n-r", type=int, default=256)
    parser.add_argument("--r-ref", type=float, default=1.0)

    args = parser.parse_args()

    df_model, df_params, normalizer, _ = load_df(args.df_run_dir)
    phi_model, _, _ = load_phi(args.phi_run_dir)

    ckpt_dir = Path(args.phi_run_dir) / "ckpt"
    steps = _sorted_steps(ckpt_dir)
    if len(steps) == 0:
        raise ValueError(f"No checkpoints found in {ckpt_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.phi_run_dir) / "frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Radial grid along x-axis
    r = np.geomspace(args.r_min, args.r_max, num=int(args.n_r)).astype(np.float32)
    x_phys = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=-1)

    std_x = np.asarray(normalizer.std[:3], dtype=np.float32)
    mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)
    x_std = (x_phys - mean_x[None, :]) / std_x[None, :]

    x_std_j = jnp.asarray(x_std)

    phi_true = plummer_phi(r)
    ar_true = plummer_ar(r)

    i_ref = int(np.argmin(np.abs(r - float(args.r_ref))))
    phi_true_ref = float(plummer_phi(np.array([float(args.r_ref)], dtype=np.float32))[0])

    import matplotlib.pyplot as plt

    ckpt_mgr = create_manager(ckpt_dir)

    for step in steps:
        restored = restore_step(ckpt_mgr, step)
        phi_params = restored["params"]

        phi_learned = np.asarray(phi_apply(phi_model, phi_params, x_std_j)).astype(np.float32)
        grad_phi_std = np.asarray(grad_phi_apply(phi_model, phi_params, x_std_j)).astype(np.float32)
        grad_phi_phys = grad_phi_std / std_x[None, :]
        ar_learned = -grad_phi_phys[:, 0]

        phi_shift = phi_learned - phi_learned[i_ref] + phi_true_ref

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

        ax1.plot(r, phi_true, label="analytic", lw=2)
        ax1.plot(r, phi_shift, label="learned (shifted)", lw=1.5)
        ax1.set_xscale("log")
        ax1.set_xlabel("r")
        ax1.set_ylabel("Phi(r)")
        ax1.set_title(f"step {step}")
        ax1.grid(True, alpha=0.2)
        ax1.legend()

        ax2.plot(r, ar_true, label="analytic", lw=2)
        ax2.plot(r, ar_learned, label="learned", lw=1.5)
        ax2.set_xscale("log")
        ax2.set_xlabel("r")
        ax2.set_ylabel("a_r(r)")
        ax2.grid(True, alpha=0.2)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(out_dir / f"phi_training_{step:06d}.png")
        plt.close(fig)

    print(f"Wrote {len(steps)} frames to {out_dir}")
    print("To make a video:")
    print(f"ffmpeg -y -r 10 -pattern_type glob -i '{out_dir}/phi_training_*.png' -c:v libx264 -vf fps=10 -pix_fmt yuv420p {out_dir}/phi_training.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
