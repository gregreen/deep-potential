from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dpjax.data import Normalizer
from dpjax.flows.realnvp import RealNVP, RealNVPConfig
from dpjax.models.potential import PotentialConfig, PotentialMLP, grad_phi_apply, phi_apply
from dpjax.utils.ckpt import create_manager, restore_latest, restore_step


def _load_df(df_run_dir: Path) -> tuple[RealNVP, dict, Normalizer]:
    df_cfg_path = df_run_dir / "config.yaml"
    if not df_cfg_path.exists():
        raise FileNotFoundError(f"Missing {df_cfg_path}")

    df_cfg = yaml.safe_load(df_cfg_path.read_text())
    flow_cfg = df_cfg.get("flow", {})
    model = RealNVP(
        RealNVPConfig(
            dim=int(flow_cfg.get("dim", 6)),
            n_coupling=int(flow_cfg.get("n_coupling", 10)),
            hidden_sizes=tuple(int(x) for x in flow_cfg.get("hidden_sizes", [128, 128])),
            s_max=float(flow_cfg.get("s_max", 2.0)),
        )
    )

    norm = Normalizer.load_npz(df_run_dir / "normalizer.npz")

    ckpt_mgr = create_manager(df_run_dir / "ckpt")
    restored = restore_latest(ckpt_mgr)
    params = restored["params"]

    return model, params, norm


def _load_phi_model(phi_run_dir: Path) -> PotentialMLP:
    cfg_path = phi_run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    pot_cfg = cfg.get("potential", {})
    return PotentialMLP(PotentialConfig(hidden_sizes=tuple(int(x) for x in pot_cfg.get("hidden_sizes", [256, 256, 256]))))


def _sorted_steps(ckpt_dir: Path) -> list[int]:
    steps: list[int] = []
    for child in ckpt_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(int(child.name))
    return sorted(steps)


def _plummer_phi(r: np.ndarray) -> np.ndarray:
    return -(1.0 + r**2) ** (-0.5)


def _plummer_ar(r: np.ndarray) -> np.ndarray:
    return -r * (1.0 + r**2) ** (-1.5)


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

    df_model, df_params, normalizer = _load_df(Path(args.df_run_dir))
    phi_model = _load_phi_model(Path(args.phi_run_dir))

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

    phi_true = _plummer_phi(r)
    ar_true = _plummer_ar(r)

    i_ref = int(np.argmin(np.abs(r - float(args.r_ref))))
    phi_true_ref = float(_plummer_phi(np.array([float(args.r_ref)], dtype=np.float32))[0])

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
