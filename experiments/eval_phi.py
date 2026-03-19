from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dpjax.data import Normalizer, iter_batches, load_eta_h5
from dpjax.flows.realnvp import RealNVP, RealNVPConfig, score_apply
from dpjax.models.potential import PotentialConfig, PotentialMLP, grad_phi_apply, phi_apply
from dpjax.paths import ensure_dir, resolve_path
from dpjax.physics.cbe import residual_A
from dpjax.utils.ckpt import create_manager, restore_latest


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


def _plummer_phi(r: np.ndarray) -> np.ndarray:
    # Matches legacy generator: Phi = -(1 + r^2)^(-1/2)
    return -(1.0 + r**2) ** (-0.5)


def _plummer_ar(r: np.ndarray) -> np.ndarray:
    # Signed radial acceleration along +r_hat: a_r = -dPhi/dr = - r * (1+r^2)^(-3/2)
    return -r * (1.0 + r**2) ** (-1.5)


# ---------------------------------------------------------------------------
# Core evaluation function – callable from both CLI and Jupyter
# ---------------------------------------------------------------------------

def run_eval_phi(
    data_path: str | Path,
    df_run_dir: str | Path,
    phi_run_dir: str | Path,
    *,
    out_dir: Optional[str | Path] = None,
    n_eval: int = 32768,
    batch_size: int = 4096,
    seed: int = 0,
    r_min: float = 1.0e-3,
    r_max: float = 10.0,
    n_r: int = 256,
    r_ref: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate trained Phi/DF on residual stats and Plummer radial curves.

    Returns
    -------
    dict
        ``{"stats": dict, "radial": dict, "out_dir": Path, "plots_dir": Path}``
    """
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)
    phi_run_dir = resolve_path(phi_run_dir)

    df_model, df_params, normalizer = _load_df(Path(df_run_dir))
    phi_model, phi_params = _load_phi(Path(phi_run_dir))

    out_dir = ensure_dir(out_dir or phi_run_dir)
    plots_dir = ensure_dir(out_dir / "plots")

    eta = load_eta_h5(data_path, dataset="eta")
    eta_std = normalizer.transform(eta)

    n_total = eta_std.shape[0]
    n_eval = int(min(n_eval, n_total))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_eval, replace=False)
    eta_eval = eta_std[idx]

    std_x = np.asarray(normalizer.std[:3], dtype=np.float32)
    mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)

    @jax.jit
    def residual_batch(eta_std_batch: jnp.ndarray) -> jnp.ndarray:
        score_std = score_apply(df_model, df_params, eta_std_batch)
        grad_phi_std = grad_phi_apply(phi_model, phi_params, eta_std_batch[:, :3])
        return residual_A(eta_std_batch, score_std, grad_phi_std, normalizer)

    rs: list[np.ndarray] = []
    for batch_np in iter_batches(eta_eval, batch_size=int(batch_size), rng=rng, shuffle=False, drop_remainder=False):
        r = residual_batch(jnp.asarray(batch_np)).astype(jnp.float32)
        rs.append(np.asarray(r))

    r_all = np.concatenate(rs, axis=0)

    stats = {
        "n_eval": int(r_all.shape[0]),
        "residual_mean": float(np.mean(r_all)),
        "residual_std": float(np.std(r_all)),
        "residual_p99_abs": float(np.percentile(np.abs(r_all), 99.0)),
        "residual_p999_abs": float(np.percentile(np.abs(r_all), 99.9)),
        "residual_max_abs": float(np.max(np.abs(r_all))),
    }

    (out_dir / "eval_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

    # Radial curves along x-axis
    r = np.geomspace(r_min, r_max, num=int(n_r)).astype(np.float32)
    x_phys = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=-1)
    x_std = (x_phys - mean_x[None, :]) / std_x[None, :]

    x_std_j = jnp.asarray(x_std)
    phi_learned = np.asarray(phi_apply(phi_model, phi_params, x_std_j)).astype(np.float32)

    grad_phi_std = np.asarray(grad_phi_apply(phi_model, phi_params, x_std_j)).astype(np.float32)
    grad_phi_phys = grad_phi_std / std_x[None, :]

    # Along x-axis, radial acceleration equals -dPhi/dx
    ar_learned = -grad_phi_phys[:, 0]

    phi_true = _plummer_phi(r)
    ar_true = _plummer_ar(r)

    # Align potential by constant offset at r_ref
    r_ref = float(r_ref)
    phi_true_ref = float(_plummer_phi(np.array([r_ref], dtype=np.float32))[0])
    # Nearest grid point
    i_ref = int(np.argmin(np.abs(r - r_ref)))
    phi_learned_shift = phi_learned - phi_learned[i_ref] + phi_true_ref

    np.savez(
        out_dir / "radial_curves_plummer.npz",
        r=r,
        phi_learned=phi_learned,
        phi_learned_shift=phi_learned_shift,
        phi_true=phi_true,
        ar_learned=ar_learned,
        ar_true=ar_true,
    )

    # Optional plotting
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(r, phi_true, label="Plummer analytic")
        plt.plot(r, phi_learned_shift, label="Learned (shifted)")
        plt.xscale("log")
        plt.xlabel("r")
        plt.ylabel("Phi(r)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "phi_r_plummer.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(r, ar_true, label="Plummer analytic")
        plt.plot(r, ar_learned, label="Learned")
        plt.xscale("log")
        plt.xlabel("r")
        plt.ylabel("a_r(r)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "ar_r_plummer.png", dpi=150)
        plt.close()

        print(f"Wrote plots to {plots_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"Plot skipped: {e}")

    return {
        "stats": stats,
        "radial": {
            "r": r, "phi_learned": phi_learned, "phi_learned_shift": phi_learned_shift,
            "phi_true": phi_true, "ar_learned": ar_learned, "ar_true": ar_true,
        },
        "out_dir": out_dir,
        "plots_dir": plots_dir,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained Phi/DF on residual stats and Plummer radial curves.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--df-run-dir", type=str, required=True)
    parser.add_argument("--phi-run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-eval", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r-min", type=float, default=1.0e-3)
    parser.add_argument("--r-max", type=float, default=10.0)
    parser.add_argument("--n-r", type=int, default=256)
    parser.add_argument("--r-ref", type=float, default=1.0)
    args = parser.parse_args()

    run_eval_phi(
        args.data, args.df_run_dir, args.phi_run_dir,
        out_dir=args.out_dir, n_eval=args.n_eval, batch_size=args.batch_size,
        seed=args.seed, r_min=args.r_min, r_max=args.r_max,
        n_r=args.n_r, r_ref=args.r_ref,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
