from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from dpjax.data import Normalizer, iter_batches, load_eta_h5
from dpjax.flows.realnvp import RealNVP, RealNVPConfig, score_apply
from dpjax.models.potential import PotentialConfig, PotentialMLP, grad_phi_apply
from dpjax.paths import ensure_dir, resolve_path
from dpjax.physics.cbe import loss_cbe_A, residual_A
from dpjax.utils.ckpt import create_manager, finalize, restore_latest, save


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


# ---------------------------------------------------------------------------
# Core training function – callable from both CLI and Jupyter
# ---------------------------------------------------------------------------

def run_phi_training(
    config: Dict[str, Any],
    data_path: str | Path,
    df_run_dir: str | Path,
    run_dir: str | Path,
    *,
    resume: bool = False,
) -> Dict[str, Any]:
    """Train the potential network Phi with a frozen DF using CBE residual.

    Parameters
    ----------
    config : dict
        Full training configuration (typically loaded from a YAML file).
    data_path : str or Path
        Path to the HDF5 data file containing ``eta``.
    df_run_dir : str or Path
        Directory of a completed DF training run (must contain
        ``config.yaml``, ``normalizer.npz``, and ``ckpt/``).
    run_dir : str or Path
        Directory for checkpoints, metrics, and config snapshots.
    resume : bool
        If ``True``, resume from the latest checkpoint in *run_dir*.

    Returns
    -------
    dict
        ``{"phi_params": ..., "phi_model": ..., "df_model": ...,
          "df_params": ..., "normalizer": ..., "final_step": int}``
    """
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)
    run_dir = ensure_dir(run_dir)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    df_model, df_params, normalizer = _load_df(Path(df_run_dir))

    eta = load_eta_h5(data_path, dataset=config.get("data", {}).get("dataset", "eta"))
    eta_std = normalizer.transform(eta)

    pot_cfg = config.get("potential", {})
    phi_model = PotentialMLP(PotentialConfig(hidden_sizes=tuple(int(x) for x in pot_cfg.get("hidden_sizes", [256, 256, 256]))))

    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 4096))
    epochs = int(train_cfg.get("epochs", 64))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 200))
    max_to_keep = int(train_cfg.get("max_to_keep", 3))

    seed = int(config.get("seed", 1))
    rng = jax.random.key(seed)

    dummy_x = jnp.zeros((1, 3), dtype=jnp.float32)
    phi_params = phi_model.init(rng, dummy_x)["params"]

    # Compute step counts to configure schedules
    n = int(eta_std.shape[0])
    steps_per_epoch = n // batch_size
    total_steps = int(epochs) * int(steps_per_epoch)

    # Configure learning rate
    lr_config = train_cfg.get("lr", 1.0e-3)
    if isinstance(lr_config, dict):
        lr_max = float(lr_config.get("max", 0.05))
        lr_final = float(lr_config.get("final", 5.0e-6))
        warmup_frac = float(lr_config.get("warmup_frac", 0.1))
        warmup_steps = int(warmup_frac * total_steps)
        
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr_max,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=lr_final
        )
        base_opt = optax.radam(lr_schedule)
    else:
        base_opt = optax.adam(float(lr_config))

    opt = optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
    opt_state = opt.init(phi_params)

    ckpt_mgr = create_manager(run_dir / "ckpt", max_to_keep=max_to_keep)
    step0 = 0
    if resume:
        restored = restore_latest(ckpt_mgr)
        phi_params = restored["params"]
        opt_state = restored["opt_state"]
        step0 = int(restored.get("step", 0))

    @jax.jit
    def train_step(phi_params, opt_state, eta_std_batch):
        x_std = eta_std_batch[:, :3]

        def loss_fn(p):
            score_std = score_apply(df_model, df_params, eta_std_batch)
            grad_phi_std = grad_phi_apply(phi_model, p, x_std)
            return loss_cbe_A(eta_std_batch, score_std, grad_phi_std, normalizer)

        loss, grads = jax.value_and_grad(loss_fn)(phi_params)
        updates, opt_state2 = opt.update(grads, opt_state, phi_params)
        phi_params2 = optax.apply_updates(phi_params, updates)
        return phi_params2, opt_state2, loss

    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists() or not resume
    with metrics_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "epoch", "loss", "residual_mean", "residual_std", "residual_p99_abs"])

        global_step = step0
        np_rng = np.random.default_rng(seed=seed)

        # +++
        pbar = tqdm(
            total=total_steps,
            initial=min(global_step, total_steps),
            dynamic_ncols=True,
            unit="step",
        )
        # +++

        for epoch in range(epochs):
            for batch_np in iter_batches(
                eta_std,
                batch_size=batch_size,
                rng=np_rng,
                shuffle=True,
                drop_remainder=True,
            ):
                eta_b = jnp.asarray(batch_np)
                phi_params, opt_state, loss = train_step(phi_params, opt_state, eta_b)

                if (global_step % log_every) == 0:
                    eta_small = eta_b[:1024]
                    score_small = score_apply(df_model, df_params, eta_small)
                    grad_phi_small = grad_phi_apply(phi_model, phi_params, eta_small[:, :3])
                    r = residual_A(eta_small, score_small, grad_phi_small, normalizer)
                    r_mean = float(jnp.mean(r))
                    r_std = float(jnp.std(r))
                    r_p99 = float(jnp.percentile(jnp.abs(r), 99.0))
                    writer.writerow([global_step, epoch, float(loss), r_mean, r_std, r_p99])
                    f.flush()
                    print(
                        f"step={global_step} epoch={epoch} loss={float(loss):.6e} "
                        f"r_mean={r_mean:.3g} r_std={r_std:.3g} |r|p99={r_p99:.3g}",
                        flush=True,
                    )
                    # +++
                    pbar.set_postfix(
                        loss=float(loss),
                        r_std=r_std,
                        r_p99=r_p99,
                        epoch=epoch,
                    )
                    # +++

                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    save(ckpt_mgr, global_step, {"params": phi_params, "opt_state": opt_state, "step": global_step})

                global_step += 1
                # +++
                pbar.update(1)
                # +++

        # +++
        pbar.close()
        # +++

    save(ckpt_mgr, global_step, {"params": phi_params, "opt_state": opt_state, "step": global_step})
    finalize(ckpt_mgr)
    print(f"Saved final checkpoint at step {global_step}.")

    return {
        "phi_params": phi_params,
        "phi_model": phi_model,
        "df_model": df_model,
        "df_params": df_params,
        "normalizer": normalizer,
        "final_step": global_step,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train Phi with frozen DF using CBE residual A.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--df-run-dir", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_phi_training(cfg, args.data, args.df_run_dir, args.run_dir, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
