from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import optax
import yaml

from dpjax.data import iter_batches, load_eta_h5
from dpjax.flows.api import load_df, score_apply
from dpjax.models.potential import PotentialConfig, PotentialMLP, grad_phi_apply, laplacian_phi_apply
from dpjax.paths import ensure_dir, resolve_path
from dpjax.physics.cbe import loss_cbe_A, loss_cbe_robust, residual_A
from dpjax.utils.ckpt import create_manager, finalize, restore_latest, save


def _mean_param_square(params: dict) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(params)
    sq_sum = sum(jnp.sum(jnp.square(x)) for x in leaves)
    n_elem = sum(x.size for x in leaves)
    return sq_sum / jnp.maximum(jnp.asarray(n_elem, dtype=jnp.float32), 1.0)


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
    logger: Optional["ExperimentLogger"] = None,
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

    df_model, df_params, normalizer, df_cfg = load_df(df_run_dir)
    flow_cfg = df_cfg.get("flow", {})

    eta = load_eta_h5(data_path, dataset=config.get("data", {}).get("dataset", "eta"))
    eta_std = normalizer.transform(eta)

    pot_cfg = config.get("potential", {})
    phi_model = PotentialMLP(PotentialConfig(hidden_sizes=tuple(int(x) for x in pot_cfg.get("hidden_sizes", [512, 512, 512, 512]))))

    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 4096))
    epochs = int(train_cfg.get("epochs", 64))
    loss_type = str(train_cfg.get("loss_type", "robust")).lower()
    alpha = float(train_cfg.get("alpha", 1.0))
    beta = float(train_cfg.get("beta", 1.0))
    lambda_mass = float(train_cfg.get("lambda_mass", 1.0))
    l2_reg = float(train_cfg.get("l2_reg", 0.1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 200))
    max_to_keep = int(train_cfg.get("max_to_keep", 3))
    n_devices = int(jax.local_device_count())
    use_sharding = bool(train_cfg.get("multi_gpu", True)) and n_devices > 1

    if use_sharding:
        if batch_size < n_devices:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= number of devices ({n_devices}) for multi-GPU training."
            )
        if (batch_size % n_devices) != 0:
            adjusted_batch_size = (batch_size // n_devices) * n_devices
            if adjusted_batch_size <= 0:
                raise ValueError(
                    f"batch_size ({batch_size}) is too small for {n_devices} devices in multi-GPU training."
                )
            print(
                f"[train_phi] Adjusting batch_size from {batch_size} to {adjusted_batch_size} "
                f"for sharded training over {n_devices} devices."
            )
            batch_size = adjusted_batch_size

    mesh = None
    replicated_sharding = None
    batch_sharding = None
    if use_sharding:
        mesh = Mesh(np.asarray(jax.local_devices()[:n_devices]), axis_names=("batch",))
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        batch_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    if loss_type not in {"robust", "mse"}:
        raise ValueError(f"Unknown train.loss_type={loss_type!r}; expected 'robust' or 'mse'.")

    seed = int(config.get("seed", 1))
    rng = jax.random.key(seed)

    dummy_x = jnp.zeros((1, 3), dtype=jnp.float32)
    phi_params = phi_model.init(rng, dummy_x)["params"]

    # Compute step counts to configure schedules
    n = int(eta_std.shape[0])
    steps_per_epoch = n // batch_size
    if steps_per_epoch <= 0:
        raise ValueError(
            f"Not enough training samples ({n}) for batch_size={batch_size}. "
            "Reduce batch_size."
        )
    total_steps = int(epochs) * int(steps_per_epoch)

    device_list = list(jax.local_devices()[:n_devices])
    per_device_batch = (batch_size // n_devices) if use_sharding else batch_size
    print(
        "[train_phi] setup: "
        f"backend={jax.default_backend()}, "
        f"use_sharding={use_sharding}, "
        f"n_devices={n_devices}, "
        f"global_batch_size={batch_size}, "
        f"per_device_batch_size={per_device_batch}, "
        f"steps_per_epoch={steps_per_epoch}, "
        f"total_steps={total_steps}"
    )
    print(f"[train_phi] local devices: {device_list}")

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

    def _to_host(tree):
        return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)

    if use_sharding:
        phi_params = jax.device_put(phi_params, replicated_sharding)
        opt_state = jax.device_put(opt_state, replicated_sharding)
        df_params = jax.device_put(df_params, replicated_sharding)
        df_params_host = _to_host(df_params)
    else:
        df_params_host = df_params

    @jax.jit
    def train_step(phi_params, opt_state, eta_std_batch):
        x_std = eta_std_batch[:, :3]
        std_x = jnp.asarray(normalizer.std[:3], dtype=eta_std_batch.dtype)

        def loss_fn(p):
            score_std = score_apply(df_model, df_params, eta_std_batch, flow_cfg)
            grad_phi_std = grad_phi_apply(phi_model, p, x_std)

            if loss_type == "mse":
                cbe_loss = loss_cbe_A(eta_std_batch, score_std, grad_phi_std, normalizer)
            else:
                residual = residual_A(eta_std_batch, score_std, grad_phi_std, normalizer)
                laplacian_phi_phys = laplacian_phi_apply(phi_model, p, x_std, std_x=std_x)
                cbe_loss = loss_cbe_robust(
                    residual,
                    laplacian_phi_phys,
                    alpha=alpha,
                    beta=beta,
                    lambda_mass=lambda_mass,
                )

            return cbe_loss + l2_reg * _mean_param_square(p)

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
        start_epoch = step0 // steps_per_epoch

        pbar = tqdm(
            total=total_steps,
            initial=min(global_step, total_steps),
            dynamic_ncols=True,
            unit="step",
            mininterval=0.5,
            smoothing=0.1,
        )

        for epoch in range(start_epoch, epochs):
            for batch_np in iter_batches(
                eta_std,
                batch_size=batch_size,
                rng=np_rng,
                shuffle=True,
                drop_remainder=True,
            ):
                eta_b = jnp.asarray(batch_np)
                if use_sharding:
                    eta_b = jax.device_put(eta_b, batch_sharding)
                phi_params, opt_state, loss = train_step(phi_params, opt_state, eta_b)
                loss_scalar = float(jax.device_get(loss))

                need_host_state = (global_step % log_every) == 0
                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    need_host_state = True

                if need_host_state and use_sharding:
                    phi_params_host = _to_host(phi_params)
                    opt_state_host = _to_host(opt_state)
                else:
                    phi_params_host = phi_params
                    opt_state_host = opt_state

                if (global_step % log_every) == 0:
                    eta_small = jnp.asarray(batch_np[:1024])
                    score_small = score_apply(df_model, df_params_host, eta_small, flow_cfg)
                    grad_phi_small = grad_phi_apply(phi_model, phi_params_host, eta_small[:, :3])
                    r = residual_A(eta_small, score_small, grad_phi_small, normalizer)
                    r_mean = float(jnp.mean(r))
                    r_std = float(jnp.std(r))
                    r_p99 = float(jnp.percentile(jnp.abs(r), 99.0))
                    writer.writerow([global_step, epoch, loss_scalar, r_mean, r_std, r_p99])
                    f.flush()
                    if logger is not None:
                        logger.log_scalars(global_step, {
                            "epoch": epoch, "loss": loss_scalar,
                            "residual_mean": r_mean, "residual_std": r_std,
                            "residual_p99_abs": r_p99,
                        })
                    pbar.set_postfix(
                        loss=loss_scalar,
                        r_std=r_std,
                        r_p99=r_p99,
                        epoch=epoch,
                    )

                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    save(ckpt_mgr, global_step, {"params": phi_params_host, "opt_state": opt_state_host, "step": global_step})

                global_step += 1
                pbar.update(1)

        pbar.close()

    if use_sharding:
        phi_params = _to_host(phi_params)
        opt_state = _to_host(opt_state)

    save(ckpt_mgr, global_step, {"params": phi_params, "opt_state": opt_state, "step": global_step})
    finalize(ckpt_mgr)
    print(f"Saved final checkpoint at step {global_step}.")

    return {
        "phi_params": phi_params,
        "phi_model": phi_model,
        "df_model": df_model,
        "df_params": df_params_host,
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
    parser.add_argument(
        "--override", type=str, default=None,
        help="JSON string of config overrides, e.g. '{\"train\": {\"epochs\": 32}}'.",
    )
    parser.add_argument("--logger", type=str, default="csv", help="Logger backend: csv, wandb, tensorboard, wandb+tb.")
    parser.add_argument("--project", type=str, default="dp-plummer", help="W&B project name.")
    parser.add_argument("--run-name", type=str, default=None, help="W&B / experiment run name.")
    args = parser.parse_args()

    import json
    import sys
    _repo = Path(__file__).resolve().parents[1]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from dpjax.config import merge_config
    from experiments.logger import ExperimentLogger

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.override:
        cfg = merge_config(cfg, json.loads(args.override))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with ExperimentLogger(
        run_dir, project=args.project, run_name=args.run_name,
        backend=args.logger, config=cfg,
    ) as logger:
        run_phi_training(
            cfg, args.data, args.df_run_dir, run_dir,
            resume=args.resume, logger=logger,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
