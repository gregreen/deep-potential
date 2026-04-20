from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import optax
import yaml

from dpjax.data import fit_normalizer, iter_batches, load_eta_h5
from dpjax.flows.api import build_flow, init_flow, log_prob_apply, log_prob_reg_apply, score_apply
from dpjax.paths import ensure_dir, resolve_path
from dpjax.utils.ckpt import create_manager, finalize, restore_latest, save

# ---------------------------------------------------------------------------
# Core training function – callable from both CLI and Jupyter
# ---------------------------------------------------------------------------

def run_df_training(
    config: Dict[str, Any],
    data_path: str | Path,
    run_dir: str | Path,
    *,
    resume: bool = False,
    logger: Optional["ExperimentLogger"] = None,
) -> Dict[str, Any]:
    """Train the DF (RealNVP) normalizing flow.

    Parameters
    ----------
    config : dict
        Full training configuration (typically loaded from a YAML file).
    data_path : str or Path
        Path to the HDF5 data file containing ``eta``.
    run_dir : str or Path
        Directory for checkpoints, metrics, and config snapshots.
    resume : bool
        If ``True``, resume from the latest checkpoint in *run_dir*.

    Returns
    -------
    dict
        ``{"params": ..., "normalizer": ..., "model": ..., "final_step": int}``
    """
    data_path = resolve_path(data_path)
    run_dir = ensure_dir(run_dir)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    eta = load_eta_h5(data_path, dataset=config.get("data", {}).get("dataset", "eta"))

    normalizer = fit_normalizer(eta, eps=float(config.get("normalizer", {}).get("eps", 1.0e-6)))
    normalizer.save_npz(run_dir / "normalizer.npz")

    eta_std = normalizer.transform(eta)

    data_cfg = config.get("data", {})
    val_frac = float(data_cfg.get("val_frac", 0.1))
    val_frac = float(np.clip(val_frac, 0.0, 0.5))

    n_total = int(eta_std.shape[0])
    n_val = int(round(n_total * val_frac))
    n_val = min(max(n_val, 0), max(n_total - 1, 0))

    if n_val > 0:
        eta_train = eta_std[:-n_val]
        eta_val = eta_std[-n_val:]
    else:
        eta_train = eta_std
        eta_val = np.empty((0, eta_std.shape[1]), dtype=np.float32)

    flow_cfg = config.get("flow", {})
    model = build_flow(flow_cfg)

    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8192))
    epochs = int(train_cfg.get("epochs", 32))
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
                f"[train_df] Adjusting batch_size from {batch_size} to {adjusted_batch_size} "
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

    seed = int(config.get("seed", 0))
    rng = jax.random.key(seed)

    params = init_flow(model, rng, flow_cfg)

    # Compute step counts to configure schedules
    n = int(eta_train.shape[0])
    steps_per_epoch = n // batch_size
    if steps_per_epoch <= 0:
        raise ValueError(
            f"Not enough training samples ({n}) for batch_size={batch_size}. "
            f"Reduce batch_size or val_frac."
        )
    total_steps = int(epochs) * int(steps_per_epoch)

    # Configure learning rate
    lr_config = train_cfg.get("lr", 1.0e-3)
    if isinstance(lr_config, dict):
        lr_max = float(lr_config.get("max", 0.02))
        lr_final = float(lr_config.get("final", 0.001))
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
    opt_state = opt.init(params)

    ckpt_mgr = create_manager(run_dir / "ckpt", max_to_keep=max_to_keep)
    step0 = 0
    if resume:
        restored = restore_latest(ckpt_mgr)
        params = restored["params"]
        restored_opt_state = restored.get("opt_state", None)
        expected_opt_state = opt.init(params)
        if restored_opt_state is None:
            print("[train_df] Resume checkpoint has no opt_state; reinitializing optimizer state.")
            opt_state = expected_opt_state
        else:
            restored_struct = jax.tree_util.tree_structure(restored_opt_state)
            expected_struct = jax.tree_util.tree_structure(expected_opt_state)
            if restored_struct != expected_struct:
                print(
                    "[train_df] Restored opt_state structure is incompatible with current optimizer; "
                    "reinitializing optimizer state and continuing from restored params."
                )
                opt_state = expected_opt_state
            else:
                opt_state = restored_opt_state
        step0 = int(restored.get("step", 0))

    if use_sharding:
        params = jax.device_put(params, replicated_sharding)
        opt_state = jax.device_put(opt_state, replicated_sharding)

    def _to_host(tree):
        return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            lp, reg = log_prob_reg_apply(model, p, batch, flow_cfg)
            return -jnp.mean(lp) + jnp.mean(reg)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    @jax.jit
    def eval_loss(params, batch):
        lp, reg = log_prob_reg_apply(model, params, batch, flow_cfg)
        return -jnp.mean(lp) + jnp.mean(reg)

    metrics_path = run_dir / "metrics.csv"
    expected_header = [
        "step",
        "epoch",
        "loss",
        "score_p50",
        "score_p99",
        "score_max_abs",
        "val_loss",
        "val_score_p99",
    ]
    if resume and metrics_path.exists():
        with metrics_path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            rows = list(reader)
        if existing_header != expected_header:
            with metrics_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=expected_header)
                writer.writeheader()
                for row in rows:
                    if row.get((existing_header or [""])[0], "") == (existing_header or [""])[0]:
                        continue
                    writer.writerow({k: row.get(k, "nan") for k in expected_header})

    if resume and metrics_path.exists():
        open_mode, write_header = "a", False
    else:
        open_mode, write_header = "w", True
    with metrics_path.open(open_mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(expected_header)

        global_step = step0
        np_rng = np.random.default_rng(seed=seed)
        n_val_eval = min(2048, int(eta_val.shape[0]))
        val_eval = eta_val[:n_val_eval] if n_val_eval > 0 else None

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
                eta_train,
                batch_size=batch_size,
                rng=np_rng,
                shuffle=True,
                drop_remainder=True,
            ):
                batch = jnp.asarray(batch_np)
                if use_sharding:
                    batch = jax.device_put(batch, batch_sharding)

                params, opt_state, loss = train_step(params, opt_state, batch)
                loss_scalar = float(jax.device_get(loss))

                need_host_state = (global_step % log_every) == 0
                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    need_host_state = True

                if need_host_state and use_sharding:
                    params_host = _to_host(params)
                    opt_state_host = _to_host(opt_state)
                else:
                    params_host = params
                    opt_state_host = opt_state

                x_small = jnp.asarray(batch_np[:1024])

                if (global_step % log_every) == 0:
                    score = score_apply(model, params_host, x_small, flow_cfg)
                    score_abs = jnp.abs(score)
                    p50 = float(jnp.percentile(score_abs, 50.0))
                    p99 = float(jnp.percentile(score_abs, 99.0))
                    smax = float(jnp.max(score_abs))

                    if val_eval is not None:
                        val_batch = jnp.asarray(val_eval)
                        val_loss = float(eval_loss(params_host, val_batch))
                        val_score = score_apply(model, params_host, val_batch, flow_cfg)
                        val_score_p99 = float(jnp.percentile(jnp.abs(val_score), 99.0))
                    else:
                        val_loss = math.nan
                        val_score_p99 = math.nan

                    writer.writerow([global_step, epoch, loss_scalar, p50, p99, smax, val_loss, val_score_p99])
                    f.flush()
                    if logger is not None:
                        logger.log_scalars(global_step, {
                            "epoch": epoch, "loss": loss_scalar,
                            "score_p50": p50, "score_p99": p99, "score_max_abs": smax,
                            "val_loss": val_loss, "val_score_p99": val_score_p99,
                        })
                    postfix = dict(nll=loss_scalar, p99=p99, smax=smax, epoch=epoch)
                    if not math.isnan(val_loss):
                        postfix["val_nll"] = val_loss
                    pbar.set_postfix(**postfix)

                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    save(ckpt_mgr, global_step, {"params": params_host, "opt_state": opt_state_host, "step": global_step})

                global_step += 1
                pbar.update(1)

        pbar.close()

    if use_sharding:
        params = _to_host(params)
        opt_state = _to_host(opt_state)

    save(ckpt_mgr, global_step, {"params": params, "opt_state": opt_state, "step": global_step})
    finalize(ckpt_mgr)
    print(f"Saved final checkpoint at step {global_step}.")

    return {
        "params": params,
        "normalizer": normalizer,
        "model": model,
        "final_step": global_step,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train DF (RealNVP / FFJORD) on eta.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--override", type=str, default=None,
        help="JSON string of config overrides, e.g. '{\"train\": {\"epochs\": 64}}'.",
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
        run_df_training(cfg, args.data, run_dir, resume=args.resume, logger=logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
