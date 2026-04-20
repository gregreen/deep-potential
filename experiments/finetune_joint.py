from __future__ import annotations

import os
import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Dict


# JAX 默认会预分配大量 GPU 显存；在共享/繁忙 GPU 上容易导致初始化 OOM。
# 这里默认关闭预分配（用户可在环境中显式设置为 "true" 来覆盖）。
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from tqdm.auto import tqdm

from dpjax.data import iter_batches, load_eta_h5
from dpjax.flows.api import load_df, log_prob_apply, score_apply
from dpjax.models.potential import grad_phi_apply, laplacian_phi_apply, load_phi
from dpjax.paths import ensure_dir, resolve_path
from dpjax.physics.cbe import loss_cbe_robust, residual_A
from dpjax.utils.ckpt import create_manager, finalize, restore_latest, save


def _mean_param_square(params: dict) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(params)
    sq_sum = sum(jnp.sum(jnp.square(x)) for x in leaves)
    n_elem = sum(x.size for x in leaves)
    return sq_sum / jnp.maximum(jnp.asarray(n_elem, dtype=jnp.float32), 1.0)


# ---------------------------------------------------------------------------
# Core fine-tuning function – callable from both CLI and Jupyter
# ---------------------------------------------------------------------------

def run_joint_finetuning(
    config: Dict[str, Any],
    data_path: str | Path,
    df_run_dir: str | Path,
    phi_run_dir: str | Path,
    run_dir: str | Path,
    *,
    resume: bool = False,
) -> Dict[str, Any]:
    """Joint fine-tuning for DF + Phi (scheme B).

    Optimizes ``L = lambda_cbe * L_cbe + lambda_nll * NLL`` with small LR.

    Parameters
    ----------
    config : dict
        Joint training configuration (typically loaded from a YAML file).
    data_path : str or Path
        Path to the HDF5 data file containing ``eta``.
    df_run_dir : str or Path
        Directory of the completed DF training run.
    phi_run_dir : str or Path
        Directory of the completed Phi training run.
    run_dir : str or Path
        Directory for joint checkpoints, metrics, and config snapshots.
    resume : bool
        If ``True``, resume from the latest joint checkpoint.

    Returns
    -------
    dict
        ``{"df_params": ..., "phi_params": ..., "df_model": ...,
          "phi_model": ..., "normalizer": ..., "final_step": int}``
    """
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)
    phi_run_dir = resolve_path(phi_run_dir)

    cfg_in = config

    run_dir = ensure_dir(run_dir)
    df_out = ensure_dir(run_dir / "df")
    phi_out = ensure_dir(run_dir / "phi")
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (df_out / "ckpt").mkdir(parents=True, exist_ok=True)
    (phi_out / "ckpt").mkdir(parents=True, exist_ok=True)

    df_model, df_params_init, normalizer, df_cfg = load_df(df_run_dir)
    flow_cfg = df_cfg.get("flow", {})
    phi_model, phi_params_init, phi_cfg = load_phi(phi_run_dir)

    # Write config snapshots for compatibility with existing eval scripts.
    # - df_out mimics train_df output
    # - phi_out mimics train_phi output
    (df_out / "config.yaml").write_text(yaml.safe_dump(df_cfg, sort_keys=False))
    (phi_out / "config.yaml").write_text(yaml.safe_dump(phi_cfg, sort_keys=False))
    shutil.copy2(Path(df_run_dir) / "normalizer.npz", df_out / "normalizer.npz")

    # Root config: merge flow/potential with joint training hyperparams.
    joint_cfg = dict(cfg_in) if isinstance(cfg_in, dict) else {}
    joint_cfg.setdefault("flow", df_cfg.get("flow", {}))
    joint_cfg.setdefault("potential", phi_cfg.get("potential", {}))
    joint_cfg.setdefault("data", cfg_in.get("data", df_cfg.get("data", {"dataset": "eta"})))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(joint_cfg, sort_keys=False))

    eta = load_eta_h5(data_path, dataset=joint_cfg.get("data", {}).get("dataset", "eta"))
    eta_std = normalizer.transform(eta)

    train_cfg = joint_cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 4096))
    epochs = int(train_cfg.get("epochs", 4))
    lr_df = float(train_cfg.get("lr_df", 1.0e-4))
    lr_phi = float(train_cfg.get("lr_phi", 1.0e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 200))
    max_to_keep = int(train_cfg.get("max_to_keep", 3))

    lambda_cbe = float(train_cfg.get("lambda_cbe", 1.0))
    lambda_nll = float(train_cfg.get("lambda_nll", 0.3))
    loss_type = str(train_cfg.get("loss_type", "robust")).lower()
    alpha = float(train_cfg.get("alpha", 1.0))
    beta = float(train_cfg.get("beta", 1.0))
    lambda_mass = float(train_cfg.get("lambda_mass", 1.0))
    l2_reg = float(train_cfg.get("l2_reg", 0.1))

    mode = str(train_cfg.get("mode", "both")).lower()
    alt_period = int(train_cfg.get("alt_period", 1))
    max_batches_per_epoch = train_cfg.get("max_batches_per_epoch", None)
    max_batches_per_epoch = int(max_batches_per_epoch) if max_batches_per_epoch is not None else None

    if mode not in {"both", "alt"}:
        raise ValueError(f"Unknown train.mode={mode!r}; expected 'both' or 'alt'.")
    if loss_type not in {"robust", "mse"}:
        raise ValueError(f"Unknown train.loss_type={loss_type!r}; expected 'robust' or 'mse'.")

    seed = int(joint_cfg.get("seed", 2))
    np_rng = np.random.default_rng(seed=seed)

    opt_df = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_df))
    opt_phi = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_phi))

    ckpt_mgr_joint = create_manager(run_dir / "ckpt", max_to_keep=max_to_keep)
    ckpt_mgr_df = create_manager(df_out / "ckpt", max_to_keep=max_to_keep)
    ckpt_mgr_phi = create_manager(phi_out / "ckpt", max_to_keep=max_to_keep)

    step0 = 0
    if resume:
        restored = restore_latest(ckpt_mgr_joint)
        df_params = restored["df_params"]
        phi_params = restored["phi_params"]
        opt_state_df = restored["df_opt_state"]
        opt_state_phi = restored["phi_opt_state"]
        step0 = int(restored.get("step", 0))
    else:
        df_params = df_params_init
        phi_params = phi_params_init
        opt_state_df = opt_df.init(df_params)
        opt_state_phi = opt_phi.init(phi_params)

    def _loss_terms(df_p: dict, phi_p: dict, eta_std_b: jnp.ndarray) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        # NLL term
        nll = -jnp.mean(log_prob_apply(df_model, df_p, eta_std_b, flow_cfg))

        # CBE term
        score_std = score_apply(df_model, df_p, eta_std_b, flow_cfg)
        grad_phi_std = grad_phi_apply(phi_model, phi_p, eta_std_b[:, :3])
        residual = residual_A(eta_std_b, score_std, grad_phi_std, normalizer)

        if loss_type == "mse":
            cbe = jnp.mean(residual**2)
        else:
            std_x = jnp.asarray(normalizer.std[:3], dtype=eta_std_b.dtype)
            laplacian_phi_phys = laplacian_phi_apply(phi_model, phi_p, eta_std_b[:, :3], std_x=std_x)
            cbe = loss_cbe_robust(
                residual,
                laplacian_phi_phys,
                alpha=alpha,
                beta=beta,
                lambda_mass=lambda_mass,
            )

        l2 = l2_reg * _mean_param_square(phi_p)

        loss = lambda_cbe * cbe + lambda_nll * nll + l2
        return loss, (nll, cbe, l2)

    @jax.jit
    def _step_both(df_p, phi_p, opt_s_df, opt_s_phi, eta_std_b):
        def loss_fn(packed):
            df_pp, phi_pp = packed
            return _loss_terms(df_pp, phi_pp, eta_std_b)

        (loss, (nll, cbe, l2)), grads = jax.value_and_grad(loss_fn, has_aux=True)((df_p, phi_p))
        grads_df, grads_phi = grads

        upd_df, opt_s_df2 = opt_df.update(grads_df, opt_s_df, df_p)
        upd_phi, opt_s_phi2 = opt_phi.update(grads_phi, opt_s_phi, phi_p)

        df_p2 = optax.apply_updates(df_p, upd_df)
        phi_p2 = optax.apply_updates(phi_p, upd_phi)
        return df_p2, phi_p2, opt_s_df2, opt_s_phi2, loss, nll, cbe, l2

    @jax.jit
    def _step_df_only(df_p, opt_s_df, phi_p, eta_std_b):
        def loss_fn(df_pp):
            return _loss_terms(df_pp, phi_p, eta_std_b)

        (loss, (nll, cbe, l2)), grads_df = jax.value_and_grad(loss_fn, has_aux=True)(df_p)
        upd_df, opt_s_df2 = opt_df.update(grads_df, opt_s_df, df_p)
        df_p2 = optax.apply_updates(df_p, upd_df)
        return df_p2, opt_s_df2, loss, nll, cbe, l2

    @jax.jit
    def _step_phi_only(phi_p, opt_s_phi, df_p, eta_std_b):
        def loss_fn(phi_pp):
            return _loss_terms(df_p, phi_pp, eta_std_b)

        (loss, (nll, cbe, l2)), grads_phi = jax.value_and_grad(loss_fn, has_aux=True)(phi_p)
        upd_phi, opt_s_phi2 = opt_phi.update(grads_phi, opt_s_phi, phi_p)
        phi_p2 = optax.apply_updates(phi_p, upd_phi)
        return phi_p2, opt_s_phi2, loss, nll, cbe, l2

    metrics_path = run_dir / "metrics.csv"
    if resume:
        metrics_mode = "a"
        write_header = (not metrics_path.exists()) or (metrics_path.stat().st_size == 0)
    else:
        metrics_mode = "w"
        write_header = True

    with metrics_path.open(metrics_mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "step",
                    "epoch",
                    "update",
                    "loss",
                    "nll",
                    "cbe",
                    "l2",
                    "residual_mean",
                    "residual_std",
                    "residual_p99_abs",
                    "score_p50",
                    "score_p99",
                    "score_max_abs",
                ]
            )

        global_step = step0

        n = eta_std.shape[0]
        steps_per_epoch = n // batch_size
        if max_batches_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, max_batches_per_epoch)
        if steps_per_epoch <= 0:
            raise ValueError("No full batches per epoch. Decrease batch_size or increase dataset size.")
        total_steps = int(epochs) * int(steps_per_epoch)

        start_epoch = step0 // steps_per_epoch
        pbar = tqdm(
            total=total_steps,
            initial=min(step0, total_steps),
            dynamic_ncols=True,
            unit="step",
            mininterval=0.5,
            smoothing=0.1,
        )

        rng = jax.random.PRNGKey(seed + start_epoch)

        global_step = step0
        for epoch in range(start_epoch, epochs):
            rng, step_rng = jax.random.split(rng)
            
            # Use numpy's standard RNG just for batch generation, seeded by jax rng
            seed_val = int(jax.random.randint(step_rng, (), 0, 1000000))
            np_rng = np.random.default_rng(seed_val)
            
            for batch_np in iter_batches(
                eta_std,
                batch_size,
                np_rng,
                shuffle=True,
                max_batches=max_batches_per_epoch,
            ):
                eta_b = jnp.asarray(batch_np)

                if mode == "both":
                    df_params, phi_params, opt_state_df, opt_state_phi, loss, nll, cbe, l2 = _step_both(
                        df_params, phi_params, opt_state_df, opt_state_phi, eta_b
                    )
                    update_tag = "both"
                elif mode == "alt":
                    phase = (global_step // max(1, alt_period)) % 2
                    if phase == 0:
                        df_params, opt_state_df, loss, nll, cbe, l2 = _step_df_only(df_params, opt_state_df, phi_params, eta_b)
                        update_tag = "df"
                    else:
                        phi_params, opt_state_phi, loss, nll, cbe, l2 = _step_phi_only(phi_params, opt_state_phi, df_params, eta_b)
                        update_tag = "phi"

                global_step += 1
                pbar.update(1)

                if (global_step % log_every) == 0:
                    eta_small = eta_b[:1024]
                    score_small = score_apply(df_model, df_params, eta_small, flow_cfg)
                    grad_phi_small = grad_phi_apply(phi_model, phi_params, eta_small[:, :3])
                    r = residual_A(eta_small, score_small, grad_phi_small, normalizer)

                    r_mean = float(jnp.mean(r))
                    r_std = float(jnp.std(r))
                    r_p99 = float(jnp.percentile(jnp.abs(r), 99.0))

                    score_abs = jnp.abs(score_small)
                    score_p50 = float(jnp.percentile(score_abs, 50.0))
                    score_p99 = float(jnp.percentile(score_abs, 99.0))
                    score_max_abs = float(jnp.max(score_abs))

                    writer.writerow(
                        [
                            global_step,
                            epoch,
                            update_tag,
                            float(loss),
                            float(nll),
                            float(cbe),
                            float(l2),
                            r_mean,
                            r_std,
                            r_p99,
                            score_p50,
                            score_p99,
                            score_max_abs,
                        ]
                    )
                    f.flush()

                    pbar.set_description(f"epoch={epoch}")
                    pbar.set_postfix({
                        "step": global_step,
                        "loss": float(loss),
                        "nll": float(nll),
                        "cbe": float(cbe),
                        "r_std": float(r_std),
                    })

                if (global_step % ckpt_every) == 0:
                    item_joint = {
                        "df_params": df_params,
                        "df_opt_state": opt_state_df,
                        "phi_params": phi_params,
                        "phi_opt_state": opt_state_phi,
                        "step": global_step,
                    }
                    save(ckpt_mgr_joint, global_step, item_joint)
                    save(ckpt_mgr_df, global_step, {"params": df_params, "opt_state": opt_state_df, "step": global_step})
                    save(ckpt_mgr_phi, global_step, {"params": phi_params, "opt_state": opt_state_phi, "step": global_step})

        pbar.close()

    # Final save
    item_joint = {
        "df_params": df_params,
        "df_opt_state": opt_state_df,
        "phi_params": phi_params,
        "phi_opt_state": opt_state_phi,
        "step": global_step,
    }
    save(ckpt_mgr_joint, global_step, item_joint)
    save(ckpt_mgr_df, global_step, {"params": df_params, "opt_state": opt_state_df, "step": global_step})
    save(ckpt_mgr_phi, global_step, {"params": phi_params, "opt_state": opt_state_phi, "step": global_step})

    # Avoid noisy async shutdown warnings from Orbax.
    finalize(ckpt_mgr_joint)
    finalize(ckpt_mgr_df)
    finalize(ckpt_mgr_phi)

    print(f"Saved final joint checkpoint at step {global_step}.")
    print(f"DF run dir:  {df_out}")
    print(f"Phi run dir: {phi_out}")

    return {
        "df_params": df_params,
        "phi_params": phi_params,
        "df_model": df_model,
        "phi_model": phi_model,
        "normalizer": normalizer,
        "final_step": global_step,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Joint fine-tuning for DF + Phi (scheme B). "
            "Optimizes L = lambda_cbe * L_cbe + lambda_nll * NLL with small LR."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--df-run-dir", type=str, required=True)
    parser.add_argument("--phi-run-dir", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_joint_finetuning(
        cfg, args.data, args.df_run_dir, args.phi_run_dir, args.run_dir,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
