from __future__ import annotations

import os
import argparse
import csv
import shutil
from pathlib import Path


# JAX 默认会预分配大量 GPU 显存；在共享/繁忙 GPU 上容易导致初始化 OOM。
# 这里默认关闭预分配（用户可在环境中显式设置为 "true" 来覆盖）。
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from dpjax.data import Normalizer, iter_batches, load_eta_h5
from dpjax.flows.realnvp import RealNVP, RealNVPConfig, log_prob_apply, score_apply
from dpjax.models.potential import PotentialConfig, PotentialMLP, grad_phi_apply
from dpjax.physics.cbe import loss_cbe_A, residual_A
from dpjax.utils.ckpt import create_manager, finalize, restore_latest, save


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_df(df_run_dir: Path) -> tuple[RealNVP, dict, Normalizer, dict]:
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

    norm_path = df_run_dir / "normalizer.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing {norm_path}")
    norm = Normalizer.load_npz(norm_path)

    ckpt_mgr = create_manager(df_run_dir / "ckpt")
    restored = restore_latest(ckpt_mgr)
    params = restored["params"]

    return model, params, norm, df_cfg


def _load_phi(phi_run_dir: Path) -> tuple[PotentialMLP, dict, dict]:
    cfg_path = phi_run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text())
    pot_cfg = cfg.get("potential", {})
    model = PotentialMLP(PotentialConfig(hidden_sizes=tuple(int(x) for x in pot_cfg.get("hidden_sizes", [256, 256, 256]))))

    ckpt_mgr = create_manager(phi_run_dir / "ckpt")
    restored = restore_latest(ckpt_mgr)
    params = restored["params"]

    return model, params, cfg


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

    cfg_in = yaml.safe_load(Path(args.config).read_text())

    run_dir = _ensure_dir(args.run_dir)
    df_out = _ensure_dir(run_dir / "df")
    phi_out = _ensure_dir(run_dir / "phi")
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (df_out / "ckpt").mkdir(parents=True, exist_ok=True)
    (phi_out / "ckpt").mkdir(parents=True, exist_ok=True)

    df_model, df_params_init, normalizer, df_cfg = _load_df(Path(args.df_run_dir))
    phi_model, phi_params_init, phi_cfg = _load_phi(Path(args.phi_run_dir))

    # Write config snapshots for compatibility with existing eval scripts.
    # - df_out mimics train_df output
    # - phi_out mimics train_phi output
    (df_out / "config.yaml").write_text(yaml.safe_dump(df_cfg, sort_keys=False))
    (phi_out / "config.yaml").write_text(yaml.safe_dump(phi_cfg, sort_keys=False))
    shutil.copy2(Path(args.df_run_dir) / "normalizer.npz", df_out / "normalizer.npz")

    # Root config: merge flow/potential with joint training hyperparams.
    joint_cfg = dict(cfg_in) if isinstance(cfg_in, dict) else {}
    joint_cfg.setdefault("flow", df_cfg.get("flow", {}))
    joint_cfg.setdefault("potential", phi_cfg.get("potential", {}))
    joint_cfg.setdefault("data", cfg_in.get("data", df_cfg.get("data", {"dataset": "eta"})))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(joint_cfg, sort_keys=False))

    eta = load_eta_h5(args.data, dataset=joint_cfg.get("data", {}).get("dataset", "eta"))
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

    mode = str(train_cfg.get("mode", "both")).lower()
    alt_period = int(train_cfg.get("alt_period", 1))
    max_batches_per_epoch = train_cfg.get("max_batches_per_epoch", None)
    max_batches_per_epoch = int(max_batches_per_epoch) if max_batches_per_epoch is not None else None

    seed = int(joint_cfg.get("seed", 2))
    np_rng = np.random.default_rng(seed=seed)

    opt_df = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_df))
    opt_phi = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_phi))

    ckpt_mgr_joint = create_manager(run_dir / "ckpt", max_to_keep=max_to_keep)
    ckpt_mgr_df = create_manager(df_out / "ckpt", max_to_keep=max_to_keep)
    ckpt_mgr_phi = create_manager(phi_out / "ckpt", max_to_keep=max_to_keep)

    step0 = 0
    if args.resume:
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

    def _loss_terms(df_p: dict, phi_p: dict, eta_std_b: jnp.ndarray) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        # NLL term
        nll = -jnp.mean(log_prob_apply(df_model, df_p, eta_std_b))

        # CBE term
        score_std = score_apply(df_model, df_p, eta_std_b)
        grad_phi_std = grad_phi_apply(phi_model, phi_p, eta_std_b[:, :3])
        cbe = loss_cbe_A(eta_std_b, score_std, grad_phi_std, normalizer)

        loss = lambda_cbe * cbe + lambda_nll * nll
        return loss, (nll, cbe)

    @jax.jit
    def _step_both(df_p, phi_p, opt_s_df, opt_s_phi, eta_std_b):
        def loss_fn(packed):
            df_pp, phi_pp = packed
            return _loss_terms(df_pp, phi_pp, eta_std_b)

        (loss, (nll, cbe)), grads = jax.value_and_grad(loss_fn, has_aux=True)((df_p, phi_p))
        grads_df, grads_phi = grads

        upd_df, opt_s_df2 = opt_df.update(grads_df, opt_s_df, df_p)
        upd_phi, opt_s_phi2 = opt_phi.update(grads_phi, opt_s_phi, phi_p)

        df_p2 = optax.apply_updates(df_p, upd_df)
        phi_p2 = optax.apply_updates(phi_p, upd_phi)
        return df_p2, phi_p2, opt_s_df2, opt_s_phi2, loss, nll, cbe

    @jax.jit
    def _step_df_only(df_p, opt_s_df, phi_p, eta_std_b):
        def loss_fn(df_pp):
            return _loss_terms(df_pp, phi_p, eta_std_b)

        (loss, (nll, cbe)), grads_df = jax.value_and_grad(loss_fn, has_aux=True)(df_p)
        upd_df, opt_s_df2 = opt_df.update(grads_df, opt_s_df, df_p)
        df_p2 = optax.apply_updates(df_p, upd_df)
        return df_p2, opt_s_df2, loss, nll, cbe

    @jax.jit
    def _step_phi_only(phi_p, opt_s_phi, df_p, eta_std_b):
        def loss_fn(phi_pp):
            return _loss_terms(df_p, phi_pp, eta_std_b)

        (loss, (nll, cbe)), grads_phi = jax.value_and_grad(loss_fn, has_aux=True)(phi_p)
        upd_phi, opt_s_phi2 = opt_phi.update(grads_phi, opt_s_phi, phi_p)
        phi_p2 = optax.apply_updates(phi_p, upd_phi)
        return phi_p2, opt_s_phi2, loss, nll, cbe

    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists() or not args.resume

    with metrics_path.open("a", newline="") as f:
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
                    "residual_mean",
                    "residual_std",
                    "residual_p99_abs",
                    "score_p50",
                    "score_p99",
                    "score_max_abs",
                ]
            )

        global_step = step0

        for epoch in range(epochs):
            for batch_np in iter_batches(
                eta_std,
                batch_size=batch_size,
                rng=np_rng,
                shuffle=True,
                drop_remainder=True,
                max_batches=max_batches_per_epoch,
            ):
                eta_b = jnp.asarray(batch_np)

                if mode == "both":
                    df_params, phi_params, opt_state_df, opt_state_phi, loss, nll, cbe = _step_both(
                        df_params, phi_params, opt_state_df, opt_state_phi, eta_b
                    )
                    update_tag = "both"
                elif mode == "alt":
                    phase = (global_step // max(1, alt_period)) % 2
                    if phase == 0:
                        df_params, opt_state_df, loss, nll, cbe = _step_df_only(df_params, opt_state_df, phi_params, eta_b)
                        update_tag = "df"
                    else:
                        phi_params, opt_state_phi, loss, nll, cbe = _step_phi_only(phi_params, opt_state_phi, df_params, eta_b)
                        update_tag = "phi"
                else:
                    raise ValueError(f"Unknown train.mode={mode!r}; expected 'both' or 'alt'.")

                if (global_step % log_every) == 0:
                    eta_small = eta_b[:1024]
                    score_small = score_apply(df_model, df_params, eta_small)
                    score_abs = jnp.abs(score_small)
                    score_p50 = float(jnp.percentile(score_abs, 50.0))
                    score_p99 = float(jnp.percentile(score_abs, 99.0))
                    score_max = float(jnp.max(score_abs))

                    grad_phi_small = grad_phi_apply(phi_model, phi_params, eta_small[:, :3])
                    r = residual_A(eta_small, score_small, grad_phi_small, normalizer)
                    r_mean = float(jnp.mean(r))
                    r_std = float(jnp.std(r))
                    r_p99 = float(jnp.percentile(jnp.abs(r), 99.0))

                    writer.writerow(
                        [
                            global_step,
                            epoch,
                            update_tag,
                            float(loss),
                            float(nll),
                            float(cbe),
                            r_mean,
                            r_std,
                            r_p99,
                            score_p50,
                            score_p99,
                            score_max,
                        ]
                    )
                    f.flush()
                    print(
                        " ".join(
                            [
                                f"step={global_step}",
                                f"epoch={epoch}",
                                f"upd={update_tag}",
                                f"loss={float(loss):.4e}",
                                f"nll={float(nll):.4e}",
                                f"cbe={float(cbe):.4e}",
                                f"r_std={r_std:.3g}",
                                f"|r|p99={r_p99:.3g}",
                                f"score|p99={score_p99:.3g}",
                            ]
                        )
                    )

                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
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

                global_step += 1

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
