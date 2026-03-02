from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from dpjax.data import fit_normalizer, iter_batches, load_eta_h5
from dpjax.flows.realnvp import RealNVP, RealNVPConfig, log_prob_apply, score_apply
from dpjax.utils.ckpt import create_manager, finalize, restore_latest, save


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DF (RealNVP) on eta.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = _ensure_dir(args.run_dir)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    eta = load_eta_h5(args.data, dataset=cfg.get("data", {}).get("dataset", "eta"))

    normalizer = fit_normalizer(eta, eps=float(cfg.get("normalizer", {}).get("eps", 1.0e-6)))
    normalizer.save_npz(run_dir / "normalizer.npz")

    eta_std = normalizer.transform(eta)

    flow_cfg = cfg.get("flow", {})
    model = RealNVP(
        RealNVPConfig(
            dim=int(flow_cfg.get("dim", 6)),
            n_coupling=int(flow_cfg.get("n_coupling", 10)),
            hidden_sizes=tuple(int(x) for x in flow_cfg.get("hidden_sizes", [128, 128])),
            s_max=float(flow_cfg.get("s_max", 2.0)),
        )
    )

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8192))
    epochs = int(train_cfg.get("epochs", 32))
    lr = float(train_cfg.get("lr", 1.0e-3))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 200))
    max_to_keep = int(train_cfg.get("max_to_keep", 3))

    seed = int(cfg.get("seed", 0))
    rng = jax.random.key(seed)

    dummy = jnp.zeros((1, 6), dtype=jnp.float32)
    params = model.init(rng, dummy, method=RealNVP.log_prob)["params"]

    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = opt.init(params)

    ckpt_mgr = create_manager(run_dir / "ckpt", max_to_keep=max_to_keep)
    step0 = 0
    if args.resume:
        restored = restore_latest(ckpt_mgr)
        params = restored["params"]
        opt_state = restored["opt_state"]
        step0 = int(restored.get("step", 0))

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            lp = log_prob_apply(model, p, batch)
            return -jnp.mean(lp)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists() or not args.resume
    with metrics_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "epoch", "loss", "score_p50", "score_p99", "score_max_abs"])

        global_step = step0
        np_rng = np.random.default_rng(seed=seed)

        for epoch in range(epochs):
            for batch_np in iter_batches(eta_std, batch_size=batch_size, rng=np_rng, shuffle=True, drop_remainder=True):
                batch = jnp.asarray(batch_np)
                params, opt_state, loss = train_step(params, opt_state, batch)

                if (global_step % log_every) == 0:
                    # Light sanity check on a small slice
                    x_small = batch[:1024]
                    score = score_apply(model, params, x_small)
                    score_abs = jnp.abs(score)
                    p50 = float(jnp.percentile(score_abs, 50.0))
                    p99 = float(jnp.percentile(score_abs, 99.0))
                    smax = float(jnp.max(score_abs))
                    writer.writerow([global_step, epoch, float(loss), p50, p99, smax])
                    f.flush()
                    print(f"step={global_step} epoch={epoch} nll={float(loss):.6f} score| p50={p50:.3g} p99={p99:.3g} max={smax:.3g}")

                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    save(ckpt_mgr, global_step, {"params": params, "opt_state": opt_state, "step": global_step})

                global_step += 1

    save(ckpt_mgr, global_step, {"params": params, "opt_state": opt_state, "step": global_step})
    finalize(ckpt_mgr)
    print(f"Saved final checkpoint at step {global_step}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
