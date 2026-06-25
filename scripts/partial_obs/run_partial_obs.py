#!/usr/bin/env python
"""End-to-end pipeline for partial-observation Deep Potential.

Orchestrates:
    1. Mock data generation (Plummer sphere with partial observations)
    2. p_obs training (unconditional normalizing flow on observed dims)
    3. p_unk pre-training (conditional density on unobserved given observed)
    4. Joint training of Φ + p_unk with CBE + entropy
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

# Allow importing from parent scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils
import fit_all  # for get_optimizer_and_schedule
import flow_matching
import flow_ot_flow_matching
import flow_matching_conditional

from partial_obs.dim_spec import DimSpec
from partial_obs.mock_data import (
    generate_plummer_partial_obs,
    save_partial_obs_data,
    load_partial_obs_data,
)
from partial_obs.pobs_model import ObservedDensityFlow
from partial_obs.punk_models import make_punk_model
from partial_obs.symmetry_potential import ProjectedPotential
from partial_obs.joint_training import (
    prepare_training_data,
    split_precomputed,
    train_partial_obs,
)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Partial-observation Deep Potential on Plummer sphere.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--dim-spec", type=str, default="xyvz",
                        help="Observed dimensions, e.g. 'xyvz'.")
    parser.add_argument("--n-samples", type=int, default=100_000,
                        help="Number of Plummer samples.")
    parser.add_argument("--r-max", type=float, default=8.0,
                        help="Plummer max radius.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--run-dir", type=str, default="runs/partial_obs",
                        help="Output directory.")

    # p_obs training
    parser.add_argument("--train-pobs", action="store_true",
                        help="Train p_obs (otherwise load).")
    parser.add_argument("--pobs-epochs", type=int, default=100,
                        help="Epochs for p_obs training.")
    parser.add_argument("--pobs-batch-size", type=int, default=5000)
    parser.add_argument("--pobs-lr", type=float, default=1e-3)
    parser.add_argument("--pobs-width", type=int, default=64,
                        help="Width of p_obs flow vector field.")

    # p_unk model
    parser.add_argument("--punk-type", type=str, default="gaussian",
                        choices=["gaussian", "gaussian_mixture",
                                 "discrete_flow", "flow"],
                        help="Type of p_unk conditional density model.")
    parser.add_argument("--punk-width", type=int, default=64,
                        help="Width of p_unk ResNet.")
    parser.add_argument("--punk-depth", type=int, default=3,
                        help="Depth of p_unk ResNet.")
    parser.add_argument("--punk-n-layers", type=int, default=3,
                        help="Number of coupling layers (discrete_flow only).")
    parser.add_argument("--pretrain-punk", action="store_true",
                        help="Pre-train p_unk on (obs, unk) pairs.")
    parser.add_argument("--pretrain-punk-epochs", type=int, default=50)
    parser.add_argument("--pretrain-punk-batch-size", type=int, default=5000)
    parser.add_argument("--pretrain-punk-lr", type=float, default=1e-3)

    # Φ model
    parser.add_argument("--symmetry", type=str, default="spherical",
                        choices=["spherical", "cylindrical"],
                        help="Symmetry prior for Φ.")
    parser.add_argument("--phi-width", type=int, default=64)
    parser.add_argument("--phi-depth", type=int, default=3)

    # Joint training
    parser.add_argument("--n-epochs", type=int, default=2000,
                        help="Joint training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Loss hyperparameters
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="CBE arcsinh sharpness.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Poisson arcsinh sharpness.")
    parser.add_argument("--lambda-poisson", type=float, default=1.0,
                        help="Poisson penalty strength.")
    parser.add_argument("--gamma-entropy", type=float, default=0.1,
                        help="Entropy regularization strength.")
    parser.add_argument("--l2-phi", type=float, default=0.01,
                        help="L2 regularization on Φ.")
    parser.add_argument("--l2-punk", type=float, default=0.01,
                        help="L2 regularization on p_unk.")

    # Misc
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Checkpoint frequency in epochs.")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate mock data, then exit.")
    parser.add_argument("--load-data-dir", type=str, default=None,
                        help="Load pre-generated data instead of generating.")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = run_dir / "data"
    model_dir = run_dir / "models"
    pobs_dir = model_dir / "pobs"
    punk_dir = model_dir / "punk"
    phi_dir = model_dir / "phi"
    joint_dir = model_dir / "joint"

    dim_spec = DimSpec.from_string(args.dim_spec)
    key = jax.random.key(args.seed)

    print("=" * 60)
    print(f"Partial-Observation Deep Potential: {dim_spec}")
    print(f"  Symmetry: {args.symmetry}")
    print(f"  p_unk type: {args.punk_type}")
    print(f"  Run directory: {run_dir}")
    print("=" * 60)

    # =====================================================================
    # Step 1: Generate or load mock data
    # =====================================================================
    print("\n--- Step 1: Mock Data ---")

    if args.load_data_dir:
        print(f"Loading pre-generated data from {args.load_data_dir}")
        data = load_partial_obs_data(Path(args.load_data_dir))
    else:
        if not (data_dir / "obs_data.h5").exists():
            print(f"Generating {args.n_samples:,} Plummer samples...")
            data = generate_plummer_partial_obs(
                n_samples=args.n_samples,
                dim_spec=dim_spec,
                r_max=args.r_max,
                seed=args.seed,
            )
            save_partial_obs_data(data, data_dir, dim_spec)
        else:
            print(f"Loading existing data from {data_dir}")
            data = load_partial_obs_data(data_dir)

    eta_obs = data["eta_obs"]
    eta_unk = data["eta_unk"]
    print(f"  Observed: {eta_obs.shape}, Unobserved: {eta_unk.shape}")

    if args.generate_only:
        print("Done (--generate-only).")
        return 0

    # =====================================================================
    # Step 2: Train or load p_obs
    # =====================================================================
    print("\n--- Step 2: p_obs Model ---")
    pobs_path = pobs_dir / "pobs_model.eqx"

    if args.train_pobs or not pobs_path.exists():
        print("Training p_obs (unconditional flow on observed dims)...")

        # Prepare data in the format expected by flow_matching
        obs_mean = np.mean(eta_obs, axis=0)
        obs_std = np.std(eta_obs, axis=0)

        train_data, val_data = utils.split_data(
            {"eta": eta_obs, "weights": np.ones(len(eta_obs), dtype=np.float32)},
            val_split=0.25,
        )

        vf_params = {
            "input_dim": dim_spec.obs_dim,
            "width": args.pobs_width,
            "depth": 3,
            "cond_dim": 0,
            "base_dist_dim": dim_spec.obs_dim,
        }

        key, subkey = jax.random.split(key)
        pobs_model = flow_ot_flow_matching.NormalizingFlow(
            key=subkey,
            data_mean=obs_mean,
            data_std=obs_std,
            vector_field_params=vf_params,
            model_dir=str(pobs_dir),
        )

        # Simple optimizer
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.pobs_lr,
            warmup_steps=500,
            decay_steps=args.pobs_epochs * (eta_obs.shape[0] // args.pobs_batch_size),
            end_value=1e-6,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(schedule),
        )

        pobs_model, loss_history = flow_matching.train_flow_matching_model(
            key=key,
            model=pobs_model,
            optimizer=optimizer,
            schedule=schedule,
            schedule_type="warmup_cosine_decay",
            lr_final=1e-6,
            train_data=train_data,
            val_data=val_data,
            norm_mean=obs_mean,
            norm_std=obs_std,
            epochs=args.pobs_epochs,
            batch_size=args.pobs_batch_size,
            time_scheduler_type="uniform",
            loss_params={},
        )

        # Wrap in ObservedDensityFlow using the trained flow
        pobs_model_wrapped = ObservedDensityFlow.from_flow(dim_spec, pobs_model)
        pobs_model_wrapped.save(pobs_path)
        print(f"  Saved to {pobs_path}")
    else:
        print(f"Loading p_obs from {pobs_path}")
        pobs_model_wrapped = ObservedDensityFlow.load(pobs_path)

    print(f"  p_obs parameters: {pobs_model_wrapped.count_parameters():,}")

    # =====================================================================
    # Step 3: Pre-train p_unk
    # =====================================================================
    print(f"\n--- Step 3: p_unk Pre-training ({args.punk_type}) ---")
    punk_path = punk_dir / "punk_pretrained.eqx"

    key, subkey = jax.random.split(key)
    punk_model = make_punk_model(
        model_type=args.punk_type,
        key=subkey,
        unk_dim=dim_spec.unk_dim,
        obs_dim=dim_spec.obs_dim,
        width_size=args.punk_width,
        depth=args.punk_depth,
        n_layers=args.punk_n_layers,
    )

    if args.pretrain_punk or not punk_path.exists():
        print(f"Pre-training {args.punk_type} p_unk on (obs, unk) pairs...")

        # Maximum likelihood pre-training
        n_train = int(len(eta_obs) * 0.75)
        train_obs = jnp.array(eta_obs[:n_train])
        train_unk = jnp.array(eta_unk[:n_train])
        val_obs = jnp.array(eta_obs[n_train:])
        val_unk = jnp.array(eta_unk[n_train:])

        optimizer = optax.adam(args.pretrain_punk_lr)
        opt_state = optimizer.init(eqx.filter(punk_model, eqx.is_array))

        @eqx.filter_jit
        def punk_pretrain_step(model, opt_state, obs_batch, unk_batch):
            def nll(m):
                return -jnp.mean(m.log_prob(unk_batch, obs_batch))
            loss, grads = eqx.filter_value_and_grad(nll)(model)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        batch_size = args.pretrain_punk_batch_size
        steps_per_epoch = n_train // batch_size

        best_val_loss = float("inf")
        for epoch in range(args.pretrain_punk_epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, n_train)

            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                idx = perm[step * batch_size:(step + 1) * batch_size]
                if len(idx) < batch_size:
                    continue
                punk_model, opt_state, loss = punk_pretrain_step(
                    punk_model, opt_state,
                    train_obs[idx], train_unk[idx],
                )
                epoch_loss += float(loss)

            # Validation
            val_nll = -jnp.mean(punk_model.log_prob(val_obs[:5000], val_unk[:5000]))
            epoch_loss /= steps_per_epoch

            if epoch % 10 == 0 or epoch == args.pretrain_punk_epochs - 1:
                print(f"  Epoch {epoch+1}: train NLL={epoch_loss:.4f}, "
                      f"val NLL={float(val_nll):.4f}")

            if float(val_nll) < best_val_loss:
                best_val_loss = float(val_nll)

        eqx.tree_serialise_leaves(punk_path, punk_model)
        print(f"  Saved to {punk_path}")
    else:
        print(f"Loading pre-trained p_unk from {punk_path}")
        punk_model = eqx.tree_deserialise_leaves(punk_path, like=punk_model)

    print(f"  p_unk parameters: {punk_model.count_parameters():,}")

    # =====================================================================
    # Step 4: Pre-compute ∇ ln p_obs
    # =====================================================================
    print("\n--- Step 4: Pre-computing ∇ ln p_obs ---")

    precomputed = prepare_training_data(
        pobs_model_wrapped,
        eta_obs,
        batch_size=1024,
    )
    precomputed_train, precomputed_val = split_precomputed(precomputed, val_frac=0.25)
    print(f"  Train: {precomputed_train['eta_obs'].shape[0]:,} samples")
    print(f"  Val:   {precomputed_val['eta_obs'].shape[0]:,} samples")

    # =====================================================================
    # Step 5: Jointly train Φ + p_unk
    # =====================================================================
    print(f"\n--- Step 5: Joint Training ({args.symmetry} symmetry) ---")

    key, subkey = jax.random.split(key)

    # Create Φ model
    phi_model = ProjectedPotential(
        key=subkey,
        symmetry_type=args.symmetry,
        width_size=args.phi_width,
        depth=args.phi_depth,
    )

    # Setup optimizer for both models
    n_train = precomputed_train["eta_obs"].shape[0]
    steps_per_epoch = n_train // args.batch_size

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=min(500, steps_per_epoch * 5),
        decay_steps=args.n_epochs * steps_per_epoch,
        end_value=1e-7,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule),
    )

    loss_params = {
        "alpha": args.alpha,
        "beta": args.beta,
        "lambda_poisson": args.lambda_poisson,
        "gamma_entropy": args.gamma_entropy,
        "l2_phi": args.l2_phi,
        "l2_punk": args.l2_punk,
    }

    phi_model, punk_model, loss_history = train_partial_obs(
        key=key,
        phi_model=phi_model,
        punk_model=punk_model,
        pobs_model=pobs_model_wrapped,
        dim_spec=dim_spec,
        precomputed_train=precomputed_train,
        precomputed_val=precomputed_val,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        loss_params=loss_params,
        checkpoint_frequency_epochs=args.checkpoint_every,
        checkpoint_dir=joint_dir,
    )

    # Save final models
    joint_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(joint_dir / "phi_final.eqx", phi_model)
    eqx.tree_serialise_leaves(joint_dir / "punk_final.eqx", punk_model)

    import json
    with open(joint_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    with open(joint_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nFinal models saved to {joint_dir}")
    print(f"  phi_final.eqx: {phi_model.count_parameters():,} parameters")
    print(f"  punk_final.eqx: {punk_model.count_parameters():,} parameters")
    print("Done!")

    return 0


if __name__ == "__main__":
    main()
