import jax.numpy as jnp
import jax
from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import optax
import numpy as np
from tqdm import trange

from flow_ot_flow_matching_conditional import ConditionalPhaseSpaceFlow, val_loss_fn, train_step, get_time_scheduler, custom_filter_spec


# ----------------- Main Training Function (Standard Flow Matching) -----------------
def train_flow_matching_model(
    key: PRNGKeyArray,
    model: ConditionalPhaseSpaceFlow,
    which_flow: str,
    optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
    schedule_type: str,
    lr_final: float,
    train_data: dict,
    val_data: dict,
    norm_mean: Array,
    norm_std: Array,
    epochs: int,
    batch_size: int,
    time_scheduler_type="uniform",
    loss_params={},
    time_logger=None,
    loss_history=None,
    checkpoint_frequency_epochs=-1,
    timeout_hours=None,
    early_stopping_loss_threshold=None
):
    """
    Trains either the spatial or conditional velocity component of a ConditionalPhaseSpaceFlow
    using standard (non-OT) flow matching.
    """
    # --- Data and Model Preparation ---
    train_x = (train_data['eta'] - norm_mean) / norm_std
    train_weights = train_data['weights']
    val_x = (val_data['eta'] - norm_mean) / norm_std
    val_weights = val_data['weights']

    if which_flow == "spatial":
        print("--- Training spatial flow n(x) with standard Flow Matching ---")
        train_x_cond, val_x_cond = None, None
        train_x, val_x = train_x[:, :3], val_x[:, :3]
        dynamics_net = model.spatial_flow.flow.bijection[0].dynamics_net
        train_label, val_label, lr_label, grads_label = 'train_pos', 'val_pos', 'lr_pos', 'global_grad_norms_pos'
        flow_save_prefix = 'flow_pos_only'
    elif which_flow == "conditional_velocity":
        print("--- Training conditional velocity flow f(v|x) with standard Flow Matching ---")
        train_x_cond, val_x_cond = train_x[:, :3], val_x[:, :3]
        train_x, val_x = train_x[:, 3:], val_x[:, 3:]
        dynamics_net = model.conditional_velocity_flow.flow.bijection[0].dynamics_net
        train_label, val_label, lr_label, grads_label = 'train_vel', 'val_vel', 'lr_vel', 'global_grad_norms_vel'
        flow_save_prefix = 'flow'
    else:
        raise ValueError("which_flow must be either 'spatial' or 'conditional_velocity'.")

    if loss_history is None:
        loss_history = {}
    if train_label not in loss_history:
        loss_history[train_label] = []
        loss_history[val_label] = []
        loss_history[lr_label] = []
        loss_history[grads_label] = []

    n_train = train_x.shape[0]
    val_batch_size = batch_size

    # --- Setup Schedulers, Optimizer, and Model Partition ---
    time_scheduler = get_time_scheduler(time_scheduler_type)

    # --- Partition Model into trainable/non-trainable parts and initialize the optimizer ---
    params, static = eqx.partition(dynamics_net, filter_spec=custom_filter_spec(dynamics_net))

    opt_state = optimizer.init(params)
    if grads_label in loss_history:
        global_grad_norms = loss_history[grads_label]
    else:
        global_grad_norms = []
    steps_per_epoch = n_train // batch_size
    start_epoch = len(loss_history[train_label])
    step = start_epoch * steps_per_epoch

    print("Starting Standard Flow Matching training...")
    print(f"Number of trainable parameters: {model.count_parameters()}")
    print(f"Number of steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}, Total training samples: {n_train}")
    print(f"Using time scheduler of type {time_scheduler_type}")
    print(f"Using loss params {loss_params}")

    # --- Training Loop ---
    avg_val_loss = 1000000.0
    for epoch in (pbar := trange(start_epoch, epochs)):
        # Shuffle training data at the beginning of each epoch
        key, key_shuffle = jax.random.split(key)
        perm = jax.random.permutation(key_shuffle, n_train)

        avg_train_loss, epoch_w = 0.0, 0.0
        epoch_lr, epoch_global_grad_norms = [], []
        for i in range(steps_per_epoch):
            if time_logger is not None:
                time_logger.start("Training | train setup")
            key, key_step, key_x0 = jax.random.split(key, 3)

            # Get batch indices from the shuffled permutation
            batch_indices = perm[i * batch_size: (i + 1) * batch_size]
            if len(batch_indices) != batch_size:
                continue # Skip incomplete batches

            # Sample x1 from data (target)
            x1_batch = train_x[batch_indices]
            x0_batch = jax.random.normal(key_x0, shape=x1_batch.shape)
            w_batch = train_weights[batch_indices]
            x1_cond_batch = train_x_cond[batch_indices] if train_x_cond is not None else None

            if time_logger is not None:
                time_logger.stop("Training | train setup")
                time_logger.start("Training | train backpropagation")

            params, opt_state, loss, grad_norm = train_step(
                params, static, opt_state, key_step, x0_batch, x1_batch, x1_cond_batch, w_batch, optimizer,
                time_scheduler=time_scheduler, schedule_type=schedule_type, val_loss=jnp.array(avg_val_loss),
                loss_params=loss_params
            )

            batch_w = float(jnp.sum(w_batch))
            avg_train_loss += float(loss.item()) * batch_w
            epoch_w += batch_w
            epoch_global_grad_norms.append(float(grad_norm.item()))
            if time_logger is not None:
                time_logger.stop("Training | train backpropagation")
                time_logger.start("Training | train lr update")

            if schedule_type == "step":
                lr = float(schedule(step)) * optax.tree.get(opt_state, "scale")
                epoch_lr.append(float(lr))
            else:
                lr = float(schedule(step))
                epoch_lr.append(lr)

            step += 1
            if time_logger is not None:
                time_logger.stop("Training | train lr update")

        # Finalize epoch stats
        avg_train_loss = avg_train_loss / epoch_w if epoch_w > 0 else 0.0

        # --- Validation Loop ---
        key, key_val_loop = jax.random.split(key)
        current_val_loss, val_w = 0.0, 0.0
        if time_logger is not None:
            time_logger.start("Training | val")

        for i in range(0, val_x.shape[0], val_batch_size):
            key_val_loop, key_step, key_x0 = jax.random.split(key_val_loop, 3)

            x1_batch = val_x[i: i + val_batch_size]
            x0_batch = jax.random.normal(key_x0, shape=x1_batch.shape)
            w_batch = val_weights[i: i + val_batch_size]
            x1_cond_batch = val_x_cond[i: i + val_batch_size] if val_x_cond is not None else None

            v_loss = val_loss_fn(params, static, key_step, x0_batch, x1_batch, x1_cond_batch, w_batch, time_scheduler, **loss_params)
            current_val_loss += float(v_loss.item()) * float(jnp.sum(w_batch))
            val_w += float(jnp.sum(w_batch))

        avg_val_loss = current_val_loss / val_w if val_w > 0 else 0.0

        # --- Logging and Checkpointing ---
        loss_history[train_label].append(avg_train_loss)
        loss_history[val_label].append(avg_val_loss)
        loss_history[lr_label].append(float(np.mean(epoch_lr)) if epoch_lr else 0.0)
        loss_history[grads_label].append(float(np.mean(epoch_global_grad_norms)) if epoch_global_grad_norms else 0.0)
        if time_logger is not None:
            time_logger.stop("Training | val")

        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"lr: {loss_history[lr_label][-1]:.4f}"
        )

        if checkpoint_frequency_epochs > 0 and (epoch + 1) % checkpoint_frequency_epochs == 0:
            dynamics_net = eqx.combine(params, static)
            if which_flow == "spatial":
                model = eqx.tree_at(lambda m: m.spatial_flow.flow.bijection[0].dynamics_net, model, dynamics_net)
            else:
                model = eqx.tree_at(lambda m: m.conditional_velocity_flow.flow.bijection[0].dynamics_net, model, dynamics_net)
            model = model.save(loss_history=loss_history, save_prefix=flow_save_prefix)

        if schedule_type == "step" and loss_history[lr_label][-1] < lr_final:
            print("Early stopping triggered by learning rate threshold.")
            break

        # If ETA is too long, break
        if timeout_hours is not None:
            elapsed_time = pbar.format_dict["elapsed"]

            # Only do this after 30 seconds
            if elapsed_time > 30 and epoch > 2:
                rate = pbar.format_dict["rate"]
                total = pbar.format_dict["total"]
                n = pbar.format_dict['n']
                remaining_time = (total - n) / rate if rate > 0 else float('inf')
                if remaining_time > timeout_hours * 3600:
                    print("Timeout limit reached, stopping training.")
                    return model, None

        # Early stopping from 50 epochs onward
        if early_stopping_loss_threshold is not None and epoch >= 50:
            if avg_val_loss > early_stopping_loss_threshold:
                print(f"Early stopping triggered by validation loss threshold: {early_stopping_loss_threshold}.")
                return model, None

    # --- Final Save ---
    dynamics_net = eqx.combine(params, static)
    if which_flow == "spatial":
        model = eqx.tree_at(lambda m: m.spatial_flow.flow.bijection[0].dynamics_net, model, dynamics_net)
    else:
        model = eqx.tree_at(lambda m: m.conditional_velocity_flow.flow.bijection[0].dynamics_net, model, dynamics_net)
    model = model.save(loss_history=loss_history, save_prefix=flow_save_prefix)

    return model, loss_history
