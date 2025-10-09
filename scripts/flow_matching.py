import jax.numpy as jnp
import jax
from jaxtyping import Array, Float, PRNGKeyArray
from jax.tree_util import tree_map_with_path, GetAttrKey

import equinox as eqx
import optax
import numpy as np
from tqdm import trange

from flow_ot_flow_matching import NormalizingFlow, val_loss_fn, train_step, uniform_scheduler, power_law_scheduler, logit_normal_scheduler


# ----------------- Main Training Function -----------------
def train_flow_matching_model(
    key: PRNGKeyArray,
    model: NormalizingFlow,
    optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
    schedule_type: str,
    lr_final: float,
    train_data: Array,
    val_data: Array,
    norm_mean: Array,
    norm_std: Array,
    epochs: int,
    batch_size: int,
    time_scheduler_type=None,
    loss_params={},
    time_logger=None,
    loss_history={'train': [], 'val': [], 'lr': []},
    checkpoint_frequency_epochs=-1,
):
    # Normalize data for training.
    train_x = (train_data['eta'] - norm_mean) / norm_std
    train_weights = train_data['weights']
    val_x = (val_data['eta'] - norm_mean) / norm_std
    val_weights = val_data['weights']

    n_train = train_x.shape[0]
    dynamics_net = model.flow.bijection[0].dynamics_net
    val_batch_size = batch_size

    # --- Setup Schedulers ---
    if time_scheduler_type == "uniform":
        time_scheduler = uniform_scheduler
    elif time_scheduler_type == "power_law":
        time_scheduler = power_law_scheduler
    elif time_scheduler_type == "logit_normal":
        time_scheduler = logit_normal_scheduler
    else:
        # Default to uniform scheduler if none is specified
        time_scheduler = uniform_scheduler

    # --- Partition Model into trainable/non-trainable parts and initialize the optimizer ---
    def custom_filter_spec(model):
        def filter_fn(path, leaf):
            final_key = path[-1]
            if isinstance(final_key, GetAttrKey):
                # Treat pos_mean and pos_std as non-trainable (static)
                if final_key.name in ["pos_mean", "pos_std"]:
                    return False
            # Treat all other JAX arrays as trainable parameters
            return isinstance(leaf, jax.Array)
        return tree_map_with_path(filter_fn, model)
    params, static = eqx.partition(dynamics_net, filter_spec=custom_filter_spec(dynamics_net))

    opt_state = optimizer.init(params)
    if "global_grad_norms" in loss_history:
        global_grad_norms = loss_history['global_grad_norms']
    else:
        global_grad_norms = []
    steps_per_epoch = n_train // batch_size

    print("Starting Standard Flow Matching training...")
    print(f"Number of trainable parameters: {model.count_parameters()}")
    print(f"Number of steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}, Total training samples: {n_train}")
    print(f"Using time scheduler of type {time_scheduler_type}")
    print(f"Using loss params {loss_params}")
    start_epoch = len(loss_history['lr'])
    step = start_epoch * steps_per_epoch  # Continue from previous step if resuming

    avg_val_loss = 1000000.0
    for epoch in (pbar := trange(start_epoch, epochs)):

        # Shuffle training data at the beginning of each epoch
        key, key_shuffle = jax.random.split(key)
        perm = jax.random.permutation(key_shuffle, n_train)

        avg_train_loss, epoch_w = 0.0, 0.0
        epoch_lr = []
        epoch_global_grad_norms = []
        for i in range(steps_per_epoch):
            if time_logger is not None:
                time_logger.start("Training | train setup")
            key, key_step, key_x0 = jax.random.split(key, 3)

            # Get batch indices from the shuffled permutation
            batch_indices = perm[i * batch_size: (i + 1) * batch_size]
            if len(batch_indices) != batch_size:
                continue # Skip incomplete batches

            # Sample x1 from data (target) and x0 from noise (source)
            x1_batch = train_x[batch_indices]
            x0_batch = jax.random.normal(key_x0, shape=x1_batch.shape)
            w_batch = train_weights[batch_indices]

            if time_logger is not None:
                time_logger.stop("Training | train setup")
                time_logger.start("Training | train backpropagation")

            # Note: avg_val_loss from the *previous* epoch is used here
            params, opt_state, loss, global_grad_norm = train_step(
                params, static, opt_state, key_step, x0_batch, x1_batch, w_batch, optimizer,
                time_scheduler=time_scheduler, schedule_type=schedule_type, val_loss=jnp.array(avg_val_loss),
                loss_params=loss_params
            )

            avg_train_loss += float(loss.item()) * float(jnp.sum(w_batch))
            epoch_w += float(jnp.sum(w_batch))
            epoch_global_grad_norms.append(float(global_grad_norm.item()))
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

        if epoch_w > 0:
            avg_train_loss /= epoch_w
        avg_global_grad_norm = np.mean(epoch_global_grad_norms) if epoch_global_grad_norms else 0.0
        global_grad_norms.append(avg_global_grad_norm)

        # --- Calculate validation loss ---
        key, key_val_loop = jax.random.split(key)
        current_val_loss, val_w = 0.0, 0.0
        if time_logger is not None:
            time_logger.start("Training | val")

        for i in range(0, val_x.shape[0], val_batch_size):
            key_val_loop, key_step, key_x0 = jax.random.split(key_val_loop, 3)

            # Sample x1 from validation data and x0 from noise
            x1_batch = val_x[i: i + val_batch_size]
            x0_batch = jax.random.normal(key_x0, shape=x1_batch.shape)
            w_batch = val_weights[i: i + val_batch_size]
            v_loss = val_loss_fn(params, static, key_step, x0_batch, x1_batch, w_batch, time_scheduler, **loss_params)
            current_val_loss += float(v_loss.item()) * float(jnp.sum(w_batch))
            val_w += float(jnp.sum(w_batch))

        if val_w > 0:
            avg_val_loss = current_val_loss / val_w

        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)
        loss_history['lr'].append(float(np.mean(epoch_lr)) if epoch_lr else 0.0)
        if time_logger is not None:
            time_logger.stop("Training | val")

        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"lr: {loss_history['lr'][-1]:.4f}"
        )
        # Checkpoint if needed
        if checkpoint_frequency_epochs > 0 and (epoch + 1) % checkpoint_frequency_epochs == 0:
            dynamics_net = eqx.combine(params, static)
            model = eqx.tree_at(lambda m: m.flow.bijection[0].dynamics_net, model, dynamics_net)
            model = model.save(loss_history=loss_history)

        if schedule_type == "step" and loss_history['lr'][-1] < lr_final:
            print("Early stopping triggered by learning rate threshold.")
            break

    dynamics_net = eqx.combine(params, static)
    model = eqx.tree_at(lambda m: m.flow.bijection[0].dynamics_net, model, dynamics_net)

    # Training is finished, let's checkpoint!
    model.save(loss_history=loss_history)
    loss_history['global_grad_norms'] = global_grad_norms
    return model, loss_history
