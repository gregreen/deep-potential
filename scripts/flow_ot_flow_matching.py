import jax.numpy as jnp
import jax
from jaxtyping import Array, Float, PRNGKeyArray
from jax.tree_util import tree_map_with_path, GetAttrKey
from typing import Optional, Dict, Any

import flowjax.distributions as flowjax_dist
import flowjax.bijections as flowjax_bij

import equinox as eqx
import diffrax
import optax
import e3nn_jax as e3nn

from pathlib import Path
from datetime import datetime
import json
import numpy as np
from tqdm import trange
import re
import bisect

import flow_vector_fields as fvf
import utils


class NormalizingFlow(eqx.Module):
    """
    A wrapper for a `flowjax` transformed distribution, representing the normalizing flow model.

    This class encapsulates the flow model, its metadata, and provides methods for
    saving and loading the model state and training history.

    Attributes:
        flow (flowjax_dist.Transformed): The underlying flowjax model. The model is equipped
        with a base distribution (a normal distribution) and a bijection (defined by a vector field) 
        checkpoint_index (int): The current index for saving checkpoints.
        model_dir (str): The directory where the model is saved.
        metadata (Dict[str, Any]): A dictionary containing metadata about the model.
    """
    flow: flowjax_dist.Transformed
    checkpoint_index: int
    model_dir: str = eqx.field(static=True, default="")
    metadata: Dict[str, Any] = eqx.field(static=True)

    def __init__(self, key, data_mean, data_std, vector_field_params={}, model_dir=None, checkpoint_index=0):
        """
        Initializes the NormalizingFlow model.

        The model consists of a base normal distribution transformed by a chain of bijections:
        1. A vector field integrator (the trainable part).
        2. An affine transformation to denormalize the output to the original data scale.

        Args:
            key (PRNGKeyArray): JAX random key for initializing the vector field.
            data_mean (Array): Mean of the training data for denormalization.
            data_std (Array): Standard deviation of the training data for denormalization.
            vector_field_params (dict): Parameters for initializing the `fvf.VectorField`.
            model_dir (str): Directory to save model checkpoints.
            checkpoint_index (int): Initial checkpoint index.
        """
        dim = data_mean.shape[0]

        base_dist = flowjax_dist.Normal(
            jnp.zeros(dim), jnp.ones(dim)
        )

        if type(vector_field_params) is not dict:
            flow_bij = vector_field_params
        else:
            flow_bij = fvf.VectorField(key, vector_field_params)
        denormalizer = flowjax_bij.Affine(
            scale=data_std, loc=data_mean
        )

        self.flow = flowjax_dist.Transformed(
            base_dist, flowjax_bij.Chain(
                [flow_bij, denormalizer]
            )
        )

        self.checkpoint_index = checkpoint_index
        self.model_dir = model_dir

        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'training_epochs': 0,
            'vector_field_type': str(self.flow).splitlines(),
            'input_dim': dim,
            'num_parameters': self.count_parameters(),
        }

    def count_parameters(self):
        """Counts the total number of trainable parameters in the flow."""
        arrays = eqx.filter(self.flow, eqx.is_array)
        return sum(x.size for x in jax.tree_util.tree_leaves(arrays))

    def log_prob(self, x):
        """Computes the log probability of the data `x`."""
        return self.flow.log_prob(x)

    def sample(self, key, num_samples):
        """Samples from the flow's distribution."""
        return self.flow.sample(key, num_samples)

    def save(self, loss_history=None):
        """
        Saves the model state and optional loss history to the model directory.

        Each save creates a new checkpoint with an incremented index.

        Args:
            loss_history (dict, optional): A dictionary containing training and
                                           validation loss history.

        Returns:
            NormalizingFlow: An updated instance of the model with an incremented
                             checkpoint index.
        """
        path = Path(self.model_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save flow model
        print(f'Saving to checkpoint with index {self.checkpoint_index}')

        name_prefix = "flow"
        eqx.tree_serialise_leaves(path / f"{name_prefix}-{self.checkpoint_index}_model.eqx", self)
        if loss_history is not None:
            with open(path / f"{name_prefix}-{self.checkpoint_index}_loss.json", "w") as f:
                json.dump(loss_history, f, indent=2)

        # Generate metadata
        with open(path / f"{name_prefix}-metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Save loss
        if loss_history is not None:
            utils.plot_loss_new(loss_history, Path(self.model_dir) / f"{name_prefix}-{self.checkpoint_index}_loss.pdf")

        # Increment checkpoint_index
        return eqx.tree_at(lambda tree: tree.checkpoint_index, self, self.checkpoint_index + 1)

    @classmethod
    def load(cls, model_dir, empty_model, load_history=True, load_index=-1):
        """
        Loads a model from a checkpoint file.

        Args:
            model_dir (Path): The directory containing the model checkpoints.
            empty_model: An instance of the model with the correct structure to load into.
            load_history (bool): If True, also loads the corresponding loss history.
            load_index (int): The checkpoint index to load. If -1, loads the latest checkpoint.

        Returns:
            tuple: A tuple containing the loaded model and the loss history (or None).
        """
        model_dir = Path(model_dir)
        name_suffix = 'flow'

        # Determine the model file to load
        checkpoint_index = -1
        if load_index == -1:
            # Find the latest checkpoint file if no index is specified.
            model_files = list(model_dir.glob(f"{name_suffix}-*_model.eqx"))
            if model_files == []:
                raise FileNotFoundError(f"No model files found in {model_dir} with prefix '{name_suffix}-' and suffix '_model.eqx'")
            else:
                # Extract suffixes and find the maximum
                suffixes = []
                for f in model_files:
                    try:
                        suffix = f.stem.split('-')[1]
                        suffix = int(suffix.split('_')[0])
                        suffixes.append(suffix)
                    except (IndexError, ValueError):
                        continue
                load_index = max(suffixes)
        checkpoint_index = load_index
        model_file = model_dir / f"{name_suffix}-{load_index}_model.eqx"
        print(f"Loading model from {model_file}")

        print(f"Loading checkpoint index {checkpoint_index}")
        # Load flow model
        model = eqx.tree_deserialise_leaves(model_file, like=empty_model)
        # Print the model standard deviation value
        print("Model loaded!")

        # Load metadata (always from metadata.json)
        metadata_file = model_dir / f"{name_suffix}-metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                model.metadata.update(json.load(f))

        loss_history = None
        if load_history:
            # Determine the loss history file to load
            if load_index == -1:
                history_file = model_dir / "loss_history.json"
            else:
                history_file = model_dir / f"{name_suffix}-{load_index}_loss.json"
            print(f"Loading loss history from {history_file}")

            if history_file.exists():
                with open(history_file, 'r') as f:
                    loss_history = json.load(f)

        return model, loss_history


class PrecomputedOTIndexLoader:
    """
    A lazy loader for OT pairing indices stored across multiple chunked files.

    This class provides a list-like interface (`loader[epoch_nr]`) to access
    precomputed indices for each training epoch. It only loads the required file
    chunk into memory when an epoch from that chunk is requested, making it
    efficient for very large datasets.

    Args:
        base_path (Path): The base path for index files (e.g., 'data/pairings_train').
                          The loader searches for files like 'data/pairings_train_0_500.npz'.
        n (int): The number of samples in the dataset, used for reshaping the loaded arrays.
    """
    def __init__(self, base_path: Path, n: int):
        self.base_path = base_path
        self.n = n
        self._file_map = {}
        self._epoch_starts = []
        self._total_epochs = 0

        self._discover_files()

        # Cache for the most recently loaded file
        self._cached_file_path = None
        self._cached_start_epoch = -1
        self._cached_x0 = None
        self._cached_x1 = None

    def _discover_files(self):
        """Finds all index files and maps epoch ranges to file paths."""
        directory = self.base_path.parent
        file_stem = self.base_path.stem

        # Regex to match filenames like 'pairings_train_epochs_0_500'
        pattern = re.compile(rf"{re.escape(file_stem)}_(\d+)_(\d+)")

        found_files = list(directory.glob(f"{file_stem}_*_*.npz"))
        if not found_files:
            raise FileNotFoundError(
                f"No precomputed index files found in {directory} "
                f"matching the pattern '{file_stem}_*_*'.npz"
            )

        for f_path in found_files:
            match = pattern.match(f_path.stem)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                self._file_map[start] = (f_path, end)

        if not self._file_map:
            raise FileNotFoundError(
                f"Could not parse any valid epoch ranges from files in {directory} "
                f"with base name '{file_stem}'."
            )
        print(found_files)

        # Sort by start epoch and determine total epochs
        self._epoch_starts = sorted(self._file_map.keys())
        last_start_epoch = self._epoch_starts[-1]
        _, self._total_epochs = self._file_map[last_start_epoch]
        print(f"Discovered {len(self._file_map)} index files, covering {self._total_epochs} total epochs.")


    def __len__(self):
        """Returns the total number of epochs available."""
        return self._total_epochs

    def __getitem__(self, epoch_nr: int):
        """
        Loads and returns the indices for a specific epoch.

        This method finds the correct file, loads it if not already cached,
        and returns the requested epoch's index data.
        """
        if not isinstance(epoch_nr, int):
            raise TypeError("Index must be an integer.")

        # Clip the epoch number to valid range
        epoch_nr = epoch_nr % self._total_epochs

        # Find which file contains the requested epoch
        # `bisect_right` finds the insertion point, and -1 gives the index of the
        # start_epoch that is less than or equal to epoch_nr.
        file_start_idx = bisect.bisect_right(self._epoch_starts, epoch_nr) - 1
        start_epoch = self._epoch_starts[file_start_idx]
        file_path, end_epoch = self._file_map[start_epoch]

        # Load file if it's not the one we have in cache
        if file_path != self._cached_file_path:
            data = np.load(file_path)
            x0 = data['x0_perm_indices']
            x1 = data['x1_perm_indices']

            # Reshape the loaded chunk
            epochs_in_file = end_epoch - start_epoch
            self._cached_x0 = jnp.array(x0).reshape(epochs_in_file, self.n)
            self._cached_x1 = jnp.array(x1).reshape(epochs_in_file, self.n)
            self._cached_file_path = file_path
            self._cached_start_epoch = start_epoch

        # Calculate the index within the loaded (cached) chunk
        local_epoch_idx = epoch_nr - self._cached_start_epoch

        return self._cached_x0[local_epoch_idx], self._cached_x1[local_epoch_idx]


# ----------------- Loss Functions and Training Step -----------------


@eqx.filter_value_and_grad
def loss_fn(params, static, key, x0, x1, weights, time_scheduler):
    # We are given the OT-paired (x0, x1), so we just proceed with Flow Matching
    t = time_scheduler(key, (x1.shape[0], 1))
    xt = t * x1 + (1 - t) * x0
    ut = x1 - x0  # Target vector field
    net = eqx.combine(params, static)
    vt = jax.vmap(net)(t.squeeze(-1), xt)  # Predicted vector field
    loss = jnp.sum(weights[:, None] * (vt - ut)**2) / jnp.sum(weights)
    return loss


@eqx.filter_jit
def val_loss_fn(params, static, key, x0, x1, weights, time_scheduler):
    # We are given the OT-paired (x0, x1), so we just proceed with Flow Matching
    t = time_scheduler(key, (x1.shape[0], 1))
    xt = t * x1 + (1 - t) * x0
    ut = x1 - x0
    net = eqx.combine(params, static)
    vt = jax.vmap(net)(t.squeeze(-1), xt)
    loss = jnp.sum(weights[:, None] * (vt - ut)**2) / jnp.sum(weights)
    return loss


@eqx.filter_value_and_grad
def loss_fn_sb(params, static, key, x0, x1, weights, time_scheduler, sb_constant):
    # We are given the OT-paired (x0, x1), so we just proceed with Flow Matching
    t_key, z_key = jax.random.split(key, 2)
    t = time_scheduler(t_key, (x1.shape[0], 1))
    mu_t = t * x1 + (1 - t) * x0
    z = jax.random.normal(z_key, shape=x1.shape)

    eps = 1e-8
    epsilon_t = sb_constant * jnp.sqrt(t * (1 - t) + eps)

    xt = mu_t + epsilon_t * z
    ut = x1 - x0 + (1 - 2 * t) / (2 * t * (1 - t) + eps) * (xt - mu_t)

    net = eqx.combine(params, static)
    vt = jax.vmap(net)(t.squeeze(-1), xt)
    loss = jnp.sum(weights[:, None] * (vt - ut)**2) / jnp.sum(weights)
    return loss


@eqx.filter_jit
def val_loss_fn_sb(params, static, key, x0, x1, weights, time_scheduler, sb_constant):
    # We are given the OT-paired (x0, x1), so we just proceed with Flow Matching
    t_key, z_key = jax.random.split(key, 2)
    t = time_scheduler(t_key, (x1.shape[0], 1))
    mu_t = t * x1 + (1 - t) * x0
    z = jax.random.normal(z_key, shape=x1.shape)

    eps = 1e-8
    epsilon_t = sb_constant * jnp.sqrt(t * (1 - t) + eps)

    xt = mu_t + epsilon_t * z
    ut = x1 - x0 + (1 - 2 * t) / (2 * t * (1 - t) + eps) * (xt - mu_t)

    net = eqx.combine(params, static)
    vt = jax.vmap(net)(t.squeeze(-1), xt)
    loss = jnp.sum(weights[:, None] * (vt - ut)**2) / jnp.sum(weights)
    return loss


@eqx.filter_jit
def train_step(params, static, opt_state, key, x0, x1, weights, optimizer, sb_constant, time_scheduler):
    if sb_constant > 0:
        loss, grads = loss_fn_sb(params, static, key, x0, x1, weights, time_scheduler, sb_constant)
    else:
        loss, grads = loss_fn(params, static, key, x0, x1, weights, time_scheduler)
    global_grad_norm = optax.global_norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss, global_grad_norm


# ----------------- Time Schedulers -----------------


def uniform_scheduler(key, shape):
    """Standard uniform sampling for t."""
    return jax.random.uniform(key, shape)


def power_law_scheduler(key, shape, p=2.0):
    """
    Samples t according to a power law: t = u^(1/p).
    p > 1.0 weighs t=1 more heavily (e.g., p=2.0 for a sqrt distribution of u).
    """
    u = jax.random.uniform(key, shape)
    return u**(1 / p)


def logit_normal_scheduler(key, shape, mu=0.0, sigma=1.0):
    """
    Logit-Normal sampling, which concentrates samples around t=0.5.
    """
    u = jax.random.uniform(key, shape)
    # Use the inverse CDF of the normal distribution (percent point function)
    x = jax.scipy.stats.norm.ppf(u, loc=mu, scale=sigma)
    return jax.scipy.special.expit(x)


# ----------------- Main Training Function -----------------


def train_ot_flow_matching_model(
    key: PRNGKeyArray,
    model: NormalizingFlow,
    optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
    schedule_type: str,
    train_data: Array,
    val_data: Array,
    norm_mean: Array,
    norm_std: Array,
    epochs: int,
    batch_size: int,
    ot_pairings_path_stem: Path,
    time_scheduler_type=None,
    sb_constant: float = 0.0,
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
    n_val = val_x.shape[0]
    dynamics_net = model.flow.bijection[0].dynamics_net
    val_batch_size = batch_size

    # --- Setup Data Loaders and Schedulers ---
    path_train_ot = ot_pairings_path_stem.with_name(ot_pairings_path_stem.name + "_train_epochs")
    x0_x1_train_indices_loader = PrecomputedOTIndexLoader(path_train_ot, n=n_train)
    path_val_ot = ot_pairings_path_stem.with_name(ot_pairings_path_stem.name + "_val_epochs")
    x0_x1_val_indices_loader = PrecomputedOTIndexLoader(path_val_ot, n=n_val)

    if time_scheduler_type == "uniform":
        time_scheduler = uniform_scheduler
    elif time_scheduler_type == "power_law":
        time_scheduler = power_law_scheduler
    elif time_scheduler_type == "logit_normal":
        time_scheduler = logit_normal_scheduler

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

    print("Starting OT Flow Matching training with pre-computed indices...")
    print(f"Number of trainable parameters: {model.count_parameters()}")
    print(f"Number of steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}, Total training samples: {n_train}")
    print(f"Using time scheduler of type {time_scheduler}")
    print(f"Using Schrodinger bridge coefficient of {sb_constant}")
    start_epoch = len(loss_history['lr'])
    step = start_epoch * steps_per_epoch  # Continue from previous step if resuming

    if schedule_type == "step":
        # This currently does not work when resuming from checkpoint
        val_loss_min = 99999999
        val_loss_avg = 99999999
        steps_since_decline = 0
        early_stopping = False

    # Pre-generate keys to ensure reproducibility with the OT pairing script.
    key_x0_train, key_x0_val = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    keys_x0_train_epoch, keys_x0_val_epoch = [], []
    for _ in range(epochs):
        key_x0_train, subkey = jax.random.split(key_x0_train)
        keys_x0_train_epoch.append(subkey)
        key_x0_val, subkey = jax.random.split(key_x0_val)
        keys_x0_val_epoch.append(subkey)

    for epoch in (pbar := trange(start_epoch, epochs)):
        if time_logger is not None:
            time_logger.start("Training | setup")
        # Derive keys for x0 generation and permutation exactly as in the pre-computation script
        key_x0_train = keys_x0_train_epoch[epoch % x0_x1_train_indices_loader._total_epochs]
        key_x0_train, _ = jax.random.split(key_x0_train, 2)
        key_x0_val = keys_x0_val_epoch[epoch % x0_x1_val_indices_loader._total_epochs]
        key_x0_val, _ = jax.random.split(key_x0_val, 2)
        key, key_loop, key_shuffle = jax.random.split(key, 3)

        # --- Reproduce the EXACT x0 and permutation for this epoch ---
        x0_train_epoch = jax.random.normal(key_x0_train, shape=train_x.shape)
        x0_val_epoch = jax.random.normal(key_x0_val, shape=val_x.shape)

        # Get the pairing indices for the current epoch
        x0_indices_epoch, x1_indices_epoch = x0_x1_train_indices_loader[epoch]
        # Additionally shuffle the indices to help the later epochs which start repeating
        perm = jax.random.permutation(key_shuffle, n_train)
        x0_indices_epoch = x0_indices_epoch[perm]
        x1_indices_epoch = x1_indices_epoch[perm]
        if time_logger is not None:
            time_logger.stop("Training | setup")

        avg_train_loss, epoch_w = 0.0, 0.0 # avg training loss is the weighted average
        epoch_lr = []
        epoch_global_grad_norms = []
        for i in range(steps_per_epoch):
            if time_logger is not None:
                time_logger.start("Training | train setup")
            key_loop, key_step = jax.random.split(key_loop)

            # These are indices into the shuffled training set
            x0_idx = x0_indices_epoch[i * batch_size: (i + 1) * batch_size]
            x1_idx = x1_indices_epoch[i * batch_size: (i + 1) * batch_size]

            # Construct the batch from the full datasets using the paired indices
            x0_batch = x0_train_epoch[x0_idx]
            x1_batch = train_x[x1_idx]
            w_batch = train_weights[x1_idx]
            if i == steps_per_epoch - 1 and x0_batch.shape[0] != batch_size:
                continue

            if time_logger is not None:
                time_logger.stop("Training | train setup")
                time_logger.start("Training | train backpropagation")
            params, opt_state, loss, global_grad_norm = train_step(
                params, static, opt_state, key_step, x0_batch, x1_batch, w_batch, optimizer,
                sb_constant=sb_constant, time_scheduler=time_scheduler
            )

            avg_train_loss += float(loss.item()) * float(np.sum(w_batch))
            epoch_w += float(np.sum(w_batch))
            epoch_global_grad_norms.append(float(global_grad_norm.item()))
            if time_logger is not None:
                time_logger.stop("Training | train backpropagation")
                time_logger.start("Training | train lr update")

            if schedule_type == "step":
                opt_state, val_loss_min, steps_since_decline, early_stopping = schedule(
                    opt_state, val_loss_min, val_loss_avg, loss_history['val'], steps_since_decline, step
                )
                lr = float(opt_state.hyperparams["learning_rate"])
                epoch_lr.append(lr)
                if early_stopping:
                    break
            else:
                lr = float(schedule(step))
                epoch_lr.append(lr)

            step += 1
            if time_logger is not None:
                time_logger.stop("Training | train lr update")

        avg_train_loss /= epoch_w
        avg_global_grad_norm = np.mean(epoch_global_grad_norms)
        global_grad_norms.append(avg_global_grad_norm)

        # Calculate validation loss
        key, key_loop, key_shuffle = jax.random.split(key, 3)
        x0_indices_epoch, x1_indices_epoch = x0_x1_val_indices_loader[epoch]
        perm = jax.random.permutation(key_shuffle, n_val)
        x0_indices_epoch = x0_indices_epoch[perm]
        x1_indices_epoch = x1_indices_epoch[perm]
        # Batch validation loss computation
        avg_val_loss, val_w = 0.0, 0.0
        if time_logger is not None:
            time_logger.start("Training | val")

        for i in range(0, val_x.shape[0], val_batch_size):
            key_loop, key_step = jax.random.split(key_loop)
            x0_idx = x0_indices_epoch[i: i + val_batch_size]
            x1_idx = x1_indices_epoch[i: i + val_batch_size]
            x0_batch = x0_val_epoch[x0_idx]
            x1_batch = val_x[x1_idx]
            w_batch = val_weights[x1_idx]
            if sb_constant > 0:
                v_loss = val_loss_fn_sb(params, static, key_loop, x0_batch, x1_batch, w_batch, time_scheduler, sb_constant)
            else:
                v_loss = val_loss_fn(params, static, key_loop, x0_batch, x1_batch, w_batch, time_scheduler)
            avg_val_loss += float(v_loss.item()) * float(np.sum(w_batch))
            val_w += float(np.sum(w_batch))
        avg_val_loss /= val_w

        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)
        loss_history['lr'].append(float(np.mean(epoch_lr)))
        if time_logger is not None:
            time_logger.stop("Training | val")

        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"lr: {loss_history['lr'][-1]:.4f}"
        )
        if schedule_type == "step" and early_stopping:
            print("Early stopping triggered.")
            break

        # Checkpoint if needed
        if checkpoint_frequency_epochs > 0 and epoch > 0 and epoch % checkpoint_frequency_epochs == 0:
            dynamics_net = eqx.combine(params, static)
            model = eqx.tree_at(lambda m: m.flow.bijection[0].dynamics_net, model, dynamics_net)
            model = model.save(loss_history=loss_history)

    dynamics_net = eqx.combine(params, static)
    model = eqx.tree_at(lambda m: m.flow.bijection[0].dynamics_net, model, dynamics_net)

    # Training is finished, let's checkpoint!
    model.save(loss_history=loss_history)
    loss_history['global_grad_norms'] = global_grad_norms
    return model, loss_history
