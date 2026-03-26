import jax.numpy as jnp
import jax
from jaxtyping import Array, Float, PRNGKeyArray
from jax.tree_util import tree_map_with_path, GetAttrKey
from typing import Optional, Dict, Any

import flowjax.distributions as flowjax_dist
import flowjax.bijections as flowjax_bij

import equinox as eqx
import optax

from pathlib import Path
from datetime import datetime
import json
import numpy as np
from tqdm import trange
import re
import bisect

import flow_vector_fields as fvf
import utils


def uniform_scheduler(key: PRNGKeyArray, shape: tuple) -> Array:
    """Standard uniform sampling for t in [0, 1]."""
    return jax.random.uniform(key, shape)


def power_law_scheduler(key: PRNGKeyArray, shape: tuple, p: float = 2.0) -> Array:
    """
    Samples t according to a power law: t = u^(1/p).

    p > 1.0 weighs t=1 more heavily (e.g., p=2.0 for a sqrt distribution of u).
    """
    u = jax.random.uniform(key, shape)
    return u**(1 / p)


def logit_normal_scheduler(
    key: PRNGKeyArray, shape: tuple, mu: float = 0.0, sigma: float = 1.0
) -> Array:
    """
    Logit-Normal sampling, which concentrates samples around t=0.5.
    """
    u = jax.random.uniform(key, shape)
    # Use the inverse CDF of the normal distribution (percent point function)
    x = jax.scipy.stats.norm.ppf(u, loc=mu, scale=sigma)
    return jax.scipy.special.expit(x)


def get_time_scheduler(scheduler_type: str):
    """
    Factory function to get a time scheduler by name.

    Args:
        scheduler_type: One of "uniform", "power_law", "logit_normal", or None

    Returns:
        The corresponding scheduler function. Defaults to uniform_scheduler if None.
    """
    schedulers = {
        "uniform": uniform_scheduler,
        "power_law": power_law_scheduler,
        "logit_normal": logit_normal_scheduler,
    }
    if scheduler_type is None:
        return uniform_scheduler
    if scheduler_type not in schedulers:
        raise ValueError(
            f"Unknown time scheduler: {scheduler_type}. "
            f"Available: {list(schedulers.keys())}"
        )
    return schedulers[scheduler_type]


# ----------------- Model Partitioning -----------------
def custom_filter_spec(model):
    """
    Creates a filter specification for eqx.partition that treats
    pos_mean and pos_std as non-trainable (static).

    This is used to freeze the normalization parameters during training
    while allowing other JAX arrays to be updated.

    Args:
        model: An equinox model to create a filter spec for

    Returns:
        A pytree with the same structure as model, containing booleans
        indicating which leaves are trainable
    """
    def filter_fn(path, leaf):
        final_key = path[-1]
        if isinstance(final_key, GetAttrKey):
            # Treat pos_mean and pos_std as non-trainable (static)
            if final_key.name in ["pos_mean", "pos_std"]:
                return False
        # Treat all other JAX arrays as trainable parameters
        return isinstance(leaf, jax.Array)
    return tree_map_with_path(filter_fn, model)


class NormalizingFlow(eqx.Module):
    """
    A wrapper for a `flowjax` transformed distribution. This now supports
    both standard and conditional flows.
    """
    flow: flowjax_dist.Transformed
    cond_mean: Optional[Array] = None
    cond_std: Optional[Array] = None
    metadata: Dict[str, Any] = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, dim: int, data_mean: Array, data_std: Array,
                 vector_field_params: dict, cond_dim: int = 0, cond_mean: Array = None, cond_std: Array = None):
        """
        Initializes the NormalizingFlow model.

        Args:
            key: JAX random key.
            dim: Dimensionality of the flow (e.g., 3 for space or velocity).
            data_mean: Mean for denormalization.
            data_std: Standard deviation for denormalization.
            vector_field_params: Parameters for the fvf.VectorField.
            cond_dim: The dimension of the conditional input. 0 for unconditional.
        """
        base_dist = flowjax_dist.Normal(jnp.zeros(dim), jnp.ones(dim))

        # Update vector field params to include conditional dimension
        vf_params = vector_field_params.copy()
        vf_params['cond_dim'] = cond_dim
        vf_params['base_dist_dim'] = dim

        flow_bij = fvf.VectorField(key, vf_params)
        denormalizer = flowjax_bij.Affine(scale=data_std, loc=data_mean)

        if cond_dim > 0:
            if cond_mean is None or cond_std is None:
                raise ValueError("cond_mean and cond_std must be provided for conditional flows.")
            self.cond_mean = cond_mean
            self.cond_std = cond_std

        self.flow = flowjax_dist.Transformed(base_dist, flowjax_bij.Chain([flow_bij, denormalizer]))

        self.metadata = {
            'input_dim': dim,
            'cond_dim': cond_dim,
            'num_parameters': self.count_parameters(),
            "vector_field_params": vf_params
        }

    def count_parameters(self):
        arrays = eqx.filter(self.flow, eqx.is_array)
        return sum(x.size for x in jax.tree_util.tree_leaves(arrays))

    def log_prob(self, x: Array, condition: Optional[Array] = None) -> Array:
        if condition is not None:
            # Normalize the condition
            condition = (condition - self.cond_mean) / self.cond_std
        return self.flow.log_prob(x, condition)

    @eqx.filter_jit
    def sample(self, key: PRNGKeyArray, num_samples: int, condition: Optional[Array] = None) -> Array:
        if condition is not None:
            # For conditional sampling, flowjax expects condition.shape[0] == num_samples
            if condition.shape[0] != num_samples:
                raise ValueError("The first dimension of the condition must match num_samples.")
            # Normalize the condition
            condition = (condition - self.cond_mean) / self.cond_std
            return self.flow.sample(key, condition=condition)
        else:
            return self.flow.sample(key, (num_samples,))


class ConditionalPhaseSpaceFlow(eqx.Module):
    """
    Represents f(x, v) = n(x) * f(v | x) using two normalizing flows.
    Now with updated save/load functionality for separate training.
    """
    spatial_flow: NormalizingFlow
    conditional_velocity_flow: NormalizingFlow
    checkpoint_index: int
    model_dir: str = eqx.field(static=True, default="")
    metadata: Dict[str, Any] = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, data_mean: Array, data_std: Array,
                 spatial_vf_params: dict, conditional_vf_params: dict,
                 model_dir: Optional[str] = None, checkpoint_index: int = 0):

        if data_mean.shape[0] != 6 or data_std.shape[0] != 6:
            raise ValueError("data_mean and data_std must be 6-dimensional.")

        key_spatial, key_velocity = jax.random.split(key)

        pos_mean, vel_mean = data_mean[:3], data_mean[3:]
        pos_std, vel_std = data_std[:3], data_std[3:]

        self.spatial_flow = NormalizingFlow(
            key_spatial, dim=3, data_mean=pos_mean, data_std=pos_std,
            vector_field_params=spatial_vf_params, cond_dim=0
        )
        self.conditional_velocity_flow = NormalizingFlow(
            key_velocity, dim=3, data_mean=vel_mean, data_std=vel_std,
            vector_field_params=conditional_vf_params, cond_dim=3,
            cond_mean=pos_mean, cond_std=pos_std
        )

        self.checkpoint_index = checkpoint_index
        self.model_dir = model_dir
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'input_dim': 6,
            'num_parameters': self.count_parameters(),
            "spatial_vf_params": spatial_vf_params,
            "conditional_vf_params": conditional_vf_params
        }

    def count_parameters(self):
        return self.spatial_flow.count_parameters() + self.conditional_velocity_flow.count_parameters()

    def log_prob(self, eta: Array) -> Array:
        """Computes the log probability of the 6D phase-space data `eta`."""
        # Split eta into x and v based on the last axis
        x, v = eta[..., :3], eta[..., 3:]
        log_prob_x = self.spatial_flow.log_prob(x)
        log_prob_v_given_x = self.conditional_velocity_flow.log_prob(v, condition=x)
        return log_prob_x + log_prob_v_given_x

    def log_prob_position(self, x: Array) -> Array:
        """Computes the log probability of the 3D position data `x`."""
        return self.spatial_flow.log_prob(x)

    def log_prob_velocity_given_position(self, eta: Array) -> Array:
        """Computes the log probability of the 3D velocity data `v` given position `x`."""
        x, v = eta[..., :3], eta[..., 3:]
        return self.conditional_velocity_flow.log_prob(v, condition=x)

    @eqx.filter_jit
    def sample(self, key: PRNGKeyArray, num_samples: int) -> Array:
        """Samples from the flow's distribution."""
        key_x, key_v = jax.random.split(key)
        x_samples = self.spatial_flow.sample(key_x, num_samples)
        v_samples = self.conditional_velocity_flow.sample(key_v, num_samples, condition=x_samples)
        return jnp.concatenate([x_samples, v_samples], axis=-1)

    @eqx.filter_jit
    def sample_position(self, key: PRNGKeyArray, num_samples: int) -> Array:
        """Samples only the position part from the spatial flow."""
        return self.spatial_flow.sample(key, num_samples)

    def save(self, save_prefix: str = "flow", loss_history: Optional[dict] = None):
        """Saves the model with a specific prefix for the checkpoint files."""
        path = Path(self.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        print(f'Saving to checkpoint with index {self.checkpoint_index} using prefix "{save_prefix}"')

        # Save the full model state
        eqx.tree_serialise_leaves(path / f"{save_prefix}-{self.checkpoint_index}_model.eqx", self)

        # Save associated loss history and metadata
        if loss_history is not None:
            with open(path / f"{save_prefix}-{self.checkpoint_index}_loss.json", "w") as f:
                json.dump(loss_history, f, indent=2)
            if "pos" in save_prefix:
                lh = {'train': loss_history['train_pos'], 'val': loss_history['val_pos'], 'lr': loss_history['lr_pos']}
                utils.plot_loss(lh, path / f"{save_prefix}-{self.checkpoint_index}_loss.pdf")
            else:
                lh = {'train': loss_history['train_vel'], 'val': loss_history['val_vel'], 'lr': loss_history['lr_vel']}
                utils.plot_loss(lh, path / f"{save_prefix}-{self.checkpoint_index}_loss.pdf")

        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        return eqx.tree_at(lambda tree: tree.checkpoint_index, self, self.checkpoint_index + 1)

    @classmethod
    def load(cls, model_dir: str, load_index: int = -1, load_prefix: str = "flow", load_history: bool = False):
        """
        Loads a model from a checkpoint file, finding the latest if index is -1.
        Can specify a prefix to filter which training run to load from.
        """
        model_dir = Path(model_dir)

        # Load metadata first to build a template model
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Create an empty shell of the model
        empty_model = cls(
            key=jax.random.key(0),
            data_mean=jnp.zeros(6), data_std=jnp.ones(6),
            spatial_vf_params=metadata['spatial_vf_params'],
            conditional_vf_params=metadata['conditional_vf_params'],
            model_dir=model_dir
        )

        if load_index == -1:
            # Find the latest checkpoint file matching the prefix
            model_files = list(model_dir.glob(f"{load_prefix}-*_model.eqx"))
            if not model_files:
                raise FileNotFoundError(f"No model files found in {model_dir} with prefix '{load_prefix}-'")

            indices = [int(re.search(r'-(\d+)_model.eqx', str(f)).group(1)) for f in model_files]
            load_index = max(indices)

        final_prefix = load_prefix if load_prefix != "*" else model_files[0].stem.split('-')[0]
        model_file = model_dir / f"{final_prefix}-{load_index}_model.eqx"
        print(f"Loading model from {model_file}")

        model = eqx.tree_deserialise_leaves(model_file, like=empty_model)
        print("Model loaded successfully!")

        model.metadata.update(metadata)

        loss_history = None
        if load_history:
            # Determine the loss history file to load
            history_file = model_dir / f"{final_prefix}-{load_index}_loss.json"
            print(f"Loading loss history from {history_file}")

            if history_file.exists():
                with open(history_file, 'r') as f:
                    loss_history = json.load(f)
        return model, loss_history

    @classmethod
    def from_components(cls,
                        spatial_instance,
                        velocity_instance,
                        model_dir: str):
        """
        Initializes a new ConditionalPhaseSpaceFlow by combining two separate instances.

        This method takes one instance that has been trained on the spatial component
        and another that has been trained on the conditional velocity component,
        and merges them into a single, complete phase-space model.

        Args:
            spatial_instance: An instance of ConditionalPhaseSpaceFlow containing the desired spatial_flow.
            velocity_instance: An instance of ConditionalPhaseSpaceFlow containing the desired conditional_velocity_flow.
            model_dir: The new directory where this combined model and its metadata will be saved.

        Returns:
            A new instance of ConditionalPhaseSpaceFlow with the combined flows.
        """
        if not isinstance(spatial_instance, cls) or not isinstance(velocity_instance, cls):
            raise TypeError("Both input instances must be of type ConditionalPhaseSpaceFlow.")

        # Create a temporary "shell" instance. We need this to get a valid object
        # structure that eqx.tree_at can operate on. The internal flows and data
        # of this shell will be immediately replaced.
        shell = cls(
            key=jax.random.key(0),
            data_mean=jnp.zeros(6),  # Dummy data
            data_std=jnp.ones(6),    # Dummy data
            spatial_vf_params=spatial_instance.metadata.get("spatial_vf_params", {}),
            conditional_vf_params=velocity_instance.metadata.get("conditional_vf_params", {}),
            model_dir=model_dir,
            checkpoint_index=0 # The new model starts at checkpoint 0
        )

        # Replace the flows in the shell with the flows from the provided instances
        # This creates a new model with the correct, trained flows.
        combined_model = eqx.tree_at(
            lambda tree: (tree.spatial_flow, tree.conditional_velocity_flow),
            shell,
            (spatial_instance.spatial_flow, velocity_instance.conditional_velocity_flow)
        )

        return combined_model


class PrecomputedOTDataLoader:
    """
    A lazy loader for OT paired noise-data samples for flow matching with minibatch OT,
    stored across multiple chunked files..

    This class provides a list-like interface (`loader[epoch_nr]`) to access
    x1 and x0 for each training epoch in a way where the samples from noise x0 are
    matched to x1 with Optimal Transport. It only loads the required file
    chunk into memory when an epoch from that chunk is requested, making it
    efficient for very large datasets.

    Args:
        base_path (Path): The base path for index files (e.g., 'data/pairings_train').
                          The loader searches for files like 'data/pairings_train_0_500.npz'.
        n (int): The number of samples in the dataset, used for reshaping the loaded arrays.
    """
    def __init__(self, base_path: Path, n: int, x1_data: Array, w1_weights: Array, x1_cond: Array = None):
        self.base_path = base_path
        self.n = n
        self.x1_data = x1_data
        self.w1_weights = w1_weights
        self.x1_cond = x1_cond
        self._file_map = {}
        self._epoch_starts = []
        self._total_epochs = 0

        self._discover_files()

        # Cache for the most recently loaded file
        self._cached_file_path = None
        self._cached_start_epoch = -1
        self._cached_idx_x0 = None
        self._cached_idx_x1 = None
        self._cached_keys = None

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
        Loads and returns minibatch OT-matched x0 and x1 of a specific epoch.

        This method finds the correct file, loads it if not already cached,
        and returns the requested epoch's data.
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
            idx_x0 = data['x0_perm_indices']
            idx_x1 = data['x1_perm_indices']
            keys_x0 = data['x0_generation_keys']

            # Reshape the loaded chunk
            epochs_in_file = end_epoch - start_epoch
            self._cached_idx_x0 = jnp.array(idx_x0).reshape(epochs_in_file, self.n)
            self._cached_idx_x1 = jnp.array(idx_x1).reshape(epochs_in_file, self.n)
            self._cached_keys = jnp.array(keys_x0, dtype=jnp.uint32).reshape(epochs_in_file, 2)  # Each key has 2 uint32 components
            self._cached_file_path = file_path
            self._cached_start_epoch = start_epoch

        # Calculate the index within the loaded (cached) chunk
        local_epoch_idx = epoch_nr - self._cached_start_epoch

        x0 = jax.random.normal(self._cached_keys[local_epoch_idx], shape=self.x1_data.shape)
        idx_x1 = self._cached_idx_x1[local_epoch_idx]
        x1_cond = self.x1_cond[idx_x1] if self.x1_cond is not None else None
        return x0[self._cached_idx_x0[local_epoch_idx]], self.x1_data[idx_x1], self.w1_weights[idx_x1], x1_cond


# ----------------- Loss Functions and Training Step -----------------
def compute_jacobian_penalty(net, t, x, key, use_exact=False, n_samples=1):
    """Compute Jacobian regularization for a single (t, x) pair."""
    if use_exact:
        # Exact full Jacobian (O(d^2))
        J = jax.jacobian(lambda x_: net(t, x_))(x)
        return jnp.sum(J**2)
    else:
        # Hutchinson trace estimator (O(d))
        keys = jax.random.split(key, n_samples)

        def single_estimate(k):
            v = jax.random.normal(k, shape=x.shape)
            Jv = jax.jvp(lambda x_: net(t, x_), (x,), (v,))[1]
            return jnp.sum(Jv**2)
        vals = jax.vmap(single_estimate)(keys)
        return jnp.mean(vals)


def _loss_fn(params, static, key, x0, x1, x1_cond, weights, time_scheduler, sb_constant, kinetic_reg, jacobian_reg):
    """
    We are given the OT-paired (x0, x1), so just proceed with Flow Matching
    This function is used for both spatial and conditional velocity flows.
    If it's spatial, then x0 and x1 are positions and x1_cond is None.
    If it's conditional velocity, then x0 and x1 are velocities and x1_cond is positions.
    """
    t = time_scheduler(key, (x1.shape[0], 1))
    xt = t * x1 + (1 - t) * x0
    ut = x1 - x0  # Target vector field

    if sb_constant > 0:
        key, subkey = jax.random.split(key, 2)
        eps = 1e-8
        epsilon_t = sb_constant * jnp.sqrt(t * (1 - t) + eps)
        z = jax.random.normal(subkey, shape=x1.shape)

        xt += epsilon_t * z
        ut += (1 - 2 * t) / (2 * t * (1 - t) + eps) * epsilon_t * z

    net = eqx.combine(params, static)
    vt = jax.vmap(net)(t.squeeze(-1), xt, condition=x1_cond)  # Predicted vector field
    # Flow Matching loss
    loss = jnp.sum(weights[:, None] * (vt - ut)**2) / jnp.sum(weights)

    # Kinetic energy regularization
    if kinetic_reg > 0:
        loss += kinetic_reg * jnp.sum(weights[:, None] * vt**2) / jnp.sum(weights)

    # Exact Jacobian regularization
    if jacobian_reg > 0:
        def single_jac_penalty(t_i, x_i, x1_cond):
            J = jax.jacobian(lambda x_: net(t_i.squeeze(), x_, condition=x1_cond))(x_i)
            return jnp.sum(J**2)

        jacobian_losses = jax.vmap(single_jac_penalty)(t, xt, x1_cond)
        loss += jacobian_reg * jnp.sum(weights * jacobian_losses) / jnp.sum(weights)
    return loss

loss_fn = eqx.filter_value_and_grad(_loss_fn)
val_loss_fn = eqx.filter_jit(_loss_fn)


@eqx.filter_jit
def train_step(params, static, opt_state, key, x0, x1, x1_cond, weights, optimizer, time_scheduler, schedule_type, val_loss, loss_params):
    loss, grads = loss_fn(params, static, key, x0, x1, x1_cond, weights, time_scheduler, **loss_params)
    global_grad_norm = optax.global_norm(grads)
    if schedule_type == "step":
        # We additionally pass the validation loss to the scheduler
        updates, opt_state = optimizer.update(grads, opt_state, params, value=val_loss)
    else:
        updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss, global_grad_norm


def train_ot_flow_matching_model(
    key: PRNGKeyArray,
    model: NormalizingFlow,
    which_flow: str,
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
    ot_pairings_dir: Path,
    time_scheduler_type=None,
    loss_params={},
    time_logger=None,
    loss_history=None,
    checkpoint_frequency_epochs=-1,
    timeout_hours=None,
    early_stopping_loss_threshold=None
):
    """
    Trains the spatial component of a ConditionalPhaseSpaceFlow
    using minibatch OT flow matching. Conditional velocity component isn't
    feasible with minibatch OT (it incurs biases at inference)
    """
    # Normalize data for training.
    train_x = (train_data['eta'] - norm_mean) / norm_std
    train_weights = train_data['weights']
    val_x = (val_data['eta'] - norm_mean) / norm_std
    val_weights = val_data['weights']

    # We make sure train_x is only positions
    print("--- Training Spatial Flow n(x) with Standard Flow Matching ---")
    train_x, val_x = train_x[:, :3], val_x[:, :3]
    dynamics_net = model.spatial_flow.flow.bijection[0].dynamics_net
    train_label, val_label, lr_label, grads_label = 'train_pos', 'val_pos', 'lr_pos', 'global_grad_norms_pos'
    flow_save_prefix = 'flow_pos_only'
    if which_flow != "spatial":
        raise ValueError("which_flow must be 'spatial'.")

    if loss_history is None:
        loss_history = {}
    if train_label not in loss_history:
        loss_history[train_label] = []
        loss_history[val_label] = []
        loss_history[lr_label] = []
        loss_history[grads_label] = []

    n_train = train_x.shape[0]
    n_val = val_x.shape[0]
    val_batch_size = batch_size

    # --- Setup Data Loaders and Schedulers ---
    path_train_ot = ot_pairings_dir / "train_epochs"
    x0_x1_train_indices_loader = PrecomputedOTDataLoader(path_train_ot, n=n_train, x1_data=train_x, w1_weights=train_weights, x1_cond=None)
    path_val_ot = ot_pairings_dir / "val_epochs"
    x0_x1_val_indices_loader = PrecomputedOTDataLoader(path_val_ot, n=n_val, x1_data=val_x, w1_weights=val_weights, x1_cond=None)

    time_scheduler = get_time_scheduler(time_scheduler_type)

    # --- Partition Model into trainable/non-trainable parts and initialize the optimizer ---
    params, static = eqx.partition(dynamics_net, filter_spec=custom_filter_spec(dynamics_net))

    opt_state = optimizer.init(params)
    if "global_grad_norms" in loss_history:
        global_grad_norms = loss_history[grads_label]
    else:
        global_grad_norms = []
    steps_per_epoch = n_train // batch_size

    print("Starting OT Flow Matching training with pre-computed indices...")
    print(f"Number of trainable parameters: {model.count_parameters()}")
    print(f"Number of steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}, Total training samples: {n_train}")
    print(f"Using time scheduler of type {time_scheduler}")
    print(f"Using loss params {loss_params}")
    start_epoch = len(loss_history[lr_label])
    step = start_epoch * steps_per_epoch  # Continue from previous step if resuming

    avg_val_loss, avg_train_loss = 1000000, 1000000
    for epoch in (pbar := trange(start_epoch, epochs)):
        if time_logger is not None:
            time_logger.start("Training | setup")
        # Derive keys for x0 generation and permutation exactly as in the pre-computation script
        x0_train_epoch, x1_train_epoch, w1_train_epoch, _ = x0_x1_train_indices_loader[epoch]
        x0_val_epoch, x1_val_epoch, w1_val_epoch, _ = x0_x1_val_indices_loader[epoch]
        key, key_loop, key_shuffle = jax.random.split(key, 3)

        # Additionally shuffle the indices to help the later epochs which start repeating
        perm = jax.random.permutation(key_shuffle, n_train)
        x0_train_epoch = x0_train_epoch[perm]
        x1_train_epoch = x1_train_epoch[perm]
        w1_train_epoch = w1_train_epoch[perm]
        if time_logger is not None:
            time_logger.stop("Training | setup")

        avg_train_loss = 0.0
        epoch_w = 0.0 # avg training loss is the weighted average
        epoch_lr = []
        epoch_global_grad_norms = []
        for i in range(steps_per_epoch):
            if time_logger is not None:
                time_logger.start("Training | train setup")
            key_loop, key_step = jax.random.split(key_loop)

            x0_batch = x0_train_epoch[i * batch_size: (i + 1) * batch_size]
            x1_batch = x1_train_epoch[i * batch_size: (i + 1) * batch_size]
            w_batch = w1_train_epoch[i * batch_size: (i + 1) * batch_size]
            if i == steps_per_epoch - 1 and x0_batch.shape[0] != batch_size:
                continue

            if time_logger is not None:
                time_logger.stop("Training | train setup")
                time_logger.start("Training | train backpropagation")
            params, opt_state, loss, global_grad_norm = train_step(
                params, static, opt_state, key_step, x0_batch, x1_batch, None, w_batch, optimizer,
                time_scheduler=time_scheduler, schedule_type=schedule_type, val_loss=jnp.array(avg_val_loss),
                loss_params=loss_params
            )

            avg_train_loss += float(loss.item()) * float(np.sum(w_batch))
            epoch_w += float(np.sum(w_batch))
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

        avg_train_loss /= epoch_w
        avg_global_grad_norm = np.mean(epoch_global_grad_norms)
        global_grad_norms.append(avg_global_grad_norm)

        # Calculate validation loss
        key, key_loop = jax.random.split(key)
        avg_val_loss = 0.0
        val_w = 0.0
        if time_logger is not None:
            time_logger.start("Training | val")

        for i in range(0, val_x.shape[0], val_batch_size):
            key_loop, key_step = jax.random.split(key_loop)
            x0_batch = x0_val_epoch[i: i + val_batch_size]
            x1_batch = x1_val_epoch[i: i + val_batch_size]
            w_batch = w1_val_epoch[i: i + val_batch_size]
            v_loss = val_loss_fn(params, static, key_loop, x0_batch, x1_batch, None, w_batch, time_scheduler, **loss_params)
            avg_val_loss += float(v_loss.item()) * float(np.sum(w_batch))
            val_w += float(np.sum(w_batch))
        avg_val_loss /= val_w

        loss_history[train_label].append(avg_train_loss)
        loss_history[val_label].append(avg_val_loss)
        loss_history[lr_label].append(float(np.mean(epoch_lr)))
        if time_logger is not None:
            time_logger.stop("Training | val")

        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"lr: {loss_history[lr_label][-1]:.4f}"
        )
        # Checkpoint if needed
        if checkpoint_frequency_epochs > 0 and epoch > 0 and epoch % checkpoint_frequency_epochs == 0:
            dynamics_net = eqx.combine(params, static)
            model = eqx.tree_at(lambda m: m.spatial_flow.flow.bijection[0].dynamics_net, model, dynamics_net)
            model = model.save(loss_history=loss_history, save_prefix=flow_save_prefix)

        if schedule_type == "step" and loss_history[lr_label][-1] < lr_final:
            print("Early stopping triggered.")
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
    model = eqx.tree_at(lambda m: m.spatial_flow.flow.bijection[0].dynamics_net, model, dynamics_net)
    model.save(loss_history=loss_history, save_prefix=flow_save_prefix) # Training is finished, let's checkpoint!
    loss_history[grads_label] = global_grad_norms

    return model, loss_history
