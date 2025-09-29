#!/usr/bin/env python

from __future__ import print_function, division

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from jax.tree_util import tree_map_with_path, GetAttrKey
import equinox as eqx
import optax
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import json
from pathlib import Path
from datetime import datetime
from tqdm import trange
from typing import Dict, Any, Optional, Tuple, List

# Custom libraries
import utils


def calc_phi_derivatives(phi_func: callable, q: Array) -> Tuple[Array, Array]:
    """Calculates higher order derivatives of the potential at q."""
    dphi_dq = jax.grad(phi_func)(q)

    # Laplacian of phi
    hessian_phi = jax.hessian(phi_func)(q)
    d2phi_dq2 = jnp.trace(hessian_phi)

    return dphi_dq, d2phi_dq2


class ResBlock(eqx.Module):
    """A standard residual block with layer normalization and two linear layers."""
    ln: eqx.nn.LayerNorm
    fn: eqx.Module

    def __init__(self, width_size: int, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key)
        self.ln = eqx.nn.LayerNorm(width_size, use_weight=True, use_bias=True)
        self.fn = eqx.nn.Sequential([
            eqx.nn.Linear(width_size, width_size, key=key1),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(width_size, width_size, key=key2),
        ])

    def __call__(self, x: Array) -> Array:
        h = self.ln(x)
        return x + self.fn(h)


class ResMLP(eqx.Module):
    """A ResNet-style MLP with an initial layer, residual blocks, and a final layer."""
    initial_layer: eqx.nn.Linear
    blocks: list[ResBlock]
    final_layer: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int, # Depth is the number of residual blocks
        key: PRNGKeyArray,
    ):
        init_key, final_key, *block_keys = jax.random.split(key, 2 + depth)

        self.initial_layer = eqx.nn.Linear(in_size, width_size, key=init_key)
        self.blocks = [ResBlock(width_size, key=k) for k in block_keys]
        self.final_layer = eqx.nn.Linear(
            width_size, out_size, key=final_key,
        )

    def __call__(self, x: Array) -> Array:
        # Initial projection and activation
        h = jax.nn.silu(self.initial_layer(x))

        # Pass through all residual blocks
        for block in self.blocks:
            h = block(h)

        # Final projection to output size
        return self.final_layer(h)


class PhiNN(eqx.Module):
    """
    A feed-forward neural network to represent the gravitational potential Phi(q).

    Attributes:
        mlp (ResMLP): The core neural network model.
        scale (Array): A non-trainable scaling factor applied to input coordinates
                       for normalization purposes.
    """
    mlp: ResMLP
    scale: Array # Not trainable

    def __init__(self, key, n_dim=3, depth=3, width=32, scale=None):
        """
        Initializes the potential model.

        Args:
            key (PRNGKeyArray): JAX random key for MLP initialization.
            n_dim (int): The dimensionality of the input space (typically 3).
            depth (int): The number of residual blocks in the MLP.
            width (int): The width of the hidden layers in the MLP.
            scale (Optional[Array]): A vector of scales for each input coordinate.
                                     If None, defaults to ones.
        """
        super().__init__()

        self.mlp = ResMLP(
            in_size=n_dim, out_size=1, width_size=width,
            depth=depth, key=key,
        )

        print(f"Initializing phi(x) model with coordinate scale {scale}")
        if scale is None:
            coord_scale = jnp.ones(n_dim)
        else:
            coord_scale = jnp.array(scale)
        self.scale = 1 / coord_scale

    def __call__(self, q):
        """Returns the gravitational potential"""
        return self.mlp(self.scale * q).squeeze()


class FrameShift(eqx.Module):
    """
    A 5-parameter model to represent a rotating frame transformation.

    This model accounts for the apparent forces and velocity shifts when moving
    from an inertial frame to a rotating frame where the distribution function `f`
    is assumed to be stationary. The parameters (e.g., angular velocity `omega`)
    can be fitted concurrently with the gravitational potential.

    Attributes:
        omega, r0, v0_x, v0_y, v0_z (Array): The parameters of the frame transformation.
        ..._trainable (bool): Static flags to control which parameters are trainable.
    """
    omega: Array
    r0: Array
    v0_x: Array
    v0_y: Array
    v0_z: Array
    n_dim: int

    # Static metadata indicating which parameters are trainable
    omega_trainable: bool
    r0_trainable: bool
    v0_x_trainable: bool
    v0_y_trainable: bool
    v0_z_trainable: bool

    def __init__(
        self, n_dim=3, omega=0.0, r0=0.0, v0_x=0.0, v0_y=0.0, v0_z=0.0,
        omega_trainable=True, r0_trainable=False, v0_x_trainable=False,
        v0_y_trainable=False, v0_z_trainable=False
    ):
        """
        Constructor for FrameShift.
        """
        self.n_dim = n_dim
        self.omega = jnp.array(omega, dtype=jnp.float32)
        self.r0 = jnp.array(r0, dtype=jnp.float32)
        self.v0_x = jnp.array(v0_x, dtype=jnp.float32)
        self.v0_y = jnp.array(v0_y, dtype=jnp.float32)
        self.v0_z = jnp.array(v0_z, dtype=jnp.float32)

        self.omega_trainable = omega_trainable
        self.r0_trainable = r0_trainable
        self.v0_x_trainable = v0_x_trainable
        self.v0_y_trainable = v0_y_trainable
        self.v0_z_trainable = v0_z_trainable

    def __call__(self, q, p):
        """
        Calculates the velocity shift `u` and acceleration `w` due to the frame shift.

        Args:
            q (Array): Position vectors (N, 3).
            p (Array): Momentum (or velocity) vectors (N, 3).

        Returns:
            A tuple (u, w) where `u` is the velocity field of the frame and `w`
            represents the fictitious forces (Coriolis, centrifugal).
        """
        qx, qy, _ = jnp.split(q, 3, axis=1)
        px, py, _ = jnp.split(p, 3, axis=1)

        ux = qy * -self.omega + self.v0_x
        uy = (qx - self.r0) * self.omega + self.v0_y
        uz = jnp.full_like(qx, self.v0_z)
        u = jnp.concatenate((ux, uy, uz), axis=1)

        w = jnp.concatenate(
            (
                (py - self.v0_y) * -self.omega,
                (px - self.v0_x) * self.omega,
                jnp.zeros_like(px),
            ),
            axis=1,
        )
        return u, w


class SelectionFunctionCorrection(eqx.Module):
    mlp: eqx.nn.MLP
    scale: Array

    def __init__(self, key, n_dim=3, depth=3, width=32, scale=None):
        self.mlp = ResMLP(
            in_size=n_dim, out_size=1, width_size=width,
            depth=depth, key=key,
        )
        self.scale = scale

    def call(self, q):
        return jax.nn.tanh(self.mlp(self.scale * q)).squeeze()


class PotentialModel(eqx.Module):
    """
    A container for the complete gravitational potential model.

    This class bundles the neural potential (PhiNN), the optional FrameShift model,
    and provides methods for saving, loading, and parameter counting.
    """
    phi_model: PhiNN
    frameshift_model: Optional[FrameShift]
    selection_function_model: Optional[SelectionFunctionCorrection]
    checkpoint_index: int
    model_dir: str = eqx.field(static=True)
    metadata: Dict[str, Any] = eqx.field(static=True)

    def __init__(self, key, model_dir, phi_params, frameshift_params=None, selection_function_params=None, checkpoint_index=0):
        key_phi, key_sf = jax.random.split(key)
        self.phi_model = PhiNN(**phi_params, key=key_phi)
        if frameshift_params:
            self.frameshift_model = FrameShift(**frameshift_params)
        else:
            self.frameshift_model = None
        if selection_function_params:
            self.selection_function_model = SelectionFunctionCorrection(**selection_function_params, key=key_sf)
        else:
            self.selection_function_model = None

        self.checkpoint_index = checkpoint_index
        self.model_dir = model_dir
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'phi_model_type': str(self.phi_model).splitlines(),
            'frameshift_model_type': str(self.frameshift_model).splitlines(),
            'num_parameters': self.count_parameters(),
        }

    def count_parameters(self):
        """Counts the total number of trainable parameters in the model."""
        return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array)))

    def save(self, loss_history=None):
        """
        Saves the model state and optional loss history to the model directory.

        Args:
            loss_history (dict, optional): Dictionary containing training diagnostics.
                                           If provided, loss plots are also generated.

        Returns:
            PotentialModel: An updated instance with an incremented checkpoint index.
        """
        path = Path(self.model_dir)
        path.mkdir(parents=True, exist_ok=True)

        print(f'Saving to checkpoint with index {self.checkpoint_index}')
        name_prefix = "potential"
        eqx.tree_serialise_leaves(path / f"{name_prefix}-{self.checkpoint_index}_model.eqx", self)
        if loss_history is not None:
            with open(path / f"{name_prefix}-{self.checkpoint_index}_loss.json", "w") as f:
                json.dump(loss_history, f, indent=2)

        with open(path / f"{name_prefix}-metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        if loss_history is not None:
            utils.plot_loss_new(loss_history, path / f"{name_prefix}-{self.checkpoint_index}_loss.pdf", w=1)
            loss_history_noreg = {
                'train': loss_history['train_noreg'],
                'val': loss_history['val_noreg'],
                'lr': loss_history['lr'],
            }
            utils.plot_loss_new(loss_history_noreg, path / f'potential-{self.checkpoint_index}_loss_noreg.pdf', w=1)

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
            A tuple containing the loaded model and the loss history (or None).
        """
        model_dir = Path(model_dir)
        name_suffix = 'potential'

        # Determine the model file to load
        checkpoint_index = -1
        if load_index == -1:
            # Find the model file with the highest suffix
            model_files = list(model_dir.glob(f"{name_suffix}-*_model.eqx"))
            if model_files:
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
            if model_files == []:
                raise FileNotFoundError(f"No model files found in {model_dir}")
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


def get_trainable_scalar_history(model: eqx.Module) -> Dict[str, float]:
    """
    Finds and returns the current values of trainable scalar parameters.

    This function traverses the model's PyTree and identifies scalar parameters
    (like `omega`) that have a corresponding `param_name_trainable` flag set to True.
    This is useful for logging the evolution of these parameters during training.

    Args:
        model (eqx.Module): The model to inspect.

    Returns:
        Dict[str, float]: A dictionary mapping parameter names to their current values.
    """
    history = {}

    def find_trainable_scalars(path, leaf):
        # We are interested in scalar JAX arrays.
        if isinstance(leaf, jax.Array) and leaf.ndim == 0:
            # The path gives us the sequence of keys to access the leaf.
            # Example path: (GetAttrKey(name='frameshift_model'), GetAttrKey(name='omega'))
            if len(path) > 0 and isinstance(path[-1], GetAttrKey):
                param_name = path[-1].name
                trainable_flag_name = f"{param_name}_trainable"

                # Navigate the model tree to find the container of the leaf
                # to check if the trainable flag exists and is True.
                container = model
                for key in path[:-1]:
                    # This assumes access is by attribute name
                    if hasattr(container, key.name):
                        container = getattr(container, key.name)
                    else:
                        # Cannot traverse further, break
                        return

                if hasattr(container, trainable_flag_name):
                    if getattr(container, trainable_flag_name):
                        history[param_name] = float(leaf)

    # Walk the model tree and apply the function to each leaf.
    jax.tree_util.tree_map_with_path(find_trainable_scalars, model)
    return history


def calc_lnsel_derivatives(selection_function, q):
    """Calculates higher order derivatives of the potential at q."""
    dlnsel_dq = jax.grad(selection_function)(q) / selection_function(q)

    return dlnsel_dq

def get_phi_loss_new(
    phi_model,
    frameshift_model,
    selection_function_model,
    q, p,
    dlnf_dq, dlnf_dp,
    alpha=1.0,
    beta=1.0,
    lambda_=1.0,
    gamma=0.0,
    mu=0.0,
    l2=0.01
):
    """
    Calculates the loss based on the collisionless Boltzmann equation (CBE).

    The core of the loss is to minimize `df/dt`, which should be zero for a
    stationary system. The loss includes several regularization terms.

    Args:
        phi_model, frameshift_model: The model components.
        q, p: Position and momentum samples (N, 3).
        df_dq, df_dp: Gradients of the distribution function `f` w.r.t. q and p.
        alpha: Controls the transition from linear to log-like loss for the CBE term.
        lambda_: Strength of the penalty on negative mass densities.
        beta: Curvature of the negative mass density penalty.
        gamma: Strength of a linear penalty on positive mass densities.
        mu: Strength of a log-like penalty on positive mass densities.
        l2: Strength of L2 weight regularization on the `phi_model`.

    Returns:
        A tuple (total_loss, loss_without_regularization).
    """
    phi_grads_fn = jax.vmap(calc_phi_derivatives, in_axes=(None, 0))
    dphi_dq, d2phi_dq2 = phi_grads_fn(phi_model, q)

    lnsel_grads_fn = jax.vmap(calc_lnsel_derivatives, in_axes=(None, 0))
    dlnsel_dq = lnsel_grads_fn(selection_function_model, q)

    if frameshift_model is None:
        dlnf_dt = jnp.sum(dlnf_dp * dphi_dq - dlnf_dq * p, axis=1)
        null_hyp = dlnf_dt
    else:
        u, w = frameshift_model(q, p)
        dlnf_dt_omega = jnp.sum((p - u) * dlnf_dq - (dphi_dq + w) * dlnf_dp, axis=1)
        null_hyp = dlnf_dt_omega
    
    null_hyp -= dlnsel_dq

    likelihood = jnp.arcsinh(alpha * jnp.abs(null_hyp)) / alpha

    # Punishing negative matter densities
    if lambda_ != 0:
        prior_neg = jnp.arcsinh(beta * jnp.clip(-d2phi_dq2, a_min=0.0)) / beta
        likelihood = likelihood + lambda_ * prior_neg

    # Punishing positive matter densities
    if mu != 0:
        prior_pos = jnp.arcsinh(jnp.clip(d2phi_dq2, a_min=0.0))
        likelihood = likelihood + mu * prior_pos

    # Punishing positive matter densities linearly
    if gamma != 0:
        prior_pos = jnp.clip(d2phi_dq2, a_min=0.0)
        likelihood = likelihood + gamma * prior_pos

    loss = jnp.log(jnp.mean(likelihood))
    loss_noreg = loss

    if l2 != 0:
        all_leaves = jax.tree_util.tree_leaves(phi_model.mlp)
        array_leaves = [p for p in all_leaves if isinstance(p, jax.Array)]

        # Calculate the sum of squares.
        sum_of_squares = jnp.sum(jnp.array([jnp.sum(p**2) for p in array_leaves]))
        # Count the total number of trainable parameters.
        param_count = sum(p.size for p in array_leaves)

        mean_of_squares = sum_of_squares / (param_count + 1e-8)
        penalty = l2 * mean_of_squares
        loss += penalty

    return loss, loss_noreg


def get_phi_loss(
    phi_model,
    frameshift_model,
    q, p,
    df_dq, df_dp,
    alpha=1.0,
    beta=1.0,
    lambda_=1.0,
    gamma=0.0,
    mu=0.0,
    l2=0.01
):
    """
    Calculates the loss based on the collisionless Boltzmann equation (CBE).

    The core of the loss is to minimize `df/dt`, which should be zero for a
    stationary system. The loss includes several regularization terms.

    Args:
        phi_model, frameshift_model: The model components.
        q, p: Position and momentum samples (N, 3).
        df_dq, df_dp: Gradients of the distribution function `f` w.r.t. q and p.
        alpha: Controls the transition from linear to log-like loss for the CBE term.
        lambda_: Strength of the penalty on negative mass densities.
        beta: Curvature of the negative mass density penalty.
        gamma: Strength of a linear penalty on positive mass densities.
        mu: Strength of a log-like penalty on positive mass densities.
        l2: Strength of L2 weight regularization on the `phi_model`.

    Returns:
        A tuple (total_loss, loss_without_regularization).
    """
    # We should vmap over q
    vmapped_calc_phi_derivatives = jax.vmap(calc_phi_derivatives, in_axes=(None, 0))
    dphi_dq, d2phi_dq2 = vmapped_calc_phi_derivatives(phi_model, q)

    if frameshift_model is None:
        df_dt = jnp.sum(df_dp * dphi_dq - df_dq * p, axis=1)
        null_hyp = df_dt
    else:
        u, w = frameshift_model(q, p)
        df_dt_omega = jnp.sum((p - u) * df_dq - (dphi_dq + w) * df_dp, axis=1)
        null_hyp = df_dt_omega

    likelihood = jnp.arcsinh(alpha * jnp.abs(null_hyp)) / alpha

    # Punishing negative matter densities
    if lambda_ != 0:
        prior_neg = jnp.arcsinh(beta * jnp.clip(-d2phi_dq2, a_min=0.0)) / beta
        likelihood = likelihood + lambda_ * prior_neg

    # Punishing positive matter densities
    if mu != 0:
        prior_pos = jnp.arcsinh(jnp.clip(d2phi_dq2, a_min=0.0))
        likelihood = likelihood + mu * prior_pos

    # Punishing positive matter densities linearly
    if gamma != 0:
        prior_pos = jnp.clip(d2phi_dq2, a_min=0.0)
        likelihood = likelihood + gamma * prior_pos

    loss = jnp.log(jnp.mean(likelihood))
    loss_noreg = loss

    if l2 != 0:
        all_leaves = jax.tree_util.tree_leaves(phi_model.mlp)
        array_leaves = [p for p in all_leaves if isinstance(p, jax.Array)]

        # Calculate the sum of squares.
        sum_of_squares = jnp.sum(jnp.array([jnp.sum(p**2) for p in array_leaves]))
        # Count the total number of trainable parameters.
        param_count = sum(p.size for p in array_leaves)

        mean_of_squares = sum_of_squares / (param_count + 1e-8)
        penalty = l2 * mean_of_squares
        loss += penalty

    return loss, loss_noreg

@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(params, static, batch, loss_params):
    model = eqx.combine(params, static)
    q, p, df_dq, df_dp = batch
    loss, loss_noreg = get_phi_loss(
        model.phi_model, model.frameshift_model, q, p, df_dq, df_dp, **loss_params
    )
    return loss, loss_noreg

@eqx.filter_jit
def loss_fn_val(params, static, batch, loss_params):
    model = eqx.combine(params, static)
    q, p, df_dq, df_dp = batch
    loss, loss_noreg = get_phi_loss(
        model.phi_model, model.frameshift_model, q, p, df_dq, df_dp, **loss_params
    )
    return loss, loss_noreg

@eqx.filter_jit
def train_step(params, static, optimizer, opt_state, batch, loss_params, schedule_type, val_loss):
    (loss, loss_noreg), grads = loss_fn(params, static, batch, loss_params)
    if schedule_type == "step":
        # We additionally pass the validation loss to the scheduler
        updates, opt_state = optimizer.update(grads, opt_state, params, value=val_loss)
    else:
        updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss, loss_noreg


def build_trainable_filter(model):
    # start with the always-non-trainable names
    non_trainable_names = set(["scale"])

    # collect all foo for which foo_trainable is equal to False
    def collect_flags(path, leaf):
        final_key = path[-1]
        if isinstance(final_key, GetAttrKey):
            name = final_key.name
            if name.endswith("_trainable"):
                # robustly check for False (python bool, numpy.bool_, jax scalar, ...)
                if leaf is False:
                    base = name[:-len("_trainable")]
                    non_trainable_names.add(base)
        return leaf  # required by tree_map_with_path

    tree_map_with_path(collect_flags, model)
    print(f"Non-trainable parameter names: {non_trainable_names}")

    # produce the boolean filter pytree (True -> trainable)
    def filter_fn(path, leaf):
        final_key = path[-1]
        if isinstance(final_key, GetAttrKey):
            name = final_key.name
            if name in non_trainable_names:
                return False
        # default: mark JAX arrays as trainable
        return isinstance(leaf, jax.Array)

    return tree_map_with_path(filter_fn, model)


def train_potential(
    key: jax.random.key,
    model: PotentialModel,
    optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
    schedule_type: str,
    lr_final: float,
    df_data: Dict[str, Array],
    n_epochs: int,
    batch_size: int,
    validation_frac: float = 0.25,
    checkpoint_frequency_epochs: int = 10,
    loss_params: Dict = None,
    loss_history={'train': [], 'val': [], 'train_noreg': [], 'val_noreg': [], 'lr': []},
):
    """
    Fits a gravitational potential and optionally a frameshift.
    """
    n_samples = df_data["eta"].shape[0]
    n_dim = df_data["eta"].shape[1] // 2

    data = (
        df_data["eta"][:, :n_dim],
        df_data["eta"][:, n_dim:],
        df_data["df_deta"][:, :n_dim],
        df_data["df_deta"][:, n_dim:],
    )

    n_val = int(validation_frac * n_samples)
    n_train = n_samples - n_val
    val_data = jax.tree.map(lambda x: x[:n_val], data)
    train_data = jax.tree.map(lambda x: x[n_val:], data)

    # We partition the model into the trainable and non-trainable parts.
    # First we collect a list of every non-trainable parameter based on the _trainable flag
    # The way it's currently set up can be vulnerable to edge cases where there are multiple
    # parameters which have the same name...
    filter_spec = build_trainable_filter(model)
    params, static = eqx.partition(model, filter_spec=filter_spec)

    opt_state = optimizer.init(params)
    steps_per_epoch = n_train // batch_size
    val_batch_size = n_val * batch_size // n_train

    print("Starting potential model training...")
    print(f"Number of trainable parameters: {model.count_parameters()}")
    print(f"Number of steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    print(f"Number of epochs: {n_epochs}, Total training samples: {n_train}")
    start_epoch = len(loss_history['lr'])
    step = start_epoch * steps_per_epoch  # Continue from previous step if resuming

    for epoch in (pbar := trange(n_epochs)):
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, n_train)
        train_data_shuffled = jax.tree.map(lambda x: x[perms], train_data)

        epoch_loss, epoch_loss_noreg, epoch_lr = [], [], []
        epoch_val_loss, epoch_val_loss_noreg = [], []
        for i in range(steps_per_epoch):
            val_batch = jax.tree.map(lambda x: x[i:i+val_batch_size], val_data)
            val_loss, val_loss_noreg = loss_fn_val(
                params, static,
                val_batch,
                loss_params
            )
            epoch_val_loss.append(val_loss.item())
            epoch_val_loss_noreg.append(val_loss_noreg.item())

            batch = jax.tree.map(lambda x: x[i*batch_size:(i+1)*batch_size], train_data_shuffled)
            params, opt_state, loss, loss_noreg = train_step(params, static, optimizer, opt_state, batch, loss_params, schedule_type, val_loss)
            epoch_loss.append(loss.item())
            epoch_loss_noreg.append(loss_noreg.item())

            if schedule_type == "step":
                lr = float(schedule(step)) * optax.tree.get(opt_state, "scale")
                epoch_lr.append(float(lr))
            else:
                lr = float(schedule(step))
                epoch_lr.append(lr)

            step += 1

        loss_history['train'].append(np.mean(epoch_loss))
        loss_history['train_noreg'].append(np.mean(epoch_loss_noreg))
        loss_history['val'].append(np.mean(epoch_val_loss))
        loss_history['val_noreg'].append(np.mean(epoch_val_loss_noreg))
        loss_history['lr'].append(np.mean(epoch_lr))
        # We also append every scalar trainable parameter's value to the history
        model_snapshot = eqx.combine(params, static)
        # Get the current values of trainable scalar parameters
        trainable_params_snapshot = get_trainable_scalar_history(model_snapshot)
        for name, value in trainable_params_snapshot.items():
            if name not in loss_history:
                loss_history[name] = []  # Initialize list if not present
            loss_history[name].append(value)

        pbar.set_description(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"Train: {loss_history['train'][-1]:.4f} | "
            f"Val: {loss_history['val'][-1]:.4f} | "
            f"lr: {loss_history['lr'][-1]:.4f}"
        )

        if checkpoint_frequency_epochs > 0 and epoch > 0 and epoch % checkpoint_frequency_epochs == 0:
            model = eqx.combine(params, static)
            model = model.save(loss_history)
            params, static = eqx.partition(model, filter_spec=filter_spec)

        if schedule_type == "step" and loss_history['lr'][-1] < lr_final:
            print("Early stopping triggered.")
            break

    model = eqx.combine(params, static)
    model.save(loss_history) # We don't increment checkpoint_index here
    return model, loss_history


def main():
    key = jax.random.PRNGKey(42)
    phi_params = {'n_dim': 3, 'depth': 3, 'width': 64}
    model = PhiNN(key, **phi_params)

    model(jnp.array([1.0, 0.0, 0.0]))

if __name__ == "__main__":
    main()
