import matplotlib

matplotlib.use("Agg")



import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.gridspec import GridSpec

from time import time, sleep
from pathlib import Path
import json
import h5py
import progressbar

import jax.numpy as jnp
import jax
import optax
from optax.contrib import reduce_on_plateau

import flow_ot_flow_matching
import utils
import flow_benchmarking
import potential_benchmarking
import flow_sampling
import potential

# limit jax gpu usage to be adaptive
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_optimizer_and_schedule(options_lr, steps_per_epoch, epochs):
    """
    Configures and returns an Optax optimizer and learning rate schedule.

    Supports three types of schedules:
    - "CosineAnnealing": A cosine schedule with warm-up and optional restarts.
    - "step": A constant schedule with warm-up, combined with reduce-on-plateau.
    - "warmup_cosine_decay": A standard warm-up followed by cosine decay (default).

    Args:
        options_lr (dict): A dictionary of learning rate options.
        steps_per_epoch (int): The number of training steps in one epoch.
        epochs (int): The total number of training epochs.

    Returns:
        tuple: A tuple containing the configured Optax optimizer and the schedule function.
    """
    if options_lr["type"] == "CosineAnnealing":
        cycle_length = epochs * steps_per_epoch // options_lr["num_cycles"]
        schedule = utils.warmup_cosine_restarts_schedule(
            peak_value=options_lr["init"],
            warmup_steps=options_lr["warmup_epochs"]*steps_per_epoch,
            cycle_length=cycle_length,
            num_cycles=options_lr["num_cycles"],
            decay_factor=options_lr["decay_factor"],
            min_value_ratio=options_lr["lr_factor"]
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(options_lr.get("global_norm_clip", 1.0)),
            optax.radam(schedule)
        )
        schedule = schedule
    elif options_lr["type"] == "step":
        n_warmup = options_lr["warmup_epochs"] * steps_per_epoch
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=options_lr["init"],
            transition_steps=n_warmup
        )
        main_schedule = optax.constant_schedule(options_lr["init"])
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, main_schedule],
            boundaries=[n_warmup]
        )
        # Accumulation defines the number of steps per evaluation
        # Patience is the number of evaluation steps before reducing lr
        optimizer = optax.chain(
            optax.clip_by_global_norm(options_lr.get("global_norm_clip", 1.0)),
            optax.radam(learning_rate=schedule),
            reduce_on_plateau(
                factor=0.5,
                patience=options_lr["patience_epochs"],
                atol=options_lr["loss_min_delta"],
                rtol=0.0,
                min_scale=options_lr["final"],
                accumulation_size=options_lr["accumulation_epochs"] * steps_per_epoch,
            )
        )
    else:
        # Schedule has warm-up for one epoch, then cosine decay
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=options_lr["init"],
            warmup_steps=options_lr["warmup_epochs"]*steps_per_epoch,
            decay_steps=epochs*steps_per_epoch,
            peak_value=options_lr["init"],
            end_value=options_lr["final"]
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(options_lr.get("global_norm_clip", 1.0)),
            optax.radam(schedule)
        )
    return optimizer, schedule


def train_flow(
    data,
    flow_dir,
    ot_pairings_path_stem=None,
    time_scheduler_type="uniform",
    seed=0,
    n_epochs=100,
    batch_size=5000,
    validation_frac=0.25,
    sb_constant=0.0,
    lr_opts={},
    vector_field_opts={},
    time_logger=None,
    reset_flow_lr=False,
    checkpoint_frequency_epochs=-1,
):
    """
    Initializes and trains a normalizing flow model for the distribution function.

    This function sets up the model, data, optimizer, and then calls the main
    training routine from `flow_ot_flow_matching`.

    Args:
        data (dict): The training data, containing 'eta' and 'weights'.
        flow_dir (str): Directory to save the trained model and checkpoints.
        ot_pairings_path_stem (str): Base path for precomputed Optimal Transport pairings.
        time_scheduler_type (str): Type of time scheduler for flow matching.
        seed (int): Random seed for reproducibility.
        n_epochs (int): Number of training epochs.
        batch_size (int): Number of samples per training batch.
        validation_frac (float): Fraction of data to use for validation.
        sb_constant (float): Schrödinger Bridge constant. If > 0, uses SB loss.
        lr_opts (dict): Dictionary of learning rate options.
        vector_field_opts (dict): Options for the vector field neural network.
        time_logger (utils.TimeLogger): Optional logger for timing operations.
        model_type (str): The type of flow model to train.
        reset_flow_lr (bool): Whether to reset the learning rate when resuming.
        checkpoint_frequency_epochs (int): Frequency (in epochs) for saving checkpoints.

    Returns:
        tuple: A tuple containing the trained flow model and the loss history.
    """
    key = jax.random.key(seed)

    train_data, val_data = utils.split_data(data, validation_frac)
    n_samples = train_data["eta"].shape[0]
    dim = train_data["eta"].shape[1]

    data_mean = np.mean(train_data["eta"], axis=0)
    data_std = np.std(train_data["eta"], axis=0)
    print(f"Using mean: {data_mean}")
    print(f"       std: {data_std}")

    # NormalizingFlow is a wrapper around a flowjax flow which itself is equipped with
    # a base distribution and a bijection.
    # The bijection is made up of an affine component responsible for data normalization,
    # then a vector field that integrates the source distribution to the target, and finally
    # an affine layer that inverts the first transformation.
    # We optimize the middle vector field using flow matching.
    key, subkey = jax.random.split(key)
    vector_field_opts["pos_mean"], vector_field_opts["pos_std"] = data_mean[:dim//2], data_std[:dim//2]
    flow_model = flow_ot_flow_matching.NormalizingFlow(
        key=subkey,
        data_mean=data_mean,
        data_std=data_std,
        vector_field_params=vector_field_opts,
        model_dir=flow_dir
    )

    # TODO: Load the latest checkpoint if it exists
    # # flow_dir defines the directory the flow is in
    # if type(flow_dir) is not str:
    #     flow_dir = Path(flow_dir)
    # # Make sure the directory exists
    # checkpoint_dir = flow_dir.parent
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set up the optimizer
    optimizer, schedule = get_optimizer_and_schedule(
        options_lr=lr_opts,
        steps_per_epoch=n_samples // batch_size,
        epochs=n_epochs
    )
    loss_history = {'train': [], 'val': [], 'lr': []}

    flow_model, loss_history = flow_ot_flow_matching.train_ot_flow_matching_model(
        key,
        flow_model,
        optimizer,
        schedule,
        lr_opts["type"],
        train_data, val_data,
        data_mean, data_std,
        n_epochs,
        batch_size,
        ot_pairings_path_stem,
        time_scheduler_type,
        sb_constant,
        time_logger=time_logger,
        loss_history=loss_history,
        checkpoint_frequency_epochs=checkpoint_frequency_epochs
    )
    # x_test = jnp.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0])
    # print(flow_model.log_prob(x_test))
    # flow_model_new, loss_history_new = flow_model.load(flow_dir, flow_model)
    # print(flow_model_new.log_prob(x_test))
    # exit()

    return flow_model, loss_history


def save_df_data(df_data, fname):
    # Make the directory if it doesn't exist
    fname.parent.mkdir(parents=True, exist_ok=True)

    kw = dict(compression="lzf", chunks=True)
    with h5py.File(fname, "w") as f:
        for key in df_data:
            f.create_dataset(key, data=df_data[key], **kw)

def load_flow(flow_dir, params, load_history=False):
    # Loads a trained NormalizingFlow model from a specified directory.
    # This currently assumes that the input dimension is 6
    flow_dir = Path(flow_dir)

    model_opts = params["df"]["vector_field_opts"]
    model_opts["pos_mean"] = jnp.zeros(3)
    model_opts["pos_std"] = jnp.zeros(3)
    flow_model = flow_ot_flow_matching.NormalizingFlow(
        key=jax.random.PRNGKey(0), # Note that this key is rendundant but needed for initialization
        data_mean=jnp.zeros(6),
        data_std=jnp.zeros(6),
        vector_field_params=model_opts,
        model_dir=flow_dir
    )
    flow_model, loss_history = flow_model.load(flow_dir, flow_model)
    if load_history:
        return flow_model, loss_history
    return flow_model


def load_potential(potential_dir, params, load_history=False):
    # Loads a trained PotentialModel from a specified directory.
    # Currently assumes a 6D input
    potential_dir = Path(potential_dir)

    # Create an empty model
    potential_model = potential.PotentialModel(
        key=jax.random.PRNGKey(0), # Note that this key is rendundant but needed for initialization 
        model_dir=potential_dir,
        phi_params=params["potential_nn_opts"],
        frameshift_params=params.get("frameshift_opts", None),
    )

    potential_model, loss_history = potential.PotentialModel.load(potential_dir, potential_model)

    if load_history:
        return potential_model, loss_history
    return potential_model


def train_potential(
    df_data,
    potential_dir,
    seed=0,
    loss_opts={},
    potential_nn_opts={},
    lr_opts={},
    n_epochs=4096,
    batch_size=1024,
    validation_frac=0.25,
    checkpoint_frequency_epochs=-1,
    frameshift_opts=None,
    selection_function_opts=None,
):
    """
    Initializes and trains the gravitational potential model.

    This function sets up the potential model, optimizer, and then runs the
    training loop using the samples generated from the normalizing flow.

    Args:
        df_data (dict): Data containing 'eta' samples and 'df_deta' gradients.
        potential_dir (Path): Directory to save the potential model.
        seed (int): Random seed.
        loss_opts (dict): Options for the loss function.
        potential_nn_opts (dict): Options for the potential neural network.
        lr_opts (dict): Options for the learning rate schedule.
        n_epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        validation_frac (float): Fraction of data for validation.
        checkpoint_frequency_epochs (int): Frequency for saving checkpoints.
        frameshift_opts (dict): Options for applying a coordinate frame shift.
        selection_function_opts (dict): Options for the selection function model.

    Returns:
        tuple: A tuple containing the trained potential model and the loss history.
    """
    key = jax.random.key(seed)
    n_samples = df_data["eta"].shape[0]

    # Make the model
    key, subkey = jax.random.split(key)
    n_val = int(validation_frac * n_samples)
    potential_nn_opts["scale"] = np.std(df_data['eta'][n_val:,:3], axis=0)
    potential_model = potential.PotentialModel(
        subkey, potential_dir, phi_params=potential_nn_opts, frameshift_params=frameshift_opts,
        selection_function_params=selection_function_opts
    )

    # Set up the optimizer
    optimizer, schedule = get_optimizer_and_schedule(
        options_lr=lr_opts,
        steps_per_epoch=(n_samples - n_val) // batch_size,
        epochs=n_epochs
    )

    potential_model, loss_history = potential.train_potential(
        key,
        potential_model,
        optimizer,
        schedule,
        lr_opts["type"],
        lr_opts.get("final", None),
        df_data,
        n_epochs,
        batch_size, validation_frac, checkpoint_frequency_epochs, loss_opts
    )
    return potential_model, loss_history


def load_params(fname):
    if fname is not None:
        with open(fname, "r") as f:
            return json.load(f)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Deep Potential: Fit potential from phase-space samples.",
        add_help=True,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=False, help="Input data.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Directory of the run. All subsequent model directories are relative to this one.",
    )
    parser.add_argument(
        "--flow-dir",
        type=str,
        default="models/df/flow",
        help="Subdirectory to store flows in.",
    )
    parser.add_argument(
        "--reset-flow-lr",
        action="store_true",
        help="Reset the learning rate of the flow in the beginning (this is useful when loading in an old model).",
    )
    parser.add_argument(
        "--df-grads-fname",
        type=str,
        default="data/df_gradients.h5",
        help="Filename in which to store the flow samples (positions and flow gradients), relative to run_dir.",
    )
    parser.add_argument(
        "--potential-dir",
        type=str,
        default="models/Phi",
        help="Subdirectory to store the potential in.",
    )
    parser.add_argument(
        "--potential-mask",
        type=str,
        required=False,
        default=None,
        help="Filename for the mask for the potential. The mask is in distance - healpix format.",
    )
    parser.add_argument(
        "--no-potential-training",
        action="store_true",
        help="Do not train the potential.",
    )
    parser.add_argument(
        "--no-flow-training",
        action="store_true",
        help="Do not train the flow, load the trained flows in instead.",
    )
    parser.add_argument(
        "--no-flow-sampling",
        action="store_true",
        help="Do not sample the flow, load the samples in instead.",
    )
    parser.add_argument(
        "--basic-flow-benchmarking",
        action="store_true",
        help="Whether to compute basic flow benchmarks after the flow training is finished.",
    )
    parser.add_argument(
        "--basic-potential-benchmarking",
        action="store_true",
        help="Whether to compute basic potential benchmarks after the potential training is finished.",
    )
    parser.add_argument("--params", type=str, help="JSON with kwargs.", default="options.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    flow_dir = run_dir / args.flow_dir
    potential_dir = run_dir / args.potential_dir
    df_grads_fname = run_dir / args.df_grads_fname

    params = load_params(run_dir / args.params)
    print("Options:")
    print(json.dumps(params, indent=2))

    time_logger = utils.TimeLogger()

    # Print the GPUs available with jax
    print(f"JAX devices: {jax.devices()}")

    # ================= Loading in training data ==================
    # Attrs contain info basic spatial limits on the data
    data, attrs = utils.load_training_data(args.input)

    # ================= Training/loading the flow =================
    if not args.no_flow_training:
        print(f'Loaded {data["eta"].shape[0]} phase-space positions.')

        # Infer the filename pattern of the OT pairing file based on the data file name
        stem = Path(args.input).stem  # 'data_1000pc'
        ot_pairings_path_stem = Path(args.input).parent / "ot_pairings" / f"pairings_{stem}"

        # Train and save normalizing flows
        print("Training normalizing flows ...")
        time_logger.start('Flow training')
        flow, loss_history = train_flow(
            data, flow_dir, **params["df"], reset_flow_lr=args.reset_flow_lr, time_logger=time_logger, ot_pairings_path_stem=ot_pairings_path_stem
        )
        time_logger.stop('Flow training')
        print(f"Training took {time_logger.get_duration('Flow training'):.2f} s.")

    if args.no_flow_training and not args.no_flow_sampling:
        flow, loss_history = load_flow(flow_dir, params, load_history=True)

    # ================= Basic flow benchmarking =================
    if args.basic_flow_benchmarking:
        # If flow has not been read in, read it
        if 'flow' not in locals():
            flow, loss_history = load_flow(flow_dir, params, load_history=True)
        validation_frac = params["df"]["validation_frac"]
        flow_benchmarking.benchmark(flow, jax.random.key(0), time_logger, *utils.split_data(data, validation_frac), loss_history)

    # ================= Sampling the flow/loading samples =================
    n_samples = params["Phi"].pop("n_samples")
    sample_batch_size = params["Phi"].pop("sample_batch_size")
    grad_batch_size = params["Phi"].pop("grad_batch_size")
    if args.no_flow_sampling:
        print("Loading DF gradients ...")
        df_data = utils.load_flow_samples(df_grads_fname)
    else:
        # Sample from the flows and calculate gradients
        print("Sampling from flows ...")
        time_logger.start('Flow sampling')
        # Cut the flow samples to the limits specified by the attributes in 
        # training data. Supports one flow
        df_data = flow_sampling.sample_from_different_flows(
            jax.random.key(params["Phi"]["seed"]),
            [flow],
            [attrs],
            n_samples,
            grad_batch_size=grad_batch_size,
            sample_batch_size=sample_batch_size,
        )
        time_logger.stop('Flow sampling')
        print(f"Sampling took {time_logger.get_duration('Flow sampling'):.2f} s.")
        save_df_data(df_data, df_grads_fname)

    # ================= Applying a spatial mask ================
    if args.potential_mask is not None:
        # Update df_data to include only the data within the mask
        mask = utils.get_mask_eta(df_data["eta"], args.potential_mask)[0]
        df_data["eta"] = df_data["eta"][mask]
        df_data["df_deta"] = df_data["df_deta"][mask]
        if "f" in df_data:
            df_data["f"] = df_data["f"][mask]

    # ================= Training the potential =================
    if not args.no_potential_training:
        print(params["Phi"])
        print("Fitting the potential ...")
        time_logger.start('Potential training')

        phi_model, loss_history = train_potential(
            df_data,
            potential_dir,
            **params["Phi"],
        )
        time_logger.stop('Potential training')
        print(f"Training took {time_logger.get_duration('Potential training'):.2f} s.")

    # ================= Basic potential benchmarking =================
    if args.basic_potential_benchmarking:
        # If potential has not been read in, read it
        if 'phi_model' not in locals():
            phi_model, loss_history = load_potential(potential_dir, params['Phi'], load_history=True)
        fname_mask = args.potential_mask
        validation_frac = params["df"]["validation_frac"]
        spherical_origin = (0, 0, 0)
        cylindrical_origin = (params['Phi'].get('frameshift_opts', {'r0': 8.277})['r0'], 0, 0)
        potential_benchmarking.benchmark_potential(
            phi_model, loss_history, fname_mask, data, attrs, df_data, spherical_origin, cylindrical_origin
        )

    return 0


if __name__ == "__main__":
    main()
