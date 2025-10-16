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
import flow_matching
import utils
import flow_benchmarking
import potential_benchmarking
import flow_sampling
import potential
import flow_ot_dataset_loader_maker

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
    data_fname=None,
    training_method="OTFlowMatching",
    ot_pairings_opts={},
    time_scheduler_type="uniform",
    seed=0,
    n_epochs=100,
    batch_size=5000,
    validation_frac=0.25,
    loss_opts={},
    lr_opts={},
    vector_field_opts={},
    time_logger=None,
    reset_flow_lr=False,
    checkpoint_frequency_epochs=-1
):
    """
    Initializes and trains a normalizing flow model for the distribution function.

    This function sets up the model, data, optimizer, and then calls the main
    training routine from `flow_ot_flow_matching`.

    Args:
        data (dict): The training data, containing 'eta' and 'weights'.
        flow_dir (str): Directory to save the trained model and checkpoints.
        training_method (str): The type of flow model to train.
        ot_pairings_opts (dict): Dictionary of the params required for precomputing
            OT pairings. If empty, OT pairings are not used.
        time_scheduler_type (str): Type of time scheduler for flow matching.
        seed (int): Random seed for reproducibility.
        n_epochs (int): Number of training epochs.
        batch_size (int): Number of samples per training batch.
        validation_frac (float): Fraction of data to use for validation.
        loss_opts (dict): Dictionary for the hyperparameters in the loss function.
            They mostly control regularization.
        lr_opts (dict): Dictionary of learning rate options.
        vector_field_opts (dict): Options for the vector field neural network.
        time_logger (utils.TimeLogger): Optional logger for timing operations.
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

    if training_method == "OTFlowMatching":
        if ot_pairings_opts == {}:
            raise ValueError("OT pairings not found, and ot_pairings_opts is empty. Cannot proceed.")
        # Infer the filename pattern of the OT pairing file based on the data file name
        ot_pairings_dir = Path(data_fname).parent / "ot_pairings" / f"pairings_{Path(data_fname).stem}_seed{ot_pairings_opts['seed']}"

        # Check if the OT pairing files exist
        ot_pairings_exist = any(ot_pairings_dir.glob("train_epochs_*_*.npz")) and any(ot_pairings_dir.glob("val_epochs_*_*.npz"))
        if not ot_pairings_exist:
            print("OT pairings not found, precomputing them now ...")
            flow_ot_dataset_loader_maker.make_ot_dataloader(
                data_fname=data_fname,
                **ot_pairings_opts
            )
            print("OT pairings precomputation done.")

    # NormalizingFlow is a wrapper around a flowjax flow which itself is equipped with
    # a base distribution and a bijection.
    # The bijection is made up of an affine component responsible for data normalization,
    # then a vector field that integrates the source distribution to the target, and finally
    # an affine layer that inverts the first transformation.
    # We optimize the middle vector field using flow matching.
    key, subkey = jax.random.split(key)
    vector_field_opts["pos_mean"], vector_field_opts["pos_std"] = data_mean[:dim//2].tolist(), data_std[:dim//2].tolist()
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

    kwargs = dict(key=key, model=flow_model, optimizer=optimizer, schedule=schedule,
                  schedule_type=lr_opts["type"], lr_final=lr_opts.get("final", None),
                  train_data=train_data, val_data=val_data,
                  norm_mean=data_mean, norm_std=data_std, epochs=n_epochs, batch_size=batch_size,
                  time_scheduler_type=time_scheduler_type, loss_params=loss_opts,
                  time_logger=time_logger, loss_history=loss_history,
                  checkpoint_frequency_epochs=checkpoint_frequency_epochs)
    if training_method == "OTFlowMatching":
        ot_pairings_dir = Path(data_fname).parent / "ot_pairings" / f"pairings_{Path(data_fname).stem}_seed{ot_pairings_opts['seed']}"
        flow_model, loss_history = flow_ot_flow_matching.train_ot_flow_matching_model(
            **kwargs, ot_pairings_dir=ot_pairings_dir,
        )
    elif training_method == "FlowMatching":
        flow_model, loss_history = flow_matching.train_flow_matching_model(
            **kwargs
        )
    else:
        raise ValueError(f"Unknown training_method: {training_method}")

    return flow_model, loss_history


def save_df_data(df_data, fname):
    # Make the directory if it doesn't exist
    fname.parent.mkdir(parents=True, exist_ok=True)

    kw = dict(compression="lzf", chunks=True)
    with h5py.File(fname, "w") as f:
        for key in df_data:
            f.create_dataset(key, data=df_data[key], **kw)


def train_potential(
    df_data,
    potential_dir,
    seed=0,
    loss_opts={},
    potential_nn_opts={},
    lr_opts={},
    n_epochs_noselfn=4096,
    n_epochs_selfn=4096,
    batch_size=1024,
    validation_frac=0.25,
    checkpoint_frequency_epochs=-1,
    frameshift_opts=None,
    selection_function_opts=None,
    benchmark_after_first_loop=True,
    benchmarking_args={}
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
    potential_nn_opts["scale"] = np.std(df_data['eta'][n_val:,:3], axis=0).tolist()
    potential_model = potential.PotentialModel(
        subkey, potential_dir, phi_params=potential_nn_opts, frameshift_params=frameshift_opts,
        selection_function_params=selection_function_opts
    )

    # We first train without selection function, then with it if applicable.
    # jobs = [(n_epochs_selfn, True)]
    jobs = []
    if n_epochs_noselfn > 0:
        jobs.append((n_epochs_noselfn, False))
    if selection_function_opts is not None:
        selection_function_opts["scale"] = potential_nn_opts["scale"]
        if n_epochs_selfn > 0:
            jobs.append((n_epochs_selfn, True))

    for i, (n_epochs, train_selfn) in enumerate(jobs):
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
            batch_size, validation_frac, checkpoint_frequency_epochs, loss_opts,
            train_selfn=train_selfn,
            reset_lr=True
        )

        if i == 0 and len(jobs) == 2:
            # Benchmark the intermediate potential if requested
            if benchmark_after_first_loop:
                print("Benchmarking the potential after the first training loop ...")
                spherical_origin = (0, 0, 0)
                cylindrical_origin = (frameshift_opts.get('r0', 8.277), 0, 0)
                potential_benchmarking.benchmark_potential(
                    potential_model, loss_history,
                    **benchmarking_args,
                    spherical_origin=spherical_origin, cylindrical_origin=cylindrical_origin,
                    checkpoint_index=potential_model.checkpoint_index - 1
                )
    return potential_model, loss_history


def load_flow(flow_dir, checkpoint_index=-1, load_history=False):
    # Loads a trained NormalizingFlow model from a specified directory.
    # This currently assumes that the input dimension is 6
    flow_model, loss_history = flow_ot_flow_matching.NormalizingFlow.load(Path(flow_dir), load_index=checkpoint_index, load_history=load_history)
    if load_history:
        return flow_model, loss_history
    return flow_model


def load_potential(potential_dir, checkpoint_index=-1, load_history=False):
    # Loads a trained PotentialModel from a specified directory.
    # Currently assumes a 6D input
    potential_model, loss_history = potential.PotentialModel.load(Path(potential_dir), load_index=checkpoint_index, load_history=load_history)

    if load_history:
        return potential_model, loss_history
    return potential_model


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Deep Potential: Fit potential from phase-space samples.",
        add_help=True,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str,
                        required=False, help="Input data.")
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
        '--flow-checkpoint-index',
        type=int,
        default=-1,
        help='If needed, the index of the flow checkpoint to load. If -1, loads the latest checkpoint.'
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
        "--potential-training",
        action="store_true",
        help="Train the potential. If not set, the potential is loaded from potential-dir.",
    )
    parser.add_argument(
        "--flow-training",
        action="store_true",
        help="Train the flow. If not set, the flow is loaded from flow-dir.",
    )
    parser.add_argument(
        "--flow-sampling",
        action="store_true",
        help="Sample from the flow. If not set, the flow samples are loaded in when necessary.",
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
    parser.add_argument(
        "--basic-potential-benchmarking-gaia-units",
        action="store_true",
        help="Whether to assume Gaia-like units or dimensioneless units for potential plotting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="If passed and non-negative, overwrite the random seed used for generating the DF and the potential."
        "Does not affect the seed used for OT pairings.",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="JSON with kwargs.",
        default="options.json"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    flow_dir = run_dir / args.flow_dir
    potential_dir = run_dir / args.potential_dir
    df_grads_fname = run_dir / args.df_grads_fname

    params = utils.load_params(run_dir / args.params)
    # Overwrite the seed if requested
    if args.seed >= 0:
        if "df" in params:
            params["df"]["seed"] = args.seed
        if "flow_sampling" in params:
            params["flow_sampling"]["seed"] = args.seed
        if "Phi" in params:
            params["Phi"]["seed"] = args.seed
    print("Options:")
    print(json.dumps(params, indent=2))

    time_logger = utils.TimeLogger()

    # Print the GPUs available with jax
    print(f"JAX devices: {jax.devices()}")

    # ================= Loading in training data ==================
    # Attrs contain info basic spatial limits on the data
    data, attrs = utils.load_training_data(args.input)

    # ================= Training/loading the flow =================
    if args.flow_training:
        benchmarking_r0 = params["df"].pop('benchmarking_r0', 8.277)
        print(f'Loaded {data["eta"].shape[0]} phase-space positions.')

        # Train and save normalizing flows
        print("Training normalizing flows ...")
        time_logger.start('Flow training')
        flow, loss_history = train_flow(
            data, flow_dir, args.input, **params["df"],
            reset_flow_lr=args.reset_flow_lr,
            time_logger=time_logger
        )
        time_logger.stop('Flow training')
        print(f"Training took {time_logger.get_duration('Flow training'):.2f} s.")

    if not args.flow_training and args.flow_sampling:
        flow, loss_history = load_flow(flow_dir, args.flow_checkpoint_index, load_history=True)

    # ================= Basic flow benchmarking =================
    if args.basic_flow_benchmarking:
        # If flow has not been read in, read it
        if 'flow' not in locals():
            flow, loss_history = load_flow(flow_dir, args.flow_checkpoint_index, load_history=True)
        validation_frac = params["df"]["validation_frac"]
        spherical_origin = (0, 0, 0)
        benchmarking_r0 = params["df"].pop('benchmarking_r0', 8.277)
        cylindrical_origin = (benchmarking_r0, 0, 0)
        flow_benchmarking.benchmark(flow, jax.random.key(0), time_logger, *utils.split_data(data, validation_frac), loss_history, spherical_origin, cylindrical_origin)

    # Exit if nothing else needs to be done
    if not args.flow_sampling and not args.potential_training and not args.basic_potential_benchmarking:
        return 0

    # ================= Sampling the flow/loading samples =================
    if not args.flow_sampling:
        print("Loading DF gradients ...")
        df_data = utils.load_flow_samples(df_grads_fname)
    else:
        # Sample from the flows and calculate gradients
        print("Sampling from flows ...")
        time_logger.start('Flow sampling')
        # Cut the flow samples to the limits specified by the attributes in 
        # training data. Supports one flow
        df_data = flow_sampling.sample_and_differentiate_from_different_flows(
            flow_list=[flow],
            attrs_list=[attrs],
            **params["flow_sampling"]
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
    if args.potential_training:
        print(params["Phi"])
        print("Fitting the potential ...")
        time_logger.start('Potential training')

        train_potential(
            df_data,
            potential_dir,
            **params["Phi"],
            benchmark_after_first_loop=args.basic_potential_benchmarking,
            benchmarking_args=dict(fname_mask=args.potential_mask, data_train=data, attrs_train=attrs, df_data=df_data,
                                   is_gaia=args.basic_potential_benchmarking_gaia_units),
        )
        time_logger.stop('Potential training')
        print(f"Training took {time_logger.get_duration('Potential training'):.2f} s.")

    # ================= Basic potential benchmarking =================
    if args.basic_potential_benchmarking:
        phi_model, loss_history = load_potential(
            potential_dir, load_history=True
        )
        fname_mask = args.potential_mask
        spherical_origin = (0, 0, 0)
        cylindrical_origin = (
            params['Phi'].get('frameshift_opts', {'r0': 8.277})['r0'],
            0, 0
        )
        potential_benchmarking.benchmark_potential(
            phi_model, loss_history,
            fname_mask, data, attrs, df_data,
            spherical_origin, cylindrical_origin,
            is_gaia=args.basic_potential_benchmarking_gaia_units,
        )

    return 0


if __name__ == "__main__":
    main()
