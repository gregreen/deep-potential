#!/usr/bin/env python

from __future__ import print_function, division

# numpy and matplotlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tensorflow & co
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
import tensorflow_addons as tfa
import sonnet as snt

# Misc imports
from time import time
import os
import json
import math

# Custom libraries
import utils


from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.internal import prefer_static


def trace_jacobian_exact_reg(
    ode_fn, state_shape, dtype, kinetic_reg=0, jacobian_reg=0, dv_dt_reg=0
):
    """Generates a function that computes `ode_fn` and trace of the jacobian.

    Augments provided `ode_fn` with explicit computation of the trace of the
    jacobian. This approach scales quadratically with the number of dimensions.
    This method computes unreduced trace, as reduction is performed inside of the
    bijector class.

    Optionally, regularization terms are subtracted from the trace, penalizing
    various measures of the transformation.

    Args:
      ode_fn: `Callable(time, state)` that computes time derivative.
      state_shape: `TensorShape` representing the shape of the state.
      dtype: `tf.DType` object representing the dtype of `state` tensor.
      dv_dt_reg: `float` indicating how strongly to penalize |dv/dt|^2.
      kinetic_reg: `float` indicating how strongly to penalize |v|^2.
      jacobian_reg: `float` indicating how strongly to penalize |grad(v)|^2.

    Returns:
      augmented_ode_fn: `Callable(time, (state, log_det_jac))` that computes
        augmented time derivative `(state_time_derivative, trace_estimation)`.
    """
    del state_shape, dtype  # Not used by trace_jacobian_exact

    def augmented_ode_fn(time, state_log_det_jac):
        """Computes both time derivative and trace of the jacobian."""
        state, _ = state_log_det_jac

        def ode_fn_with_time(x):
            return ode_fn(time, x)
        batch_shape = [prefer_static.size0(state)]

        if dv_dt_reg > 0:
            watched_vars = [time, state]
        elif (kinetic_reg > 0) or (jacobian_reg > 0):
            watched_vars = [state]
        else:
            watched_vars = []

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
            # g.watch([time, state])
            g.watch(watched_vars)
            state_time_derivative, diag_jac = tfp_math.diag_jacobian(
                xs=state, fn=ode_fn_with_time, sample_shape=batch_shape
            )
            # tfp_math.diag_jacobian returns lists
            if isinstance(state_time_derivative, list):
                state_time_derivative = state_time_derivative[0]
            if isinstance(diag_jac, list):
                diag_jac = diag_jac[0]

        trace_value = diag_jac

        # Calculate regularization terms
        if (dv_dt_reg > 0) or (jacobian_reg > 0):
            delv_delx = g.batch_jacobian(state_time_derivative, state)

        if dv_dt_reg > 0:
            print(f"Using dv/dt regularization: {dv_dt_reg}.")
            delv_delt = g.gradient(state_time_derivative, time)
            vnabla_v = tf.linalg.matvec(delv_delx, state_time_derivative)
            dv_dt = delv_delt + vnabla_v
            # print('dv/dt :', dv_dt)
            trace_value = trace_value - dv_dt_reg * dv_dt**2

        if kinetic_reg > 0:
            print(f"Using kinetic regularization: {kinetic_reg}.")
            # print('v :', state_time_derivative.shape)
            trace_value = trace_value - kinetic_reg * state_time_derivative**2

        if jacobian_reg > 0:
            print(f"Using Jacobian regularization: {jacobian_reg}.")
            jacobian_norm2 = tf.math.reduce_sum(delv_delx**2, axis=-1)
            # print('|J|^2 :', jacobian_norm2.shape)
            trace_value = trace_value - jacobian_reg * jacobian_norm2

        return state_time_derivative, trace_value

    return augmented_ode_fn


class ForceFieldModel(snt.Module):
    def __init__(self, n_dim, n_hidden, hidden_size, name="dz_dt"):
        super(ForceFieldModel, self).__init__(name=name)

        self._n_dim = n_dim

        output_sizes = [hidden_size] * n_hidden + [n_dim]
        self._nn = snt.nets.MLP(output_sizes, activation=tf.math.tanh, name="mlp")

    def __call__(self, t, x):
        """
        Returns the vector dx/dt.

        Inputs:
          t (tf.Tensor): Scalar representing time.
          x (tf.Tensor): Spatial coordinates at which to evaluate
            dx/dt. Shape = (n_points, n_dim).
        """
        # Concatenate time and position vectors
        tx = tf.concat([tf.broadcast_to(t, [x.shape[0], 1]), x], 1)
        # Return dz_dt(t,x)
        return self._nn(tx)

    def augmented_field(self, t, y):
        """
        Returns the vector dy/dt, where y = (x, s), and s is the
        path length. This is useful when regularizing a neural ODE
        by path length.

        Inputs:
          t (tf.Tensor): Scalar representing time.
          y (tf.Tensor): Concatenation of spatial coordinates and path
            length. Shape = (n_points, n_dim+1).
        """
        x, s = tf.split(y, [self._n_dim, 1], axis=1)
        dx_dt = self.__call__(t, x)
        ds_dt = tf.math.sqrt(tf.math.reduce_sum(dx_dt**2, axis=1))
        dy_dt = tf.concat([dx_dt, tf.expand_dims(ds_dt, 1)], 1)
        return dy_dt


class FFJORDFlow(tfd.TransformedDistribution):
    def __init__(
        self,
        n_dim,
        n_hidden,
        hidden_size,
        n_bij,
        reg_kw=dict(),
        rtol=1.0e-7,
        atol=1.0e-5,
        base_mean=None,
        base_std=None,
        name="DF",
    ):
        self._n_dim = n_dim
        self._n_hidden = n_hidden
        self._hidden_size = hidden_size
        self._n_bij = n_bij
        self._name = name

        # ODE solver
        self.ode_solver = tfp.math.ode.DormandPrince(rtol=rtol, atol=atol)

        if len(reg_kw):
            print("Using regularization.")

            def trace_augmentation_fn(*args):
                return trace_jacobian_exact_reg(*args, **reg_kw)

        else:
            trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact

        # Force fields guiding transformations
        self.dz_dt = [
            ForceFieldModel(n_dim, n_hidden, hidden_size) for k in range(n_bij)
        ]

        # Initialize bijector
        bij = [
            tfb.FFJORD(
                state_time_derivative_fn=self.dz_dt[k],
                ode_solve_fn=self.ode_solver.solve,
                trace_augmentation_fn=trace_augmentation_fn,
            )
            for k in range(n_bij)
        ]
        bij = tfb.Chain(bij)

        # Multivariate normal base distribution
        self.base_mean = tf.Variable(
            tf.zeros([n_dim]) if base_mean is None else base_mean,
            trainable=False,
            name="base_mean",
        )
        self.base_std = tf.Variable(
            tf.ones([n_dim]) if base_std is None else base_std,
            trainable=False,
            name="base_std",
        )
        base_dist = tfd.MultivariateNormalDiag(
            loc=self.base_mean, scale_diag=self.base_std
        )

        # Initialize FFJORD
        super(FFJORDFlow, self).__init__(
            distribution=base_dist, bijector=bij, name=name
        )

        # Initialize flow by taking a sample
        self.sample([1])

        self.n_var = sum([int(tf.size(v)) for v in self.trainable_variables])
        print(f"# of trainable variables: {self.n_var}")

    def calc_trajectories(self, n_samples, t_eval):
        if t_eval[-1] < 1.0:
            t_eval = np.hstack([t_eval, 1.0])

        x0 = self.distribution.sample([n_samples])

        res = []
        for dzdt in self.dz_dt:
            res.append(self.ode_solver.solve(dzdt, 0, x0, t_eval))
            x0 = res[-1].states[-1]

        return res

    def save_specs(self, spec_name_base):
        """Saves the specs of the model that are required for initialization to a json"""
        d = dict(
            n_dim=self._n_dim,
            n_hidden=self._n_hidden,
            hidden_size=self._hidden_size,
            n_bij=self._n_bij,
            name=self._name,
        )
        with open(spec_name_base + "_spec.json", "w") as f:
            json.dump(d, f)

        return spec_name_base

    @classmethod
    def load(cls, checkpoint_name):
        """Load FFJORDFlow from a checkpoint and a spec file"""
        # Get spec file name
        if (
            checkpoint_name.find("-") == -1
            or not checkpoint_name.rsplit("-", 1)[1].isdigit()
        ):
            raise ValueError(
                "FFJORDFlow checkpoint name doesn't follow the correct syntax."
            )
        spec_name = checkpoint_name.rsplit("-", 1)[0] + "_spec.json"

        # Load network specs
        with open(spec_name, "r") as f:
            kw = json.load(f)
        flow = cls(**kw)

        # Restore variables
        checkpoint = tf.train.Checkpoint(flow=flow)
        checkpoint.restore(checkpoint_name).expect_partial()

        print(f"loaded {flow} from {checkpoint_name}")
        return flow

    @classmethod
    def load_latest(cls, checkpoint_dir):
        """Load the latest FFJORDFlow from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(
                f"Couldn't load a valid FFJORDFlow from {repr(checkpoint_dir)}"
            )
        return FFJORDFlow.load(latest)


def train_flow(
    flow,
    data,
    optimizer=None,
    batch_size=32,
    n_epochs=1,
    lr_type="step",
    lr_init=2.0e-2,
    lr_final=1.0e-4,
    lr_patience=32,
    lr_min_delta=0.01,
    warmup_proportion=0.1,
    validation_frac=0.25,
    checkpoint_every=None,
    checkpoint_hours=None,
    checkpoint_at_end=False,
    max_checkpoints=None,
    checkpoint_dir=r"checkpoints/ffjord",
    checkpoint_name="ffjord",
    neptune_run=None,
    reset_flow_lr=False,
):
    """
    Trains a flow using the given data.

    Inputs:
      flow (NormalizingFlow): Normalizing flow to be trained.
      data (dict of np.array): Observed points. Shape = (# of points, # of dim).
      optimizer (tf.keras.optimizers.Optimizer or str): Optimizer to use.
          Defaults to the Rectified Adam implementation from
          tensorflow_addons. If a string, will try to interpret and
          construct optimizer.
      batch_size (int): Number of points per training batch. Defaults to 32.
      n_epochs (int): Number of training epochs. Defaults to 1.
      checkpoint_dir (str): Directory for checkpoints. Defaults to
          'checkpoints/ffjord/'.
      checkpoint_name (str): Name to save checkpoints under. Defaults
          to 'ffjord'.
      checkpoint_every (int): Checkpoint every N steps. Defaults to 128.

    Returns:
      loss_history (list of floats): Loss after each training iteration.
    """

    # Split training/validation sample.
    # Handle the case where training/validation split has been manually done
    if "eta_train" in data and "eta_val" in data:
        print("Flow training/validation batches were passed in manually..")
        n_samples = data["eta_train"].shape[0]
        n_val = data["eta_val"].shape[0]
        val_batch_size = int(n_val / (n_samples + n_val) * batch_size)

        data_train = tf.constant(data["eta_train"])
        data_val = tf.constant(data["eta_val"])
    else:
        print("Forming flow training/validation batches..")
        n_samples = data["eta"].shape[0]
        n_val = int(validation_frac * n_samples)

        val_batch_size = int(validation_frac * batch_size)
        n_samples -= n_val

        data_val = tf.constant(data["eta"][:n_val])
        data_train = tf.constant(data["eta"][n_val:])

    print(f"Train/validation split: {data_train.shape[0]}/{data_val.shape[0]}")

    # Create Tensorflow datasets
    batches = tf.data.Dataset.from_tensor_slices(data_train)
    batches = batches.shuffle(n_samples, reshuffle_each_iteration=True)
    batches = batches.repeat(n_epochs)
    batches = batches.batch(batch_size, drop_remainder=True)

    val_batches = tf.data.Dataset.from_tensor_slices(data_val)
    val_batches = val_batches.shuffle(n_val, reshuffle_each_iteration=True)
    val_batches = val_batches.repeat(n_epochs)
    val_batches = val_batches.batch(val_batch_size, drop_remainder=True)

    n_steps = (n_epochs * n_samples) // batch_size
    unrounded_steps_per_epoch = (
        n_samples / batch_size
    )  # Due to the way batches are repeated, the number of steps per epoch is not necessarily an integer..
    epoch_counter, rounded_epoch_duration = unrounded_steps_per_epoch, 0

    def get_optimizer():
        if isinstance(optimizer, str):
            if lr_type == "exponential":
                lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    lr_init, n_steps, lr_final / lr_init, staircase=False
                )
            elif lr_type == "step":
                lr_schedule = lr_init
                steps_since_decline = 0
            else:
                raise ValueError(f'Unknown lr_type: "{lr_type}" ("exponential" or "step")')
            if optimizer == "RAdam":
                opt = tfa.optimizers.RectifiedAdam(
                    lr_schedule, total_steps=n_steps, warmup_proportion=warmup_proportion
                )
            elif optimizer == "SGD":
                opt = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.5)
            else:
                raise ValueError(f'Unrecognized optimizer: "{optimizer}"')
        else:
            opt = optimizer
        return opt
    opt = get_optimizer()
    steps_since_decline = 0

    print(f"Optimizer: {opt}")

    train_loss_history = []
    val_loss_history = []
    lr_history = []

    t0 = time()

    # Set up checkpointing
    step = tf.Variable(0, name="step")
    val_loss_min = tf.Variable(np.inf, name="loss_min")

    if checkpoint_every is not None:
        checkpoint = tf.train.Checkpoint(
            opt=opt, flow=flow, step=step, loss_min=val_loss_min
        )
        chkpt_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            max_to_keep=max_checkpoints,
            keep_checkpoint_every_n_hours=checkpoint_hours,
        )

        # Look for latest extisting checkpoint
        latest = chkpt_manager.latest_checkpoint
        if latest is not None:
            print(f"Restoring from checkpoint {latest} ...")
            checkpoint.restore(latest)
            print(f"Beginning from step {int(step)}.")

            if reset_flow_lr:
                # Reset the learning rate, and also forgo loading loss history
                # since the history is used for learning rate scheduling
                # (TODO: this is bad practice!)
                print("Resetting flow learning rate.")
                opt = get_optimizer()
                steps_since_decline = 0
            else:
                # Try to load loss history
                loss_fname = f"{latest}_loss.txt"
                (
                    train_loss_history,
                    val_loss_history,
                    lr_history,
                    _,
                    _,
                ) = utils.load_loss_history(loss_fname)

        # Convert from # of epochs to # of steps between checkpoints
        # checkpoint_steps = checkpoint_every * n_samples // batch_size
        checkpoint_steps = math.ceil(checkpoint_every * n_samples / batch_size)

    # Keep track of whether this is the first step.
    # Were it not for checkpointing, we could use i == 0.
    traced = False

    @tf.function
    def training_step(batch):
        print(f"Tracing training_step with batch shape {batch.shape} ...")
        variables = flow.trainable_variables
        with tf.GradientTape() as g:
            g.watch(variables)
            train_loss = -tf.reduce_mean(flow.log_prob(batch))
        grads = g.gradient(train_loss, variables)
        # tf.print([(v.name,tf.norm(dv)) for v,dv in zip(variables,grads)])
        grads, global_norm = tf.clip_by_global_norm(grads, 10.0)
        # grads,global_norm = tf.clip_by_global_norm(grads, 100.)
        # tf.print('\nglobal_norm =', global_norm)
        # tf.print([(v.name,tf.norm(v)) for v in grads])
        # tf.print('loss =', loss)
        opt.apply_gradients(zip(grads, variables))
        return train_loss

    @tf.function
    def validation_step(batch):
        print(f"Tracing validation_step with batch shape {batch.shape} ...")
        val_loss = -tf.reduce_mean(flow.log_prob(batch))
        return val_loss

    update_bar = utils.get_training_progressbar_fn(n_steps, train_loss_history, opt)
    t1 = None

    # Main training loop
    # First we check for stopping conditions, then perform the training step
    # and update the progress bar. The order is important in case we load in
    # a checkpoint which has already satisfied the stopping conditions.
    for i, (y, y_val) in enumerate(zip(batches, val_batches), int(step)):
        if i >= n_steps:
            # Break if too many steps taken. This can occur
            # if we began from a checkpoint.
            break

        # Adjust learning rate?
        if lr_type == "step":
            n_smooth = max(lr_patience // 8, 1)
            if len(train_loss_history) >= n_smooth:
                train_loss_avg = np.mean(train_loss_history[-n_smooth:])
                val_loss_avg = np.mean(val_loss_history[-n_smooth:])
            else:
                train_loss_avg = np.inf
                val_loss_avg = np.inf

            if val_loss_avg < val_loss_min - lr_min_delta:
                steps_since_decline = 0
                print(f"New minimum loss: {val_loss_avg}.")
                val_loss_min.assign(val_loss_avg)
            elif steps_since_decline >= lr_patience:
                # Reduce learning rate
                old_lr = float(opt.lr)
                new_lr = 0.5 * old_lr
                print(f"Reducing learning rate from {old_lr} to {new_lr}.")
                print(f"   (loss threshold: {float(val_loss_min-lr_min_delta)})")

                # Terminate if the learning rate is below the threshold
                if new_lr < lr_final:
                    print(
                        "Learning rate below threshold. Checkpointing and terminating ..."
                    )
                    step.assign(i + 1)
                    chkpt_fname = chkpt_manager.save()
                    print(f"  --> {chkpt_fname}")
                    utils.save_loss_history(
                        f"{chkpt_fname}_loss.txt",
                        train_loss_history,
                        val_loss_history=val_loss_history,
                        lr_history=lr_history,
                    )
                    fig = utils.plot_loss(
                        train_loss_history,
                        val_loss_hist=val_loss_history,
                        lr_hist=lr_history,
                    )
                    fig.savefig(f"{chkpt_fname}_loss.pdf")
                    plt.close(fig)
                    break
                opt.lr.assign(new_lr)
                steps_since_decline = 0
            else:
                steps_since_decline += 1

        train_loss = training_step(y)
        val_loss = validation_step(y_val)

        train_loss_history.append(float(train_loss))
        val_loss_history.append(float(val_loss))
        lr_history.append(float(opt._decayed_lr(tf.float32)))

        if neptune_run is not None:
            neptune_run["train/train_loss"].append(train_loss_history[-1])
            neptune_run["train/val_loss"].append(val_loss_history[-1])
            neptune_run["train/lr"].append(lr_history[-1])

            epoch_counter -= 1
            rounded_epoch_duration += 1
            if epoch_counter <= -0.5:
                # we say that a new epoch has started when
                # integer*unrounded_steps_per_epoch \in [-0.5, 0.5]
                neptune_run["train/epoch_train_loss"].append(
                    np.mean(train_loss_history[-rounded_epoch_duration:])
                )
                neptune_run["train/epoch_val_loss"].append(
                    np.mean(val_loss_history[-rounded_epoch_duration:])
                )
                neptune_run["train/epoch_lr"].append(
                    np.mean(lr_history[-rounded_epoch_duration:])
                )
                epoch_counter += unrounded_steps_per_epoch
                rounded_epoch_duration = 0

        # Progress bar
        update_bar(i)

        if not traced:
            # Get time after gradients function is first traced
            traced = True
            t1 = time()

        # Checkpoint periodically or at the very end
        if (checkpoint_at_end and i == n_steps - 1) or (checkpoint_every is not None) and i and not (i % checkpoint_steps):
            print("Checkpointing ...")
            step.assign(i + 1)
            chkpt_fname = chkpt_manager.save()
            print(f"  --> {chkpt_fname}")
            utils.save_loss_history(
                f"{chkpt_fname}_loss.txt",
                train_loss_history,
                val_loss_history=val_loss_history,
                lr_history=lr_history,
            )
            fig = utils.plot_loss(
                train_loss_history, val_loss_hist=val_loss_history, lr_hist=lr_history
            )
            fig.savefig(f"{chkpt_fname}_loss.pdf")
            plt.close(fig)

    t2 = time()
    train_loss_avg = np.mean(train_loss_history[-50:])
    n_steps = len(train_loss_history)
    print(f"<train loss> = {train_loss_avg: >7.5f}")
    if t1 is not None:
        print(f"tracing time: {t1-t0:.2f} s")
        print(f"training time: {t2-t1:.1f} s ({(t2-t1)/(n_steps-1):.4f} s/step)")

    if neptune_run is not None:
        neptune_run.stop()

    # Save the trained model
    # checkpoint = tf.train.Checkpoint(flow=flow)
    # checkpoint.save(checkpoint_prefix + '_final')

    return train_loss_history, val_loss_history, lr_history


def main():
    from scipy.stats import multivariate_normal

    flow = FFJORDFlow(n_dim=2, n_hidden=5, hidden_size=20, n_bij=1)

    # If directory checkpoints exists, delete it
    if os.path.exists("checkpoints/ffjord"):
        import shutil

        shutil.rmtree("checkpoints/ffjord")

    n_samples = 8 * 1024
    mu = [[-2, 0.0], [2, 0.0]]
    cov = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
    data = {}
    data['eta'] = tf.concat(
        [
            np.random.multivariate_normal(m, c, n_samples // 2).astype("f4")
            for m, c in zip(mu, cov)
        ],
        axis=0,
    )
    # Shuffle the data
    data['eta'] = tf.random.shuffle(data['eta'])

    plt.hist2d(data['eta'][:, 0], data['eta'][:, 1], bins=128, cmap="Blues")
    plt.savefig("data.png")

    train_flow(
        flow, data, batch_size=1024, n_epochs=16, checkpoint_every=256,
        optimizer='RAdam', checkpoint_at_end=True, lr_init=0.02, lr_patience=16
    )

    fname = "checkpoints/ffjord/"
    flow.save_specs('checkpoints/ffjord/ffjord')
    flow2 = FFJORDFlow.load_latest(fname)

    samples = flow.sample(32 * 1024)
    plt.hist2d(samples[:, 0], samples[:, 1], bins=128, cmap="Blues")
    plt.savefig("samples.png")

    x = tf.random.normal([5, 2])

    theoretical_y = np.log(
        (multivariate_normal.pdf(x, mean=mu[0], cov=cov[0]) +
         multivariate_normal.pdf(x, mean=mu[1], cov=cov[1]))/2
    )
    y = flow.log_prob(x)
    y2 = flow2.log_prob(x)
    print(theoretical_y)
    print(y.numpy())
    print(y2.numpy())

    # for i in range(10):
    #    eta = tf.random.normal([1024,2])
    #    df_deta = calc_flow_gradients(flow, eta)
    #    print(i)

    return 0


if __name__ == "__main__":
    main()
