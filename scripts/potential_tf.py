#!/usr/bin/env python

from __future__ import print_function, division

# Tensorflow & co
import tensorflow as tf

print(f"Tensorflow version {tf.__version__}")
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_probability as tfp

print(f"Tensorflow Probability version {tfp.__version__}")
tfb = tfp.bijectors
tfd = tfp.distributions
import sonnet as snt

import numpy as np
import matplotlib.pyplot as plt

import math
import json
import os
from time import time

# Custom libraries
import utils

def calc_df_deta(f_func, q, p):
    """Calculates gradients of distribution function at q and p"""
    with tf.GradientTape(persistent=True) as g:
        g.watch([q, p])
        f = f_func(q, p)

    df_dq = g.gradient(f, q, unconnected_gradients="zero")
    df_dp = g.gradient(f, p, unconnected_gradients="zero")

    return f, df_dq, df_dp


def calc_phi_derivatives(phi_func, q, return_phi=False):
    """Calculates derivatives of the potential at q."""
    # We have to use an unstacking and re-stacking trick,
    # which comes from https://github.com/xuzhiqin1990/laplacian/
    with tf.GradientTape(persistent=True) as gg:
        q_unstacked = tf.unstack(q, axis=1)
        gg.watch(q_unstacked)

        with tf.GradientTape() as g:
            q_stacked = tf.stack(q_unstacked, axis=1)
            g.watch(q_stacked)
            phi = phi_func(q_stacked)

        dphi_dq = g.gradient(phi, q_stacked)
        dphi_dq_unstacked = tf.unstack(dphi_dq, axis=1)

    # Laplacian of phi
    d2phi_dq2 = 0
    for di, qi in zip(dphi_dq_unstacked, q_unstacked):
        d2phi_dq2 += gg.gradient(di, qi)

    if return_phi:
        return phi, dphi_dq, d2phi_dq2

    return dphi_dq, d2phi_dq2


def get_phi_loss_gradients(
    phi,
    frameshift,
    params,
    q,
    p,
    f=None,
    df_dq=None,
    df_dp=None,
    lam=1.0,
    mu=0,
    xi=1.0,
    delf_delt_scale=1.0,
    l2=tf.constant(0.01),
    return_grads=True,
    return_loss_noreg=False,
):
    """
    Calculates both the loss and the gradients of the loss w.r.t. the
    given parameters.

    The loss describes how far the system is from satisfying both the
    Collisionless Boltzmann Equation (CBE) and the stationarity condition in
    the lab or a moving frame (described by frameshift, default is a rotating
    frame).
    Assuming the Hamiltonian is

        H = p^2 / 2 + phi(q),

    CBE + stationarity read

        0 = (v - u)*df/dx - (dPhi/dx + w)*df/dv,

    where for rotation, u = Omega x (x - xc) and w = Omega x v.
    Loss is a function of d2phi_dq2 and CBE+stationarity.

    Let
        n := # of points at which to evaluate loss terms,
        d := # of spatial dimensions.

    Inputs:
        f (callable): The distribution function. Takes q and p, each
            (n,d) tensors, and returns a (n,) tensor.
        phi (callable): The gravitational potential. Takes q, a
            (n,d) tensor, and returns a (n,) tensor.
        frameshift (callable): an object for indicating which
            frame stationarity is enforced in. Takes q, p, both (n,d)
            tensors, and returns two (n,d) tensors, corresponding
            to u and w defined by the frameshift. If set to None,
            stationarity is enforced in lab frame.
        params (tuple of tf.Variable): The gradients will be taken
            w.r.t these parameters. Usually they are parameters of
            phi and frameshift.
        q (tf.Tensor): Spatial coordinates of shape (n,d) at which
            to compute the loss and gradients.
        p (tf.Tensor): Momenta of shape (n,d) at which to compute
            the loss and gradients.
        lam (scalar): Constant that determines how strongly to
            penalize negative matter densities. Larger values
            translate to stronger penalties. Defaults to 1.

        NOTE: weight_samples, sigma_q, sigma_p, eps_w aren't implemented

    Outputs:
        loss (tf.Tensor): Scalar tensor.
        dloss_dparam (list of tf.Tensor): The gradient of the loss
            w.r.t. each parameter.
    """

    # If df/dq and df/dp are not provided, then calculate them
    # from using the distribution function.
    if f is not None:
        _, df_dq, df_dp = calc_df_deta(f, q, p)
    elif (df_dq is None) or (df_dp is None):
        raise ValueError("If f is not provided, then df_dq and df_dp must be provided.")

    c = xi
    print(f"c = {c}")

    with tf.GradientTape(watch_accessed_variables=False) as g:
        if return_grads:
            g.watch(params)

        # Calculate derivatives of phi w.r.t. q
        dphi_dq, d2phi_dq2 = calc_phi_derivatives(phi, q)

        if frameshift is None:
            # partial f / partial t = {H,f}
            df_dt = tf.reduce_sum(df_dp * dphi_dq - df_dq * p, axis=1)

            null_hyp = df_dt
        else:
            u, w = frameshift(q, p)

            # TODO: add weighing

            null_hyp = tf.reduce_sum((p - u) * df_dq - (dphi_dq + w) * df_dp, axis=1)

        # Average over sampled points in phase space
        likelihood = tf.math.asinh(c * tf.math.abs(null_hyp)) / c
        # likelihood = tf.math.asinh(tf.math.abs(null_hyp))
        tf.print("likelihood:", tf.reduce_mean(likelihood))

        if phi._name == "PhiNNAnalytic":
            tf.print(type(phi))
            # tf.print(f'mn (amp, a, b): ({phi._mn_amp.print_tensor()}, {phi._mn_a}, {phi._mn_b})')
            # tf.print(f'a: {phi._mn_a.numpy()}')
            # print(f'nm (amp, a, b): ({phi._mn_amp.numpy():.5f}, {phi._mn_a.numpy():.5f}, {phi._mn_b.numpy():.5f})')

        if lam != 0:
            prior_neg = tf.math.asinh(tf.clip_by_value(-d2phi_dq2, 0.0, np.inf))
            # prior_neg = tf.clip_by_value(-d2phi_dq2, 0., np.inf)
            tf.print("prior_neg:", tf.reduce_mean(lam * prior_neg))
            # pneg_mean = tf.math.reduce_mean(prior_neg)
            # pneg_max = tf.math.reduce_max(prior_neg)
            # L_mean = tf.math.reduce_mean(likelihood)
            # r = tf.norm(q, axis=1)
            # r_max = tf.math.reduce_max(r)
            # tf.print('     lambda =', lam)
            # tf.print('        <L> =', L_mean)
            # tf.print('  <penalty> =', pneg_mean)
            # tf.print('penalty_max =', pneg_max)
            # tf.print('      r_max =', r_max)
            # tf.print('')
            likelihood = likelihood + lam * prior_neg

        if mu != 0:
            prior_pos = tf.math.asinh(tf.clip_by_value(d2phi_dq2, 0.0, np.inf))
            tf.print("prior_pos:", tf.reduce_mean(mu * prior_pos))
            likelihood = likelihood + mu * prior_pos

        # Regularization penalty
        # penalty = 0.
        # for p in params:
        #     penalty += tf.reduce_sum(p**2)

        # tf.print('likelihood:', tf.reduce_mean(likelihood))
        loss = tf.math.log(tf.reduce_mean(likelihood))
        tf.print("loss (before penalty):", loss)
        loss_noreg = tf.identity(loss)
        #    likelihood
        #    + lam*prior_neg
        #    + mu*prior_pos
        #    # + reg*penalty
        # )

        # L2 penalty on all weights with l2penalty in their name, except for biases.
        if l2 != 0:
            print(f"l2 = {l2}")
            penalty = 0
            for p in params:
                if ("l2penalty" in p.name) and not ("b:0" in p.name):
                    print(f"L2 penalty on {p}")
                    penalty += l2 * tf.reduce_mean(p**2)
            tf.print("L2 penalty:", penalty)
            loss += penalty

    # Gradients of loss w.r.t. NN parameters
    if return_grads:
        dloss_dparam = g.gradient(loss, params)
        if return_loss_noreg:
            return loss, loss_noreg, dloss_dparam
        return loss, dloss_dparam

    if return_loss_noreg:
        return loss, loss_noreg
    return loss


class PhiNN(snt.Module):
    """
    Feed-forward neural network to represent the gravitational
    potential.

    Note on checkpointing: Both PhiNN and FrameShift rely on a combination of
    tf.Checkpoint and a custom spec saving system. This is caused by
    restoration from a tf Checkpoint requiring an already initialized instance
    of the object. Initializing the object can't use data stored in the
    checkpoint, and must find its metadata elsewhere, hence from the spec file.
    """

    def __init__(self, n_dim=3, n_hidden=3, hidden_size=32, scale=None, name="Phi"):
        """
        Constructor for PhiNN.

        Inputs:
            n_dim (int): Dimensionality of space.
            n_hidden (int): Number of hidden layers.
            hidden_size (int): Number of neurons in each hidden layer.
            scale (float-array-like): Typical scale of coordinates along
                each dimension. This will be used to rescale the
                coordinates, before passing them into the neural network.
                Defaults to None, in which case the scales are 1.
        """
        super(PhiNN, self).__init__(name=name)

        self._n_dim = n_dim
        self._n_hidden = n_hidden
        self._hidden_size = hidden_size
        self._name = name

        # Coordinate scaling
        if scale is None:
            coord_scale = np.ones((1, n_dim), dtype="f4")
        else:
            print(f"Using coordinate scale: {scale}")
            coord_scale = np.reshape(scale, (1, n_dim)).astype("f4")
        self._scale = tf.Variable(
            1 / coord_scale, trainable=False, name="coord_scaling"
        )

        # Variables that have "l2penalty" in them are penalized
        self._layers = [
            snt.Linear(hidden_size, name=f"hidden_{i}_l2penalty")
            for i in range(n_hidden)
        ]
        self._layers.append(snt.Linear(1, with_bias=False, name="Phi_l2penalty"))
        self._activation = tf.math.tanh

        # Initialize
        self.__call__(tf.zeros([1, n_dim]))

    def __call__(self, q):
        """Returns the gravitational potential"""
        # Transform coordinates to standard frame
        q = self._scale * q
        # Run the coordinates through the neural net
        for layer in self._layers[:-1]:
            q = layer(q)
            q = self._activation(q)
        # No activation on the final layer
        q = self._layers[-1](q)
        return q

    def save_specs(self, spec_name_base):
        """Saves the specs of the model that are required for initialization to a json"""
        d = dict(
            n_dim=self._n_dim,
            n_hidden=self._n_hidden,
            hidden_size=self._hidden_size,
            name=self._name,
        )
        with open(spec_name_base + "_spec.json", "w") as f:
            json.dump(d, f)

        return spec_name_base

    @classmethod
    def load(cls, checkpoint_name):
        """Load PhiNN from a checkpoint and a spec file"""
        # Get spec file name
        if (
            checkpoint_name.find("-") == -1
            or not checkpoint_name.rsplit("-", 1)[1].isdigit()
        ):
            raise ValueError("PhiNN checkpoint name doesn't follow the correct syntax.")
        spec_name = checkpoint_name.rsplit("-", 1)[0] + "_spec.json"

        # Load network specs
        with open(spec_name, "r") as f:
            kw = json.load(f)
        phi_nn = cls(**kw)

        # Restore variables
        checkpoint = tf.train.Checkpoint(phi=phi_nn)
        checkpoint.restore(checkpoint_name).expect_partial()

        print(f"loaded {phi_nn} from {checkpoint_name}")
        return phi_nn

    @classmethod
    def load_latest(cls, checkpoint_dir):
        """Load the latest PhiNN from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(f"Couldn't load a valid PhiNN from {repr(checkpoint_dir)}")
        return PhiNN.load(latest)

    @classmethod
    def load_checkpoint_with_id(cls, checkpoint_dir, id):
        """Load the PhiNN with a specified id from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(f"Couldn't load a valid PhiNN from {repr(checkpoint_dir)}")
        latest = latest[: latest.rfind("-")] + f"-{id}"
        return PhiNN.load(latest)


class PhiNNGuided(snt.Module):
    """
    Feed-forward neural network to represent the gravitational
    potential. This one has a trainable smooth quadratic potential added
    on top of the NN. The potential parameter is currently independent of
    the frameshift omega

    Note on checkpointing: Both PhiNN and FrameShift rely on a combination of
    tf.Checkpoint and a custom spec saving system. This is caused by
    restoration from a tf Checkpoint requiring an already initialized instance
    of the object. Initializing the object can't use data stored in the
    checkpoint, and must find its metadata elsewhere, hence from the spec file.
    """

    def __init__(
        self,
        n_dim=3,
        n_hidden=3,
        hidden_size=32,
        scale=None,
        name="PhiNNGuided",
        r_c=8.3,
    ):
        """
        Constructor for PhiNN.

        Inputs:
            n_dim (int): Dimensionality of space.
            n_hidden (int): Number of hidden layers.
            hidden_size (int): Number of neurons in each hidden layer.
            scale (float-array-like): Typical scale of coordinates along
                each dimension. This will be used to rescale the
                coordinates, before passing them into the neural network.
                Defaults to None, in which case the scales are 1.
        """
        super(PhiNNGuided, self).__init__(name=name)

        self._n_dim = n_dim
        self._n_hidden = n_hidden
        self._hidden_size = hidden_size
        self._name = name

        # The guiding variables R2 and omega**2*R2/2 are calculated w.r.t. (r_c, 0, 0)
        self._r_c = tf.Variable(r_c, trainable=False, name="r_c", dtype=tf.float32)
        self._dxy = tf.Variable(
            np.array((-r_c, 0)).astype("f4"),
            trainable=False,
            name="dxy",
            dtype=tf.float32,
        )
        self._omega = tf.Variable(
            0.0001, trainable=True, name="guiding_omega", dtype=tf.float32
        )

        # Coordinate scaling
        if scale is None:
            coord_scale = np.ones((1, n_dim), dtype="f4")
        else:
            print(f"Using coordinate scale: {scale}")
            coord_scale = np.reshape(scale, (1, n_dim)).astype("f4")
        coord_scale = np.concatenate(
            (coord_scale, np.reshape(r_c**2, (1, 1))), axis=1
        ).astype("f4")
        self._scale = tf.Variable(
            1 / coord_scale, trainable=False, name="coord_scaling"
        )

        # Variables that have "l2penalty" in them are penalized
        self._layers = [
            snt.Linear(hidden_size, name=f"hidden_{i}_l2penalty")
            for i in range(n_hidden)
        ]
        self._layers.append(snt.Linear(1, with_bias=False, name="Phi_l2penalty"))
        self._activation = tf.math.tanh

        # Initialize
        self.__call__(tf.zeros([1, n_dim]))

    def __call__(self, q):
        """Returns the gravitational potential"""
        xy, z = tf.split(q, [2, 1], axis=1)
        xy = xy + self._dxy
        R2 = tf.math.reduce_sum(xy**2, axis=1)
        R2 = tf.expand_dims(R2, 1)

        q = tf.concat([q, R2], axis=1)
        guiding_term = self._omega**2 * R2 / 2

        # Transform coordinates to standard frame
        q = self._scale * q
        # Run the coordinates through the neural net
        for layer in self._layers[:-1]:
            q = layer(q)
            q = self._activation(q)
        # No activation on the final layer
        q = self._layers[-1](q)
        return q + guiding_term

    def save_specs(self, spec_name_base):
        """Saves the specs of the model that are required for initialization to a json"""
        d = dict(
            n_dim=self._n_dim,
            n_hidden=self._n_hidden,
            hidden_size=self._hidden_size,
            name=self._name,
        )
        with open(spec_name_base + "_spec.json", "w") as f:
            json.dump(d, f)

        return spec_name_base

    @classmethod
    def load(cls, checkpoint_name):
        """Load PhiNNGuided from a checkpoint and a spec file"""
        # Get spec file name
        if (
            checkpoint_name.find("-") == -1
            or not checkpoint_name.rsplit("-", 1)[1].isdigit()
        ):
            raise ValueError(
                "PhiNNGuided checkpoint name doesn't follow the correct syntax."
            )
        spec_name = checkpoint_name.rsplit("-", 1)[0] + "_spec.json"

        # Load network specs
        with open(spec_name, "r") as f:
            kw = json.load(f)
        phi_nn = cls(**kw)

        # Restore variables
        checkpoint = tf.train.Checkpoint(phi=phi_nn)
        checkpoint.restore(checkpoint_name).expect_partial()

        print(f"loaded {phi_nn} from {checkpoint_name}")
        return phi_nn

    @classmethod
    def load_latest(cls, checkpoint_dir):
        """Load the latest PhiNNGuided from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(f"Couldn't load a valid PhiNN from {repr(checkpoint_dir)}")
        return PhiNNGuided.load(latest)

    @classmethod
    def load_checkpoint_with_id(cls, checkpoint_dir, id):
        """Load the PhiNNGuided with a specified id from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(
                f"Couldn't load a valid PhiNNGuided from {repr(checkpoint_dir)}"
            )
        latest = latest[: latest.rfind("-")] + f"-{id}"
        return PhiNNGuided.load(latest)


class FrameShift(tf.Module):
    """
    A 5-parameter model to represent the rotating frame in which f is stationary.
    The rotation axis is (0, 0, Omega) and passes through a point at position
    x_c = (r_c, 0, 0). LSR moves with speed (u_x, u_y, u_z).
    Note that there is a degeneracy between r_c, u_y and Omega

    The infinitesimal flow is given by
        x -> x + dt*u = x + dt*Omega x (x - x_c),
        v -> v + dt*w = v + dt*Omega x v.
    """

    def __init__(
        self,
        n_dim=3,
        omega0=0.0,
        omega0_trainable=True,
        r_c0=0.0,
        r_c0_trainable=False,
        u_x0=0.0,
        u_x0_trainable=True,
        u_y0=0.0,
        u_y0_trainable=True,
        u_z0=0.0,
        u_z0_trainable=True,
        name="FrameShift",
    ):
        """
        Constructor for FrameShift.

        Inputs:
            n_dim (int): Dimensionality of space.
            omega0, ..., u_z0: default values of the parameters that defined the frame shift.
        """
        super(FrameShift, self).__init__(name=name)

        self._n_dim = n_dim
        self._name = name

        # Set the variables.
        self._omega = tf.Variable(
            omega0, trainable=omega0_trainable, name="omega", dtype=tf.float32
        )
        self._r_c = tf.Variable(
            r_c0, trainable=r_c0_trainable, name="r_c", dtype=tf.float32
        )
        self._u_x = tf.Variable(
            u_x0, trainable=u_x0_trainable, name="u_x", dtype=tf.float32
        )
        self._u_y = tf.Variable(
            u_y0, trainable=u_y0_trainable, name="u_y", dtype=tf.float32
        )
        self._u_z = tf.Variable(
            u_z0, trainable=u_z0_trainable, name="u_z", dtype=tf.float32
        )

    def __call__(self, q, p):
        """Returns u and w, the flows defined by the frame shift"""
        n = q.shape[0]

        qx, qy, _ = tf.unstack(q, axis=1)
        px, py, _ = tf.unstack(p, axis=1)
        # Add rotation and LSR vel to u (while shifting q by -r_c)
        ux = tf.add(tf.multiply(qy, -self._omega), self._u_x)
        uy = tf.add(tf.multiply(tf.subtract(qx, self._r_c), self._omega), self._u_y)
        uz = tf.repeat(self._u_z, n)
        u = tf.stack((ux, uy, uz), axis=1)

        # Add rotation to w
        w = tf.stack(
            (
                tf.multiply(tf.subtract(py, self._u_y), -self._omega),
                tf.multiply(tf.subtract(px, self._u_x), self._omega),
                tf.zeros(n),
            ),
            axis=1,
        )

        return u, w

    def save_specs(self, spec_name_base):
        """Saves the specs of the model that are required for initialization to a json"""
        d = dict(n_dim=self._n_dim, name=self._name)
        with open(spec_name_base + "_fspec.json", "w") as f:
            json.dump(d, f)

        return spec_name_base

    @classmethod
    def load(cls, checkpoint_name, verbose=True):
        """Load FrameShift from a checkpoint and a spec file"""
        # Get spec file name
        if (
            checkpoint_name.find("-") == -1
            or not checkpoint_name.rsplit("-", 1)[1].isdigit()
        ):
            raise ValueError(
                "FrameShift checkpoint name doesn't follow the correct syntax."
            )
        spec_name = checkpoint_name.rsplit("-", 1)[0] + "_fspec.json"

        # Load network specs
        with open(spec_name, "r") as f:
            kw = json.load(f)
        fs = cls(**kw)

        # Restore variables
        checkpoint = tf.train.Checkpoint(frameshift=fs)
        checkpoint.restore(checkpoint_name).expect_partial()

        if verbose:
            print(f"loaded {kw} from {checkpoint_name}")
        return fs

    @classmethod
    def load_latest(cls, checkpoint_dir, verbose=True):
        """Load the latest FrameShift from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(
                f"Couldn't load a valid FrameShift from {repr(checkpoint_dir)}"
            )
        return FrameShift.load(latest, verbose)

    @classmethod
    def load_checkpoint_with_id(cls, checkpoint_dir, id, verbose=True):
        """Load the FrameShift with a specified id from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(
                f"Couldn't load a valid FrameShift from {repr(checkpoint_dir)}"
            )
        latest = latest[: latest.rfind("-")] + f"-{id}"
        return FrameShift.load(latest, verbose)

    def debug(self):
        print(
            f"name={self.name}\n\
  u_=({self._u_x.numpy():.8f}, {self._u_y.numpy():.8f}, {self._u_z.numpy():.8f})\n\
  omega={self._omega.numpy():.8f}, r_c={self._r_c.numpy():.8f}"
        )


def train_potential(
    df_data,
    phi_model,
    frameshift_model=None,
    optimizer=None,
    n_epochs=4096,
    batch_size=1024,
    lr_type="step",
    lr_init=2.0e-2,
    lr_final=None,
    lr_patience=32,
    lr_min_delta=0.01,
    warmup_proportion=0.1,
    validation_frac=0.25,
    checkpoint_every=None,
    checkpoint_hours=None,
    max_checkpoints=None,
    checkpoint_dir=r"checkpoints/Phi",
    checkpoint_name="Phi",
    xi=1.0,
    lam=1.0,
    mu=0,
    l2=0,
):
    """
    Fits a gravitational potential and a optionally a frameshift based on the
    given data. Potential is fit to satisfy CBE and frameshift represents the
    frame at which stationarity is best enforced in.

    Let
        n := # of points at which to evaluate loss terms,
        d := # of spatial dimensions.

    Inputs:
        df_data (dict of np.Array): The data to train the models on.
            'eta' contains 2*d columns, 0,...,d-1 corresponding to q and
            d,...,2*d to p. 'df_deta' contains 2*d columns, 0,...,d-1 are
            df/dq and d,...2*d are df/dp. The data is in Cartesian coordinates
            with q = x and p = v (mass = 1)
        phi_model (callable): The gravitational potential. Takes q, a
            (n,d) tensor, and returns a (n,) tensor.
        frameshift_model (callable): an object for indicating which
            frame stationarity is enforced in. Takes q, p, both (n,d)
            tensors, and returns two (n,d) tensors, corresponding
            to u and w defined by the frameshift. If set to None,
            stationarity is enforced in lab frame.
        Batch settings: Self-explanatory, the data is split into a training/
            validation set given by validation_frac.
        Optimizer settings: Self-explanatory, The optimizer
            currently uses a step-wise decreasing learning rate which halves
            every time the change in loss goes below lr_min_delta (with some
            inertia given by lr_patience), and stops when lr goes below lr_final.
            There is also a warm-up period for the lr given by warmup_proportion.
        Checkpointing settings: Self-explanatory, the checkpoints are saved to
            checkpoint_dir with a filename base given by checkpoint name.
            TODO: CheckpointManager currently doesn't clean all the auxilliary
            files when checkpoint number exceeds max_checkpoints.
        Loss settings: These influence how loss is calculated.
            xi: Scale above which outliers are suppressed,
            lam: Penalty for negative matter densities,
            mu: Penalty for positive matter densities,
            l2: L2 penalty on weights in the models.

    Outputs:
        loss_history (list of floats): Records losses at every training step.
    """

    print("df_data type:", type(df_data))
    print(type(optimizer))
    # Split training/validation sample
    n_samples = df_data["eta"].shape[0]
    n_dim = df_data["eta"].shape[1] // 2
    data = np.stack(
        [
            df_data["eta"][:, :n_dim].astype("f4"),  # q
            df_data["eta"][:, n_dim:].astype("f4"),  # p
            df_data["df_deta"][:, :n_dim].astype("f4"),  # df/dq
            df_data["df_deta"][:, n_dim:].astype("f4"),  # df/dp
        ],
        axis=1,
    )
    n_val = int(validation_frac * n_samples)
    val_batch_size = int(validation_frac * batch_size)
    n_samples -= n_val
    val = data[:n_val]
    data = data[n_val:]

    # Create Tensorflow datasets
    batches = tf.data.Dataset.from_tensor_slices(data)
    batches = batches.shuffle(n_samples, reshuffle_each_iteration=True)
    batches = batches.repeat(n_epochs + 1)
    batches = batches.batch(batch_size, drop_remainder=True)

    val_batches = tf.data.Dataset.from_tensor_slices(val)
    val_batches = val_batches.shuffle(n_val, reshuffle_each_iteration=True)
    val_batches = val_batches.repeat(n_epochs + 1)
    val_batches = val_batches.batch(val_batch_size, drop_remainder=True)

    phi_param = phi_model.trainable_variables  # Returns a tuple
    n_variables_phi = sum([int(tf.size(param)) for param in phi_param])
    print(f"{n_variables_phi} variables in the gravitational potential model.")

    frameshift_param = ()
    if frameshift_model is not None:
        frameshift_param = frameshift_model.trainable_variables
        n_variables_frameshift = sum(
            [int(tf.size(param)) for param in frameshift_param]
        )
        print(f"{n_variables_frameshift} variables in the frameshift model.")

    # Estimate typical scale of flows (with constant gravitational potential)
    # delf_delt_scale = np.percentile(
    #    np.abs(np.sum(
    #        df_data['eta'][:,n_dim:] * df_data['df_deta'][:,:n_dim],
    #        axis=1
    #    )),
    #    50.
    # )
    # print(f'Using del(f)/del(t) ~ {delf_delt_scale}')

    # Optimizer
    n_steps = n_epochs * n_samples // batch_size
    print(f"{n_steps} steps planned.")

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

    print(f"Optimizer: {opt}")

    loss_history = []
    loss_noreg_history = []
    val_loss_history = []
    val_loss_noreg_history = []
    lr_history = []
    potential_params_hist = {}
    params_fname = f"{os.path.join(checkpoint_dir, checkpoint_name)}_params.csv"

    # Set up checkpointing
    step = tf.Variable(0, name="step")
    loss_min = tf.Variable(np.inf, name="loss_min")

    if checkpoint_every is not None:
        if frameshift_model is None:
            checkpoint = tf.train.Checkpoint(
                opt=opt, phi=phi_model, step=step, loss_min=loss_min
            )
        else:
            checkpoint = tf.train.Checkpoint(
                opt=opt,
                phi=phi_model,
                frameshift=frameshift_model,
                step=step,
                loss_min=loss_min,
            )
        print("maxx", max_checkpoints)
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

            # Try to load loss history
            loss_fname = f"{latest}_loss.txt"
            (
                loss_history,
                val_loss_history,
                lr_history,
                loss_noreg_history,
                val_loss_noreg_history,
            ) = utils.load_loss_history(loss_fname)
            potential_params_hist = utils.load_potential_params(params_fname, remove_lr=True)

        # Convert from # of epochs to # of steps between checkpoints
        checkpoint_steps = math.ceil(checkpoint_every * n_samples / batch_size)
        # print(n_samples, batch_size, n_epochs)
        # print(checkpoint_steps, (checkpoint_every * n_samples) // batch_size, n_epochs, n_steps)

    @tf.function
    def training_step(batch):
        print(f"Tracing training_step with batch shape {batch.shape} ...")

        # Unpack the data from the batch
        q_b, p_b, df_dq_b, df_dp_b = [tf.squeeze(x) for x in tf.split(batch, 4, axis=1)]

        # Calculate the loss and its gradients w.r.t. the parameters
        loss, loss_noreg, dloss_dparam = get_phi_loss_gradients(
            phi_model,
            frameshift_model,
            phi_param + frameshift_param,
            q_b,
            p_b,
            df_dq=df_dq_b,
            df_dp=df_dp_b,
            xi=xi,
            delf_delt_scale=1,  # delf_delt_scale,
            lam=lam,
            mu=mu,
            l2=l2,
            return_loss_noreg=True,
        )

        dloss_dparam, global_norm = tf.clip_by_global_norm(dloss_dparam, 1.0)
        tf.print("\nglobal norm:", global_norm)

        # Take step using optimizer
        opt.apply_gradients(zip(dloss_dparam, phi_param + frameshift_param))

        return loss, loss_noreg

    @tf.function
    def validation_step(batch):
        print(f"Tracing validation step with batch shape {batch.shape} ...")

        # Unpack the data from the batch
        q_b, p_b, df_dq_b, df_dp_b = [tf.squeeze(x) for x in tf.split(batch, 4, axis=1)]

        # Calculate the loss and its gradients w.r.t. the parameters
        loss, loss_noreg = get_phi_loss_gradients(
            phi_model,
            frameshift_model,
            phi_param + frameshift_param,
            q_b,
            p_b,
            df_dq=df_dq_b,
            df_dp=df_dp_b,
            xi=xi,
            delf_delt_scale=1,  # delf_delt_scale,
            lam=lam,
            mu=mu,
            l2=l2,
            return_grads=False,
            return_loss_noreg=True,
        )

        return loss, loss_noreg

    # Set up checkpointing
    step = tf.Variable(0, name="step")
    checkpoint_prefix = os.path.join(checkpoint_dir, checkpoint_name)
    if checkpoint_every is not None:
        if frameshift_model is None:
            checkpoint = tf.train.Checkpoint(opt=opt, phi=phi_model, step=step)
        else:
            checkpoint = tf.train.Checkpoint(
                opt=opt, phi=phi_model, frameshift=frameshift_model, step=step
            )

        # Look for latest extisting checkpoint
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is not None:
            print(f"Restoring from checkpoint {latest} ...")
            checkpoint.restore(latest)
            print(f"Beginning from step {int(step)}.")

    # Keep track of whether this is the first step.
    # Were it not for checkpointing, we could use i == 0.
    traced = False

    update_bar = utils.get_training_progressbar_fn(n_steps, loss_history, opt)
    t0 = time()

    # Main training loop
    for i, (y, y_val) in enumerate(zip(batches, val_batches), int(step)):
        if i >= n_steps:
            # Break if too many steps taken. This can occur
            # if we began from a checkpoint.
            break

        # Take one step
        loss, loss_noreg = training_step(y)
        val_loss, val_loss_noreg = validation_step(y_val)

        # Logging
        loss_history.append(float(loss))
        loss_noreg_history.append(float(loss_noreg))
        val_loss_history.append(float(val_loss))
        val_loss_noreg_history.append(float(val_loss_noreg))
        lr_history.append(float(opt._decayed_lr(tf.float32)))
        potential_params_hist = utils.append_to_potential_params_history(
            phi_model, frameshift_model, potential_params_hist
        )
        update_bar(i)

        # Adjust learning rate?
        if lr_type == "step":
            n_smooth = max(lr_patience // 8, 1)
            if len(loss_history) >= n_smooth:
                loss_avg = np.mean(loss_history[-n_smooth:])
            else:
                loss_avg = np.inf

            if loss_avg < loss_min - lr_min_delta:
                steps_since_decline = 0
                print(f"New minimum loss: {loss_avg}.")
                loss_min.assign(loss_avg)
            elif steps_since_decline >= lr_patience:
                # Reduce learning rate
                old_lr = float(opt.lr)
                new_lr = 0.5 * old_lr
                print(f"Reducing learning rate from {old_lr} to {new_lr}.")
                print(f"   (loss threshold: {float(loss_min-lr_min_delta)})")
                opt.lr.assign(new_lr)
                steps_since_decline = 0

                if lr_final is not None and new_lr <= lr_final:
                    print(
                        "Learning rate has been reduced below the threshold. Terminating ..."
                    )
                    chkpt_fname = chkpt_manager.save()
                    print(f"  --> {chkpt_fname}")
                    break
            else:
                steps_since_decline += 1

        if not traced:
            # Get time after gradients function is first traced
            traced = True
            t1 = time()
        else:
            t1 = None

        # Checkpoint
        if (checkpoint_every is not None) and i and not (i % checkpoint_steps):
            print("Checkpointing ...")
            step.assign(i + 1)
            chkpt_fname = chkpt_manager.save()
            print(f"  --> {chkpt_fname}")
            utils.save_loss_history(
                f"{chkpt_fname}_loss.txt",
                loss_history,
                val_loss_history=val_loss_history,
                lr_history=lr_history,
                loss_noreg_history=loss_noreg_history,
                val_loss_noreg_history=val_loss_noreg_history,
            )
            print(os.path.join(checkpoint_dir, checkpoint_name))
            utils.save_potential_params_history(
                params_fname, potential_params_hist, lr_history
            )

            fig = utils.plot_loss(
                loss_history, val_loss_hist=val_loss_history, lr_hist=lr_history
            )
            fig.savefig(f"{chkpt_fname}_loss.pdf")
            plt.close(fig)

            fig = utils.plot_loss(
                loss_noreg_history,
                val_loss_hist=val_loss_noreg_history,
                lr_hist=lr_history,
            )
            fig.savefig(f"{chkpt_fname}_loss_noreg.pdf")
            plt.close(fig)

    t2 = time()
    loss_avg = np.mean(loss_history[-50:])
    n_steps = len(loss_history)
    print(f"<loss> = {loss_avg: >7.5f}")
    if t1 is not None:
        print(f"tracing time: {t1-t0:.2f} s")
        print(f"training time: {t2-t1:.1f} s ({(t2-t1)/(n_steps-1):.4f} s/step)")

    return loss_history


def main():
    x = tf.random.normal([7, 3])
    phi_nn = PhiNN(hidden_size=128)
    y0 = phi_nn(x)
    fname = phi_nn.save("models/Phi")

    phi_nn_1 = PhiNN.load(fname)
    y1 = phi_nn_1(x)

    print(y0)
    print(y1)

    # print(phi_nn.trainable_variables)
    # print(phi_nn_1.trainable_variables)

    return 0


if __name__ == "__main__":
    main()
