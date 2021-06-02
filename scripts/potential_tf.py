#!/usr/bin/env python

from __future__ import print_function, division

# Tensorflow & co
import tensorflow as tf
print(f'Tensorflow version {tf.__version__}')
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_probability as tfp
print(f'Tensorflow Probability version {tfp.__version__}')
tfb = tfp.bijectors
tfd = tfp.distributions
import sonnet as snt

import numpy as np
import matplotlib.pyplot as plt

import json
import os
from time import time

# Custom libraries
from utils import *


def calc_df_deta(f_func, q, p):
    #print('Tracing calc_df_deta ...')

    # Calculate gradients of distribution function
    with tf.GradientTape(persistent=True) as g:
        g.watch([q, p])
        f = f_func(q, p)

    df_dq = g.gradient(f, q, unconnected_gradients='zero')
    df_dp = g.gradient(f, p, unconnected_gradients='zero')

    return f, df_dq, df_dp


def calc_phi_derivatives(phi_func, q, return_phi=False):
    # Calculate derivatives of the potential.
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
    for di,qi in zip(dphi_dq_unstacked, q_unstacked):
        d2phi_dq2 += gg.gradient(di, qi)

    if return_phi:
        return phi, dphi_dq, d2phi_dq2
    
    return dphi_dq, d2phi_dq2


def calc_loss_terms(phi_func, q, p, df_dq, df_dp):
    """
    Calculates both {H,f} and the Laplacian of the potential at
    a set of points in phase space. The Hamiltonian is assumed
    to be of the form,

        H = p^2 / 2 + phi(q) .

    Let
        n := # of points at which to evaluate loss terms,
        k := # of spatial dimensions.

    Inputs:
      phi_func (function): Potential. phi_func(q) -> shape-(n,) tensor.
      q (tf.Tensor): Positions at which to evaluate terms. Shape = (n,k).
      p (tf.Tensor): Momenta at which to evaluate terms. Shape = (n,k).
      f_func (Optional[function]): Distribution function. f_func(q,p) -> shape-(n,)
          tensor.

    Outputs:
      df_dt (tf.Tensor): {H,f} evaluated at the points (q,p).
          Shape = (n,).
      d2phi_dq2 (tf.Tensor): Laplacian of the potential, evaluated
          at the points (q,p). Shape = (n,).
    """

    # Calculate necessary derivatives
    dphi_dq, d2phi_dq2 = calc_phi_derivatives(phi_func, q)

    # partial f / partial t = {H,f}
    df_dt = tf.reduce_sum(df_dp * dphi_dq - df_dq * p, axis=1)

    return df_dt, d2phi_dq2


def get_phi_loss_gradients(phi, params, q, p,
                           f=None, df_dq=None, df_dp=None,
                           lam=1., mu=0,
                           xi=1., delf_delt_scale=1.,
                           sigma_q=tf.constant(1.0),
                           sigma_p=tf.constant(1.0),
                           eps_w=tf.constant(0.1),
                           weight_samples=False):
    """
    Calculates both the loss and the gradients of the loss w.r.t. the
    given parameters.

    In the following, let n be the number of points and d be the
    number of spatial parameters.

    Inputs:
        f (callable): The distribution function. Takes q and p, each
            (n,d) tensors, and returns a (n,) tensor.
        phi (callable): The gravitational potential. Takes q, a
            (n,d) tensor, and returns a (n,) tensor.
        params (list of tf.Variable): The gradients will be taken
            w.r.t these parameters.
        q (tf.Tensor): Spatial coordinates at which to compute the
            loss and gradients.
        p (tf.Tensor): Momenta at which to compute the loss and
            gradients.
        lam (scalar): Constant that determines how strongly to
            penalize negative matter densities. Larger values
            translate to stronger penalties. Defaults to 1.
    
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
        raise ValueError(
            'If f is not provided, then df_dq and df_dp must be provided.'
        )

    c = xi / delf_delt_scale
    print(f'c = {c}')

    with tf.GradientTape() as g:
        g.watch(params)

        # Calculate derivatives of phi w.r.t. q
        dphi_dq, d2phi_dq2 = calc_phi_derivatives(phi, q)

        # partial f / partial t = {H,f}
        df_dt = tf.reduce_sum(df_dp * dphi_dq - df_dq * p, axis=1)

        # Weight each point differently
        if weight_samples:
            print('Re-weighting samples.')
            w = tf.stop_gradient(
                tf.reduce_sum(
                    dphi_dq**2 / sigma_p**2 + p**2 / sigma_q**2,
                    axis=1
                )
            )
            w = 1 / (w + eps_w)
            #w = w / tf.reduce_mean(w)
            df_dt = w * df_dt

        # Average over sampled points in phase space
        likelihood = tf.math.asinh(c * tf.math.abs(df_dt)) / c

        if lam != 0:
            prior_neg = tf.math.asinh(
                tf.clip_by_value(-d2phi_dq2, 0., np.inf)
            )
            print(f'prior_neg = {prior_neg}')
            #pneg_mean = tf.math.reduce_mean(prior_neg)
            #pneg_max = tf.math.reduce_max(prior_neg)
            #L_mean = tf.math.reduce_mean(likelihood)
            #r = tf.norm(q, axis=1)
            #r_max = tf.math.reduce_max(r)
            #tf.print('     lambda =', lam)
            #tf.print('        <L> =', L_mean)
            #tf.print('  <penalty> =', pneg_mean)
            #tf.print('penalty_max =', pneg_max)
            #tf.print('      r_max =', r_max)
            #tf.print('')
            likelihood = likelihood + lam * prior_neg

        if mu != 0:
            prior_pos = tf.math.asinh(
                tf.clip_by_value(d2phi_dq2, 0., np.inf)
            )
            print(f'prior_pos = {prior_pos}')
            likelihood = likelihood + mu * prior_pos

        # Regularization penalty
        # penalty = 0.
        # for p in params:
        #     penalty += tf.reduce_sum(p**2)

        loss = tf.reduce_mean(likelihood)
        #    likelihood
        #    + lam*prior_neg
        #    + mu*prior_pos
        #    # + reg*penalty
        #)

    # Gradients of loss w.r.t. NN parameters
    dloss_dparam = g.gradient(loss, params)

    return loss, dloss_dparam


class PhiNN(snt.Module):
    """
    Feed-forward neural network to represent the gravitational
    potential.
    """

    def __init__(self, n_dim=3, n_hidden=3, hidden_size=32, name='Phi'):
        """
        Constructor for PhiNN.

        Inputs:
            n_dim (int): Dimensionality of space.
            n_hidden (int): Number of hidden layers.
            n_features (int): Number of neurons in each hidden layer.
            build (bool): Whether to create the weights and biases.
                Defaults to True. This option exists so that the
                deserializer can set the weights and biases on its
                own.
        """
        super(PhiNN, self).__init__(name=name)

        self._n_dim = n_dim
        self._n_hidden = n_hidden
        self._hidden_size = hidden_size
        self._name = name

        self._layers = [
            snt.Linear(hidden_size, name=f'hidden_{i}')
            for i in range(n_hidden)
        ]
        self._layers.append(snt.Linear(1, with_bias=False, name='Phi'))
        #self._activation = tf.math.sigmoid
        self._activation = tf.math.tanh

        # Initialize
        self.__call__(tf.zeros([1,n_dim]))

    def __call__(self, x):
        for layer in self._layers[:-1]:
            x = layer(x)
            x = self._activation(x)
        x = self._layers[-1](x)
        return x
    
    def save(self, fname_base):
        # Save variables
        checkpoint = tf.train.Checkpoint(phi=self)
        fname_out = checkpoint.save(fname_base)

        # Save the specs of the neural network
        d = dict(
            n_dim=self._n_dim,
            n_hidden=self._n_hidden,
            hidden_size=self._hidden_size,
            name=self._name
        )
        with open(fname_out+'_spec.json', 'w') as f:
            json.dump(d, f)

        return fname_out

    @classmethod
    def load(cls, fname):
        # Load network specs
        with open(fname+'_spec.json', 'r') as f:
            kw = json.load(f)
        phi_nn = cls(**kw)

        # Restore variables
        checkpoint = tf.train.Checkpoint(phi=phi_nn)
        #latest = tf.train.latest_checkpoint(fname_base)
        #print(f'Restoring from {latest}')
        checkpoint.restore(fname)

        return phi_nn


def train_potential(
            df_data, phi_model,
            n_epochs=4096,
            batch_size=1024,
            checkpoint_every=None,
            checkpoint_dir=r'checkpoints/Phi',
            checkpoint_name='Phi',
            xi=1.,   # Scale above which outliers are suppressed
            lam=1.,  # Penalty for negative matter densities
            mu=0,    # Penalty for positive matter densities
            lr_init=1.e-3,
            lr_final=1.e-6
        ):
    n_samples = df_data['eta'].shape[0]
    n_dim = df_data['eta'].shape[1] // 2
    data = tf.stack(
        [
            df_data['eta'][:,:n_dim],     # q
            df_data['eta'][:,n_dim:],     # p
            df_data['df_deta'][:,:n_dim], # df/dq
            df_data['df_deta'][:,n_dim:]  # df/dp
        ],
        axis=1
    )
    data = tf.data.Dataset.from_tensor_slices(data)

    phi_param = phi_model.trainable_variables
    n_variables = sum([int(tf.size(param)) for param in phi_param])
    print(f'{n_variables} variables in the gravitational potential model.')

    # Estimate typical scale of flows (with constant gravitational potential)
    delf_delt_scale = np.percentile(
        np.abs(np.sum(
            df_data['eta'][:,n_dim:] * df_data['df_deta'][:,:n_dim],
            axis=1
        )),
        50.
    )
    print(f'Using del(f)/del(t) ~ {delf_delt_scale}')

    # Optimizer
    n_steps = n_epochs * (n_samples // batch_size)
    print(f'{n_steps} steps planned.')
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        lr_init,
        n_steps,
        lr_final/lr_init,
        staircase=False
    )
    opt = tfa.optimizers.RectifiedAdam(
        lr_schedule,
        total_steps=n_steps,
        warmup_proportion=0.1
    )

    # Set up batches of data
    batches = data.shuffle(n_samples, reshuffle_each_iteration=True)
    batches = batches.repeat(n_epochs)
    batches = batches.batch(batch_size, drop_remainder=True)

    @tf.function
    def training_step(batch):
        print(f'Tracing training_step with batch shape {batch.shape} ...')

        # Unpack the data from the batch
        q_b, p_b, df_dq_b, df_dp_b = [
            tf.squeeze(x) for x in tf.split(batch, 4, axis=1)
        ]

        # Calculate the loss and its gradients w.r.t. the parameters
        loss, dloss_dparam = get_phi_loss_gradients(
            phi_model, phi_param,
            q_b, p_b,
            df_dq=df_dq_b,
            df_dp=df_dp_b,
            xi=xi,
            delf_delt_scale=delf_delt_scale,
            lam=lam,
            mu=mu,
            weight_samples=False
        )

        #dloss_dparam,global_norm = tf.clip_by_global_norm(dloss_dparam, 1.)
        #tf.print('\nglobal norm:', global_norm)

        # Take step using optimizer
        opt.apply_gradients(zip(dloss_dparam, phi_param))

        return loss

    # Set up checkpointing
    step = tf.Variable(0, name='step')
    checkpoint_prefix = os.path.join(checkpoint_dir, checkpoint_name)
    if checkpoint_every is not None:
        checkpoint = tf.train.Checkpoint(opt=opt, phi=phi_model, step=step)

        # Look for latest extisting checkpoint
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is not None:
            print(f'Restoring from checkpoint {latest} ...')
            checkpoint.restore(latest)
            print(f'Beginning from step {int(step)}.')

    # Keep track of whether this is the first step.
    # Were it not for checkpointing, we could use i == 0.
    traced = False

    loss_history = []
    update_bar = get_training_progressbar_fn(n_steps, loss_history, opt)
    t0 = time()

    for i,b in enumerate(batches, int(step)):
        if i >= n_steps:
            # Break if too many steps taken. This can occur
            # if we began from a checkpoint.
            break

        # Take one step
        loss = training_step(b)

        # Logging
        loss_history.append(float(loss))
        update_bar(i)

        if not traced:
            # Get time after gradients function is first traced
            traced = True
            t1 = time()

        # Checkpoint
        if (checkpoint_every is not None) and i and not (i % checkpoint_every):
            step.assign(i+1)
            checkpoint.save(checkpoint_prefix)

    t1 = time()
    print(f'Elapsed time: {t1-t0:.1f} s')

    return loss_history


def main():
    x = tf.random.normal([7,3])
    phi_nn = PhiNN(hidden_size=128)
    y0 = phi_nn(x)
    fname = phi_nn.save('models/Phi')

    phi_nn_1 = PhiNN.load(fname)
    y1 = phi_nn_1(x)

    print(y0)
    print(y1)

    #print(phi_nn.trainable_variables)
    #print(phi_nn_1.trainable_variables)

    return 0

if __name__ == '__main__':
    main()

