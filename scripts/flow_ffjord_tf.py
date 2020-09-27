#!/usr/bin/env python

from __future__ import print_function, division

# numpy and matplotlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

# Custom libraries
from utils import *


class ForceFieldModel(snt.Module):
    def __init__(self, n_dim, n_hidden, hidden_size, name='dz_dt'):
        super(ForceFieldModel, self).__init__(name=name)

        output_sizes = [hidden_size] * n_hidden + [n_dim]
        self._nn = snt.nets.MLP(
            output_sizes,
            activation=tf.math.tanh,
            name='mlp'
        )

    def __call__(self, t, x):
        # Concatenate time and position vectors
        tx = tf.concat([tf.broadcast_to(t, x.shape), x], -1)
        # Return dz_dt(t,x)
        return self._nn(tx)


def create_flow(n_dim, n_hidden, hidden_size, exact=False):
    dz_dt = ForceFieldModel(n_dim, n_hidden, hidden_size)
    ode_solver = tfp.math.ode.DormandPrince(atol=1.e-5)

    if exact:
        trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact
    else:
        trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson

    bij = tfb.FFJORD(
        state_time_derivative_fn=dz_dt,
        ode_solve_fn=ode_solver.solve,
        trace_augmentation_fn=trace_augmentation_fn
    )

    base_dist = tfd.MultivariateNormalDiag(
        loc=np.zeros(n_dim, dtype='f4')
    )

    dist = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=bij
    )

    # Initialize flow
    dist.sample([1])

    n_var = sum([int(tf.size(v)) for v in dist.trainable_variables])
    print(f'# of trainable variables: {n_var}')

    return dist


def train_flow(flow, data,
               optimizer=None,
               batch_size=32,
               n_epochs=1,
               checkpoint_every=128,
               checkpoint_dir=r'checkpoints/ffjord',
               checkpoint_name='ffjord'):
    """
    Trains a flow using the given data.

    Inputs:
      flow (NormalizingFlow): Normalizing flow to be trained.
      data (tf.Tensor): Observed points. Shape = (# of points, # of dim).
      optimizer (tf.keras.optimizers.Optimizer): Optimizer to use.
          Defaults to the Rectified Adam implementation from
          tensorflow_addons.
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

    n_samples = data.shape[0]
    batches = tf.data.Dataset.from_tensor_slices(data)
    batches = batches.repeat(n_epochs)
    batches = batches.shuffle(n_samples, reshuffle_each_iteration=True)
    batches = batches.batch(batch_size, drop_remainder=True)

    n_steps = n_epochs * n_samples // batch_size

    if optimizer is None:
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            2.e-2,
            n_steps,
            0.05,
            staircase=False
        )
        opt = tfa.optimizers.RectifiedAdam(
            lr_schedule,
            total_steps=n_steps,
            warmup_proportion=0.1
        )
    else:
        opt = optimizer

    loss_history = []
    update_bar = get_training_progressbar_fn(n_steps, loss_history, opt)

    t0 = time()

    # Set up checkpointing
    step = tf.Variable(0, name='step')
    checkpoint_prefix = os.path.join(checkpoint_dir, checkpoint_name)
    if checkpoint_every is not None:
        checkpoint = tf.train.Checkpoint(opt=opt, flow=flow, step=step)

        # Look for latest extisting checkpoint
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is not None:
            print(f'Restoring from checkpoint {latest} ...')
            checkpoint.restore(latest)
            print(f'Beginning from step {int(step)}.')

    # Keep track of whether this is the first step.
    # Were it not for checkpointing, we could use i == 0.
    traced = False

    @tf.function
    def training_step(batch):
        print(f'Tracing training_step with batch shape {batch.shape} ...')
        variables = flow.trainable_variables
        with tf.GradientTape() as g:
            g.watch(variables)
            loss = -tf.reduce_mean(flow.log_prob(batch))
        grads = g.gradient(loss, variables)
        opt.apply_gradients(zip(grads, variables))
        return loss

    for i,y in enumerate(batches, int(step)):
        if i >= n_steps:
            # Break if too many steps taken. This can occur
            # if we began from a checkpoint.
            break

        loss = training_step(y)

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

    t2 = time()
    loss_avg = np.mean(loss_history[-50:])
    n_steps = len(loss_history)
    print(f'<loss> = {loss_avg: >7.5f}')
    print(f'tracing time: {t1-t0:.2f} s')
    print(f'training time: {t2-t1:.1f} s ({(t2-t1)/(n_steps-1):.4f} s/step)')

    # Save the trained model
    checkpoint = tf.train.Checkpoint(flow=flow)
    checkpoint.save(checkpoint_prefix + '_final')
    
    return loss_history


def load_flow_params(flow, checkpoint_fname):
    checkpoint = tf.train.Checkpoint(flow=flow)
    checkpoint.restore(checkpoint_fname)


def calc_f_gradients(f, eta):
    with tf.GradientTape() as g:
        g.watch(eta)
        f_eta = f(eta)
    df_deta = g.gradient(f_eta, eta)
    return df_deta


def main():
    flow = create_flow(2, 4, 16)

    n_samples = 128*1024
    mu = [[-2., 0.], [2., 0.]]
    cov = [
        [[1., 0.],
         [0., 1.]],
        [[1., 0.],
         [0., 1.]]
    ]
    data = tf.concat([
        np.random.multivariate_normal(m, c, n_samples//2).astype('f4')
        for m,c in zip(mu, cov)
    ], axis=0)

    train_flow(
        flow, data,
        batch_size=1024,
        n_epochs=16,
        checkpoint_every=256
    )
    load_flow_params(flow, 'checkpoints/ffjord/ffjord_final-1')

    #for i in range(10):
    #    eta = tf.random.normal([1024,2])
    #    df_deta = calc_flow_gradients(flow, eta)
    #    print(i)

    return 0

if __name__ == '__main__':
    main()

