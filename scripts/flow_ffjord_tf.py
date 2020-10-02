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
import json

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


class FFJORDFlow(tfd.TransformedDistribution):
    def __init__(self, n_dim, n_hidden, hidden_size,
                 exact=True, atol=1.e-5, name='DF'):
        self._n_dim = n_dim
        self._n_hidden = n_hidden
        self._hidden_size = hidden_size
        self._name = name

        # Force field guiding transformation
        dz_dt = ForceFieldModel(n_dim, n_hidden, hidden_size)

        ode_solver = tfp.math.ode.DormandPrince(atol=atol)

        if exact:
            trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact
        else:
            # Stochastic estimate of the Jacobian. Better scaling with
            # number of dimensions, but noisy.
            trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson

        # Initialize bijector
        bij = tfb.FFJORD(
            state_time_derivative_fn=dz_dt,
            ode_solve_fn=ode_solver.solve,
            trace_augmentation_fn=trace_augmentation_fn
        )

        # Multivariate normal base distribution
        base_dist = tfd.MultivariateNormalDiag(
            loc=np.zeros(n_dim, dtype='f4')
        )

        # Initialize FFJORD
        super(FFJORDFlow, self).__init__(
            distribution=base_dist,
            bijector=bij,
            name=name
        )

        # Initialize flow by taking a sample
        self.sample([1])

        self.n_var = sum([int(tf.size(v)) for v in self.trainable_variables])
        print(f'# of trainable variables: {self.n_var}')
    
    def save(self, fname_base):
        # Save variables
        checkpoint = tf.train.Checkpoint(flow=self)
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
    def load(cls, fname, **kwargs):
        # Load network specs
        with open(fname+'_spec.json', 'r') as f:
            kw = json.load(f)
        kw.update(kwargs)
        flow = cls(**kw)

        # Restore variables
        checkpoint = tf.train.Checkpoint(flow=flow)
        checkpoint.restore(fname)

        return flow


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
            0.005,
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
        #tf.print([(v.name,tf.norm(v)) for v in grads])
        grads,global_norm = tf.clip_by_global_norm(grads, 1.)
        #tf.print('\n',global_norm)
        #tf.print([(v.name,tf.norm(v)) for v in grads])
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
    #checkpoint = tf.train.Checkpoint(flow=flow)
    #checkpoint.save(checkpoint_prefix + '_final')
    
    return loss_history


def save_flow(flow, fname_base):
    checkpoint = tf.train.Checkpoint(flow=flow)
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


def calc_f_gradients(f, eta):
    with tf.GradientTape() as g:
        g.watch(eta)
        f_eta = f(eta)
    df_deta = g.gradient(f_eta, eta)
    return df_deta


def main():
    #flow = create_flow(2, 4, 16)
    flow = FFJORDFlow(2, 4, 16)

    n_samples = 32*1024
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
        n_epochs=1,
        checkpoint_every=256
    )

    fname = flow.save('checkpoints/ffjord/ffjord_test')
    flow2 = FFJORDFlow.load(fname)

    x = tf.random.normal([5,2])
    y = flow.log_prob(x)
    y2 = flow2.log_prob(x)
    print(y)
    print(y2)

    #for i in range(10):
    #    eta = tf.random.normal([1024,2])
    #    df_deta = calc_flow_gradients(flow, eta)
    #    print(i)

    return 0

if __name__ == '__main__':
    main()

