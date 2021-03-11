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


from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.internal import prefer_static

def trace_jacobian_exact_reg(ode_fn, state_shape, dtype,
                             kinetic_reg=None, jacobian_reg=None,
                             dv_dt_reg=None):
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
        ode_fn_with_time = lambda x: ode_fn(time, x)
        batch_shape = [prefer_static.size0(state)]

        if dv_dt_reg is not None:
            watched_vars = [time, state]
        elif (kinetic_reg is not None) or (jacobian_reg is not None):
            watched_vars = [state]
        else:
            watched_vars = []

        with tf.GradientTape(watch_accessed_variables=False,
                             persistent=True) as g:
          #g.watch([time, state])
          g.watch(watched_vars)
          state_time_derivative, diag_jac = tfp_math.diag_jacobian(
              xs=state, fn=ode_fn_with_time, sample_shape=batch_shape)
          # tfp_math.diag_jacobian returns lists
          if isinstance(state_time_derivative, list):
              state_time_derivative = state_time_derivative[0]
          if isinstance(diag_jac, list):
              diag_jac = diag_jac[0]

        trace_value = diag_jac

        # Calculate regularization terms
        if (dv_dt_reg is not None) or (jacobian_reg is not None):
            delv_delx = g.batch_jacobian(state_time_derivative, state)

        if dv_dt_reg is not None:
            delv_delt = g.gradient(state_time_derivative, time)
            vnabla_v = tf.linalg.matvec(delv_delx, state_time_derivative)
            dv_dt = delv_delt + vnabla_v
            #print('dv/dt :', dv_dt)
            trace_value = trace_value - dv_dt_reg * dv_dt**2

        if kinetic_reg is not None:
            #print('v :', state_time_derivative.shape)
            trace_value = trace_value - kinetic_reg * state_time_derivative**2

        if jacobian_reg is not None:
            jacobian_norm2 = tf.math.reduce_sum(delv_delx**2, axis=-1)
            #print('|J|^2 :', jacobian_norm2.shape)
            trace_value = trace_value - jacobian_reg * jacobian_norm2

        return state_time_derivative, trace_value

    return augmented_ode_fn


class ForceFieldModel(snt.Module):
    def __init__(self, n_dim, n_hidden, hidden_size, name='dz_dt'):
        super(ForceFieldModel, self).__init__(name=name)

        self._n_dim = n_dim

        output_sizes = [hidden_size] * n_hidden + [n_dim]
        self._nn = snt.nets.MLP(
            output_sizes,
            activation=tf.math.tanh,
            name='mlp'
        )

    def __call__(self, t, x):
        """
        Returns the vector dx/dt.

        Inputs:
          t (tf.Tensor): Scalar representing time.
          x (tf.Tensor): Spatial coordinates at which to evaluate
            dx/dt. Shape = (n_points, n_dim).
        """
        # Concatenate time and position vectors
        tx = tf.concat([tf.broadcast_to(t, [x.shape[0],1]), x], 1)
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
        x,s = tf.split(y, [self._n_dim,1], axis=1)
        dx_dt = self.__call__(t, x)
        ds_dt = tf.math.sqrt(tf.math.reduce_sum(dx_dt**2, axis=1))
        dy_dt = tf.concat([dx_dt, tf.expand_dims(ds_dt, 1)], 1)
        return dy_dt


class FFJORDFlow(tfd.TransformedDistribution):
    def __init__(self, n_dim, n_hidden, hidden_size,
                 reg_kw=dict(), atol=1.e-5, name='DF'):
        self._n_dim = n_dim
        self._n_hidden = n_hidden
        self._hidden_size = hidden_size
        self._name = name

        # Force field guiding transformation
        self.dz_dt = ForceFieldModel(n_dim, n_hidden, hidden_size)

        self.ode_solver = tfp.math.ode.DormandPrince(atol=atol)

        #if exact:
        #    trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact
        #else:
        #    # Stochastic estimate of the Jacobian. Better scaling with
        #    # number of dimensions, but noisy.
        #    trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson
        
        if len(reg_kw):
            print('Using regularization.')
            def trace_augmentation_fn(*args):
                return trace_jacobian_exact_reg(*args, **reg_kw)
        else:
            trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact

        # Initialize bijector
        bij = tfb.FFJORD(
            state_time_derivative_fn=self.dz_dt,
            ode_solve_fn=self.ode_solver.solve,
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

    def calc_path_length(self, x, forward=True):
        if forward:
            dxs_dt = self.dz_dt.augmented_field
        else:
            dxs_dt = lambda t,x: -self.dz_dt.augmented_field(1.-t,x)

        # Concatenate the initial path length (0) to x
        xs0 = tf.concat([x, tf.zeros([self._n_dim,1])], 1)

        # Solve the trajectory of x, keeping track of the path length
        res = self.ode_solver.solve(
            self.dz_dt.augmented_field,
            0., xs0,
            solution_times=[1.]
        )

        # Extract the final position and the path length
        xs_T = tf.squeeze(res.states, axis=0)
        x_T,s_T = tf.split(xs_T, [self._n_dim,1], axis=1)

        return x_T, s_T

    def calc_trajectories(self, n_samples, t_eval):
        x0 = self.distribution.sample([n_samples])
        res = self.ode_solver.solve(self.dz_dt, 0, x0, t_eval)
        return res
    
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
               lr_init=2.e-2,
               lr_final=1.e-4,
               checkpoint_every=None,
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
    batches = batches.shuffle(n_samples, reshuffle_each_iteration=True)
    batches = batches.repeat(n_epochs+1)
    batches = batches.batch(batch_size, drop_remainder=True)

    n_steps = n_epochs * n_samples // batch_size

    if optimizer is None:
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

        # Convert from # of epochs to # of steps between checkpoints
        checkpoint_steps = checkpoint_every * n_samples // batch_size

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
        grads,global_norm = tf.clip_by_global_norm(grads, 10.)
        #tf.print('\nglobal_norm =', global_norm)
        #tf.print([(v.name,tf.norm(v)) for v in grads])
        tf.print('loss =', loss)
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
        if (checkpoint_every is not None) and i and not (i % checkpoint_steps):
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
    flow = FFJORDFlow(2, 4, 16)

    n_samples = 8*1024
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

