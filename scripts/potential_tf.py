#!/usr/bin/env python

from __future__ import print_function, division

import tensorflow as tf
print(f'Tensorflow version {tf.__version__}')
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_probability as tfp
print(f'Tensorflow Probability version {tfp.__version__}')
tfb = tfp.bijectors
tfd = tfp.distributions

import numpy as np
import matplotlib.pyplot as plt


from serializers_tf import (
    serialize_variable, deserialize_variable,
    weights_as_list, set_weights_w_list
)


def calc_df_deta(f_func, q, p):
    # Calculate gradients of distribution function
    with tf.GradientTape(persistent=True) as g:
        g.watch([q, p])
        f = f_func(q, p)

    df_dq = g.gradient(f, q, unconnected_gradients='zero')
    df_dp = g.gradient(f, p, unconnected_gradients='zero')

    return f, df_dq, df_dp


@tf.function
def calc_phi_derivatives(phi_func, q):
    print('Tracing calc_phi_derivatives')
    print(f'q.shape = {q.shape}')

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

    return dphi_dq, d2phi_dq2


@tf.function
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
    print('Tracing calc_loss_terms')
    print(f'q.shape = {q.shape}')
    print(f'p.shape = {p.shape}')

    # Calculate necessary derivatives
    dphi_dq, d2phi_dq2 = calc_phi_derivatives(phi_func, q)

    # partial f / partial t = {H,f}
    df_dt = tf.reduce_sum(df_dp * dphi_dq - df_dq * p, axis=1)

    return df_dt, d2phi_dq2


@tf.function
def get_phi_loss_gradients(phi, params, q, p,
                           f=None, df_dq=None, df_dp=None,
                           lam=tf.constant(1.0),
                           mu=tf.constant(0.01),
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
    print('Tracing get_phi_loss_gradients')
    print(f'q.shape = {q.shape}')
    print(f'p.shape = {p.shape}')

    # If df/dq and df/dp are not provided, then calculate them
    # from using the distribution function.
    if f is not None:
        _, df_dq, df_dp = calc_df_deta(f, q, p)
    elif (df_dq is None) or (df_dp is None):
        raise ValueError(
            'If f is not provided, then df_dq and df_dp must be provided.'
        )

    with tf.GradientTape() as g:
        g.watch(params)

        # Calculate derivatives of phi w.r.t. q
        dphi_dq, d2phi_dq2 = calc_phi_derivatives(phi, q)

        # partial f / partial t = {H,f}
        df_dt = tf.reduce_sum(df_dp * dphi_dq - df_dq * p, axis=1)

        # Weight each point differently
        if weight_samples:
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
        likelihood = tf.math.asinh(tf.math.abs(df_dt))
        prior_neg = tf.math.asinh(
            tf.clip_by_value(-d2phi_dq2, 0., np.inf)
        )
        prior_pos = tf.math.asinh(
            tf.clip_by_value(d2phi_dq2, 0., np.inf)
        )

        # Regularization penalty
        # penalty = 0.
        # for p in params:
        #     penalty += tf.reduce_sum(p**2)

        loss = tf.reduce_mean(
            likelihood
            + lam*prior_neg
            + mu*prior_pos
            # + reg*penalty
        )

    # Gradients of loss w.r.t. NN parameters
    dloss_dparam = g.gradient(loss, params)

    return loss, dloss_dparam


class PhiNN(tf.Module):
    """
    Feed-forward neural network to represent the gravitational
    potential.
    """

    def __init__(self, n_dim=3, n_hidden=3, n_features=32, build=True):
        """
        Constructor for PhiNN.

        Inputs:
            n_dim (int): Number of spatial dimensions in the input.
            n_hidden (int): Number of hidden layers.
            n_features (int): Number of neurons in each hidden layer.
            build (bool): Whether to create the weights and biases.
                Defaults to True. This option exists so that the
                deserializer can set the weights and biases on its
                own.
        """
        self._n_dim = n_dim
        self._n_hidden = n_hidden
        self._n_features = n_features

        if build:
            self._W = [
                tf.Variable(
                    tf.random.truncated_normal([n_dim,n_features]),
                    name='weight_1'
                )
            ]
            self._W += [
                tf.Variable(
                    tf.random.truncated_normal([n_features,n_features]),
                    name=f'weight_{i+1}'
                )
                for i in range(n_hidden-1)
            ]
            self._W += [
                tf.Variable(
                    tf.random.truncated_normal([n_features,1]),
                    name=f'weight_{n_hidden+1}'
                )
            ]
            self._b = [
                tf.Variable(
                    tf.random.truncated_normal([1,n_features]),
                    name=f'bias_{i+1}'
                )
                for i in range(n_hidden)
            ]

    @tf.function
    def __call__(self, q):
        """
        Returns the potential at the coordinates q.

        Let n = number of input points, d = number
        of spatial dimensions.

        Inputs:
            q (tf.Tensor): Shape = (n,d).

        Outputs:
            The potential at each input point. Tensor of
            shape (n,).
        """
        print('Tracing PhiNN.__call__')
        print(f'q.shape = {q.shape}')
        o = q
        for i in range(self._n_hidden):
            o = tf.tensordot(o, self._W[i], (1,0))
            o = tf.math.sigmoid(o + self._b[i])
        o = tf.tensordot(o, self._W[-1], (1,0))
        o = tf.squeeze(o, axis=[1])
        return o

    def serialize(self):
        """
        Returns a JSON-serializable dictionary containing the
        information necessary to reconstruct the neural network.
        """
        d = dict(
            n_dim=self._n_dim,
            n_hidden=self._n_hidden,
            n_features=self._n_features,
            W=[],
            b=[]
        )
        for W in self._W:
            d['W'].append(serialize_variable(W))
        for b in self._b:
            d['b'].append(serialize_variable(b))
        return d

    @classmethod
    def deserialize(cls, d):
        """
        Returns a PhiNN constructed from a dictionary of the
        form created by PhiNN.serialize.
        """
        phi_nn = cls(
            n_dim=d['n_dim'],
            n_hidden=d['n_hidden'],
            n_features=d['n_features'],
            build=False
        )
        phi_nn._W = [deserialize_variable(W) for W in d['W']]
        phi_nn._b = [deserialize_variable(b) for b in d['b']]
        return phi_nn


def main():
    return 0

if __name__ == '__main__':
    main()

