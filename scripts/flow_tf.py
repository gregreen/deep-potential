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
import matplotlib
import matplotlib.pyplot as plt

import re

from time import time

from serializers_tf import (
    serialize_variable, deserialize_variable,
    weights_as_list, set_weights_w_list
)


class RQSBijector(tf.Module):
    def __init__(self, n_bins=8, n_features=16, n_hidden=3):
        # Number of dimensions to output
        self._n_outputs = None
        # Number of bins (knots + 1) in the spline
        self._n_bins = n_bins
        # Number of hidden layers used to determine spline parameters
        self._n_hidden = n_hidden
        # Number of neurons per hidden layer
        self._n_features = n_features
        # The parameters are initialized on the first call
        self._built = False

    def _bin_positions(self, x):
        """
        Activation function for the bin widths or heights.

        The widths (heights) should approximately add up to 2, so we
        use the softmax function.
        """
        # Reshape x: (# of points, # of dimensions to output, # of knots)
        x = tf.reshape(x, [-1, self._n_outputs, self._n_bins])
        # The extra factors involving 1.e-2 ensure that each bin has
        # positive width (height).
        return tf.math.softmax(x, axis=-1) * (2-self._n_bins*1e-2) + 1e-2

    def _slopes(self, x):
        """
        Activation function for slopes at the knot points.

        The slopes must be positive in order for the transformation
        to be a bijection, so we use the softplus transformation.
        """
        # Reshape x: (# of points, # of dimensions to output, # of knots)
        x = tf.reshape(x, [-1, self._n_outputs, self._n_bins - 1])
        # The zero-point added into x causes the activation
        # for x = 0 to be 1. The extra term added onto the activation
        # ensures that the slope is always sufficiently positive.
        return tf.math.softplus(x + 0.541324854612918) + 1e-2

    def _build(self, n_outputs):
        # Initialize the neural network that determines
        # the knot positions and slopes when the first
        # input is seen.
        self._n_outputs = n_outputs
        self._layers = []
        for i in range(self._n_hidden):
            # A stack of Dense + Leaky ReLU
            self._layers.append(tf.keras.layers.Dense(
                self._n_features,
                name=f'dense_{i+1}'
            ))
            self._layers.append(tf.keras.layers.LeakyReLU(
                alpha=0.2,
                name=f'activation_{i+1}'
            ))
        # Three separate layers determine the bin widths and heights,
        # and knot slopes
        self._bin_widths = tf.keras.layers.Dense(
            self._n_outputs * self._n_bins,
            activation=self._bin_positions,
            name='w'
        )
        self._bin_heights = tf.keras.layers.Dense(
            self._n_outputs * self._n_bins,
            activation=self._bin_positions,
            name='h'
        )
        self._knot_slopes = tf.keras.layers.Dense(
            self._n_outputs * (self._n_bins - 1),
            activation=self._slopes,
            name='s'
        )
        self._built = True

    def __call__(self, x, n_outputs):
        """
        Returns a RQS whose knot positions and slopes are determined by x.
        """
        if not self._built:
            self._build(n_outputs)
        # Pass x through a series of hidden NN layers
        z = x
        for layer in self._layers:
            z = layer(z)
        # Pass output into three different layers to determine
        # RQS parameters
        return tfb.RationalQuadraticSpline(
            bin_widths=self._bin_widths(z),
            bin_heights=self._bin_heights(z),
            knot_slopes=self._knot_slopes(z),
            #validate_args=True
        )

    def serialize(self):
        """
        Converts the RQS bijector into an object that can
        be converted to JSON.
        """
        o = {}
        o['n_hidden'] = self._n_hidden
        o['n_features'] = self._n_features
        o['n_bins'] = self._n_bins
        o['built'] = self._built

        # If network has not been built, no more info to add
        if not self._built:
            return o

        o['n_outputs'] = self._n_outputs

        # Check if weights have been initalized
        if not len(self._bin_widths.weights):
            return o

        o['dense'] = [weights_as_list(lay) for lay in self._layers[::2]]
        o['bin_widths'] = weights_as_list(self._bin_widths)
        o['bin_heights'] = weights_as_list(self._bin_heights)
        o['knot_slopes'] = weights_as_list(self._knot_slopes)

        return o

    @classmethod
    def deserialize(cls, d):
        """
        RQS factory that takes in a serialized version
        of the model, and returns an RQSBijector instance.
        """
        rqs = cls(
            n_bins=d['n_bins'],
            n_features=d['n_features'],
            n_hidden=d['n_hidden']
        )

        if d['built']:
            rqs._build(d['n_outputs'])

        if 'bin_widths' in d:
            # Feed in tensor to initialize layers
            if d['n_hidden']:
                n_dim_in = len(d['dense'][0][0])
            else:
                n_dim_in = len(d['bin_widths'][0])
            x = tf.ones([1, n_dim_in], dtype=tf.float32)
            _ = rqs(x, d['n_outputs'])

            # Set weights
            for w,l in zip(d['dense'], rqs._layers[::2]):
                set_weights_w_list(l, w)
            set_weights_w_list(rqs._bin_widths, d['bin_widths'])
            set_weights_w_list(rqs._bin_heights, d['bin_heights'])
            set_weights_w_list(rqs._knot_slopes, d['knot_slopes'])

        return rqs


def trainable_lu_factorization(event_size, batch_shape=(), name=None):
    with tf.name_scope(name or 'trainable_lu_factorization'):
        event_size = tf.convert_to_tensor(
            event_size,
            dtype_hint=tf.int32,
            name='event_size'
        )
        batch_shape = tf.convert_to_tensor(
            batch_shape,
            dtype_hint=event_size.dtype,
            name='batch_shape'
        )
        random_matrix = tf.random.uniform(
            shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
            dtype=tf.float32
        )
        random_orthonormal = tf.linalg.qr(random_matrix)[0]
        lower_upper, permutation = tf.linalg.lu(random_orthonormal)
        lower_upper = tf.Variable(
            initial_value=lower_upper,
            trainable=True,
            name='lower_upper'
        )
        # Initialize a non-trainable variable for the permutation indices so
        # that its value isn't re-sampled from run-to-run.
        permutation = tf.Variable(
            initial_value=permutation,
            trainable=False,
            name='permutation'
        )
    return lower_upper, permutation


class NormalizingFlow(object):
    """
    Represents a normalizing flow, with a unit Gaussian prior
    and a bijector consisting of interleaved invertible 1x1
    convolutions and Rational Quadratic Splines.
    """

    def __init__(self, n_dim, n_units, rqs=None, lu_fact=None, log_scale=None):
        """
        Randomly initializes the normalizing flow.

        If rqs and lu_fact are provided, then they will be used
        to create the normalizing flow, instead of randomly
        initializing the bijectors.
        """
        self._n_dim = n_dim
        self._n_units = n_units
        self.build(
            n_dim, n_units,
            rqs=rqs,
            lu_fact=lu_fact,
            log_scale=log_scale
        )

    def build(self, n_dim, n_units, rqs=None, lu_fact=None, log_scale=None):
        # Base distribution: p(x)
        self.dist = tfd.MultivariateNormalDiag(
            loc=np.zeros(n_dim, dtype='f4')
        )

        # Generate bijectors first, so that they are accessible later
        if rqs is None:
            self.rqs = [RQSBijector(n_bins=8) for i in range(2*n_units)]
        else:
            self.rqs = rqs

        if lu_fact is None:
            self.lu_fact = [
                trainable_lu_factorization(n_dim)
                for i in range(n_units)
            ]
        else:
            self.lu_fact = lu_fact

        if log_scale is None:
            self.log_scale = [
                tf.Variable(
                    initial_value=0.1*tf.random.truncated_normal([n_dim]),
                    trainable=True,
                    name='log_scale'
                )
                for i in range(n_units+1)
            ]
        else:
            self.log_scale = log_scale

        # Bijection: x -> y
        chain = []
        for i in range(n_units):
            chain.append(tfb.ScaleMatvecDiag(
                tf.exp(self.log_scale[i]),
                validate_args=True
            ))
            chain.append(tfb.RealNVP(
                fraction_masked=0.5,
                bijector_fn=self.rqs[2*i],
                validate_args=True
            ))
            chain.append(tfb.RealNVP(
                fraction_masked=-0.5,
                bijector_fn=self.rqs[2*i+1],
                validate_args=True
            ))
            chain.append(tfb.ScaleMatvecLU(
                *self.lu_fact[i],
                validate_args=True
            ))
        chain.append(tfb.ScaleMatvecDiag(
            tf.exp(self.log_scale[n_units]),
            validate_args=True
        ))
        self.bij = tfb.Chain(chain[::-1])

        # Transformed distribution: p(y)
        self.nvp = tfd.TransformedDistribution(
            distribution=self.dist,
            bijector=self.bij
        )

    def serialize(self):
        """
        Returns a JSON-serializable object representing the
        normalizing flow.
        """
        o = {}
        o['n_dim'] = self._n_dim
        o['n_units'] = self._n_units
        o['rqs'] = [rqs.serialize() for rqs in self.rqs]
        o['lu_fact'] = [
            [serialize_variable(v) for v in lu]
            for lu in self.lu_fact
        ]
        o['log_scale'] = [serialize_variable(v) for v in self.log_scale]
        return o

    @classmethod
    def deserialize(cls, d):
        """
        Factory to generate a normalizing flow from a dictionary (of
        the same format as created by `serialize`).
        """
        rqs = [RQSBijector.deserialize(v) for v in d['rqs']]
        lu_fact = [
            [deserialize_variable(v) for v in lu]
            for lu in d['lu_fact']
        ]
        log_scale = [deserialize_variable(v) for v in d['log_scale']]
        return cls(
            d['n_dim'],
            d['n_units'],
            rqs=rqs,
            lu_fact=lu_fact,
            log_scale=log_scale
        )


def get_flow_plot_fn(flow, p_true_fn=None):
    """
    Returns a function that produces a visualization of the flow.
    """
    # Sample x
    x_sample = flow.dist.sample(sample_shape=[1000])

    # Get input grid
    x = np.linspace(-2., 2., 300)
    y = np.linspace(-2., 2., 300)
    x,y = np.meshgrid(x, y)
    s = x.shape
    xy_grid = np.stack([x,y], axis=-1)
    xy_grid.shape = (-1, 2)
    xy_grid = tf.Variable(xy_grid.astype('f4'))

    if p_true_fn is None:
        p_true_img = None
    else:
        p_true_img = p_true_fn(xy_grid)
        p_true_img /= np.sum(p_true_img)
        p_true_img.shape = s

    def plot_fn():
        # Sample y, and get x through inverse
        y_sample = flow.bij.forward(x_sample)
        c = flow.nvp.log_prob(y_sample)

        fig,ax_arr = plt.subplots(
            2,3,
            figsize=(15,10),
            subplot_kw=dict(aspect='equal')
        )
        fig.subplots_adjust(
            left=0.05, right=0.99,
            bottom=0.05, top=0.95
        )

        ax1,ax2,ax3 = ax_arr[0]
        ax1.scatter(x_sample[:,0], x_sample[:,1], alpha=0.3, s=3, c=c)
        ax1.set_title('x')
        ax1.set_xlim(-3., 3.)
        ax1.set_ylim(-3., 3.)
        ax2.scatter(y_sample[:,0], y_sample[:,1], alpha=0.3, s=3, c=c)
        ax2.set_title('y')
        ax2.set_xlim(-3., 3.)
        ax2.set_ylim(-3., 3.)
        for xx,yy in zip(x_sample[::4],y_sample[::4]):
            dxy = yy-xx
            ax3.arrow(
                xx[0], xx[1],
                0.2*dxy[0], 0.2*dxy[1],
                # head_length=0.1,
                head_width=0.1,
                alpha=0.3
            )
            ax3.set_xlim(-3., 3.)
            ax3.set_ylim(-3., 3.)
        ax3.set_title(r'$0.2 \left( y-x \right)$')

        # Image of distribution
        lnp_img = flow.nvp.log_prob(xy_grid).numpy()
        lnp_img.shape = s

        p_img = np.exp(lnp_img)
        p_img /= np.sum(p_img)

        if p_true_img is None:
            vmax = np.max(p_img)
        else:
            vmax = 1.2 * np.max(p_true_img)

        ax1,ax2,ax3 = ax_arr[1]
        ax1.imshow(
            lnp_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmax=np.max(lnp_img),
            vmin=np.max(lnp_img) - 25.
        )
        ax1.set_title(r'$\ln p \left( y \right)$')
        ax2.imshow(
            p_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmin=0.,
            vmax=vmax
        )
        ax2.set_title(r'$p \left( y \right)$')

        if p_true_img is None:
            ax3.axis('off')
        else:
            ax3.imshow(
                p_true_img,
                extent=(-2., 2., -2., 2.),
                interpolation='nearest',
                vmin=0.,
                vmax=vmax
            )
            ax3.set_title(r'$p_{\mathrm{true}} \left( y \right)$')

        return fig

    return plot_fn


def plot_inv1x1conv(bij, ax, label_y_axis=True):
    """
    Plots a representation of an invertible 1x1
    convolution to the provided axes.
    """
    x = np.linspace(-0.8, 0.8, 10).astype('f4')
    y = np.linspace(-0.8, 0.8, 10).astype('f4')
    x,y = np.meshgrid(x,y)
    xy = tf.stack([x.flat, y.flat], axis=1)

    xy_p = bij(xy)

    ax.scatter(xy[:,0], xy[:,1], c='k', alpha=0.1, s=4)
    ax.scatter(xy_p[:,0], xy_p[:,1], s=4)
    for xy_i,xyp_i in zip(xy, xy_p):
        ax.plot(
            [xy_i[0], xyp_i[0]],
            [xy_i[1], xyp_i[1]],
            c='b',
            alpha=0.1
        )

    if isinstance(bij, tfb.ScaleMatvecLU):
        ax.set_title('ScaleMatvecLU')
    elif isinstance(bij, tfb.ScaleMatvecDiag):
        ax.set_title('ScaleMatvecDiag')

    ax.set_xlabel(r'$x_0$')

    ax.set_ylabel(r'$x_1$', labelpad=-2)
    if not label_y_axis:
        ax.set_yticklabels([])

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)


def plot_nsf(bij, ax, label_y_axis=True):
    """
    Plots a representation of a normalizing spline flow
    to the provided axes.
    """
    cmap = matplotlib.cm.get_cmap('viridis')
    for x0 in tf.linspace(-1., 1., 5):
        c = cmap(0.5*(x0+1))
        x = tf.stack([
            x0*tf.ones([1000]),
            tf.linspace(-1.2, 1.2, 1000)
        ], axis=1)
        y = bij(x)
        ax.plot(x[:,1], y[:,1], c=c)
        ax.set_title('RealNVP-RQS')

    ax.grid('on', alpha=0.25)
    ax.set_xlabel(r'$x_1$')
    ax.set_title(r'RQS')

    ax.set_ylabel(r'$x_1^{\prime}$', labelpad=-2)
    if not label_y_axis:
        ax.set_yticklabels([])


def plot_prob(nvp, ax_p=None, ax_lnp=None):
    # Get input grid
    x = np.linspace(-2., 2., 300)
    y = np.linspace(-2., 2., 300)
    x,y = np.meshgrid(x, y)
    s = x.shape
    xy_grid = np.stack([x,y], axis=-1)
    xy_grid.shape = (-1, 2)
    xy_grid = tf.Variable(xy_grid.astype('f4'))

    # Image of distribution
    lnp_img = nvp.log_prob(xy_grid).numpy()
    lnp_img.shape = s

    p_img = np.exp(lnp_img)
    p_img /= np.sum(p_img)

    ax = []

    if ax_p is not None:
        ax_p.imshow(
            p_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmax=np.max(p_img),
            vmin=0.
        )
        ax.append(ax_p)

    if ax_lnp is not None:
        ax_lnp.imshow(
            lnp_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmax=np.max(lnp_img),
            vmin=np.max(lnp_img) - 25.
        )
        ax.append(ax_lnp)

    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])


def plot_bijections(flow):
    """
    Returns a figure that visualizes the bijections
    in the provided flow.
    """
    n_bij = len(flow.bij.bijectors)
    fig,ax_arr = plt.subplots(
        3,n_bij,
        figsize=(1+3*n_bij,10),
        subplot_kw=dict(aspect='equal')
    )
    fig.subplots_adjust(
        wspace=0.16, hspace=0.2,
        left=0.03, right=0.99,
        bottom=0.02, top=0.97
    )

    for i,b in enumerate(flow.bij.bijectors[::-1]):
        if isinstance(b, tfb.RealNVP):
            plot_nsf(b, ax_arr[0,i], label_y_axis=(i == 0))
        elif (isinstance(b, tfb.ScaleMatvecLU)
              or isinstance(b, tfb.ScaleMatvecDiag)):
            plot_inv1x1conv(b, ax_arr[0,i], label_y_axis=(i == 0))
        else:
            ax_arr[0,i].axis('off')

        bij_p = tfb.Chain(flow.bij.bijectors[-(i+1):])
        nvp_p = tfd.TransformedDistribution(
            distribution=flow.dist,
            bijector=bij_p
        )
        plot_prob(nvp_p, ax_p=ax_arr[1,i], ax_lnp=ax_arr[2,i])

    ax_arr[1,0].set_ylabel(r'$p$', fontsize=18)
    ax_arr[2,0].set_ylabel(r'$\ln p$', fontsize=18)

    return fig


def get_flow_loss_gradient_func(flow):
    """
    Returns a function that calculates the
    loss and gradients (dloss/dparams) of
    the provided flow, given a set of
    observations y.
    """
    @tf.function
    def calc_flow_loss_gradients(y):
        print('Tracing calc_flow_loss_gradients')

        with tf.GradientTape() as g:
            g.watch(flow.bij.trainable_variables)
            log_p = flow.nvp.log_prob(y)
            loss = -tf.reduce_mean(log_p)

        grads = g.gradient(loss, flow.bij.trainable_variables)

        return loss, grads

    return calc_flow_loss_gradients


def get_training_callback(
    flow, every=500,
    fname='nvp_{i:05d}.png',
    p_true_fn=None,
    **kwargs
):
    """
    Returns a standard callback function that can be passed
    to train_flow. Every <every> steps callback prints the
    step number, loss and learning rate, and plots the flow.

    Inputs:
        flow (NormalizingFlow): Normalizing flow to be trained.
        every (int): The callback will run every <every> steps.
            Defaults to 500.
        fname (str): Pattern (using the new Python formatting
            language) used to generate the filename. Can use
            <i>, <n_steps> and <every>.
        p_true_fn (callable): Function that takes coordinates,
            and returns the true probability density. Defaults
            to None.
    """
    plt_fn = kwargs.get(
        'plt_fn',
        get_flow_plot_fn(flow, p_true_fn=p_true_fn)
    )

    def training_callback(i, n_steps, loss_history, opt):
        if (i % every == 0) or (i == n_steps - 1):
            loss_avg = np.mean(loss_history[-50:])
            lr = float(opt._decayed_lr(tf.float32))
            print(
                f'Step {i+1: >5d} of {n_steps}: '
                f'<loss> = {loss_avg: >7.5f} , '
                f'lr = {lr:.4g}'
            )
            if plt_fn is not None:
                fig = plt_fn()
                namespace = dict(i=i, n_steps=n_steps, every=every)
                fig.savefig(fname.format(**namespace), dpi=100)
                plt.close(fig)

    return training_callback


def train_flow(flow, data, optimizer=None, batch_size=32,
               n_epochs=1, callback=None):
    """
    Trains a flow using the given data.

    Inputs:
      flow (NormalizingFlow): Normalizing flow to be trained.
      data (tf.Tensor): Observed points. Shape = (# of points, # of dim).
      optimizer (tf.keras.optimizers.Optimizer): Optimizer to use.
          Defaults to the Rectified Adam implementation from
          tensorflow_addons.
      batch_size (int): Number of points per training batch.
      n_epochs (int): Number of training epochs.
      callback (callable): Function that will be called
          at the end of each iteration. Function
          signature: f(i, n_steps, loss_history, opt), where i is the
          iteration index (int), n_steps (int) is the total number of
          steps, loss_history is a list of floats, containing the loss
          at each iteration so far, and opt is the optimizer.

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

    t0 = time()

    calc_loss_gradients = get_flow_loss_gradient_func(flow)

    for i,y in enumerate(batches):
        loss, grads = calc_loss_gradients(y)
        opt.apply_gradients(zip(grads, flow.bij.trainable_variables))

        if i == 0:
            # Get time after gradients function is first traced
            t1 = time()

        loss_history.append(float(loss))
        callback(i, n_steps, loss_history, opt)

    t2 = time()
    loss_avg = np.mean(loss_history[-50:])
    n_steps = len(loss_history)
    print(f'<loss> = {loss_avg: >7.5f}')
    print(f'tracing time: {t1-t0:.2f} s')
    print(f'training time: {t2-t1:.1f} s ({(t2-t1)/n_steps:.4f} s/step)')

    return loss_history


def main():
    return 0

if __name__ == '__main__':
    main()
