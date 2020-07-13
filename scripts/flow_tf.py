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


def weights_as_list(layer):
    return [w.tolist() for w in layer.get_weights()]


def set_weights_w_list(layer, weights):
    layer.set_weights([np.array(w, dtype='f4') for w in weights])


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
            validate_args=True
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

        o['dense'] = [weights_as_list(l) for l in self._layers[::2]]
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


def serialize_variable(v):
    """
    Returns a JSON-serializable dictionary representing a
    tf.Variable.
    """
    return dict(
        dtype=v.dtype.name,
        shape=list(v.shape),
        values=v.numpy().tolist(),
        name=re.sub('\:[0-9]+$', '', v.name),
        trainable=v.trainable
    )


def deserialize_variable(d):
    """
    Returns a tf.Variable, constructed using a dictionary
    of the form returned by `serialize_variable`.
    """
    return tf.Variable(
        np.array(d['values'], dtype=d['dtype']),
        name=d['name'],
        trainable=d['trainable']
    )


class NormalizingFlow(object):
    """
    Represents a normalizing flow, with a unit Gaussian prior
    and a bijector consisting of interleaved invertible 1x1
    convolutions and Rational Quadratic Splines.
    """

    def __init__(self, n_dim, n_units, rqs=None, lu_fact=None):
        """
        Randomly initializes the normalizing flow.

        If rqs and lu_fact are provided, then they will be used
        to create the normalizing flow, instead of randomly
        initializing the bijectors.
        """
        self._n_dim = n_dim
        self._n_units = n_units
        self.build(n_dim, n_units, rqs=rqs, lu_fact=lu_fact)

    def build(self, n_dim, n_units, rqs=None, lu_fact=None):
        # Base distribution: p(x)
        self.dist = tfd.MultivariateNormalDiag(
            loc=np.zeros(n_dim, dtype='f4')
        )

        # Generate bijectors first, so that they are accessible later
        if rqs is None:
            self.rqs = [RQSBijector(n_bins=8) for i in range(n_units)]
        else:
            self.rqs = rqs
        
        if lu_fact is None:
            self.lu_fact = [
                trainable_lu_factorization(n_dim)
                for i in range(n_units+1)
            ]
        else:
            self.lu_fact = lu_fact

        # Bijection: x -> y
        chain = []
        for i in range(n_units):
            chain.append(tfb.ScaleMatvecLU(
                *self.lu_fact[i],
                validate_args=True
            ))
            chain.append(tfb.RealNVP(
                fraction_masked=0.5,
                bijector_fn=self.rqs[i],
                validate_args=True
            ))
        chain.append(tfb.ScaleMatvecLU(
            *self.lu_fact[n_units],
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
        return cls(d['n_dim'], d['n_units'], rqs=rqs, lu_fact=lu_fact)


def get_plot_fn(flow):
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

        r = np.sqrt(xy_grid[:,0]**2+xy_grid[:,1]**2)
        p_true_img = np.exp(-0.5*((r-1.)/0.1)**2) / r
        p_true_img /= np.sum(p_true_img)
        p_true_img.shape = s

        ax1,ax2,ax3 = ax_arr[1]
        ax1.imshow(
            lnp_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmax=np.max(lnp_img),
            vmin=np.max(lnp_img) - 30.
        )
        ax1.set_title(r'$\ln p \left( y \right)$')
        ax2.imshow(
            p_img,
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmax=1.2*np.max(p_true_img),
            vmin=0.
        )
        ax2.set_title(r'$p \left( y \right)$')
        ax3.imshow(
            p_true_img / np.sum(p_true_img),
            extent=(-2., 2., -2., 2.),
            interpolation='nearest',
            vmax=1.2*np.max(p_true_img),
            vmin=0.
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

    if isinstance(b, tfb.ScaleMatvecLU):
        ax.set_title('invertible 1x1 convolution')
    elif isinstance(b, ActNorm):
        ax.set_title('activation normalization')

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
        ax.set_title('RQS')

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
            plot_nsf(b, ax_arr[0,i], label_y_axis=(i==0))
        elif isinstance(b, tfb.ScaleMatvecLU) or isinstance(b, ActNorm):
            plot_inv1x1conv(b, ax_arr[0,i], label_y_axis=(i==0))
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


def get_loss_gradient_func(flow):
    """
    Returns a function that calculates the
    loss and gradients (dloss/dparams) of
    the provided flow, given a set of
    observations y.
    """
    @tf.function
    def get_loss_gradients(y):
        print('Tracing get_loss_gradients')

        with tf.GradientTape() as g:
            g.watch(flow.bij.trainable_variables)
            log_p = flow.nvp.log_prob(y)
            loss = -tf.reduce_mean(log_p)

        grads = g.gradient(loss, flow.bij.trainable_variables)

        return loss, grads

    return get_loss_gradients


def main():
    return 0

if __name__ == '__main__':
    main()

