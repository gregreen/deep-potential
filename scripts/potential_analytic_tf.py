#!/usr/bin/env python

from __future__ import print_function, division

# Tensorflow & co
import tensorflow as tf
print(f'Tensorflow version {tf.__version__}')
from tensorflow import keras
import tensorflow_probability as tfp
print(f'Tensorflow Probability version {tfp.__version__}')
tfb = tfp.bijectors
tfd = tfp.distributions
import sonnet as snt

import numpy as np

import json
from time import time

# Custom libraries
import utils



class PhiNNAnalytic(snt.Module):
    """
    Feed-forward neural network to represent the gravitational
    potential. This one is made up of several analytic components:
    1. three Miyamoto-Nagai disk: variables mni_amp, mni_a, mni_b where i goes
       from 1 to 3
    2. NFW potential representing the halo: variables halo_amp, halo_a
    3. Power-law density spherical potential with an exponential cut-off
       representing the bulge: variables bulge_amp, bulge_rcut, bulge_alpha

    Note on checkpointing: Both PhiNN and FrameShift rely on a combination of
    tf.Checkpoint and a custom spec saving system. This is caused by
    restoration from a tf Checkpoint requiring an already initialized instance
    of the object. Initializing the object can't use data stored in the
    checkpoint, and must find its metadata elsewhere, hence from the spec file.
    """

    def __init__(
        self, n_dim=3, dz=0., dz_trainable=False,
        mn1_amp=1., mn1_amp_trainable=False,
        mn1_a=3., mn1_a_trainable=False,
        mn1_b=0.5, mn1_b_trainable=False,
        mn2_amp=1., mn2_amp_trainable=False,
        mn2_a=3., mn2_a_trainable=False,
        mn2_b=0.5, mn2_b_trainable=False,
        mn3_amp=1., mn3_amp_trainable=False,
        mn3_a=3., mn3_a_trainable=False,
        mn3_b=0.5, mn3_b_trainable=False,
        halo_amp=0., halo_amp_trainable=False,
        halo_a=16., halo_a_trainable=False,
        #bulge_amp=0., bulge_amp_trainable=False,
        #bulge_rcut=2., bulge_rcut_trainable=False,
        #bulge_alpha=2., bulge_alpha_trainable=False,
        name='PhiNNAnalytic', r_c=8.3
    ):
        """
        Constructor for PhiNNAnalytic.
        Amplitudes are in units of [(100km/s)^2 kpc] and represent the total mass of the component in some units. Densities inferred from the amplitudes are in units of [(100kms/s/kpc)^2].
        To convert amplitudes to total mass in Msun, divide by 4 pi G. This conversion turns out to be 2.3250854e9 / 4 / pi.
        To convert densities to Msun / pc^3, divide by 4 pi G. This conversion turns out to be 2.3250854 / 4 / pi.

        Inputs:
            n_dim (int): Dimensionality of space.
        """
        super(PhiNNAnalytic, self).__init__(name=name)

        self._n_dim = n_dim
        self._name = name

        # Variables for describing the central coordinate system
        self._r_c = tf.Variable(r_c, trainable=False, name='r_c', dtype=tf.float32)
        self._dxy = tf.Variable(np.array((-r_c, 0)).astype('f4'), trainable=False, name='dxy', dtype=tf.float32)
        self._dz = tf.Variable(dz, trainable=dz_trainable, name='dz', dtype=tf.float32)

        # Variables that have "l2penalty" in them are penalized

        # Disk
        # Miyamoto Nagai variables.
        # https://docs.galpy.org/en/v1.8.1/reference/potentialmiyamoto.html
        # MWPotential2014: scaleRadius=a=3.0[kpc], scaleHeight=b=0.28[kpc], mass=amp=6.819386e10[...]
        self._mn1_logamp = tf.Variable(np.log(mn1_amp), trainable=mn1_amp_trainable, name='miyamoto_nagai_amplitude_1', dtype=tf.float32)
        self._mn1_loga = tf.Variable(np.log(mn1_a), trainable=mn1_a_trainable, name='miyamoto_nagai_loga_1', dtype=tf.float32)
        self._mn1_logb = tf.Variable(np.log(mn1_b), trainable=mn1_b_trainable, name='miyamoto_nagai_logb_1', dtype=tf.float32)
        self._mn2_logamp = tf.Variable(np.log(mn2_amp), trainable=mn2_amp_trainable, name='miyamoto_nagai_amplitude_2', dtype=tf.float32)
        self._mn2_loga = tf.Variable(np.log(mn2_a), trainable=mn2_a_trainable, name='miyamoto_nagai_loga_2', dtype=tf.float32)
        self._mn2_logb = tf.Variable(np.log(mn2_b), trainable=mn2_b_trainable, name='miyamoto_nagai_logb_2', dtype=tf.float32)
        self._mn3_logamp = tf.Variable(np.log(mn3_amp), trainable=mn3_amp_trainable, name='miyamoto_nagai_amplitude_3', dtype=tf.float32)
        self._mn3_loga = tf.Variable(np.log(mn3_a), trainable=mn3_a_trainable, name='miyamoto_nagai_loga_3', dtype=tf.float32)
        self._mn3_logb = tf.Variable(np.log(mn3_b), trainable=mn3_b_trainable, name='miyamoto_nagai_logb_3', dtype=tf.float32)

        # Halo
        # NFW potential.
        # https://docs.galpy.org/en/v1.8.1/reference/potentialnfw.html
        # MWPotential2014: densitynorm=8.48683e6, a=16
        self._halo_logamp = tf.Variable(np.log(halo_amp), trainable=halo_amp_trainable, name='halo_nfw_amplitude', dtype=tf.float32)
        self._halo_loga = tf.Variable(np.log(halo_a), trainable=halo_a_trainable, name='halo_nfw_loga', dtype=tf.float32)

        # Bulge
        # Power-law density spherical potential with an exponential cut-off.
        # https://docs.galpy.org/en/v1.8.1/reference/potentialpowerspherwcut.html?highlight=PowerSphericalPotentialwCutoff
        # MWPotential2014: densityNorm=2.22694e8, alpha=1.8, rc=1.9
        #self._bulge_logamp = tf.Variable(np.log(bulge_amp), trainable=bulge_amp_trainable, name='bulge_pspc_amp', dtype=tf.float32)
        #self._bulge_logrcut = tf.Variable(np.log(bulge_rcut), trainable=bulge_rcut_trainable, name='bulge_pspc_logrcut', dtype=tf.float32)
        #self._bulge_alpha = tf.Variable(bulge_alpha, trainable=bulge_alpha_trainable, name='bulge_pspc_alpha', dtype=tf.float32)

        # Initialize
        self.__call__(tf.zeros([1,n_dim]))

    def __call__(self, q):
        """Returns the gravitational potential of the analytic model"""
        xy,z = tf.split(q, [2,1], axis=1)
        z = z + self._dz
        xy = xy + self._dxy
        R2 = tf.expand_dims(tf.math.reduce_sum(xy**2, axis=1), 1) # w.r.t. galactic centre
        r2 = R2 + z**2 # w.r.t galactic centre

        # We assume amp to be in units of M_sun. Hence, to convert to internal units of [(100km/s)^2 kpc], we multiply by 4.3e-7.
        coef = 4.3e-10 # Converts [M_sun * G] to [(100km/s)^2 kpc]

        # Miyamoto-Nagai contribution
        pot_mn1 = -coef*tf.exp(self._mn1_logamp) / tf.math.sqrt(R2 + (tf.math.sqrt(z**2 + tf.math.exp(2*self._mn1_logb)) + tf.math.exp(self._mn1_loga))**2)
        pot_mn2 = -coef*tf.exp(self._mn2_logamp) / tf.math.sqrt(R2 + (tf.math.sqrt(z**2 + tf.math.exp(2*self._mn2_logb)) + tf.math.exp(self._mn2_loga))**2)
        pot_mn3 = -coef*tf.exp(self._mn3_logamp) / tf.math.sqrt(R2 + (tf.math.sqrt(z**2 + tf.math.exp(2*self._mn3_logb)) + tf.math.exp(self._mn3_loga))**2)

        # Halo contribution
        pot_halo = -coef*tf.exp(self._halo_logamp)*tf.math.xlogy(1./r2**0.5, 1. + r2**0.5/tf.math.exp(self._halo_loga))
        #pot_halo = tf.exp(self._halo_logamp) / 4 / np.pi / r2**0.5 / (tf.math.exp(self._halo_loga) + r2**0.5)**2

        # Bulge contribution
        #pot_bulge = tf.exp(self._bulge_logamp)*2.* np.pi*tf.exp((3. - self._bulge_alpha)*self._bulge_logrcut) *\
        #    (1/tf.math.exp(self._bulge_logrcut)*tf.math.exp(tf.math.lgamma(1. - self._bulge_alpha/2.)) *
        #     tf.math.igamma(1. - self._bulge_alpha/2., (r2**0.5/tf.math.exp(self._bulge_logrcut))**2.) -
        #     tf.math.exp(tf.math.lgamma(1.5 - self._bulge_alpha/2.)) *
        #     tf.math.igamma(1.5 - self._bulge_alpha/2., (r2**0.5/tf.math.exp(self._bulge_logrcut))**2.)/r2**0.5)
        #pot_bulge = tf.exp(self._bulge_logamp) / r2**(self._bulge_alpha / 2) * tf.math.exp(-r2 / tf.math.exp(2 * self._bulge_logrcut))

        return pot_mn1 + pot_mn2 + pot_mn3 + pot_halo# + pot_bulge

    def save_specs(self, spec_name_base):
        """Saves the specs of the model that are required for initialization to a json"""
        d = dict(
            n_dim=self._n_dim,
            name=self._name
        )
        with open(spec_name_base + '_spec.json', 'w') as f:
            json.dump(d, f)

        return spec_name_base

    @classmethod
    def load(cls, checkpoint_name):
        """Load PhiNNAnalytic from a checkpoint and a spec file"""
        # Get spec file name
        if checkpoint_name.find('-') == -1 or not checkpoint_name.rsplit('-', 1)[1].isdigit():
            raise ValueError("PhiNNAnalytic checkpoint name doesn't follow the correct syntax.")
        spec_name = checkpoint_name.rsplit('-', 1)[0] + "_spec.json"

        # Load specs
        with open(spec_name, 'r') as f:
            kw = json.load(f)
        phi_nn = cls(**kw)

        # Restore variables
        checkpoint = tf.train.Checkpoint(phi=phi_nn)
        checkpoint.restore(checkpoint_name).expect_partial() 

        print(f'loaded {phi_nn} from {checkpoint_name}')
        return phi_nn

    @classmethod
    def load_latest(cls, checkpoint_dir):
        """Load the latest PhiNNAnalytic from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(f"Couldn't load a valid PhiNNAnalytic from {repr(checkpoint_dir)}")
        return PhiNNAnalytic.load(latest)

    @classmethod
    def load_checkpoint_with_id(cls, checkpoint_dir, id):
        """Load the PhiNNAnalytic with a specified id from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(f"Couldn't load a valid PhiNNAnalytic from {repr(checkpoint_dir)}")
        latest = latest[:latest.rfind('-')] + f'-{id}'
        return PhiNNAnalytic.load(latest)


class PhiNNAnalyticBarmodel(snt.Module):
    """
    Feed-forward neural network to represent the gravitational
    potential. This one is made up of several analytic components:
    1. Miyamoto-Nagai disk: variables mn_amp, mn_a, mn_b
    2. NFW potential representing the halo: variables halo_amp, halo_a
    3. Power-law density spherical potential with an exponential cut-off
       representing the bulge: variables bulge_amp, bulge_rcut, bulge_alpha

    Note on checkpointing: Both PhiNN and FrameShift rely on a combination of
    tf.Checkpoint and a custom spec saving system. This is caused by
    restoration from a tf Checkpoint requiring an already initialized instance
    of the object. Initializing the object can't use data stored in the
    checkpoint, and must find its metadata elsewhere, hence from the spec file.
    """

    def __init__(
        self, n_dim=3,
        dz=0., dz_trainable=False,
        mn_amp=1., mn_amp_trainable=False,
        mn_a=3., mn_a_trainable=False,
        mn_b=0.5, mn_b_trainable=False,
        halo_amp=0., halo_amp_trainable=False,
        halo_a=16., halo_a_trainable=False,
        #A=[0.4, 0.1], A_trainable=False,
        #B=[0.01, 0.01], B_trainable=False,
        #C=0.186, C_trainable=False,
        name='PhiNNAnalyticBarmodel', r_c=8.3
    ):
        """
        Constructor for PhiNNAnalyticBarmodel.
        Amplitudes are in units of [(100km/s)^2 kpc] and represent the total mass of the component in some units. Densities inferred from the amplitudes are in units of [(100kms/s/kpc)^2].
        To convert amplitudes to total mass in Msun, divide by 4 pi G. This conversion turns out to be 2.3250854e9 / 4 / pi.
        To convert densities to Msun / pc^3, divide by 4 pi G. This conversion turns out to be 2.3250854 / 4 / pi.

        Inputs:
            n_dim (int): Dimensionality of space.
        """
        super(PhiNNAnalyticBarmodel, self).__init__(name=name)

        self._n_dim = n_dim
        self._name = name

        # Variables for describing the central coordinate system
        self._r_c = tf.Variable(r_c, trainable=False, name='r_c', dtype=tf.float32)
        self._dxy = tf.Variable(np.array((-r_c, 0)).astype('f4'), trainable=False, name='dxy', dtype=tf.float32)
        self._dz = tf.Variable(dz, trainable=dz_trainable, name='dz', dtype=tf.float32)

        # Variables that have "l2penalty" in them are penalized

        # Disk
        # Miyamoto Nagai variables.
        # https://docs.galpy.org/en/v1.8.1/reference/potentialmiyamoto.html
        # MWPotential2014: scaleRadius=a=3.0[kpc], scaleHeight=b=0.28[kpc], mass=amp=6.819386e10[...]
        self._mn_logamp = tf.Variable(np.log(mn_amp), trainable=mn_amp_trainable, name='miyamoto_nagai_amplitude', dtype=tf.float32)
        self._mn_loga = tf.Variable(np.log(mn_a), trainable=mn_a_trainable, name='miyamoto_nagai_loga', dtype=tf.float32)
        self._mn_logb = tf.Variable(np.log(mn_b), trainable=mn_b_trainable, name='miyamoto_nagai_logb', dtype=tf.float32)

        # # Harmonic components of the disk
        #self._A = tf.Variable(A, trainable=A_trainable, name='fourier_A', dtype=tf.float32)
        #self._B = tf.Variable(B, trainable=B_trainable, name='fourier_B', dtype=tf.float32)
        #self._C = tf.Variable(C, trainable=C_trainable, name='fourier_C', dtype=tf.float32)

        # Halo
        # NFW potential.
        # https://docs.galpy.org/en/v1.8.1/reference/potentialnfw.html
        # MWPotential2014: densitynorm=8.48683e6, a=16
        self._halo_logamp = tf.Variable(np.log(halo_amp), trainable=halo_amp_trainable, name='halo_nfw_amplitude', dtype=tf.float32)
        self._halo_loga = tf.Variable(np.log(halo_a), trainable=halo_a_trainable, name='halo_nfw_loga', dtype=tf.float32)

        # Initialize
        self.__call__(tf.zeros([1,n_dim]))

    def __call__(self, q):
        """Returns the gravitational potential of the analytic model"""
        xy,z = tf.split(q, [2,1], axis=1)
        x,y = tf.split(xy, [1,1], axis=1)
        z = z + self._dz
        xy = xy + self._dxy
        R2 = tf.expand_dims(tf.math.reduce_sum(xy**2, axis=1), 1) # w.r.t. galactic centre
        r2 = R2 + z**2 # w.r.t galactic centre
        phi = tf.math.atan2(y, x)

        # Miyamoto-Nagai contribution
        pot_mn = -tf.exp(self._mn_logamp) / tf.math.sqrt(R2 + (tf.math.sqrt(z**2 + tf.math.exp(2*self._mn_logb)) + tf.math.exp(self._mn_loga))**2)
        fourier_mn = 1
        for i in range(self._A.shape[0]):
            fourier_mn += self._A[i]/(1 + i)**2 * tf.math.cos(2*(i + 1)*(phi - self._C))
            fourier_mn += self._B[i]/(1 + i)**2 * tf.math.sin(2*(i + 1)*(phi - self._C))
        pot_mn *= fourier_mn

        # Halo contribution
        pot_halo = -tf.exp(self._halo_logamp)*tf.math.xlogy(1./r2**0.5, 1. + r2**0.5/tf.math.exp(self._halo_loga))

        return pot_mn + pot_halo

    def save_specs(self, spec_name_base):
        """Saves the specs of the model that are required for initialization to a json"""
        d = dict(
            n_dim=self._n_dim,
            name=self._name
        )
        with open(spec_name_base + '_spec.json', 'w') as f:
            json.dump(d, f)

        return spec_name_base

    @classmethod
    def load(cls, checkpoint_name):
        """Load PhiNNAnalyticBarmodel from a checkpoint and a spec file"""
        # Get spec file name
        if checkpoint_name.find('-') == -1 or not checkpoint_name.rsplit('-', 1)[1].isdigit():
            raise ValueError("PhiNNAnalyticBarmodel checkpoint name doesn't follow the correct syntax.")
        spec_name = checkpoint_name.rsplit('-', 1)[0] + "_spec.json"
        
        # Load specs
        with open(spec_name, 'r') as f:
            kw = json.load(f)
        phi_nn = cls(**kw)

        # Restore variables
        checkpoint = tf.train.Checkpoint(phi=phi_nn)
        checkpoint.restore(checkpoint_name).expect_partial() 

        print(f'loaded {phi_nn} from {checkpoint_name}')
        return phi_nn

    @classmethod
    def load_latest(cls, checkpoint_dir):
        """Load the latest PhiNNAnalyticBarmodel from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(f"Couldn't load a valid PhiNNAnalyticBarmodel from {repr(checkpoint_dir)}")
        return PhiNNAnalyticBarmodel.load(latest)

    @classmethod
    def load_checkpoint_with_id(cls, checkpoint_dir, id):
        """Load the PhiNNAnalytic with a specified id from a specified checkpoint directory"""
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise ValueError(f"Couldn't load a valid PhiNNAnalyticBarmodel from {repr(checkpoint_dir)}")
        latest = latest[:latest.rfind('-')] + f'-{id}'
        return PhiNNAnalyticBarmodel.load(latest)