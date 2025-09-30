#!/usr/bin/env python
#
# Plummer sphere distribution function.
# Allows sampling from and evaluation of the DF.
#

import numpy as np

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def get_1d_sampler(p, x_min, x_max, n=1024):
    """
    Given a 1D probability distribution p(x), returns a function that samples from it.

    Parameters
    ----------
    p : function
        Probability distribution function (not necessarily normalized).
    x_min : float
        Minimum x value.
    x_max : float
        Maximum x value.
    n : int
        Number of points to use for interpolation. Default is 100.
    
    Returns
    -------
    sample : function
        Function that takes a shape argument and returns samples from p(x)
        with the given shape.
    """
    x = np.linspace(x_min, x_max, n)
    p_x = p(x)

    P_x = cumulative_trapezoid(p_x, x)
    P_x /= P_x[-1]
    P_x = np.hstack([0., P_x])

    x_of_P = interp1d(P_x, x)

    def sample(shape=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(size=shape)
        return x_of_P(u)

    return sample


def draw_from_sphere(n, rng=None):
    """
    Draws n points uniformly from the unit sphere.

    Parameters
    ----------
    n : int
        Number of points to draw.
    rng : np.random.Generator, optional
        Random number generator to use. If None, uses default_rng().
        Default is None.
    
    Returns
    -------
    points : ndarray, shape (n, 3)
        Points drawn from the unit sphere.
    """
    if rng is None:
        rng = np.random.default_rng()
    phi = np.random.uniform(0., 2*np.pi, size=n)
    theta = np.arccos(rng.uniform(-1., 1., size=n))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x,y,z], axis=1)


class PlummerSphere(object):
    """
    Plummer sphere distribution function.
    """

    def __init__(self):
        self._v_sampler = get_1d_sampler(
            lambda v: v**2 * (1 - v**2 / 2)**(7/2),
            0., np.sqrt(2.)-1.e-8,
            n=1000
        )
        #self._r_sampler = get_1d_sampler(
        #    lambda r: r**2 * (1+r**2)**(-5/2),
        #    0, 10
        #)
        self.df_norm = 24*np.sqrt(2.) / (7*np.pi**3)

    def psi(self, r):
        return 1 / np.sqrt(1 + r**2)

    def phi(self, r):
        return -self.psi(r)

    def rho(self, r):
        return 3/(4*np.pi) * (1+r**2)**(-5/2)
    
    def energy(self, x, v):
        r = np.sqrt(np.sum(x**2, axis=1))
        v2 = np.sum(v**2, axis=1)
        return 0.5*v2 + self.phi(r)

    def sample_r(self, n, r_max=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # return self._r_sampler(n)
        u = rng.uniform(size=n)
        if r_max is not None:
            u_max = r_max**3 / (r_max**2 + 1)**(3/2)
            u *= u_max
        r = 1 / np.sqrt(u**(-2/3) - 1)
        return r

    def sample_df(self, n, r_max=None, rng=None):
        """
        Samples n particles from the Plummer sphere distribution function.
        Returns positions and velocities.
        
        Parameters
        ----------
        n : int
            Number of particles to sample.
        r_max : float, optional
            Maximum radius to sample. If None, samples from 0 to infinity.
            Default is None.
        rng : np.random.Generator, optional
            Random number generator to use. If None, uses default_rng().
            Default is None.
        
        Returns
        -------
        x : ndarray, shape (n, 3)
            Positions of sampled particles.
        v : ndarray, shape (n, 3)
            Velocities of sampled particles.
        """
        r = self.sample_r(n, r_max=r_max, rng=rng)
        x = r[:,None] * draw_from_sphere(n)

        psi = self.psi(r)
        v = np.sqrt(psi) * self._v_sampler(n, rng=rng)
        v = v[:,None] * draw_from_sphere(n)

        return x, v

    def df(self, x, v):
        """
        Evaluates the distribution function at given positions and velocities.

        Parameters
        ----------
        x : ndarray, shape (n, 3)
            Positions.
        v : ndarray, shape (n, 3)
            Velocities.
        
        Returns
        -------
        f : ndarray, shape (n,)
            Distribution function values at the given positions and velocities.
        """
        r = np.sqrt(np.sum(x**2, axis=1))
        v2 = np.sum(v**2, axis=1)
        E = self.psi(r) - 0.5*v2
        return self.df_norm * np.clip(E, 0., np.inf)**(7/2)
