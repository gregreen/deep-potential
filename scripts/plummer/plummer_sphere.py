#!/usr/bin/env python
#
# Plummer sphere distribution function.
# Allows sampling from and evaluation of the DF.
#

import numpy as np
import scipy
import jax.numpy as jnp
import jax

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


class UnitPlummerSphere:
    """
    Unit Plummer sphere distribution function,
    with G = M = a = 1.
    """

    def __init__(self, r_max=None):
        self._v_sampler = get_1d_sampler(
            lambda v: v**2 * (1 - v**2 / 2)**(7/2),
            0., np.sqrt(2.)-1.e-8,
            n=1000
        )
        self.df_norm = 24*jnp.sqrt(2.) / (7*jnp.pi**3)
        self.r_max = r_max
        if r_max is not None:
            self.df_norm *= (r_max**2 + 1)**(3/2) / r_max**3

    def psi(self, r):
        # Relative potential at radius r. (psi = -phi)
        return 1 / jnp.sqrt(1 + r**2)

    def phi(self, r):
        return -self.psi(r)

    def rho(self, r):
        return 3/(4*jnp.pi) * (1 + r**2)**(-5/2)

    def energy(self, x, v):
        r = jnp.sqrt(jnp.sum(x**2, axis=1))
        v2 = jnp.sum(v**2, axis=1)
        return 0.5*v2 + self.phi(r)

    def sample_r(self, n, r_max=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # return self._r_sampler(n)
        u = rng.uniform(size=n)
        if r_max is not None:
            u_max = r_max**3 / (r_max**2 + 1)**(3/2)
            u *= u_max
        r = 1 / jnp.sqrt(u**(-2/3) - 1)
        return r

    def sample_df(self, n, rng=None):
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
        r = self.sample_r(n, r_max=self.r_max, rng=rng)
        x = r[:,None] * draw_from_sphere(n, rng=rng)

        psi = self.psi(r)
        v = np.sqrt(psi) * self._v_sampler(n, rng=rng)
        v = v[:,None] * draw_from_sphere(n, rng=rng)

        return np.stack([x, v], axis=1).reshape(n, 6)

    def df(self, eta):
        """
        Evaluates the distribution function at a given phase-space position.
        """
        x, v = eta[:, :3], eta[:, 3:]
        r = jnp.sqrt(jnp.sum(x**2, axis=1))
        v2 = jnp.sum(v**2, axis=1)
        E = self.psi(r) - 0.5 * v2
        return self.df_norm * jnp.clip(E, 0., jnp.inf)**(7/2)

    def df_spatial(self, x):
        """
        n(x) = \int f(x,v) d^3v
        """
        r = jnp.sqrt(jnp.sum(x**2, axis=1))
        return self.rho(r) / self.mass_total()

    def df_given_x(self, eta):
        """
        Evaluates f(v | x) at given phase-space positions.
        """
        return self.df(eta) / self.df_spatial(eta[:, :3])

    def mass_enclosed(self, r):
        return r**3 / (1 + r**2)**(3/2)

    def v_escape(self, r):
        """
        Escape velocity at radius r.
        """
        return np.sqrt(2 * self.psi(r))

    def mass_total(self):
        if self.r_max is not None:
            return self.mass_enclosed(self.r_max)
        else:
            return 1.0


class PlummerSphere:
    """
    Plummer sphere distribution function with arbitrary
    mass and scale radius.

    Instantiates a UnitPlummerSphere, and scales the outputs
    accordingly, using the following transformations:

      x' = k x,
      v' = gamma v,

    which then imply

      f(x', v') = f(x, v) / (k^3 gamma^3),
      Phi(r') = gamma^2 Phi(r),
      rho(r') = (gamma/k)^2 rho(r),
      M(<r') = k gamma^2 M(<r).

    These transformations ensure that the new distribution function
    and potential satisfy the collisionless Boltzmann equation.

    Parameters
    ----------
    k : float
        Spatial scaling factor.
    gamma : float
        Velocity scale factor.
    """
    def __init__(self, k, gamma, r_max=None):
        r_max_scaled = r_max / k if r_max is not None else None
        self.unit_plummer = UnitPlummerSphere(r_max=r_max_scaled)
        self.r_max = r_max
        self.k = k
        self.gamma = gamma
        self.eta_scaling = jnp.array([k, k, k, gamma, gamma, gamma]) # eta is (x,v)
        self.df_rescale = 1 / (k**3 * gamma**3)
        self.rho_rescale = gamma**2 / k**2
        self.phi_rescale = gamma**2
        self.mass_rescale = k * gamma**2

    @classmethod
    def from_mass_scale(cls, M, a):
        """
        Constructs a Plummer sphere from total mass and scale radius.

        Parameters
        ----------
        M : float
            Total mass.
        a : float
            Scale radius.
        """
        k = a
        gamma = np.sqrt(M / a)
        return cls(k, gamma)

    @classmethod
    def from_density_scale(cls, rho_0, a):
        """
        Constructs a Plummer sphere from central density and scale radius.

        Parameters
        ----------
        rho_0 : float
            Central density.
        a : float
            Scale radius.
        """
        k = a
        gamma = a * np.sqrt(4*np.pi*rho_0/3)
        return cls(k, gamma)

    def psi(self, r):
        return self.unit_plummer.psi(r / self.k) * self.phi_rescale

    def phi(self, r):
        return -self.psi(r)

    def rho(self, r):
        return self.unit_plummer.rho(r / self.k) * self.rho_rescale

    def rho_from_x(self, x):
        r = jnp.linalg.norm(x, axis=1)
        return self.rho(r)

    def phi_from_x(self, x):
        r = jnp.linalg.norm(x, axis=1)
        return self.phi(r)

    def acc_from_x(self, x):
        def phi_from_single_x(x):
            r = jnp.linalg.norm(x)
            return self.phi(r)
        return -jax.vmap(jax.grad(phi_from_single_x))(x)

    def energy(self, x, v):
        r = np.sqrt(np.sum(x**2, axis=1))
        v2 = np.sum(v**2, axis=1)
        return 0.5*v2 + self.phi(r)

    def df(self, eta):
        return self.df_rescale * self.unit_plummer.df(eta / self.eta_scaling)

    def df_gradients(self, eta):
        def df_single(eta_single):
            return self.df(eta_single[None,:]).reshape()
        return jax.vmap(jax.grad(df_single))(eta)

    def df_spatial(self, x):
        return self.unit_plummer.df_spatial(x / self.k) / self.k**3

    def df_spatial_gradients(self, x):
        def n_single(x_single):
            return self.df_spatial(x_single[None,:]).reshape()
        return jax.vmap(jax.grad(n_single))(x)

    def df_given_x(self, eta):
        return self.df(eta) / self.df_spatial(eta[:, :3])

    def df_given_x_gradients(self, eta):
        def fvx_single(eta_single):
            return self.df_given_x(eta_single[None,:]).reshape()
        return jax.vmap(jax.grad(fvx_single))(eta)

    def sample_df(self, n, rng=None):
        eta = self.unit_plummer.sample_df(n, rng=rng)
        eta[:,:3] *= self.k
        eta[:,3:] *= self.gamma
        return eta

    def v_escape(self, r):
        return np.sqrt(2*self.psi(r))

    def mass_enclosed(self, r):
        """
        Mass enclosed within radius r for the scaled Plummer sphere.

        Parameters
        ----------
        r : float
            Radius at which to compute the enclosed mass.

        Returns
        -------
        M_enc : float
            Enclosed mass.
        """
        return self.unit_plummer.mass_enclosed(r / self.k) * self.mass_rescale

    def mass_total(self):
        """
        Total mass of the Plummer sphere.
        """
        return self.unit_plummer.mass_total() * self.mass_rescale


class PlummerSphereSelfn(PlummerSphere):
    def __init__(
        self, k, gamma, r_max=None,
        dustmap_fname=None, dustmap_n_samples=1024*1024, dustmap_batch_size=128*1024, M_mean=5.0, M_sigma=2.0, m_max=18.0
    ):
        super().__init__(k, gamma, r_max=r_max)

        self.dustmap_fname = dustmap_fname
        self.dustmap_n_samples = dustmap_n_samples
        self.dustmap_batch_size = dustmap_batch_size
        self.M_mean = M_mean
        self.M_sigma = M_sigma
        self.m_max = m_max

        # --- Dustmap handling ---
        self.has_dustmap = dustmap_fname is not None
        if self.has_dustmap:
            data = np.load(dustmap_fname)

            self.rho_dust_interp = jax.scipy.interpolate.RegularGridInterpolator(
                data["rho_dust_interp_grid"],
                data["rho_dust_interp_values"],
                bounds_error=False, fill_value=0., method='linear'
            )

            self.integrated_dust_interp = jax.scipy.interpolate.RegularGridInterpolator(
                data["integrated_dust_interp_grid"],
                data["integrated_dust_interp_values"],
                bounds_error=False, fill_value=0., method='linear'
            )

            # --- Estimate normalization factor for selection function ---
            n_batches = int(np.ceil(self.dustmap_n_samples / self.dustmap_batch_size))
            rng = np.random.default_rng(42)
            p_obs_vals = []
            print(f"Estimating the normalization constant of the DF with {self.dustmap_n_samples} samples...")
            for i in range(n_batches):
                n_sample_batch = min(self.dustmap_batch_size, self.dustmap_n_samples - i*self.dustmap_batch_size)
                eta_batch = super().sample_df(n_sample_batch, rng=rng)
                p_obs_batch = self.p_obs(eta_batch[:, :3])
                p_obs_vals.append(p_obs_batch)
            p_obs_vals = np.concatenate(p_obs_vals, axis=0)
            self.df_normalization = 1 / jnp.mean(p_obs_vals)
            print(f"Estimated DF normalization constant: {self.df_normalization}")
        else:
            self.df_normalization = 1.0

    def p_obs(self, x):
        r = jnp.linalg.norm(x, axis=1)
        dm = 5 * jnp.log10(r/0.01) # Assumes r has units of kpc
        if not self.has_dustmap:
            A = 0.
        else:
            A = self.integrated_dust_interp(x)
        p_obs = jnp.exp(jax.scipy.special.log_ndtr((self.m_max - self.M_mean - dm - A) / self.M_sigma))
        return p_obs

    def p_obs_gradients(self, x):
        def p_obs_single(x_single):
            return self.p_obs(x_single[None,:3]).reshape()
        return jax.vmap(jax.grad(p_obs_single))(x)

    def sample_df(self, n, rng=None):
        """
        Samples n particles from the Plummer sphere distribution function,
        accounting for the selection function using rejection sampling.
        Depending on the scale lengths of the Plummer sphere and the dustmap,
        a lot of samples could be rejected (95% or more).
        """
        if rng is None:
            rng = np.random.default_rng()

        samples = []
        n_tries = 0
        n_accepted = 0
        while n_accepted < n:
            eta_batch = super().sample_df(n, rng=rng)
            p_obs_batch = self.p_obs(eta_batch[:, :3])
            u = rng.uniform(size=len(p_obs_batch))
            accepted = eta_batch[(u < p_obs_batch)]
            samples.append(accepted)
            n_tries += len(eta_batch)
            n_accepted += len(accepted)
        samples = jnp.concatenate(samples, axis=0)
        print(f"Acceptance rate: {len(samples)/n_tries:.4f}")
        samples = samples[:n]
        return samples

    def df(self, eta):
        df = super().df(eta) * self.df_normalization
        p_obs = self.p_obs(eta[:,:3])
        return df * p_obs

    def df_gradients(self, eta):
        def df_single(eta_single):
            return self.df(eta_single[None,:]).reshape()
        return jax.vmap(jax.grad(df_single))(eta)

    def df_spatial(self, x):
        df_spatial = super().df_spatial(x) * self.df_normalization
        p_obs = self.p_obs(x)
        return df_spatial * p_obs

    def df_spatial_gradients(self, x):
        def n_single(x_single):
            return self.df_spatial(x_single[None,:]).reshape()
        return jax.vmap(jax.grad(n_single))(x)

    def df_given_x(self, eta):
        df_given_x = super().df_given_x(eta)
        return df_given_x

    def df_given_x_gradients(self, eta):
        def fvx_single(eta_single):
            return self.df_given_x(eta_single[None,:]).reshape()
        return jax.vmap(jax.grad(fvx_single))(eta)
