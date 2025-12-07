#!/usr/bin/env python3

import numpy as np

import jax
import jax.numpy as jnp

import h5py

from astropy_healpix import HEALPix, npix_to_nside
from astropy import units

from tqdm.auto import tqdm

import gc


def gen_gaussian_process(
    shape,                 # (X, Y, Z)
    power_law_index=-3.0,  # alpha in P(k) ∝ k^alpha
    field_variance=1.0,    # target variance of the field
    mean=0.0,              # target mean of the field
    rng=None
):
    """
    Generates a 3D Gaussian random field with a power-law power spectrum.
    This is a numpy implementation that is slow but can generate large fields
    that would not fit in GPU memory.

    Parameters
    ----------
    shape: tuple of ints
        The shape of the output 3D field (n_x, n_y, n_z).
    power_law_index: float
        The power-law index alpha for the power
        spectrum P(k) ∝ k^alpha.
    field_variance: float
        The desired variance of the output field.
    mean: float
        The desired mean of the output field.
    rng: np.random.Generator or None
        Random number generator instance. If None, a new default
        generator is created.
    
    Returns
    -------
    f: np.ndarray
        The generated 3D Gaussian random field.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_x, n_y, n_z = shape

    # 1) Real white noise (mean 0, var 1)
    xi = rng.standard_normal(shape)

    # 2) FFT to k-space (Hermitian symmetry guaranteed because xi is real)
    Xi = np.fft.fftn(xi)

    # 3) Build isotropic k grid and spectral envelope
    kx, ky, kz = np.meshgrid(
        np.fft.fftfreq(n_x, d=1.0),
        np.fft.fftfreq(n_y, d=1.0),
        np.fft.fftfreq(n_z, d=1.0),
        indexing="ij"
    )
    k = np.sqrt(kx**2 + ky**2 + kz**2)

    # Amplitude envelope A(k) so that power ∝ k^alpha
    #   ⇒ |A|^2 ∝ k^alpha
    #   ⇒ A ∝ k^{alpha/2}
    A = np.zeros_like(k, dtype=float)
    mask = (k > 0)
    A[mask] = k[mask]**(0.5 * power_law_index)

    # Zero out DC in the shaped random part
    A[0, 0, 0] = 0.0

    # Apply envelope
    F = A * Xi

    # 4) Back to real space
    f = np.fft.ifftn(F).real

    # 5a) Rescale to target variance (around zero)
    f *= np.sqrt(field_variance / f.var())

    # 5b) Set the mean
    f += mean

    return f


def gen_gaussian_process_jax(
    shape,                 # (X, Y, Z)
    power_law_index=-3.0,  # alpha in P(k) ∝ k^alpha
    field_variance=1.0,    # target variance of the field
    mean=0.0,              # target mean of the field
    rng_key=None
):
    """
    Generates a 3D Gaussian random field with a power-law power spectrum.
    This is a JAX implementation that should be faster, as long as
    the field fits in GPU memory.

    Parameters
    ----------
    shape: tuple of ints
        The shape of the output 3D field (n_x, n_y, n_z).
    power_law_index: float
        The power-law index alpha for the power
        spectrum P(k) ∝ k^alpha.
    field_variance: float
        The desired variance of the output field.
    mean: float
        The desired mean of the output field.
    rng: np.random.Generator or None
        Random number generator instance. If None, a new default
        generator is created.
    
    Returns
    -------
    f: np.ndarray
        The generated 3D Gaussian random field.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    elif isinstance(rng_key, int):
        rng_key = jax.random.PRNGKey(rng_key)
    
    n_x, n_y, n_z = shape

    # 1) Real white noise (mean 0, var 1)
    xi = jax.random.normal(rng_key, shape)

    # 2) FFT to k-space (Hermitian symmetry guaranteed because xi is real)
    Xi = jnp.fft.fftn(xi)

    # 3) Build isotropic k grid and spectral envelope
    kx, ky, kz = jnp.meshgrid(
        jnp.fft.fftfreq(n_x, d=1.0),
        jnp.fft.fftfreq(n_y, d=1.0),
        jnp.fft.fftfreq(n_z, d=1.0),
        indexing="ij"
    )
    k = jnp.sqrt(kx**2 + ky**2 + kz**2)

    # Amplitude envelope A(k) so that power ∝ k^alpha
    #   ⇒ |A|^2 ∝ k^alpha
    #   ⇒ A ∝ k^{alpha/2}
    A = jnp.zeros_like(k)
    mask = (k > 0)
    A = A.at[mask].set(k[mask]**(0.5 * power_law_index))

    # Zero out DC in the shaped random part
    A = A.at[0, 0, 0].set(0.0)

    # Apply envelope
    F = A * Xi

    # 4) Back to real space
    f = jnp.fft.ifftn(F).real
    f = jnp.array(f)

    # 5) Rescale to target variance and set mean
    f = (f - jnp.mean(f)) / jnp.std(f) * jnp.sqrt(field_variance) + mean
    
    return f


def gen_field_interpolator(field, extent, method='linear',
                           fill_value=0., use_jax=False):
    """
    Generates an interpolator for a 3D field defined on a regular grid.

    Parameters
    ----------
    field : ndarray
        3D array of shape (n_x, n_y, n_z) representing the field values on a
        regular grid.
    extent : list of tuples
        [(x_min, x_max), (y_min, y_max), (z_min, z_max)] defining the physical
        extent of the grid.
    method : str, optional
        Interpolation method, one of 'linear', 'nearest', 'slinear', 'cubic',
        'quintic' or 'pchip'. Default is 'linear'.
    fill_value : float, optional
        Value to use for points outside the interpolation domain.
        Default is 0.0.

    Returns
    -------
    interpolator : function
        A function that takes an array of shape (M, 3) with (x, y, z)
        coordinates and returns the interpolated field values at those points.
    """
    if use_jax:
        import jax.numpy as jnp
        from jax.scipy.interpolate import RegularGridInterpolator
    else:
        from scipy.interpolate import RegularGridInterpolator

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = extent
    n_x, n_y, n_z = field.shape

    # Create grid points
    if use_jax:
        x = jnp.linspace(x_min, x_max, n_x)
        y = jnp.linspace(y_min, y_max, n_y)
        z = jnp.linspace(z_min, z_max, n_z)
    else:
        x = np.linspace(x_min, x_max, n_x)
        y = np.linspace(y_min, y_max, n_y)
        z = np.linspace(z_min, z_max, n_z)

    # Create the interpolator
    interpolator = RegularGridInterpolator(
        (x, y, z),
        field,
        bounds_error=False,
        fill_value=fill_value,
        method=method
    )

    return interpolator


def integrate_field_radially(field_interpolator, xyz, n_steps=128):
    """
    Integrates a field radially out to the locations (x,y,z), taking
    a given number of steps along each ray.

    Parameters
    ----------
    field_interpolator : function
        A function that takes an array of shape (M, 3) with (x, y, z)
        coordinates and returns the interpolated field values at those points.
    xyz : ndarray
        An array of shape (N, 3) containing the (x, y, z) coordinates of the
        points where the integration is to be performed.
    n_steps : int, optional
        The number of steps to use in the radial integration. Default is 128.
    
    Returns
    -------
    integral : ndarray
        An array of shape (N,) containing the integrated field values along
        the radial paths to each point in `xyz`.
    """

    r_max = np.linalg.norm(xyz, axis=1)
    xyz_unit = xyz / r_max[:,None]
    dr = r_max / n_steps

    xyz_current = np.zeros_like(xyz)
    integral = np.zeros(xyz.shape[0], dtype='f8')
    
    from tqdm.notebook import tqdm

    for i in tqdm(range(n_steps)):
        xyz_current = xyz_unit * dr[:,None] * (i+0.5)
        integral += dr * field_interpolator(xyz_current)
    
    return integral


def save_interpolator(interp, fn):
    """
    Saves a RegularGridInterpolator to an HDF5 file.

    Parameters
    ----------
    interp : RegularGridInterpolator
        The interpolator to save.
    fn : str
        Filename of the HDF5 file.
    """
    with h5py.File(fn, 'w') as f:
        f.create_dataset(
            'values',
            data=interp.values,
            chunks=True,
            compression='gzip',
            compression_opts=5
        )
        for i, grid in enumerate(interp.grid):
            f.create_dataset(f'grid_{i}', data=grid)
        f.attrs['method'] = interp.method
        f.attrs['fill_value'] = interp.fill_value


def load_interpolator(fn, jax=False):
    """
    Loads a RegularGridInterpolator from an HDF5 file.

    Parameters
    ----------
    fn : str
        Filename of the HDF5 file.
    jax : bool, optional
        Whether to use JAX for the interpolator. Default is False.
    
    Returns
    -------
    interp : RegularGridInterpolator
        The loaded interpolator.
    """
    if jax:
        from jax.scipy.interpolate import RegularGridInterpolator
    else:
        from scipy.interpolate import RegularGridInterpolator
    
    with h5py.File(fn, 'r') as f:
        values = f['values'][:]
        grid = [f[f'grid_{i}'][:] for i in range(3)]
        method = f.attrs['method']
        fill_value = f.attrs['fill_value']
    
    if jax:
        values = jnp.array(values)
        grid = tuple(jnp.array(g) for g in grid)
    
    interp = RegularGridInterpolator(
        grid,
        values,
        bounds_error=False,
        fill_value=fill_value,
        method=method
    )
    
    return interp


class HealpixRadialInterpolator(object):
    """
    Interpolator for 3D fields, stored internally using a HEALPix map (at
    constant angular resolution) at each radial distance.

    To determine the value at a given (x,y,z) point, linear interpolation is
    first conducted in the radial direction to determine the field values in
    the 4 nearest HEALPix pixels, and then bilinear interpolation is done on
    the sphere to obtain the final value. This is not the most principled
    interpolation scheme, but it is fast and simple to implement.
    """

    def __init__(self, values, r_max):
        """
        Creates an interpolator for the 3D field.

        Parameters
        ----------
        values : jnp.ndarray, shape (n_pix, n_r_steps)
            Field values on the healpix grid.
        r_max : float
            Maximum radius of grid.
        """
        n_ang,n_r = values.shape
        self.nside = npix_to_nside(n_ang)
        self.n_r_steps = n_r
        self.r_max = r_max
        self.values = values
        self.hpix = HEALPix(nside=self.nside, order='nested')

    # Create interpolator that takes (x,y,z) and returns integral at that point
    def __call__(self, xyz_query):
        """
        Interpolates the integrated field values at the given query points.

        Parameters
        ----------
        xyz_query : jnp.ndarray, shape (N, 3)
            Query points in Cartesian coordinates.
        
        Returns
        -------
        v_interp : jnp.ndarray, shape (N,)
            Interpolated field values at the query points.
        """
        r = jnp.linalg.norm(xyz_query, axis=1)
        lat = jnp.arcsin(jnp.clip(xyz_query[:,2] / r, -1.0, 1.0))
        lon = jnp.arctan2(xyz_query[:,1], xyz_query[:,0]) % (2*jnp.pi)

        # Get angular interpolation weights
        hp_idx,hp_weight = self.hpix.bilinear_interpolation_weights(
            lon*units.rad,
            lat*units.rad
        )
        # hp_idx and hp_weight have shape (4, n_query)

        # Remove negative values in hp_weight (when there are 3 neighbors)
        hp_weight = jnp.maximum(hp_weight, 0.0)
        hp_weight = hp_weight / jnp.sum(hp_weight, axis=0, keepdims=True)

        # Interpolate in radius using linear interpolation
        r_indices = r / self.r_max * (self.n_r_steps - 1)
        r_indices_clipped = jnp.clip(r_indices, 0, self.n_r_steps - 1 - 1e-6)
        r_idx_lower = jnp.floor(r_indices_clipped).astype(int)
        r_idx_upper = r_idx_lower + 1
        t = r_indices_clipped - r_idx_lower

        # Loop over the 4 angular neighbors and combine
        for i in range(4):
            v_lower = self.values[hp_idx[i], r_idx_lower]
            v_upper = self.values[hp_idx[i], r_idx_upper]
            if i == 0:
                v_interp = hp_weight[i] * ((1 - t) * v_lower + t * v_upper)
            else:
                v_interp += hp_weight[i] * ((1 - t) * v_lower + t * v_upper)
        
        return v_interp

    def save(self, fname):
        """
        Saves the interpolator to an HDF5 file.
        """
        import h5py

        with h5py.File(fname, 'w') as f:
            f.create_dataset('values', data=np.array(self.values))
            f.attrs['r_max'] = self.r_max
    
    @classmethod
    def load(cls, fname):
        """
        Loads the interpolator from an HDF5 file.
        """
        import h5py

        with h5py.File(fname, 'r') as f:
            values = jnp.array(f['values'][:])
            r_max = f.attrs['r_max']
        
        return cls(values, r_max)


def integrate_field_radially_healpix_jax(
    field_interpolator,
    nside,
    r_max,
    n_r_steps=128
):
    """
    JAX radial integration on a healpix angular grid.

    Parameters
    ----------
    field_interpolator : callable
        Function taking (N,3) array -> (N,) field values.
    nside : int
        Healpix nside parameter defining angular resolution.
    r_max : float
        Maximum radius to integrate out to.
    n_r_steps : int
        Number of radial steps.

    Returns
    -------
    radial_integrals : jnp.ndarray, shape (n_pix, n_r_steps)
        Integrated field values on the healpix grid.
    """
    from astropy_healpix import HEALPix
    from astropy import units

    hpix = HEALPix(nside=nside, order='nested')

    lon,lat = hpix.healpix_to_lonlat(np.arange(hpix.npix))
    lon = lon.to_value(units.rad)
    lat = lat.to_value(units.rad)
    xyz_dirs = jnp.array([
        jnp.cos(lat) * jnp.cos(lon),
        jnp.cos(lat) * jnp.sin(lon),
        jnp.sin(lat)
    ]).T  # (n_pix, 3)
    r_vals = jnp.linspace(0, r_max, n_r_steps)

    # Outer product of angles and radii to get target points
    xyz_targets = xyz_dirs[:,None,:]*r_vals[None,:,None] # (healpix_index,r_index,3)

    # Evaluate density at grid points
    density = field_interpolator(xyz_targets.reshape(-1, 3))

    # Reshape to (healpix_index, r_index)
    density = density.reshape(hpix.npix, n_r_steps)
    
    # Implement trapezoid rule at every distance
    integral_at_r = (
        jnp.cumsum(density, axis=1)
        - 0.5*(density[:,0:1]+density)
    ) * (r_max / n_r_steps) # (n_pix, n_r_steps)

    # Create interpolator
    integral_interpolator = HealpixRadialInterpolator(
        integral_at_r,
        r_max
    )
    
    return integral_interpolator


def gen_mock_dust_map_naive(
    extent,
    n_modes,
    power_law_index=-3.0,
    mean_ln_density=-1.0,
    ln_density_variance=1.0,
    n_integration_steps=256,
    rng=None
):
    """
    Generates a mock dust map by creating a Gaussian random field
    with the specified properties. This is a "naive" implementation that
    does not use JAX, and which stores the integrated dust map on a Cartesian
    grid. It is memory-intensive and slow, but simple.

    Parameters
    ----------
    extent : list of tuples
        [(x_min, x_max), (y_min, y_max), (z_min, z_max)] defining the physical
        extent of the grid.
    n_modes : tuple of ints
        Number of Fourier modes in each dimension (n_x, n_y, n_z).
    power_law_index : float, optional
        Power-law index for the power spectrum of the log density field.
        Default is -3.0.
    mean_ln_density : float, optional
        Mean of the log density field. Default is -1.0.
    ln_density_variance : float, optional
        Variance of the log density field. Default is 1.0.
    n_integration_steps : int, optional
        Number of steps to use in the radial integration. Default is 256.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None (the default), a
        new generator is created.
    
    Returns
    -------
    density_interpolator : RegularGridInterpolator
        Interpolator for the 3D density field.
    integrated_dust_interpolator : RegularGridInterpolator
        Interpolator for the 3D extinction map.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Create a Gaussian random field representing the log density
    print('Generating Gaussian random field for ln(density)...')
    ln_density = gen_gaussian_process(
        shape=n_modes,
        power_law_index=power_law_index,
        field_variance=ln_density_variance,
        mean=mean_ln_density,
        rng=rng
    )

    # Create interpolator for the (linear) density field
    print('Interpolating density field...')
    density_interpolator = gen_field_interpolator(
        np.exp(ln_density),
        extent=extent,
        method='linear',
        use_jax=False
    )

    # Integrate the field to create the integrated dust map
    # First, create (x,y,z) grid at which to evaluate the integrated map
    print('Integrating density field radially...')
    img_shape = [3*n for n in n_modes]
    xyz_grid = np.stack(np.meshgrid(*[
        np.linspace(x0,x1,s) for (x0,x1),s in zip(extent, img_shape)
    ], indexing='ij'), axis=-1)  # shape (Nx, Ny, Nz, 3)
    xyz_grid.shape = (-1, 3)  # shape (N, 3)
    integrated_dust_map = integrate_field_radially(
        density_interpolator,
        xyz_grid,
        n_steps=n_integration_steps
    )
    integrated_dust_map.shape = img_shape

    # Create interpolator for the integrated dust map
    print('Interpolating integrated dust map...')
    integrated_dust_interpolator = gen_field_interpolator(
        integrated_dust_map,
        extent=extent,
        method='linear',
        use_jax=False
    )

    # Return both interpolators
    return density_interpolator, integrated_dust_interpolator


def gen_mock_dust_map_healpix(
    extent,
    n_modes,
    nside,
    power_law_index=-3.0,
    mean_ln_density=-1.0,
    ln_density_variance=1.0,
    n_integration_steps=256,
    rng_key=None
):
    """
    Generates a mock dust map by creating a Gaussian random field
    with the specified properties. This is a "naive" implementation that
    does not use JAX, and which stores the integrated dust map on a Cartesian
    grid. It is memory-intensive and slow, but simple.

    Parameters
    ----------
    extent : list of tuples
        [(x_min, x_max), (y_min, y_max), (z_min, z_max)] defining the physical
        extent of the grid.
    n_modes : tuple of ints
        Number of Fourier modes in each dimension (n_x, n_y, n_z).
    power_law_index : float, optional
        Power-law index for the power spectrum of the log density field.
        Default is -3.0.
    mean_ln_density : float, optional
        Mean of the log density field. Default is -1.0.
    ln_density_variance : float, optional
        Variance of the log density field. Default is 1.0.
    n_integration_steps : int, optional
        Number of steps to use in the radial integration. Default is 256.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None (the default), a
        new generator is created.
    
    Returns
    -------
    density_interpolator : RegularGridInterpolator
        Interpolator for the 3D density field.
    integrated_dust_interpolator : HealpixRadialInterpolator
        Interpolator for the 3D extinction map.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    elif isinstance(rng_key, int):
        rng_key = jax.random.PRNGKey(rng_key)

    # Create a Gaussian random field representing the log density
    print('Generating Gaussian random field for ln(density)...')
    ln_density = gen_gaussian_process_jax(
        shape=n_modes,
        power_law_index=power_law_index,
        field_variance=ln_density_variance,
        mean=mean_ln_density,
        rng_key=rng_key
    )

    # Create interpolator for the (linear) density field
    print('Interpolating density field...')
    density_interpolator = gen_field_interpolator(
        jnp.exp(ln_density),
        extent=extent,
        method='linear',
        use_jax=True
    )

    # Integrate the field to create the integrated dust map
    print('Integrating density field radially...')
    integrated_dust_interpolator = integrate_field_radially_healpix_jax(
        density_interpolator,
        nside,
        r_max=extent[0][1],  # assuming cubic box
        n_r_steps=n_integration_steps
    )

    # Return both interpolators
    return density_interpolator, integrated_dust_interpolator


def main():

    # Example usage of the mock dust map generation functions
    extent = [(-8.,8.)] * 3
    n_modes = (32,) * 3
    nside = 32

    rng = np.random.default_rng()
    test_xyz = rng.uniform(-8., 8., size=(64, 3))

    print("Generating mock dust map using naive method...")
    rho_interp, A_interp = gen_mock_dust_map_naive(
        extent,
        n_modes,
        n_integration_steps=64
    )
    print("Saving interpolators...")
    save_interpolator(rho_interp, 'example_rho_dust_naive.h5')
    save_interpolator(A_interp, 'example_A_dust_naive.h5')
    print("Testing loading interpolators...")
    rho_loaded = load_interpolator('example_rho_dust_naive.h5')
    A_loaded = load_interpolator('example_A_dust_naive.h5')
    print("Evaluating loaded interpolators...")
    rho_0 = rho_loaded(test_xyz)
    rho_1 = rho_interp(test_xyz)
    A_0 = A_loaded(test_xyz)
    A_1 = A_interp(test_xyz)
    np.testing.assert_allclose(rho_0, rho_1, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(A_0, A_1, atol=1e-5, rtol=1e-5)
    print("Done.")

    print("Generating mock dust map using HEALPix method...")
    rho_interp, A_interp = gen_mock_dust_map_healpix(
        extent,
        n_modes,
        nside,
        n_integration_steps=64
    )
    print("Saving interpolators...")
    save_interpolator(rho_interp, 'example_rho_dust_healpix.h5')
    A_interp.save('example_A_dust_healpix.h5')
    print("Testing loading interpolators...")
    rho_loaded = load_interpolator('example_rho_dust_healpix.h5')
    A_loaded = HealpixRadialInterpolator.load('example_A_dust_healpix.h5')
    print("Evaluating loaded interpolators...")
    rho_0 = rho_loaded(test_xyz)
    rho_1 = rho_interp(test_xyz)
    A_0 = A_loaded(test_xyz)
    A_1 = A_interp(test_xyz)
    np.testing.assert_allclose(rho_0, rho_1, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(A_0, A_1, atol=1e-5, rtol=1e-5)
    print("Done.")

if __name__ == '__main__':
    main()