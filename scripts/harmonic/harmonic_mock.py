#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import h5py


def sample_normal(rng, n, sigma, mu):
    return rng.normal(loc=mu, scale=sigma, size=n)

def sample_exponential(rng, n, x0):
    return rng.exponential(size=n, scale=x0)

def gen_mock_data(n, r_c, omega, sigma_r, omega_z, sigma_z, r_inner, r_outer, seed=5):
    """
    Mock data for the potential
    phi(x, y, z) = omega^2 * ((x - r_c)^2/2 + y^2)/2 +
                   omega_z^2 * z^2/2.

    This is a harmonic potential in x-y-z.
    The mock data is generated for r_inner <= (x^2+y^2+z^2)^0.5 <= r_outer (imitating DR3)
    for a rotating clump of stars with a gaussian profile in x-y-z-vx-vy-vz.
    The clump is stationary in a rotating frame (0,0,omega).
    
    The distribution function is
    f(x, y, z, vx, vy, vz) = g(x, sigma_r) *
                             g(y, sigma_r) *
                             g(z, sigma_r) *
                             g(vx, |omega|*sigma_r) *
                             g(vy - omega*r_c, |omega|*sigma_r) *
                             g(vz, |omega_z|*sigma_z),
    where g(x, sigma_r) = exp(-x^2/(2sigma_r^2))/sqrt(2*pi)/sigma_r is the normal distr.
    """
    
    rng = np.random.default_rng(seed=seed)
    
    # Generate mock data while respecting the radial cuts
    eta = np.zeros((0, 6))
    ps = []
    while len(eta) <= n:
        new_eta = np.zeros((n, 6))
        new_eta[:, 0] = sample_normal(rng, n, sigma_r, 0) #                        x
        new_eta[:, 1] = sample_normal(rng, n, sigma_r, 0) #                        y
        new_eta[:, 2] = sample_normal(rng, n, sigma_z, 0) #                        z
        new_eta[:, 3] = sample_normal(rng, n, np.abs(omega)*sigma_r, 0) #          v_x
        new_eta[:, 4] = sample_normal(rng, n, np.abs(omega)*sigma_r, -omega*r_c) #  v_y
        new_eta[:, 5] = sample_normal(rng, n, np.abs(omega_z)*sigma_z, 0) #        v_z

        r2 = np.sum(new_eta[:, :3]**2, axis=1)
        idx = (r_inner**2 <= r2) & (r2 <= r_outer**2)
        ps.append(np.sum(~idx)/len(idx))
        new_eta = new_eta[idx]
        eta = np.concatenate((eta, new_eta))
    print(f'{np.mean(ps):.3f} of generated datapoints fell outside the radial cuts')
    eta = eta[:n, :]

    return eta

def noisify(eta, error_r, error_v, seed=1301):
    rng = np.random.default_rng(seed)
    
    eta_noisy = np.zeros(eta.shape, dtype='f8')
    eta_noisy[:,:3] = eta[:,:3] + error_r * rng.normal(size=(eta.shape[0], 3))
    eta_noisy[:,3:] = eta[:,3:] + error_v * rng.normal(size=(eta.shape[0], 3))

    return eta_noisy


def save_selection(eta, t_snap, fname, n_max=None):
    x0 = np.array([1., 0., 0.])
    r0 = 1.

    r2 = np.sum((eta[:,-1,:3] - x0[None,:])**2, axis=1)
    idx = (r2 < r0**2)

    eta_sel = eta[idx,-1]

    if n_max is not None:
        if eta_sel.shape[0] < n_max:
            print('Not as many sources as requested in selection!')
        eta_sel = eta_sel[:n_max]

    print(f'{len(eta_sel)} particles selected.')

    with h5py.File(fname, 'w') as f:
        dset = f.create_dataset('eta', data=eta_sel, compression='lzf')
        dset.attrs['axes'] = ('particle', 'dimension')
        dset.attrs['time'] = t_snap[-1]
        dset.attrs['dimensions'] = ('x', 'y', 'z', 'vx', 'vy', 'vz')

    return eta_sel


def save_nonstationary(eta, t_snap, fname, t_idx=4):
    t_nonstationary = t_snap[t_idx]
    eta_nonstationary = eta[:,t_idx]
    print(f'Using t = {t_nonstationary}.')

    with h5py.File(fname, 'w') as f:
        dset = f.create_dataset('eta', data=eta_nonstationary, compression='lzf')
        dset.attrs['axes'] = ('particle', 'dimension')
        dset.attrs['time'] = t_nonstationary
        dset.attrs['dimensions'] = ('x', 'y', 'z', 'vx', 'vy', 'vz')

def radial_cut(eta, r_inner, r_outer, n):
    r2 = np.sum(eta[:, :3]**2, axis=1)
    idx = (r_inner**2 <= r2) & (r2 <= r_outer**2)
    return eta[idx][:n]

def main():
    n = 2**21          # 131072 stars
    r_c = 8.3          # 8.3 kpc       literature
    omega = -0.25      # -25 km/s/kpc  taken from one of the rotating fits for DR3
    sigma_r = 0.45     # 400 pc        chosen by gut feeling
    sigma_z = 0.35     # 0.35 kp       literature
    omega_z = -0.4
    r_inner = 0.063    # 63 pc         inner limit of pop3 from DR3  (2.0 < M_G < 4.0)
    r_outer = 3.0#0.398    # 398 pc        outer limit of pop3 from DR3  (2.0 < M_G < 4.0)


    error_r = 0.01 # 10 pc
    error_v = 0.05 # 10 km/s

    #u = (-0.128, -2.222, -0.077)  # Constant offset to apply to the mock data
    u = (0, 0, 0)

    data_dir = 'data/'

    # Generate the mock data
    eta = gen_mock_data(int(1.2*n), r_c=r_c, omega=omega, sigma_r=sigma_r, omega_z=omega_z, sigma_z=sigma_z, r_inner=r_inner - error_r*5, r_outer=r_outer + error_r*5)
    for i in range(3):
        eta[:, 3+i] += u[i]
    
    # Save to HDF5
    with h5py.File(f'{data_dir}s_Harmonic.h5', 'w') as f:
        dset = f.create_dataset('eta', data=radial_cut(eta, r_inner, r_outer, n), compression='lzf')
        dset.attrs['r_c'] = r_c
        dset.attrs['sigma_r'] = sigma_r
        dset.attrs['sigma_z'] = sigma_z
        dset.attrs['omega'] = omega
        dset.attrs['omega_z'] = omega_z
        dset.attrs['r_inner'] = r_inner
        dset.attrs['r_outer'] = r_outer
        dset.attrs['parallax_min'] = 1/r_outer
        dset.attrs['parallax_max'] = 1/r_inner
        dset.attrs['spatial_unit'] = 'kpc'
        dset.attrs['velocity_unit'] = 'km/cs'
    print('generated and saved mock data')

    # Generate noisy eta. Small caveat is that the noise might bring datapoints outsided of radial cuts
    noisy_eta = noisify(eta, error_r, error_v)
    
    # Save to HDF5
    with h5py.File(f'{data_dir}s_Harmonic_noisy.h5', 'w') as f:
        dset = f.create_dataset('eta', data=radial_cut(noisy_eta, r_inner, r_outer, n), compression='lzf')
        dset.attrs['r_c'] = r_c
        dset.attrs['sigma_r'] = sigma_r
        dset.attrs['sigma_z'] = sigma_z
        dset.attrs['omega'] = omega
        dset.attrs['omega_z'] = omega_z
        dset.attrs['r_inner'] = r_inner
        dset.attrs['r_outer'] = r_outer
        dset.attrs['parallax_min'] = 1/r_outer
        dset.attrs['parallax_max'] = 1/r_inner
        dset.attrs['spatial_unit'] = 'kpc'
        dset.attrs['velocity_unit'] = 'km/cs'
    print('generated and saved noisy mock data')



    n = 2**17         # 131072 stars
    r_c = 8.3         # 8.3 kpc        literature
    sigma_r = 4.5     # 4.50 kpc       big gaia sample
    sigma_z = 0.35    # 350 pc         big gaia sample
    omega = -0.25     # -25 km/s/kpc   taken from one of the rotating fits for DR3
    omega_z = -0.4
    r_inner = 0.63    # 630 pc         big gaia sample
    r_outer = 3.98    # 3.98 kpc       big gaia sample

    u = (-0.128, -2.222, -0.077)  # Constant offset to apply to the mock data

    # Generate the mock data
    eta = gen_mock_data(int(1.2*n), r_c=r_c, omega=omega, sigma_r=sigma_r, omega_z=omega_z, sigma_z=sigma_z, r_inner=r_inner - error_r*5, r_outer=r_outer + error_r*5)
    for i in range(3):
        eta[:, 3+i] += u[i]

    # Save to HDF5
    with h5py.File(f'{data_dir}Harmonic_big.h5', 'w') as f:
        dset = f.create_dataset('eta', data=radial_cut(eta, r_inner, r_outer, n), compression='lzf')
        dset.attrs['r_c'] = r_c
        dset.attrs['sigma_r'] = sigma_r
        dset.attrs['sigma_z'] = sigma_z
        dset.attrs['omega'] = omega
        dset.attrs['omega_z'] = omega_z
        dset.attrs['r_inner'] = r_inner
        dset.attrs['r_outer'] = r_outer
        dset.attrs['parallax_min'] = 1/r_outer
        dset.attrs['parallax_max'] = 1/r_inner
        dset.attrs['spatial_unit'] = 'kpc'
        dset.attrs['velocity_unit'] = 'km/cs'
    print('generated and saved mock data')

    # Generate noisy eta. Small caveat is that the noise might bring datapoints outsided of radial cuts
    noisy_eta = noisify(eta, error_r, error_v)
    
    # Save to HDF5
    with h5py.File(f'{data_dir}Harmonic_noisy_big.h5', 'w') as f:
        dset = f.create_dataset('eta', data=radial_cut(noisy_eta, r_inner, r_outer, n), compression='lzf')
        dset.attrs['r_c'] = r_c
        dset.attrs['sigma_r'] = sigma_r
        dset.attrs['sigma_z'] = sigma_z
        dset.attrs['omega'] = omega
        dset.attrs['omega_z'] = omega_z
        dset.attrs['r_inner'] = r_inner
        dset.attrs['r_outer'] = r_outer
        dset.attrs['parallax_min'] = 1/r_outer
        dset.attrs['parallax_max'] = 1/r_inner
        dset.attrs['spatial_unit'] = 'kpc'
        dset.attrs['velocity_unit'] = 'km/cs'
    print('generated and saved noisy mock data')

    return 0

if __name__ == '__main__':
    main()

