#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import ticker as mticker

import galpy.potential as gpot
from galpy.orbit import Orbit

import h5py
import os.path
from tqdm import tqdm

from flow_analysis_axisymm import cart2cyl


def gen_mock_data(n, batch_size=128, seed=5):
    pot = gpot.MiyamotoNagaiPotential(amp=1., a=1., b=0.1)

    # Initial scale radius and height of tracer particles
    hr, hz = 1., 0.1

    # Integrator settings
    n_tdyn = 128            # Number of dynamical times to integrate system
    n_per_tdyn = 16         # Minimum number of steps per dynamical time
    n_steps_max = 1024*64   # Maximum number of steps (circuit-breaker)

    # Create empty array to hold snapshots
    dtype = [(k,'f4') for k in ('R','vR','vT','z','vz')]
    t_snap = [0, 1]
    while t_snap[-1] < n_tdyn:
        t_snap.append(t_snap[-1] * 2)

    n_batches = n // batch_size
    x = np.empty((n_batches, batch_size, len(t_snap)), dtype=dtype)

    # Draw the initial conditions
    rng = np.random.default_rng(seed)

    R = rng.gamma(2, scale=hr, size=n)
    v_c = pot.vcirc(R)
    v_R = 0.05 * v_c * rng.normal(size=n)
    v_T = v_c * (1. - 0.1 * np.abs(rng.normal(size=n)))
    #v_T = v_c * (1. + 0.1 * rng.normal(size=n))
    z = rng.exponential(hz, n) * rng.choice([-1,1], n)
    v_z = 0.05 * v_c * rng.normal(size=n)

    # [R,vR,vT,z,vz,phi]
    x0 = np.stack([R,v_R,v_T,z,v_z], axis=1)

    # Calculate orbital timescales
    T = 2*np.pi / np.vstack([
            pot.verticalfreq(R),  # Vertical frequency
            pot.epifreq(R),       # Epicyclic frequency
            pot.omegac(R)         # Orbital frequency
    ])
    tdyn_min = np.min(T, axis=0)
    tdyn_max = np.max(T, axis=0)

    # Sort by orbital timescale
    idx = np.argsort(tdyn_min)
    x0 = x0[idx]
    tdyn_min = tdyn_min[idx]
    tdyn_max = tdyn_max[idx]

    # Loop over batches of points
    x0.shape = (n_batches, batch_size, 5)
    tdyn_min.shape = (n_batches, batch_size)
    tdyn_max.shape = (n_batches, batch_size)

    #xt_batches = []
    #t_batches = []

    for i,(xb,T0,T1) in tqdm(enumerate(zip(x0,tdyn_min,tdyn_max)), total=n_batches):
        o = Orbit(xb)

        dt = np.nanpercentile(T0, 5.) / n_per_tdyn
        T_max = n_tdyn * np.max(T1)
        nt = T_max / dt
        if nt > n_steps_max:
            dt = T_max / n_steps_max
            nt = n_steps_max

        #print(f'Using T_max, dt = ({T_max:.5g}, {dt:.5g}) : {nt:.0f} steps.')

        t = np.arange(0., T_max+0.5*dt, 4*dt)

        o.integrate(t, pot, method='symplec6_c', dt=dt)

        xt = o.getOrbit()
        #xt_batches.append(o.getOrbit())
        #t_batches.append(t)

        for j,ts in enumerate(t_snap):
            t_idx = int(ts / n_tdyn * (xt.shape[1] - 1))
            for k_idx,(key,_) in enumerate(dtype):
                x[key][i,:,j] = xt[:,t_idx,k_idx]

    #for i,xt in enumerate(xt_batches):
    #    for j,t in enumerate(t_snap):
    #        for k_idx,(key,_) in enumerate(dtype):
    #            x[key][i,:,j] = xt[:,t_idx,k_idx]

    x.shape = (n_batches*batch_size, -1)

    # Save Cartesian representation of snapshots
    phi = rng.uniform(0., 2*np.pi, size=x.shape) # Random azimuth

    eta = np.empty(x.shape + (6,))
    eta[...,0] = x['R'] * np.cos(phi)
    eta[...,1] = x['R'] * np.sin(phi)
    eta[...,2] = x['z']
    eta[...,3] = x['vR'] * np.cos(phi) - x['vT'] * np.sin(phi)
    eta[...,4] = x['vR'] * np.sin(phi) + x['vT'] * np.cos(phi)
    eta[...,5] = x['vz']

    # Randomly shuffle input, so that it's not ordered by dynamical time
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    x = x[idx]
    eta = eta[idx]

    return x, eta, t_snap


def save_full(x, eta, t_snap, fname_cyl, fname_cart):
    with h5py.File(fname_cyl, 'w') as f:
        dset = f.create_dataset('x', data=x, compression='lzf')
        dset.attrs['axes'] = ('particle', 'time')
        dset.attrs['times'] = t_snap


    with h5py.File(fname_cart, 'w') as f:
        dset = f.create_dataset('eta', data=eta, compression='lzf')
        dset.attrs['axes'] = ('particle', 'time', 'dimension')
        dset.attrs['times'] = t_snap
        dset.attrs['dimensions'] = ('x', 'y', 'z', 'vx', 'vy', 'vz')


def load_cyl(fname):
    with h5py.File(fname, 'r') as f:
        x = f['x'][:]
        t_snap = f['x'].attrs['times'][:]
    return x, t_snap


def load_cart(fname):
    with h5py.File(fname, 'r') as f:
        eta = f['eta'][:]
        t_snap = f['eta'].attrs['times'][:]
    return eta, t_snap


def save_stationary(eta, t_snap, fname):
    with h5py.File(fname, 'w') as f:
        dset = f.create_dataset('eta', data=eta[:,-1], compression='lzf')
        dset.attrs['axes'] = ('particle', 'dimension')
        dset.attrs['time'] = t_snap[-1]
        dset.attrs['dimensions'] = ('x', 'y', 'z', 'vx', 'vy', 'vz')


def save_noisy(eta, t_snap, fname, seed=1301):
    rng = np.random.default_rng(seed)

    # Observer position: (x,y,z)
    x0 = np.array([1,0,0])

    # Distance from observer to each star
    eta0 = np.hstack([x0, np.zeros_like(x0)])
    eta_cent = eta[:,-1,:] - eta0[None,:]
    r = np.linalg.norm(eta_cent[:,:3], axis=1)

    # Distance error: similar to 0.1 mas in MW, but clipped at 10%
    sigma_r = np.clip(0.1 * r**2, 0., 0.1*r)
    # RV error: similar to 0.5 km/s in MW (same fraction of circular velocity)
    sigma_vr = 0.5 / 200 * 0.6
    # v_T error: 1/4000th of circular velocity at r = 1 (similar to 0.1 mas/yr)
    sigma_vt = 0.6 / 4000 * r

    # Draw position & velocity errors in spherical basis
    n = eta.shape[0]

    dx_sph = np.zeros((n,3))
    dx_sph[:,0] = sigma_r * rng.normal(size=n) # distance

    dv_sph = np.empty((n,3))
    dv_sph[:,0] = sigma_vr * rng.normal(size=n) # radial velocity
    dv_sph[:,1:] = sigma_vt[:,None] * rng.normal(size=(n,2)) # transverse vel

    # Convert position & velocity errors to Cartesian basis
    R = np.linalg.norm(eta_cent[:,:2], axis=1)
    cos_phi = eta_cent[:,0] / R
    sin_phi = eta_cent[:,1] / R
    cos_theta = eta_cent[:,2] / r
    sin_theta = R / r
    rot_mat = np.array([
        [sin_theta*cos_phi,  cos_theta*cos_phi, -sin_phi],
        [sin_theta*sin_phi,  cos_theta*sin_phi,  cos_phi],
        [cos_theta,         -sin_theta,          np.zeros(n)]
    ]) # shape = (row, col, point)
    dx = np.einsum('ijn,nj->ni', rot_mat, dx_sph)
    dv = np.einsum('ijn,nj->ni', rot_mat, dv_sph)

    # Check: dx parallel to x
    cth = (
        np.sum(eta_cent[:,:3]*dx, axis=1) / (
            np.linalg.norm(eta_cent[:,:3], axis=1)
          * np.linalg.norm(dx, axis=1)
        )
    )
    print(fr'dx∥x : {np.all(np.abs(np.abs(cth)-1) < 1e-5)}')
    print(fr'  (|dx·x| ∈ {np.percentile(np.abs(cth),[0,1,50,99,100])})')

    # Check: dv parallel (when vt = 0) or perpendicular to x (when vr = 0)
    cth = (
        np.sum(eta_cent[:,:3]*dv, axis=1) / (
            np.linalg.norm(eta_cent[:,:3], axis=1)
          * np.linalg.norm(dv, axis=1)
        )
    )
    print(fr'dv∥x : {np.all(np.abs(np.abs(cth)-1) < 1e-5)}')
    print(fr'  (|dv·x| ∈ {np.percentile(np.abs(cth),[0,1,50,99,100])})')
    print(fr'dv⟂x : {np.all(np.abs(cth) < 1e-5)}')
    print(fr'  (dv·x ∈ {np.percentile(cth,[0,1,50,99,100])})')

    # Add errors into positions & velocities
    deta = np.concatenate([dx, dv], axis=1)
    eta_noisy = eta[:,-1,:] + deta

    #sigma_x = 0.10
    #sigma_v = 0.05

    #eta_noisy = np.empty((n,6), dtype='f8')
    #eta_noisy[:,:3] = eta[:,-1,:3] + sigma_x * rng.normal(size=(n,3))
    #eta_noisy[:,3:] = eta[:,-1,3:] + sigma_v * rng.normal(size=(n,3))

    # Save to HDF5
    with h5py.File(fname, 'w') as f:
        dset = f.create_dataset('eta', data=eta_noisy, compression='lzf')
        dset.attrs['axes'] = ('particle', 'dimension')
        dset.attrs['time'] = t_snap[-1]
        dset.attrs['dimensions'] = ('x', 'y', 'z', 'vx', 'vy', 'vz')

        dset = f.create_dataset('deta', data=deta, compression='lzf')

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


def plot_snapshot_projections(eta_vals, eta_labels, fname):
    n_t = len(eta_vals)
    h_row = 1.4
    margin_bottom = 0.6
    margin_top = 0.2
    h = h_row*n_t + margin_bottom + margin_top

    fig,ax_arr = plt.subplots(n_t, 4, figsize=(8,h))

    pot = gpot.MiyamotoNagaiPotential(amp=1., a=1., b=0.1)
    R = np.linspace(0., 5., 100)
    v_c = pot.vcirc(R)

    for eta,label,(ax_R,ax_z,ax_T,ax_xy) in zip(eta_vals,eta_labels,ax_arr):
        print(f'Plotting {label} ...')
        x = cart2cyl(eta)
        ax_R.hist2d(
            x['R'],
            x['vR'],
            bins=128,
            range=((0.,5.),(-0.22,0.22)),
            rasterized=True
        )
        ax_z.hist2d(
            x['z'],
            x['vz'],
            bins=128,
            range=((-0.15,0.15),(-0.15,0.15)),
            rasterized=True
        )
        ax_T.hist2d(
            x['R'],
            x['vT'],
            bins=128,
            range=((0.,5.),(0.0,0.75)),
            rasterized=True
        )
        ax_T.plot(
            R, v_c,
            alpha=0.5,
            c='w', lw=1,
            label=r'$v_{\mathrm{circ}}$'
        )
        ax_xy.hist2d(
            eta[:,0],
            eta[:,1],
            bins=128,
            range=((-5.,5.),(-5.,5.)),
            rasterized=True
        )
        ax_xy.text(
            0.95, 0.95,
            label,
            transform=ax_xy.transAxes,
            ha='right',
            va='top',
            c='w'
        )

    for ax in ax_arr[:-1].flat:
        ax.set_xticklabels([])

    ax_arr[-1,0].set_xlabel(r'$R$')
    ax_arr[-1,1].set_xlabel(r'$z$')
    ax_arr[-1,2].set_xlabel(r'$R$')
    ax_arr[-1,3].set_xlabel(r'$x$')

    ax_arr[-1,2].legend(loc='lower right', frameon=False, labelcolor='w')
    #for text in l.get_texts():
    #    text.set_color('w')

    labels = ('v_R', 'v_z', 'v_T', 'y')
    for i,l in enumerate(labels):
        for ax in ax_arr[:,i]:
            ax.set_ylabel(rf'${l}$', labelpad=(6 if l=='v_T' else -4))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        #for ax,k in zip(ax_arr[:,i],t_idx):
        #    t = t_snap[k]
        #    ax.text(
        #        0.95, 0.95,
        #        rf'${t:.0f} \ t_{{\mathrm{{dyn}}}}$',
        #        transform=ax.transAxes,
        #        ha='right',
        #        va='top',
        #        c='w'
        #    )

    # ax = fig.add_axes([0,0,1,1], zorder=-1)
    # ax.set_xticks([])
    # ax.set_yticks([])

    fig.subplots_adjust(
        wspace=0.45, hspace=0.10,
        left=0.08, right=0.98,
        bottom=margin_bottom/h, top=1-margin_top/h
    )

    for fmt in ('svg', 'pdf', 'png'):
        fig.savefig(fname.format(fmt=fmt))
    plt.close(fig)


def main():
    #n_init = 3 * 1024 * 1024
    #n = 512 * 1024
    n_init = 64 * 1024
    n = 64 * 1024
    data_dir = 'tmp'
    fig_dir = 'tmp'

    # Generate the mock data
    x, eta, t_snap = gen_mock_data(n_init, batch_size=128)

    # Save various versions of the mock data
    fname = os.path.join(data_dir, 'Miyamoto_Nagai_selection.h5')
    eta_sel = save_selection(eta, t_snap, fname, n_max=n)

    eta = eta[:n]

    fname_cyl = os.path.join(data_dir, 'Miyamoto_Nagai_cylindrical.h5')
    fname_cart = os.path.join(data_dir, 'Miyamoto_Nagai_cartesian.h5')
    save_full(x, eta, t_snap, fname_cyl, fname_cart)

    fname = os.path.join(data_dir, 'Miyamoto_Nagai_stationary.h5')
    save_stationary(eta, t_snap, fname)

    fname = os.path.join(data_dir, 'Miyamoto_Nagai_nonstationary.h5')
    save_nonstationary(eta, t_snap, fname, t_idx=4)

    fname = os.path.join(data_dir, 'Miyamoto_Nagai_noisy.h5')
    eta_noisy = save_noisy(eta, t_snap, fname)

    #x, t_snap = load_cyl(fname_cyl)
    #eta, _ = load_cart(fname_cart)

    # Plots of the mock data
    print('Plotting projections of snapshots ...')
    fname = os.path.join(fig_dir, r'mn_nonstationary_df.{fmt}')
    eta_vals = [eta[:,k] for k in range(9)] + [eta_noisy, eta_sel]
    labels = (
        [fr'${t}\,t_{{\mathrm{{dyn}}}}$' for t in t_snap]
      + [fr'$\mathrm{{{s}}}$' for s in ('noisy', 'selection')]
    )
    plot_snapshot_projections(eta_vals, labels, fname)

    return 0

if __name__ == '__main__':
    main()

