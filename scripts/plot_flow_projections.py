#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import h5py
import progressbar
import os
from glob import glob
from pathlib import Path

import tensorflow as tf
print(f'Tensorflow version {tf.__version__}')

import flow_ffjord_tf


dpi = 200


def calc_coords(eta):
    # Cylindrical
    R = np.linalg.norm(eta[:,:2], axis=1)
    z = eta[:,2]
    phi = np.arctan2(eta[:,1], eta[:,0])
    cos_phi = eta[:,0] / R
    sin_phi = eta[:,1] / R
    vR =  eta[:,3] * cos_phi + eta[:,4] * sin_phi
    vT = -eta[:,3] * sin_phi + eta[:,4] * cos_phi
    vz = eta[:,5]
    cyl = {'R':R, 'z':z, 'phi':phi, 'vR':vR, 'vz':vz, 'vT':vT}

    # Cartesian (z is already in cylindrical)
    cart = {'x':eta[:,0], 'y':eta[:,1], 'vx':eta[:,3], 'vy':eta[:,4]}

    # Spherical (phi is already in cylindrical)
    r = np.linalg.norm(eta[:,:3], axis=1)
    vr = np.sum(eta[:,:3]*eta[:,3:], axis=1) / r
    costheta = z / r
    vth = (z*vr - r*vz) / R
    sph = {'r':r, 'cth':costheta, 'vr':vr, 'vth':vth}

    return dict(**cart, **cyl, **sph)


def load_training_data(fname, load_attrs=False):
    with h5py.File(fname, 'r') as f:
        eta = f['eta'][:]
        if load_attrs:
            attrs = dict(f['eta'].attrs.items())
            return eta, attrs
    return eta


def load_flows(fname_patterns):
    flow_list = []

    fnames = []
    for fn in fname_patterns:
        fnames += glob(fn)
    fnames = sorted(fnames)
    #fnames = [fn[:-6] for fn in fnames]

    print(f'Found {len(fnames)} flows.')

    for i,fn in enumerate(fnames):
        print(f'Loading flow {i+1} of {len(fnames)} ...')
        if os.path.isdir(fn):
            print(f'  Loading latest checkpoint from directory {fn} ...')
            flow = flow_ffjord_tf.FFJORDFlow.load_latest(fn)
        else:
            print(f'  Loading {fn} ...')
            flow = flow_ffjord_tf.FFJORDFlow.load(fn[:-6])
        flow_list.append(flow)

    return flow_list


def sample_from_flows(flow_list, n_samples, batch_size=1024):
    n_flows = len(flow_list)

    # Sample from ensemble of flows
    n_batches = n_samples // (n_flows * batch_size)
    n_samples_rounded = n_flows * n_batches * batch_size
    print(f'Rounding down # of samples: {n_samples} -> {n_samples_rounded}')
    eta = np.empty((n_samples_rounded,6), dtype='f4')
    eta[:] = np.nan # Make it obvious if there are unfilled values at the end

    bar = progressbar.ProgressBar(max_value=n_batches*n_flows)

    batch_idx = 0

    for i,flow in enumerate(flow_list):
        #print(f'Sampling from flow {i+1} of {n_flows} ...')

        @tf.function
        def sample_batch():
            print(f'Tracing sample_batch for flow {i+1} of {n_flows} ...')
            return flow.sample([batch_size])

        for k in range(n_batches):
            j0 = batch_idx * batch_size
            eta[j0:j0+batch_size] = sample_batch().numpy()
            batch_idx += 1
            bar.update(batch_idx)

    return eta


def plot_1d_marginals(coords_train, coords_sample, fig_dir,
                      loss=None, coordsys='cart', fig_fmt=('svg',)):
    if coordsys == 'cart':
        labels = ['$x$', '$y$', r'$z$', '$v_x$', '$v_y$', '$v_z$']
        keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    elif coordsys == 'cyl':
        labels = ['$R$', '$z$', r'$\phi$', '$v_R$', '$v_z$', '$v_T$']
        keys = ['R', 'z', 'phi', 'vR', 'vz', 'vT']
    elif coordsys == 'sph':
        labels = ['$r$', r'$\cos \theta$', r'$\phi$', '$v_r$', r'$v_{\theta}$', r'$v_{\phi}$']
        keys = ['r', 'cth', 'phi', 'vr', 'vth', 'vT']
    else:
        raise ValueError(f'Unknown coordsys: {coordsys}.')

    fig,ax_arr = plt.subplots(2,3, figsize=(6,4), dpi=120)

    for i,(ax,l,k) in enumerate(zip(ax_arr.flat,labels,keys)):
        xlim = np.percentile(coords_train[k], [1., 99.])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0]-0.2*w, xlim[1]+0.2*w]
        if k == 'R':
            xlim[0] = max(xlim[0], 0.)
        elif k == 'phi':
            xlim = [-np.pi, np.pi]
        elif k == 'cth':
            xlim = [-1, 1]

        kw = dict(range=xlim, bins=101, density=True)
        ax.hist(coords_train[k], label=r'$\mathrm{train}$', alpha=0.7, **kw)
        ax.hist(
            coords_sample[k],
            histtype='step',
            alpha=0.8,
            label=r'$\mathrm{NF}$',
            **kw
        )
        ax.set_xlim(xlim)

        ax.set_xlabel(l, labelpad=0)
        ax.set_yticklabels([])

    ax_arr.flat[0].legend()

    if loss is not None:
        ax = ax_arr.flat[1]
        ax.text(
            0.02, 0.98, rf'$\left< \ln p \right> = {-loss:.3f}$',
            ha='left', va='top',
            transform=ax.transAxes
        )

    fig.subplots_adjust(
        wspace=0.1,
        hspace=0.3,
        left=0.03,
        right=0.97,
        bottom=0.12,
        top=0.97
    )

    # Make sure fig_dir exists
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'DF_marginals_{coordsys}.{fmt}')
        fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def plot_2d_marginal(coords_train, coords_sample,
                     eta_train, eta_sample,
                     fig_dir, dim1, dim2,
                     fig_fmt=('svg',)):
    labels = [
        '$R$', '$z$', r'$\phi$', '$v_R$', '$v_z$', '$v_{\phi}$',
        '$x$', '$y$', '$v_x$', '$v_y$',
        '$r$', r'$\cos \theta$', '$v_r$', r'$v_{\theta}$'
    ]
    keys = [
        'R', 'z', 'phi', 'vR', 'vz', 'vT',
        'x', 'y', 'vx', 'vy',
        'r', 'cth', 'vr', 'vth'
    ]

    def extract_dims(dim):
        return coords_train[dim], coords_sample[dim]
        #if dim in keys[:-4]:
        #    return coords_train[dim], coords_sample[dim]
        #elif dim in keys[-4:]:
        #    d = {'x':0, 'y':1, 'vx':3, 'vy':4}[dim]
        #    return eta_train[:,d], eta_sample[:,d]

    x_train, x_sample = extract_dims(dim1)
    y_train, y_sample = extract_dims(dim2)

    labels = {k:l for k,l in zip(keys,labels)}

    fig,(ax_t,ax_s,ax_d,cax_d) = plt.subplots(
        1,4,
        figsize=(6,2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[1,1,1,0.05])
    )

    lims = []
    for i,(k,z) in enumerate([(dim1,x_train),(dim2,y_train)]):
        xlim = np.percentile(z, [1., 99.])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0]-0.2*w, xlim[1]+0.2*w]
        if k == 'R':
            xlim[0] = max(xlim[0], 0.)
        elif k == 'phi':
            xlim = [-np.pi, np.pi]
        elif k == 'cth':
            xlim = [-1, 1]
        lims.append(xlim)

    kw = dict(range=lims, bins=128, rasterized=True)

    n_train = len(x_train)
    n_sample = len(x_sample)

    nt,_,_,_ = ax_t.hist2d(x_train, y_train, **kw)
    norm = Normalize(vmin=0, vmax=np.max(nt)*n_sample/n_train)
    ns,_,_,_ = ax_s.hist2d(x_sample, y_sample, norm=norm, **kw)

    dn = ns/n_sample - nt/n_train
    dn /= np.sqrt(ns * (n_train/n_sample)) / n_train
    vmax = 5.
    #dn /= np.max(nt)/n_train
    #vmax = 0.2
    im = ax_d.imshow(
        dn.T,
        extent=lims[0]+lims[1],
        cmap='coolwarm_r',
        vmin=-vmax, vmax=vmax,
        origin='lower', aspect='auto',
        rasterized=True
    )

    cb = fig.colorbar(
        im, cax=cax_d,
        label=r'$\mathrm{Poisson\ significance} \ \left( \sigma \right)$'
        #label=r'$\mathrm{fraction\ of\ max\ density}$'
    )

    ax_s.set_yticklabels([])
    ax_d.set_yticklabels([])

    for ax in (ax_s,ax_t,ax_d):
        ax.set_xlabel(labels[dim1], labelpad=0)

    ax_t.set_ylabel(labels[dim2], labelpad=2)

    ax_t.set_title(r'$\mathrm{training\ data}$', fontsize=10)
    ax_s.set_title(r'$\mathrm{normalizing\ flow\ (NF)}$', fontsize=10)
    ax_d.set_title(r'$\mathrm{NF - training}$', fontsize=10)

    fig.subplots_adjust(
        left=0.11,
        right=0.88,
        bottom=0.22,
        top=0.88,
        wspace=0.16
    )

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'DF_marginal_{dim1}_{dim2}.{fmt}')
        fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def plot_2d_slice(coords_train, coords_sample, fig_dir, dim1, dim2, dimz, z, dz, attrs=None, fig_fmt=('svg',), verbose=False): 
    labels = [
        '$R$', '$z$', r'$\phi$', '$v_R$', '$v_z$', '$v_{\phi}$',
        '$x$', '$y$', '$v_x$', '$v_y$',
        '$r$', r'$\cos \theta$', '$v_r$', r'$v_{\theta}$'
    ]
    
    keys = [
        'R', 'z', 'phi', 'vR', 'vz', 'vT',
        'x', 'y', 'vx', 'vy',
        'r', 'cth', 'vr', 'vth'
    ]
    
    idx_train = (coords_train[dimz] > z - dz) & (coords_train[dimz] < z + dz)
    idx_sample = (coords_sample[dimz] > z - dz) & (coords_sample[dimz] < z + dz)
    x_train, x_sample = coords_train[dim1][idx_train], coords_sample[dim1][idx_sample]
    y_train, y_sample = coords_train[dim2][idx_train], coords_sample[dim2][idx_sample]

    labels = {k:l for k,l in zip(keys,labels)}

    fig,(ax_t,ax_s,ax_d,cax_d) = plt.subplots(
        1,4,
        figsize=(6,2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[1,1,1,0.05])
    )

    lims = []
    for i,(k,val) in enumerate([(dim1,x_train),(dim2,y_train)]):
        xlim = np.percentile(val, [1., 99.])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0]-0.2*w, xlim[1]+0.2*w]
        if k == 'R':
            xlim[0] = max(xlim[0], 0.)
        elif k == 'phi':
            xlim = [-np.pi, np.pi]
        elif k == 'cth':
            xlim = [-1, 1]
        lims.append(xlim)

    kw = dict(range=lims, bins=64, rasterized=True)

    n_train = len(x_train)
    n_sample = len(x_sample)

    nt,_,_,_ = ax_t.hist2d(x_train, y_train, **kw)
    norm = Normalize(vmin=0, vmax=np.max(nt)*n_sample/n_train)
    ns,_,_,_ = ax_s.hist2d(x_sample, y_sample, norm=norm, **kw)

    if attrs is not None:
        # Visualise the boundaries of the population
        cartesian_keys = ['x', 'y', 'z']
        r_inner, r_outer = 1/attrs['parallax_max'], 1/attrs['parallax_min'] # [kpc], [kpc]
        kw = dict(linestyle=(0, (5, 3)), lw=0.5, color='white')
        if (dim1 in cartesian_keys) and (dim2 in cartesian_keys):
            # Plot circles
            for ax in [ax_t, ax_s, ax_d]:
                circ = plt.Circle((0, 0), r_inner, fill=False, **kw)
                ax.add_patch(circ)
                circ = plt.Circle((0, 0), r_outer, fill=False, **kw)
                ax.add_patch(circ)
        if dim1 in ['R']:
            for ax in [ax_t, ax_s, ax_d]:
                ax.axvline(r_inner, **kw)
                ax.axvline(r_outer, **kw)
        if dim2 in ['R']:
            for ax in [ax_t, ax_s, ax_d]:
                ax.axhline(r_inner, **kw)
                ax.axhline(r_outer, **kw)

    dn = ns/n_sample - nt/n_train
    dn /= np.sqrt(ns * (n_train/n_sample)) / n_train
    vmax = 5.
    #dn /= np.max(nt)/n_train
    #vmax = 0.2
    im = ax_d.imshow(
        dn.T,
        extent=lims[0]+lims[1],
        cmap='coolwarm_r',
        vmin=-vmax, vmax=vmax,
        origin='lower', aspect='auto',
        rasterized=True
    )

    cb = fig.colorbar(
        im, cax=cax_d,
        label=r'$\mathrm{Poisson\ significance} \ \left( \sigma \right)$'
        #label=r'$\mathrm{fraction\ of\ max\ density}$'
    )

    ax_s.set_yticklabels([])
    ax_d.set_yticklabels([])

    for ax in (ax_s,ax_t,ax_d):
        ax.set_xlabel(labels[dim1], labelpad=0)

    ax_t.set_ylabel(labels[dim2], labelpad=2)


    ax_d.set_title(r'$\mathrm{NF - training}$', fontsize=10)
    if verbose:
        ax_t.set_title(f'$\mathrm{{training\ data}}$\n$n={len(x_train)}$', fontsize=10)
        ax_s.set_title(f'$\mathrm{{normalizing\ flow\ (NF)}}$\n$n={len(x_sample)}$', fontsize=10)
        # Print additional info on the subplots
        fig.suptitle(f'${z-dz:.2f}\leq${labels[dimz]}$\leq{z+dz:.2f}$', fontsize=10)

        fig.subplots_adjust(
            left=0.16,
            right=0.83,
            bottom=0.18,
            top=0.74,
            wspace=0.16
        )
    else:
        ax_t.set_title(r'$\mathrm{training\ data}$', fontsize=10)
        ax_s.set_title(r'$\mathrm{normalizing\ flow\ (NF)}$', fontsize=10)
        
        fig.subplots_adjust(
            left=0.11,
            right=0.88,
            bottom=0.22,
            top=0.88,
            wspace=0.16
        )

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'DF_slice_{dim1}_{dim2}.{fmt}')
        fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def evaluate_loss(flow_list, eta_train, batch_size=1024):
    n_flows = len(flow_list)
    n_samples = eta_train.shape[0]

    # Sample from ensemble of flows
    #n_batches = n_samples // batch_size
    #eta_batches = np.reshape(eta_train, (n_batches,batch_size,6)).astype('f4')
    #eta_batches = [
    #    eta_train[i0:i0+batch_size].astype('f4')
    #    for i0 in range(0,n_samples,batch_size)
    #]
    #n_batches = len(eta_batches)
    n_batches = n_samples // batch_size
    if np.mod(n_samples, batch_size) > 0:
        n_batches += 1

    loss = []
    bar = progressbar.ProgressBar(max_value=n_batches*n_flows)

    for i,flow in enumerate(flow_list):
        loss_i = []
        weight_i = []

        @tf.function
        def logp_batch(eta):
            print('Tracing logp_batch ...')
            return -tf.math.reduce_mean(flow.log_prob(eta))

        for k in range(0,n_samples,batch_size):
            eta = eta_train[k:k+batch_size].astype('f4')
            loss_i.append(logp_batch(tf.constant(eta)).numpy())
            weight_i.append(eta.shape[0])
            bar.update(i*n_batches + k//batch_size + 1)

        loss.append(np.average(loss_i, weights=weight_i))

    loss_std = np.std(loss)
    loss_mean = np.mean(loss)

    return loss_mean, loss_std


def plot_1d_slice(coords_train, coords_sample, fig_dir, dim1, dimy, dimz, y, dy, z, dz, attrs=None, fig_fmt=('svg',), verbose=False): 
    labels = [
        '$R$', '$z$', r'$\phi$', '$v_R$', '$v_z$', '$v_{\phi}$',
        '$x$', '$y$', '$v_x$', '$v_y$',
        '$r$', r'$\cos \theta$', '$v_r$', r'$v_{\theta}$'
    ]
    
    keys = [
        'R', 'z', 'phi', 'vR', 'vz', 'vT',
        'x', 'y', 'vx', 'vy',
        'r', 'cth', 'vr', 'vth'
    ]
    
    idx_train = (coords_train[dimz] > z - dz) & (coords_train[dimz] < z + dz) &\
                (coords_train[dimy] > y - dy) & (coords_train[dimy] < y + dy)
    idx_sample = (coords_sample[dimz] > z - dz) & (coords_sample[dimz] < z + dz) &\
                 (coords_sample[dimy] > y - dy) & (coords_sample[dimy] < y + dy)
    x_train, x_sample = coords_train[dim1][idx_train], coords_sample[dim1][idx_sample]

    labels = {k:l for k,l in zip(keys,labels)}

    fig,(ax_h, ax_r) = plt.subplots(
        1,2,
        figsize=(6,3),
        dpi=200,
        gridspec_kw=dict(width_ratios=[1,1])
    )

    lim_min, lim_max = 99999., -99999.
    for i,(k,val) in enumerate([(dim1,x_train)]):
        xlim = np.percentile(val, [1., 99.])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0]-0.05*w, xlim[1]+0.05*w]
        if k == 'R':
            xlim[0] = max(xlim[0], 0.)
        elif k == 'phi':
            xlim = [-np.pi, np.pi]
        elif k == 'cth':
            xlim = [-1, 1]
        lim_min = min(lim_min, xlim[0])
        lim_max = max(lim_max, xlim[1])

    kw = dict(range=(lim_min, lim_max), bins=64)

    n_train = len(x_train)
    n_sample = len(x_sample)
    
    nt,bins,_ = ax_h.hist(x_train, histtype='step', **kw, label='train')
    ns,*_ = ax_h.hist(x_sample, histtype='step', **kw, weights=np.ones_like(x_sample)*n_train/n_sample, label='sample')
    ns *= n_sample/n_train
    ax_h.legend(loc='lower right', frameon=False, fontsize=8)
    ax_h.set_ylabel('frequency')

    if attrs is not None:
        # Visualise the boundaries of the population
        valid_keys = ['x', 'y', 'z', 'R']
        r_inner, r_outer = 1/attrs['parallax_max'], 1/attrs['parallax_min'] # [kpc], [kpc]
        kw = dict(linestyle=(0, (5, 3)), lw=1.0, color='black', zorder=0)
        if (dim1 in valid_keys):
            for ax in [ax_h, ax_r]:
                ax.axvline(r_inner, **kw)
                ax.axvline(r_outer, **kw)
                if dim1 != 'R':
                    ax.axvline(-r_inner, **kw)
                    ax.axvline(-r_outer, **kw)

    dn = ns/n_sample - nt/n_train
    dn /= np.sqrt(ns * (n_train/n_sample)) / n_train
    ax_r.plot(bins[:-1], dn, label=r"gaia_vr, $\varpi$ > 0.2 mas", drawstyle='steps-post')
    ax_r.yaxis.tick_right()
    ax_r.yaxis.set_label_position("right")
    ax_r.set_ylabel(r'$\mathrm{Poisson\ significance} \ \left( \sigma \right)$')
    ax_r.axhline(0, ls='--', lw=1., color='black', zorder=0)

    for ax in (ax_h,ax_r):
        ax.set_xlabel(labels[dim1], labelpad=0)

    ax_r.set_title(r'$\mathrm{NF - training}$', fontsize=10)
    
    if verbose:
        ax_h.set_title(f'$n_\mathrm{{train}}={len(x_train)},\quad n_\mathrm{{sample}}={len(x_sample)}$', fontsize=10)
        # Print additional info on the subplots
        fig.suptitle(f'${y-dy:.2f}\leq${labels[dimy]}$\leq{y+dy:.2f},\quad\
                       {z-dz:.2f}\leq${labels[dimz]}$\leq{z+dz:.2f}$', fontsize=10)

        fig.subplots_adjust(
            left=0.11,
            right=0.88,
            bottom=0.18,
            top=0.84,
            wspace=0.16
        )
    else:
        fig.subplots_adjust(
            left=0.11,
            right=0.88,
            bottom=0.22,
            top=0.88,
            wspace=0.16
        )

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'DF_slice_{dim1}.{fmt}')
        fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def plot_slices(flow_list, eta_train, n_pix=256, batch_size=128):
    R2 = eta_train[:,0]**2 + eta_train[:,1]**2
    R_max = 1.2 * np.sqrt(np.percentile(R2, 99.))

    x = np.linspace(-R_max, R_max, n_pix)
    x,y = np.meshgrid(x, x)
    s = x.shape
    x.shape = (x.size,)
    y.shape = (x.size,)

    p = np.empty_like(x)

    for flow in flow_list:
        @tf.function
        def p_batch(eta_batch):
            print('Tracing p_batch ...')
            return flow.prob(eta_batch)

        for i0 in range(0,x.size,batch_size):
            eta = np.zeros((x.size,6), dtype='f4')
            eta[:,0] = x[i0:i0+batch_size]
            eta[:,1] = y[i0:i0+batch_size]
            eta = tf.constant(eta)
            p[i0:i0+batch_size] += p_batch(eta)

    p /= len(flow_list)

    # Plot in (x,y)-plane


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Deep Potential: Plot results for Plummer sphere.',
        add_help=True
    )
    parser.add_argument(
        '-i', '--input',
        type=str, required=True,
        help='Filename of input training data (particle phase-space coords).'
    )
    parser.add_argument(
        '--flows',
        type=str, nargs='+',
        required=True,
        help='Flow model filename pattern(s).'
    )
    parser.add_argument(
        '--save-samples',
        type=str,
        metavar='*.h5',
        help='Save samples to this filename after generating them.'
    )
    parser.add_argument(
        '--load-samples',
        type=str,
        metavar='*.h5',
        help='Load samples, instead of generating them.'
    )
    parser.add_argument(
        '--fig-dir',
        type=str,
        default='plots',
        help='Directory to put figures in.'
    )
    parser.add_argument(
        '--oversample',
        type=int,
        default=1,
        help='Draw oversample*(# of training samples) samples from flows.'
    )
    parser.add_argument(
        '--origin',
        type=float,
        nargs=3,
        default=(0.,0.,0.),
        help='Origin of coordinate system in (x,y,z) to subtract from coords.'
    )
    parser.add_argument(
        '--fig-fmt',
        type=str,
        nargs='+',
        default=('svg',),
        help='Formats in which to save figures (svg, png, pdf, etc.).'
    )
    parser.add_argument(
        '--dark',
        action='store_true',
        help='Use dark background for figures.'
    )
    parser.add_argument(
        '--load-attrs',
        action='store_true',
        help='Load attributes of the training data for visualisation (e.g. cut boundaries).'
    )
    args = parser.parse_args()

    if args.dark:
        plt.style.use('dark_background')

    print('Loading training data ...')
    if args.load_attrs:
        eta_train, attrs_train = load_training_data(args.input, load_attrs=True)
    else:
        eta_train, attrs_train = load_training_data(args.input, load_attrs=False), None
    n_train = eta_train.shape[0]

    print(attrs_train, args.load_attrs)
    print(f'  --> Training data shape = {eta_train.shape}')

    print('Loading flows ...')
    flows = load_flows(args.flows)

    if args.load_samples is None:
        print('Evaluating loss ...')
        loss_mean, loss_std = evaluate_loss(flows, eta_train)
        print(f'  --> loss = {loss_mean:.5f} +- {loss_std:.5f}')
        print('Sampling from flows ...')
        eta_sample = sample_from_flows(flows, args.oversample*n_train)
        print('  --> Saving samples ...')
        if args.save_samples is not None:
            Path(os.path.split(args.save_samples)[0]).mkdir(parents=True, exist_ok=True)
            with h5py.File(args.save_samples, 'w') as f:
                dset = f.create_dataset(
                    'eta',
                    data=eta_sample,
                    chunks=True,
                    compression='lzf'
                )
                dset.attrs['loss_training'] = loss_mean
                dset.attrs['loss_std_training'] = loss_std
    else:
        print('Loading pre-generated samples ...')
        with h5py.File(args.load_samples, 'r') as f:
            eta_sample = f['eta'][:]
            loss_mean = f['eta'].attrs['loss_training']
            loss_std = f['eta'].attrs['loss_std_training']
        print(f'  --> loss = {loss_mean:.5f} +- {loss_std:.5f}')
        print(f'  --> {len(eta_sample)} samples')
    print(f'  --> {np.count_nonzero(np.isnan(eta_sample))} NaN values')

    # Subtract off origin position
    for k,x0 in enumerate(args.origin):
        eta_train[:,k] -= x0
        eta_sample[:,k] -= x0

    print('Calculating cylindrical & spherical coordinates ...')
    coords_train = calc_coords(eta_train)
    coords_sample = calc_coords(eta_sample)

    print('Plotting 1D marginal distributions ...')
    for coordsys in ['cart', 'cyl', 'sph']:
        plot_1d_marginals(
            coords_train, coords_sample, args.fig_dir,
            loss=loss_mean, coordsys=coordsys,
            fig_fmt=args.fig_fmt
        )

    print('Plotting 2D marginal distributions ...')

    dims = [
        ('r', 'vr'),
        ('phi', 'cth'),
        ('vT', 'vth'),
        ('R', 'z'),
        ('R', 'vz'),
        ('R', 'vR'),
        ('R', 'vT'),
        ('z', 'vz'),
        ('vz', 'vT'),
        ('vR', 'vz'),
        ('vR', 'vT'),
        ('x', 'y'),
        ('x', 'z'),
        ('y', 'z'),
        ('vx', 'vy'),
        ('vx', 'vz'),
        ('vy', 'vz')
    ]

    for dim1,dim2 in dims:
        print(f'  --> ({dim1}, {dim2})')
        plot_2d_marginal(
            coords_train, coords_sample,
            eta_train, eta_sample,
            args.fig_dir, dim1, dim2,
            fig_fmt=args.fig_fmt
        )

    dims = [
        ('phi', 'cth', 'R', np.mean(coords_train['R']), 0.05),
        ('R', 'phi', 'z', 0., 0.05),
        ('x', 'y', 'z', 0., 0.05),
        ('y', 'z', 'x', 0., 0.05),
        ('x', 'z', 'y', 0., 0.05),
    ]

    for dim1, dim2, dimz, z, dz in dims:
        print(f'  --> ({dim1}, {dim2}, {dimz}={z:.2f}+-{dz:.2f})')
        plot_2d_slice(
            coords_train, coords_sample,
            args.fig_dir, dim1, dim2,
            dimz, z, dz, attrs=attrs_train,
            fig_fmt=args.fig_fmt,
            verbose=True
        )


    dims = [
        ('x', 'y', 'z', 0., 0.05, 0., 0.05),
        ('y', 'x', 'z', 0., 0.05, 0., 0.05),
        ('x', 'y', 'z', 0., 0.05, 0., 0.05),
    ]

    for dim1, dimy, dimz, y, dy, z, dz in dims:
        print(f'  --> ({dim1}, {dimy}={y}+-{dy}, {dimz}={z}+-{dz})')
        plot_1d_slice(
            coords_train, coords_sample,
            args.fig_dir, dim1, dimy,
            dimz, y, dy, z, dz, attrs=attrs_train,
            fig_fmt=args.fig_fmt,
            verbose=True
        )
    return 0


if __name__ == '__main__':
    main()

