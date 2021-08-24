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

import tensorflow as tf
print(f'Tensorflow version {tf.__version__}')

import flow_ffjord_tf


fig_fmt = 'png'
dpi = 120


def cart2cyl(eta):
    R = np.linalg.norm(eta[:,:2], axis=1)
    z = eta[:,2]
    phi = np.arctan2(eta[:,1], eta[:,0])
    cos_phi = eta[:,0] / R
    sin_phi = eta[:,1] / R
    vR =  eta[:,3] * cos_phi + eta[:,4] * sin_phi
    vT = -eta[:,3] * sin_phi + eta[:,4] * cos_phi
    vz = eta[:,5]
    return {'R':R, 'z':z, 'phi':phi, 'vR':vR, 'vz':vz, 'vT':vT}


def load_training_data(fname):
    with h5py.File(fname, 'r') as f:
        eta = f['eta'][:]
    return eta


def load_flows(fname_patterns):
    flow_list = []

    fnames = []
    for fn in fname_patterns:
        fnames += glob(fn)
    fnames = sorted(fnames)
    fnames = [fn[:-6] for fn in fnames]

    print(f'Found {len(fnames)} flows.')

    for i,fn in enumerate(fnames):
        print(f'Loading flow {i+1} of {len(fnames)} ...')
        print(fn)
        flow = flow_ffjord_tf.FFJORDFlow.load(fname=fn)
        flow_list.append(flow)

    return flow_list


def sample_from_flows(flow_list, n_samples, batch_size=1024):
    n_flows = len(flow_list)

    # Sample from ensemble of flows
    n_batches = n_samples // (n_flows * batch_size)
    eta = np.empty((n_samples,6), dtype='f4')

    bar = progressbar.ProgressBar(max_value=n_batches*n_flows)

    for i,flow in enumerate(flow_list):
        #print(f'Sampling from flow {i+1} of {n_flows} ...')

        @tf.function
        def sample_batch():
            print('Tracing sample_batch ...')
            return flow.sample([batch_size])

        for k in range(n_batches):
            j0 = (i*n_batches + k) * batch_size
            eta[j0:j0+batch_size] = sample_batch().numpy()
            bar.update(i*n_batches+k+1)

    return eta


def plot_1d_marginals(cyl_train, cyl_sample, fig_dir, loss=None):
    labels = ['$R$', '$z$', r'$\phi$', '$v_R$', '$v_z$', '$v_T$']
    keys = ['R', 'z', 'phi', 'vR', 'vz', 'vT']

    fig,ax_arr = plt.subplots(2,3, figsize=(12,8), dpi=120)

    for i,(ax,l,k) in enumerate(zip(ax_arr.flat,labels,keys)):
        xlim = np.percentile(cyl_train[k], [1., 99.])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0]-0.2*w, xlim[1]+0.2*w]
        if k == 'R':
            xlim[0] = max(xlim[0], 0.)
        elif k == 'phi':
            xlim = (-np.pi, np.pi)

        kw = dict(range=xlim, bins=101, density=True)
        ax.hist(cyl_train[k], label=r'$\mathrm{train}$', **kw)
        ax.hist(cyl_sample[k], alpha=0.5, label=r'$\mathrm{NF}$', **kw)

        ax.set_xlabel(l)
        ax.set_yticklabels([])
    
    ax_arr.flat[0].legend()

    if loss is not None:
        ax = ax_arr.flat[1]
        ax.text(
            0.02, 0.98, rf'$\left< \ln p \right> = {loss:.4f}$',
            ha='left', va='top',
            transform=ax.transAxes
        )

    fig.subplots_adjust(wspace=0.1)

    fig.subplots_adjust(
        wspace=0.1,
        hspace=0.3,
        left=0.03,
        right=0.97,
        bottom=0.12,
        top=0.97
    )

    fname = os.path.join(fig_dir, f'DF_marginals.{fig_fmt}')
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def plot_2d_marginal(cyl_train, cyl_sample, fig_dir, dim1, dim2):
    labels = ['$R$', '$z$', r'$\phi$', '$v_R$', '$v_z$', '$v_T$']
    keys = ['R', 'z', 'phi', 'vR', 'vz', 'vT']

    labels = {k:l for k,l in zip(keys,labels)}
    
    fig,(ax_t,ax_s,ax_d,cax_d) = plt.subplots(
        1,4,
        figsize=(12,4),
        dpi=120,
        gridspec_kw=dict(width_ratios=[1,1,1,0.05])
    )

    lims = []
    for i,k in enumerate([dim1,dim2]):
        xlim = np.percentile(cyl_train[k], [1., 99.])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0]-0.2*w, xlim[1]+0.2*w]
        if k == 'R':
            xlim[0] = max(xlim[0], 0.)
        elif k == 'phi':
            xlim = (-np.pi, np.pi)
        lims.append(xlim)

    kw = dict(range=lims, bins=40)

    n_train = cyl_train['R'].shape[0]
    n_sample = cyl_sample['R'].shape[0]

    nt,_,_,_ = ax_t.hist2d(cyl_train[dim1], cyl_train[dim2], **kw)
    norm = Normalize(vmin=0, vmax=np.max(nt)*n_sample/n_train)
    ns,_,_,_ = ax_s.hist2d(cyl_sample[dim1], cyl_sample[dim2], norm=norm, **kw)

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
        origin='lower', aspect='auto'
    )

    cb = fig.colorbar(
        im, cax=cax_d,
        label=r'$\mathrm{Poisson\ significance} \ \left( \sigma \right)$'
        #label=r'$\mathrm{fraction\ of\ max\ density}$'
    )

    ax_s.set_yticklabels([])
    ax_d.set_yticklabels([])

    for ax in (ax_s,ax_t,ax_d):
        ax.set_xlabel(labels[dim1])

    ax_t.set_ylabel(labels[dim2])

    ax_t.set_title(r'$\mathrm{training\ data}$')
    ax_s.set_title(r'$\mathrm{normalizing\ flow}$')
    ax_d.set_title(r'$\mathrm{NF - training}$')

    fig.subplots_adjust(
        left=0.07,
        right=0.92,
        bottom=0.15,
        top=0.88
    )

    fname = os.path.join(fig_dir, f'DF_marginal_{dim1}_{dim2}.{fig_fmt}')
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def evaluate_loss(flow_list, eta_train, batch_size=1024):
    n_flows = len(flow_list)
    n_samples = eta_train.shape[0]

    # Sample from ensemble of flows
    n_batches = n_samples // batch_size
    eta_batches = np.reshape(eta_train, (n_batches,batch_size,6)).astype('f4')
    
    loss = []
    bar = progressbar.ProgressBar(max_value=n_batches*n_flows)

    for i,flow in enumerate(flow_list):
        loss_i = []

        @tf.function
        def logp_batch(eta):
            print('Tracing logp_batch ...')
            return -tf.math.reduce_mean(flow.log_prob(eta))
        
        for k,eta in enumerate(eta_batches):
            loss_i.append(logp_batch(tf.constant(eta)).numpy())
            bar.update(i*n_batches+k+1)

        loss.append(np.mean(loss_i))

    loss_std = np.std(loss)
    loss_mean = np.mean(loss)

    return loss_mean, loss_std


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
        '--fig-dir',
        type=str,
        default='plots',
        help='Directory to put figures in.'
    )
    parser.add_argument(
        '--oversample',
        type=int,
        default=1,
        help='Sample X*(# of training samples) from flows.'
    )
    args = parser.parse_args()

    print('Loading training data ...')
    eta_train = load_training_data(args.input)
    n_train = eta_train.shape[0]
    cyl_train = cart2cyl(eta_train)
    print(eta_train.shape)

    print('Loading flows ...')
    flows = load_flows(args.flows)
    
    print('Evaluating loss ...')
    loss_mean, loss_std = evaluate_loss(flows, eta_train)
    print(f'loss = {loss_mean:.5f} +- {loss_std:.5f}')

    print('Sampling from flows ...')
    eta_sample = sample_from_flows(flows, args.oversample*n_train)

    print('Converting to cylindrical coordinates ...')
    cyl_sample = cart2cyl(eta_sample)

    print('Plotting 1D marginal distributions ...')
    plot_1d_marginals(cyl_train, cyl_sample, args.fig_dir, loss=loss_mean)

    print('Plotting 2D marginal distributions ...')
    print('  --> (R, z)')
    plot_2d_marginal(cyl_train, cyl_sample, args.fig_dir, 'R', 'z')
    print('  --> (R, vz)')
    plot_2d_marginal(cyl_train, cyl_sample, args.fig_dir, 'R', 'vz')
    print('  --> (R, vR)')
    plot_2d_marginal(cyl_train, cyl_sample, args.fig_dir, 'R', 'vR')
    print('  --> (R, vz)')
    plot_2d_marginal(cyl_train, cyl_sample, args.fig_dir, 'z', 'vz')
    print('  --> (R, vT)')
    plot_2d_marginal(cyl_train, cyl_sample, args.fig_dir, 'R', 'vT')

    return 0


if __name__ == '__main__':
    main()

