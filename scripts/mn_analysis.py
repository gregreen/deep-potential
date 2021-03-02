#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import scipy.stats

import matplotlib
matplotlib.use('Agg')
#matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator
from matplotlib.gridspec import GridSpec

import tensorflow as tf
print(f'Tensorflow version {tf.__version__}')

import progressbar
from glob import glob
import json
import os
import os.path

import flow_ffjord_tf
import potential_tf
import toy_systems


def batch_calc_df_deta(f, eta, batch_size):
    df_deta = np.empty_like(eta)
    n_data = eta.shape[0]

    @tf.function
    def calc_grads(batch):
        print(f'Tracing calc_grads with shape = {batch.shape}')
        return flow_ffjord_tf.calc_f_gradients(f, batch)

    bar = None
    for k in range(0,n_data,batch_size):
        if k != 0:
            if bar is None:
                bar = progressbar.ProgressBar(max_value=n_data)
            bar.update(k)
        b0,b1 = k, k+batch_size
        eta_k = tf.constant(eta[b0:b1])
        df_deta[b0:b1] = calc_grads(eta_k).numpy()

    bar.update(n_data)

    return df_deta


def vec2ang(x):
    phi = np.arctan2(x[...,1], x[...,0])
    theta = np.arctan2(x[...,2], np.sqrt(x[...,0]**2+x[...,1]**2))
    return theta, phi


def wrap_path(x, dx_max):
    dx = np.diff(x, axis=0)
    idx = np.any(np.abs(dx) > dx_max[None,:], axis=1)
    idx = np.where(idx)[0]
    y = np.insert(x, idx+1, np.nan, axis=0)
    return y


def calc_prob_rv_grid(r_lim, v_lim, bins):
    r = np.linspace(r_lim[0], r_lim[1], 2*bins[0]+1)
    v = np.linspace(v_lim[0], v_lim[1], 2*bins[1]+1)

    r = 0.5 * (r[:-1] + r[1:])
    v = 0.5 * (v[:-1] + v[1:])

    rr,vv = np.meshgrid(r, v)

    psi = 1. / np.sqrt(1+rr**2)
    E = psi - vv**2 / 2
    df = np.clip(E, 0., np.inf)**(7/2)
    A = 24 * np.sqrt(2.) / (7 * np.pi**3)

    n = A * (4*np.pi)**2 * rr**2 * vv**2 * df

    # Downsample by a factor of 2
    n = 0.5 * (n[:-1:2] + n[1::2])
    n = 0.5 * (n[:,:-1:2] + n[:,1::2])
    r = 0.5 * (r[:-1:2] + r[1::2])
    v = 0.5 * (v[:-1:2] + v[1::2])

    return n, r, v


def plot_flow_trajectories(flow, n_samples=100, n_t=100):
    # Integrate paths
    t_eval = np.linspace(0., 1., n_t)
    res = flow.calc_trajectories(n_samples, t_eval)
    eta_path = res.states.numpy() # shape = (time, sample, dimension)

    # Set up figure
    fig = plt.figure(figsize=(8,4), dpi=200)
    gs = GridSpec(2,4)
    ax_rv = fig.add_subplot(gs[:,:3])
    ax_angles_q = fig.add_subplot(gs[0,3])
    ax_angles_p = fig.add_subplot(gs[1,3])
    
    # v vs. r
    r_lim = (0., 7.)
    v_lim = (0., 3.)
    
    # Background ideal probabilities
    bins = (140, 60)
    p_rv, r, v = calc_prob_rv_grid(r_lim, v_lim, bins)
    p_sorted = np.sort(p_rv.flat)
    P_cumulative = np.cumsum(p_sorted)
    P_cumulative /= P_cumulative[-1]
    P_levels = [0.01, 0.1, 0.5, 0.9]
    idx_levels = np.searchsorted(P_cumulative, P_levels)
    p_levels = p_sorted[idx_levels]
    level_labels = {pl: rf'{100*(1-P):.0f}\%' for pl,P in zip(p_levels,P_levels)}
    cs = ax_rv.contour(r, v, p_rv, p_levels, alpha=0.1)
    ax_rv.clabel(cs, fmt=level_labels)

    # Trajectories
    r = np.sqrt(np.sum(eta_path[:,:,:3]**2, axis=2))
    v = np.sqrt(np.sum(eta_path[:,:,3:]**2, axis=2))
    for k in range(n_samples):
        ax_rv.plot(r[:,k], v[:,k], c='k', alpha=0.1, lw=1.)
    ax_rv.scatter(
        r[0], v[0],
        s=4,
        alpha=0.1,
        edgecolors='k',
        facecolors='none',
        lw=0.5
    )
    ax_rv.scatter(r[-1], v[-1], s=9, alpha=0.5, edgecolors='none')
    ax_rv.set_xlim(r_lim)
    ax_rv.set_ylim(v_lim)
    ax_rv.set_xlabel(r'$r$')
    ax_rv.set_ylabel(r'$v$', labelpad=0)

    # Zero-energy line
    r_E0 = np.linspace(r_lim[0], r_lim[1], 100)
    v_E0 = np.sqrt(2.) * (1+r_E0**2)**(-1/4)
    ax_rv.plot(r_E0, v_E0, c='k', ls='--', alpha=0.2)
    ax_rv.text(
        r_E0[-1]*0.98, v_E0[-1]*0.95, r'$E = 0$',
        ha='right', va='top',
        fontsize=12, c='k', alpha=0.2
    )

    # Angular plots
    tp_q = np.stack(vec2ang(eta_path[:,:,:3]), axis=2)
    tp_p = np.stack(vec2ang(eta_path[:,:,3:]), axis=2)
    dtp_max = np.array([1., np.pi])
    for ax,tp,lab in ((ax_angles_q,tp_q,'q'),(ax_angles_p,tp_p,'p')):
        for k in range(n_samples):
            #tp_plot = tp[:,k,:]
            tp_plot = wrap_path(tp[:,k,:], dtp_max)
            ax.plot(tp_plot[:,1], tp_plot[:,0], c='k', alpha=0.1, lw=1.)
        ax.scatter(
            tp[-1,:,1], tp[-1,:,0],
            s=9, alpha=0.5, edgecolors='none'
        )
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1., 1.)
        ax.set_xlabel(rf'$\varphi_{lab}$', labelpad=0)
        ax.set_ylabel(rf'$\theta_{lab}$', labelpad=0)
    
    # Adjust figure
    fig.subplots_adjust(
        left=0.06, right=0.99,
        bottom=0.12, top=0.97,
        wspace=0.4, hspace=0.3
    )

    return fig


def plot_flow_trajectories_multiple(flows, fname_pattern):
    n_flows = len(flows)

    # Plot flow trajectories
    for i,flow in enumerate(flows):
        print(f'Plotting trajectories of flow {i+1} of {len(flows)} ...')
        fig = plot_flow_trajectories(flow)
        fig.savefig(fname_pattern.format(i), dpi=200)
        plt.close(fig)


def plot_flow_slices(flows, fname):
    fname_prefix, ext = os.path.splitext(fname)
    ext = ext.lstrip('.')

    # Slices through distribution
    x_max = 2.2
    n_bins = 200
    q = np.linspace(-x_max, x_max, n_bins).astype('f4')

    x,y = np.meshgrid(q,q)
    s = x.shape
    x.shape = (x.size,)
    y.shape = (y.size,)
    coords = np.zeros([x.size, 6], dtype='f4')

    idx_x = [0, 0, 1, 3, 3, 4]
    idx_y = [1, 2, 2, 4, 5, 5]

    def gen_figure(img_stack, islog, isdiff=False):
        fig, ax_arr = plt.subplots(2,3, figsize=(9,6.5), dpi=200)

        for img,ax,ix,iy in zip(img_stack,ax_arr.flat,idx_x,idx_y):
            if isdiff:
                if islog:
                    vmax = 1.
                    vmin = -vmax
                else:
                    vmax = 1.5
                    vmin = 0.5
                cmap = 'bwr_r'
            else:
                vmax = np.max(img)
                if islog:
                    vmin = max([np.min(img), vmax-10])
                else:
                    vmin = 0.
                cmap = 'viridis'

            ax.imshow(
                img,
                interpolation='nearest',
                origin='lower',
                vmax=vmax,
                vmin=vmin,
                extent=(-x_max,x_max,-x_max,x_max),
                cmap=cmap
            )

            if ix < 3:
                ax.set_xlabel(f'$x_{ix}$')
            else:
                ax.set_xlabel(f'$v_{ix}$')

            if iy < 3:
                ax.set_ylabel(f'$x_{iy}$', labelpad=0)
            else:
                ax.set_ylabel(f'$v_{iy}$', labelpad=0)

        title = r'$\mathrm{{Slices\ through\ origin}} \left( {} \right)$'
        if islog:
            title = title.format(r'\mathrm{log\ scale}')
        else:
            title = title.format(r'\mathrm{linear\ scale}')
        fig.suptitle(
            title,
            fontsize=16,
            y=0.94,
            va='bottom'
        )

        fig.subplots_adjust(
            wspace=0.2,
            hspace=0.18,
            left=0.07,
            right=0.98,
            bottom=0.10,
            top=0.92
        )

        return fig

    img_avg_stack = [np.zeros(s, dtype='f4') for i in range(6)]

    log_df_fns = [flow.log_prob for flow in flows]

    img_ideal_stack = []

    for k,log_df in enumerate(log_df_fns):
        if k == 0:
            print('Ideal DF ...')
        else:
            print(f'Flow {k} of {len(flows)} ...')

        img_stack = []

        for i,(ix,iy) in enumerate(zip(idx_x,idx_y)):
            print(f'Generating image {i+1} of 6 ...')
            coords[:] = 0.
            coords[:,ix] = x
            coords[:,iy] = y

            img = log_df(coords).numpy()
            img.shape = s

            img_stack.append(img)

            if k != 0:
                img_avg_stack[i] += img / len(flows)
        
        if k == 0:
            img_ideal_stack = img_stack
            fn = f'{fname_prefix}_{{linlog}}_ideal.{ext}'
        else:
            fn = f'{fname_prefix}_{{linlog}}_{k-1:02d}.{ext}'

        fig = gen_figure(img_stack, True)
        fig.savefig(fn.format(linlog='log'))
        plt.close(fig)

        fig = gen_figure([np.exp(img) for img in img_stack], False)
        fig.savefig(fn.format(linlog='lin'))
        plt.close(fig)

    fig = gen_figure(img_avg_stack, True)
    fig.savefig(f'{fname_prefix}_log_ensemble.{ext}')
    plt.close(fig)

    fig = gen_figure([np.exp(img) for img in img_avg_stack], False)
    fig.savefig(f'{fname_prefix}_lin_ensemble.{ext}')
    plt.close(fig)

    dimg_stack = [img-img0 for img,img0 in zip(img_avg_stack,img_ideal_stack)]
    fig = gen_figure(dimg_stack, True, isdiff=True)
    fig.savefig(f'{fname_prefix}_log_diff.{ext}')
    plt.close(fig)


def plot_flow_projections_ensemble(flows, fname_proj, fname_hist,
                                   n_samples=1024*1024, batch_size=1024):
    """Plot projections of flows along each axis."""

    fname_base_proj, ext_proj = os.path.splitext(fname_proj)
    ext_proj = ext_proj.lstrip('.')

    fname_base_hist, ext_hist = os.path.splitext(fname_hist)
    ext_hist = ext_hist.lstrip('.')

    n_batches = n_samples // batch_size
    eta_ensemble = []
    n_flows = len(flows)

    for i,flow in enumerate(flows):
        print(f'Plotting projections of flow {i+1} of {n_flows} ...')

        @tf.function
        def sample_batch():
            print('Tracing sample_batch ...')
            return flow.sample([batch_size])

        eta = []
        bar = progressbar.ProgressBar(max_value=n_batches)
        for k in range(n_batches):
            eta.append(sample_batch().numpy())
            bar.update(k+1)
        eta = np.concatenate(eta, axis=0)

        eta_ensemble.append(eta[:n_samples//n_flows])

        fig = plot_flow_histograms(eta)
        fig.savefig(f'{fname_base_hist}_{i}.{ext_hist}', dpi=150)
        plt.close(fig)

        fig = plot_flow_projections(eta)
        fig.savefig(f'{fname_base_proj}_{i}.{ext_proj}', dpi=150)
        plt.close(fig)

    print('Plotting projections of ensemble of flows ...')
    eta_ensemble = np.concatenate(eta_ensemble, axis=0)
    print(eta_ensemble.shape)

    fig = plot_flow_histograms(eta_ensemble)
    fig.savefig(f'{fname_base_hist}_ensemble.{ext_hist}', dpi=100)
    plt.close(fig)

    fig = plot_flow_projections(eta_ensemble)
    fig.savefig(f'{fname_base_proj}_ensemble.{ext_proj}', dpi=100)
    plt.close(fig)


def plot_flow_histograms(eta):
    n_bins = 60
    r_max = 6.

    r = np.sqrt(np.sum(eta[:,:3]**2, axis=1))

    fig,ax = plt.subplots(1,1, figsize=(8,4), dpi=150)

    ax.hist(r, bins=n_bins, range=(0., r_max))

    r_range = np.linspace(0., r_max, 500)
    prob_r = r_range**2 * 3. * (1+r_range**2)**(-5/2)
    n_samples = eta.shape[0]
    n_r = prob_r * r_max / n_bins * n_samples
    ax.plot(r_range, n_r, c='orange', alpha=0.7)

    ax.set_ylabel(r'$N$')
    ax.set_xlabel(r'$r$')
    ax.set_yticklabels([])

    return fig


def plot_flow_projections(eta):
    fig,ax_arr = plt.subplots(
        3,3,
        figsize=(13,12),
        subplot_kw=dict(aspect='equal')
    )
    fig.subplots_adjust(wspace=0.30, hspace=0.25)

    xlim = (-2., 2.)
    vlim = (-1.5, 1.5)

    for k,(i,j) in enumerate([(0,1), (0,2), (1,2)]):
        ax_arr[0,k].hist2d(eta[:,i], eta[:,j], bins=51, range=[xlim,xlim])
        ax_arr[1,k].hist2d(eta[:,i+3], eta[:,j+3], bins=51, range=[vlim,vlim])

        ax_arr[0,k].set_xlabel(rf'$x_{i}$')
        ax_arr[0,k].set_ylabel(rf'$x_{j}$', labelpad=-5)
        ax_arr[1,k].set_xlabel(rf'$v_{i}$')
        ax_arr[1,k].set_ylabel(rf'$v_{j}$', labelpad=-5)

    r = np.sqrt(np.sum(eta[:,:3]**2, axis=1))
    v = np.sqrt(np.sum(eta[:,3:]**2, axis=1))
    ax_arr[2,0].hist2d(r, v, bins=51, range=[(0.,5.),(0.,1.5)])
    ax_arr[2,0].set_xlabel(r'$r$')
    ax_arr[2,0].set_ylabel(r'$v$', labelpad=0)
    
    bins = 11
    v0 = eta.shape[0] / bins**2
    dv = 0.5*v0

    theta, phi = vec2ang(eta[:,:3])
    ax_arr[2,1].hist2d(
        phi, np.sin(theta),
        bins=bins,
        vmin=v0-dv, vmax=v0+dv,
        cmap='bwr_r'
    )
    ax_arr[2,1].set_xlabel(r'$\varphi_x$')
    ax_arr[2,1].set_ylabel(r'$\sin \theta_x$', labelpad=-5)

    theta, phi = vec2ang(eta[:,3:])
    ax_arr[2,2].hist2d(
        phi, np.sin(theta),
        bins=bins,
        vmin=v0-dv, vmax=v0+dv,
        cmap='bwr_r'
    )
    ax_arr[2,2].set_xlabel(r'$\varphi_v$')
    ax_arr[2,2].set_ylabel(r'$\sin \theta_v$', labelpad=-5)

    for a in ax_arr[2]:
        a.set_aspect('auto')

    return fig


def phi_Miyamoto_Nagai(R, z, a, b):
    return -1 / np.sqrt(R**2 + (np.sqrt(z**2 + b**2) + a)**2)


def rho_Miyamoto_Nagai(R, z, a, b):
    z_eff = np.sqrt(z**2 + b**2)
    return (
        b**2 / (4*np.pi)
        * (a*R**2 + (3*z_eff+a) * (z_eff+a)**2)
        / ((R**2 + (z_eff+a)**2)**(5/2) * z_eff**3)
    )


class MiyamotoNagaiDisk(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def rho(self, R, z):
        z_eff = np.sqrt(z**2 + self.b**2)
        return (
            self.b**2 / (4*np.pi)
            * (self.a*R**2 + (3*z_eff+self.a) * (z_eff+self.a)**2)
            / ((R**2 + (z_eff+self.a)**2)**(5/2) * z_eff**3)
        )

    def phi(self, R, z):
        return -1 / np.sqrt(R**2 + (np.sqrt(z**2 + self.b**2) + self.a)**2)

    def force(self, R, z):
        phi3 = np.abs(self.phi(R, z))**3
        F_R = -R * phi3
        F_z = -z * (1 + self.a/np.sqrt(z**2 + self.b**2)) * phi3
        return F_R, F_z


def force_residuals(phi_nn, q, mn_disk, batch_size=256):
    @tf.function
    def dphi_dq_batch(q_batch):
        with tf.GradientTape() as g:
            g.watch(q_batch)
            phi = phi_nn(q_batch)
        dphi_dq_batch = g.gradient(phi, q_batch)
        return dphi_dq_batch

    F = np.empty(q.shape, dtype='f8')
    n_batches = int(np.ceil(q.shape[0]/batch_size))
    bar = progressbar.ProgressBar(max_value=n_batches)
    for k in range(n_batches):
        i0 = k * batch_size
        i1 = (k+1) * batch_size
        F[i0:i1] = -dphi_dq_batch(tf.constant(q[i0:i1].astype('f4'))).numpy()
        bar.update(k+1)

    #theta = np.arctan2(q[...,1], q[...,0])
    q_mod = np.sqrt(q[...,0]**2 + q[...,1]**2)
    cos_th = q[...,0] / q_mod
    sin_th = q[...,1] / q_mod

    F_R = cos_th * F[...,0] + sin_th * F[...,1]
    F_phi = -sin_th * F[...,0] + cos_th * F[...,1]
    F_z = F[...,2]

    R = np.sqrt(q[...,0]**2 + q[...,1]**2)
    F_R_true, F_z_true = mn_disk.force(R, q[...,2])
    
    dF = np.sqrt(
        (F_R - F_R_true)**2
      + (F_z - F_z_true)**2
      + F_phi**2
    )

    return dF, {'R': (F_R_true, F_R), 'z': (F_z_true, F_z), 'phi': F_phi}


def plot_force_residuals_slices(phi_nn, fname, x_max=5., grid_size=256):
    # (x,y)-plane
    x = np.linspace(-x_max, x_max, grid_size)
    y = np.linspace(-x_max, x_max, grid_size)
    xlim = (x[0], x[-1])
    ylim = (y[0], y[-1])
    x,y = np.meshgrid(x, y)
    s = x.shape
    x.shape = (x.size,)
    y.shape = (y.size,)
    xyz = np.stack([x,y,np.zeros_like(x)], axis=1)

    mn_disk = MiyamotoNagaiDisk(1., 0.1)
    dF,F = force_residuals(phi_nn, xyz, mn_disk)
    F_mod = np.sqrt(F['R'][0]**2 + F['z'][0]**2)
    dFR = F['R'][1] - F['R'][0]
    dFz = F['z'][1] - F['z'][0]
    F_mod.shape = s
    dF.shape = s
    dFR.shape = s
    dFz.shape = s

    fn_base, fn_ext = os.path.splitext(fname)

    spec = [
        (dF/F_mod,
         'total_norm',
         r'$\left| \vec{F}^{\ast}-\vec{F} \right|\ /\ \left|\vec{F}\right|$',
         'viridis'
        ),
        (dF,
         'total',
         r'$\left| \vec{F}^{\ast}-\vec{F} \right|$',
         'viridis'
        ),
        (dFR,
         'R',
         r'$F^{\ast}_R-F_R$',
         'bwr_r'
        ),
        (dFz,
         'z',
         r'$F^{\ast}_z-F_z$',
         'bwr_r'
        )
    ]

    for img,desc,label,cmap in spec:
        kw = dict(extent=xlim+ylim, cmap=cmap)
        if cmap == 'viridis':
            kw['vmin'] = 0.
        elif cmap == 'bwr_r':
            vmax = 1.1 * np.percentile(img, 99.)
            kw['vmax'] = vmax
            kw['vmin'] = -vmax

        fig = plt.figure(figsize=(7,5.6), dpi=200)

        fig.subplots_adjust(
            left=0.10, right=0.92,
            bottom=0.10, top=0.92
        )

        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(img, **kw)
        #ax.set_title(r'$\mathrm{Force\ residuals}$', pad=2)
        ax.set_xlabel(r'$x$', labelpad=-1)
        ax.set_ylabel(r'$y$', labelpad=-2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        cb = fig.colorbar(
            im, ax=ax,
            label=label
        )

        fig.savefig(f'{fn_base}_{desc}{fn_ext}', dpi=250)
        plt.close(fig)


def plot_force_residuals(phi_nn, df_data, fname):
    mn_disk = MiyamotoNagaiDisk(1., 0.1)
    dF,F = force_residuals(phi_nn, df_data['eta'][:,:3], mn_disk)

    fig = plt.figure(figsize=(8,8), dpi=200)

    fig.suptitle(r'$\mathrm{Force\ residuals}$')
    fig.subplots_adjust(
        left=0.12, right=0.88,
        bottom=0.10, top=0.92,
        wspace=0.1, hspace=0.2
    )

    ax = fig.add_subplot(2,2,1)
    xlim = np.percentile(F['R'][0], [0., 100.])
    w = xlim[1] - xlim[0]
    xlim = (xlim[0]-0.2*w, xlim[1]+0.2*w)
    #ax.scatter(*F['R'], s=2, alpha=0.01, edgecolors='none')
    ax.hexbin(*F['R'], extent=xlim+xlim, lw=0.2, gridsize=200, bins='log')
    ax.plot(xlim, xlim, c='w', lw=1., alpha=0.2)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_xlabel(r'$F_{R}$')
    ax.set_ylabel(r'$F_{R}^{\ast}$')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax = fig.add_subplot(2,2,2)
    xlim = np.percentile(F['z'][0], [0., 100.])
    w = xlim[1] - xlim[0]
    xlim = (xlim[0]-0.2*w, xlim[1]+0.2*w)
    #ax.scatter(*F['z'], s=2, alpha=0.01, edgecolors='none')
    ax.hexbin(*F['z'], extent=xlim+xlim, lw=0.2, gridsize=200, bins='log')
    ax.plot(xlim, xlim, c='w', lw=1., alpha=0.2)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_xlabel(r'$F_{z}$')
    ax.set_ylabel(r'$F_{z}^{\ast}$')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax = fig.add_subplot(2,2,3)
    vmax = 1.5 * np.percentile(np.abs(F['phi']), 99.)
    ax.hist(F['phi'], bins=101, range=(-vmax, vmax), density=True)
    ax.set_xlabel(r'$F_{\varphi}^{\ast}$')

    ax = fig.add_subplot(2,2,4)
    ax.hist(dF, bins=100, density=True)
    ax.set_xlabel(r'$\left| \vec{F}^{\ast} - \vec{F} \right|$')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    fig.savefig(fname, dpi=250)


def plot_phi_slices(phi_nn, fname,
                    x_max=5., z_max=1., grid_size=30):
    fig = plt.figure(figsize=(10,8), dpi=200)
    gs = GridSpec(3,3, height_ratios=[1,0.5,0.5])
    fig.subplots_adjust(
        bottom=0.06, top=0.96,
        left=0.06, right=0.98,
        wspace=0.05, hspace=0.25
    )
    ax_phi_xy = fig.add_subplot(gs[0,0])
    ax_rho_xy = fig.add_subplot(gs[0,1])
    ax_lnrho_xy = fig.add_subplot(gs[0,2])
    ax_phi_xz = fig.add_subplot(gs[1,0])
    ax_rho_xz = fig.add_subplot(gs[1,1])
    ax_lnrho_xz = fig.add_subplot(gs[1,2])
    ax_phi_Rz = fig.add_subplot(gs[2,0])
    ax_rho_Rz = fig.add_subplot(gs[2,1])
    ax_lnrho_Rz = fig.add_subplot(gs[2,2])

    # (x,y)-plane
    x = np.linspace(-x_max, x_max, 2*grid_size+1)
    y = np.linspace(-x_max, x_max, 2*grid_size+1)
    xlim = (x[0], x[-1])
    ylim = (y[0], y[-1])
    x,y = np.meshgrid(x, y)
    s = x.shape
    x.shape = (x.size,)
    y.shape = (y.size,)
    xyz = np.stack([x,y,np.zeros_like(x)], axis=1)
    q_grid = tf.constant(xyz.astype('f4'))

    phi,_,d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_nn, q_grid,
        return_phi=True
    )
    phi_img = np.reshape(phi.numpy(), s)
    rho_img = np.reshape(d2phi_dq2.numpy() / (4*np.pi), s)
    ax_phi_xy.imshow(phi_img, extent=xlim+ylim)
    ax_phi_xy.set_title(r'$\Phi$')
    ax_phi_xy.set_xlabel(r'$x$', labelpad=-1)
    ax_phi_xy.set_ylabel(r'$y$', labelpad=-2)
    ax_phi_xy.set_xlim(xlim)
    ax_phi_xy.set_ylim(ylim)

    ax_rho_xy.set_title(r'$\rho$')
    ax_rho_xy.imshow(rho_img, vmin=0., extent=xlim+ylim)
    ax_rho_xy.set_xlabel(r'$x$', labelpad=-1)
    ax_rho_xy.set_yticklabels([])
    ax_rho_xy.set_xlim(xlim)
    ax_rho_xy.set_ylim(ylim)
    ax_rho_xy.tick_params(axis='y', which='both', right=True, left=False)

    ax_lnrho_xy.set_title(r'$\ln \rho$')
    ax_lnrho_xy.imshow(np.log(rho_img), extent=xlim+ylim)
    ax_lnrho_xy.set_xlabel(r'$x$', labelpad=-1)
    ax_lnrho_xy.set_yticklabels([])
    ax_lnrho_xy.set_xlim(xlim)
    ax_lnrho_xy.set_ylim(ylim)
    ax_lnrho_xy.tick_params(axis='y', which='both', right=True, left=False)

    # (x,z)-plane
    x = np.linspace(-x_max, x_max, 2*grid_size+1)
    z = np.linspace(-z_max, z_max, grid_size+1)
    xlim = (x[0], x[-1])
    ylim = (z[0], z[-1])
    x,z = np.meshgrid(x, z)
    s = x.shape
    x.shape = (x.size,)
    z.shape = (z.size,)
    xyz = np.stack([x,np.zeros_like(x),z], axis=1)
    q_grid = tf.constant(xyz.astype('f4'))

    phi,_,d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_nn, q_grid,
        return_phi=True
    )
    phi_img = np.reshape(phi.numpy(), s)
    rho_img = np.reshape(d2phi_dq2.numpy() / (4*np.pi), s)

    ax_phi_xz.imshow(phi_img, extent=xlim+ylim, aspect='auto')
    ax_phi_xz.set_xlabel(r'$x$', labelpad=-1)
    ax_phi_xz.set_ylabel(r'$z$', labelpad=-2)
    ax_phi_xz.set_xlim(xlim)
    ax_phi_xz.set_ylim(ylim)

    ax_rho_xz.imshow(rho_img, vmin=0., extent=xlim+ylim, aspect='auto')
    ax_rho_xz.set_xlabel(r'$x$', labelpad=-1)
    ax_rho_xz.set_yticklabels([])
    ax_rho_xz.set_xlim(xlim)
    ax_rho_xz.set_ylim(ylim)
    ax_rho_xz.tick_params(axis='y', which='both', right=True, left=False)

    ax_lnrho_xz.imshow(np.log(rho_img), extent=xlim+ylim, aspect='auto')
    ax_lnrho_xz.set_xlabel(r'$x$', labelpad=-1)
    ax_lnrho_xz.set_yticklabels([])
    ax_lnrho_xz.set_xlim(xlim)
    ax_lnrho_xz.set_ylim(ylim)
    ax_lnrho_xz.tick_params(axis='y', which='both', right=True, left=False)

    ## (y,z)-plane
    #y = np.linspace(-x_max, x_max, 2*grid_size+1)
    #z = np.linspace(-0.2*x_max, 0.2*x_max, grid_size+1)
    #xlim = (y[0], y[-1])
    #ylim = (z[0], z[-1])
    #y,z = np.meshgrid(y, z)
    #s = y.shape
    #y.shape = (y.size,)
    #z.shape = (z.size,)
    #xyz = np.stack([np.zeros_like(y),y,z], axis=1)
    #q_grid = tf.constant(xyz.astype('f4'))

    #phi,_,d2phi_dq2 = potential_tf.calc_phi_derivatives(
    #    phi_nn, q_grid,
    #    return_phi=True
    #)
    #phi_img = np.reshape(phi.numpy(), s)
    #rho_img = np.reshape(d2phi_dq2.numpy() / (4*np.pi), s)

    #ax_phi_yz.imshow(phi_img, extent=xlim+ylim, aspect='auto')
    #ax_phi_yz.set_xlabel(r'$y$', labelpad=-1)
    #ax_phi_yz.set_ylabel(r'$z$', labelpad=-2)
    #ax_phi_yz.set_xlim(xlim)
    #ax_phi_yz.set_ylim(ylim)

    ##ax_rho_yz.imshow(np.log(rho_img), extent=xlim+ylim, aspect='auto')
    #ax_rho_yz.imshow(rho_img, vmin=0., extent=xlim+ylim, aspect='auto')
    #ax_rho_yz.set_xlabel(r'$y$', labelpad=-1)
    #ax_rho_yz.set_yticklabels([])
    #ax_phi_yz.set_xlim(xlim)
    #ax_phi_yz.set_ylim(ylim)
    #ax_rho_yz.tick_params(axis='y', which='both', right=True, left=False)

    # (R,z)-plane (averaged over azimuth)
    theta = np.linspace(0., 2*np.pi, 32+1)[:-1]
    img_shape = (grid_size+1, 2*grid_size+1)
    phi_img = np.zeros(img_shape, dtype='f8')
    rho_img = np.zeros(img_shape, dtype='f8')
    for th in theta:
        R = np.linspace(-x_max, x_max, 2*grid_size+1)
        z = np.linspace(-z_max, z_max, grid_size+1)
        xlim = (R[0], R[-1])
        ylim = (z[0], z[-1])
        R,z = np.meshgrid(R, z)
        s = R.shape
        R.shape = (R.size,)
        z.shape = (z.size,)
        xyz = np.stack([np.cos(th)*R,np.sin(th)*R,z], axis=1)
        q_grid = tf.constant(xyz.astype('f4'))

        phi,_,d2phi_dq2 = potential_tf.calc_phi_derivatives(
            phi_nn, q_grid,
            return_phi=True
        )
        phi_img += np.reshape(phi.numpy(), s)
        rho_img += np.reshape(d2phi_dq2.numpy() / (4*np.pi), s)
    phi_img /= len(theta)
    rho_img /= len(theta)

    ax_phi_Rz.imshow(phi_img, extent=xlim+ylim, aspect='auto')
    ax_phi_Rz.set_xlabel(r'$R$', labelpad=-1)
    ax_phi_Rz.set_ylabel(r'$z$', labelpad=-2)
    ax_phi_Rz.set_xlim(xlim)
    ax_phi_Rz.set_ylim(ylim)

    ax_rho_Rz.imshow(rho_img, vmin=0., extent=xlim+ylim, aspect='auto')
    ax_rho_Rz.set_xlabel(r'$R$', labelpad=-1)
    ax_rho_Rz.set_yticklabels([])
    ax_rho_Rz.set_xlim(xlim)
    ax_rho_Rz.set_ylim(ylim)
    ax_rho_Rz.tick_params(axis='y', which='both', right=True, left=False)

    ax_lnrho_Rz.imshow(np.log(rho_img), extent=xlim+ylim, aspect='auto')
    ax_lnrho_Rz.set_xlabel(r'$R$', labelpad=-1)
    ax_lnrho_Rz.set_yticklabels([])
    ax_lnrho_Rz.set_xlim(xlim)
    ax_lnrho_Rz.set_ylim(ylim)
    ax_lnrho_Rz.tick_params(axis='y', which='both', right=True, left=False)

    for a in (ax_phi_xy,ax_rho_xy,ax_lnrho_xy):
        a.xaxis.set_major_locator(MultipleLocator(4.))
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_major_locator(MultipleLocator(4.))
        a.yaxis.set_minor_locator(AutoMinorLocator())

    for a in (ax_phi_xz,ax_rho_xz,ax_lnrho_xz,ax_phi_Rz,ax_rho_Rz,ax_lnrho_Rz):
        a.xaxis.set_major_locator(MultipleLocator(4.))
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_major_locator(MaxNLocator(2))
        a.yaxis.set_minor_locator(AutoMinorLocator())

    fig.savefig(fname, dpi=250)
    plt.close(fig)


def plot_phi(phi_nn, df_data, fname,
             r_max=13., x_max=5., r_trunc=None,
             n_samples=1024, grid_size=51):
    idx = np.arange(df_data['eta'].shape[0])
    np.random.shuffle(idx)
    idx = idx[:n_samples]
    q = df_data['eta'][idx,:3]
    R = np.sqrt(q[:,0]**2 + q[:,1]**2)
    z = q[:,2]
    q = tf.constant(q.astype('f4'))

    fig = plt.figure(figsize=(9,2.5), dpi=200)
    gs_left = GridSpec(1,2, left=0.07, right=0.50, wspace=0.28)
    gs_right = GridSpec(1,2, left=0.55, right=0.99, wspace=0.05)
    fig.subplots_adjust(bottom=0.17, top=0.90)

    # add plots to the nested structure
    ax_phisc = fig.add_subplot(gs_left[0,0])
    ax_rhosc = fig.add_subplot(gs_left[0,1])
    ax2 = fig.add_subplot(gs_right[0,0])
    ax3 = fig.add_subplot(gs_right[0,1])

    # Truth vs. estimate
    phi_est = phi_nn(q).numpy()
    phi_theory = phi_Miyamoto_Nagai(R, z, 1., 0.1)
    phi_0 = np.median(phi_theory - phi_est)
    ax_phisc.scatter(
        phi_theory, phi_est+phi_0,
        alpha=0.08,
        s=3,
        edgecolors='none'
    )
    xlim = np.min(phi_theory), np.max(phi_theory)
    ax_phisc.plot(xlim, xlim, c='k', alpha=0.25, lw=1.)
    ax_phisc.set_xlim(xlim)
    ax_phisc.set_ylim(xlim)
    ax_phisc.set_xlabel(r'$\Phi$', labelpad=-1)
    ax_phisc.set_ylabel(r'$\Phi_{\theta^{\ast}}$', labelpad=0)

    _, d2phi_dq2 = potential_tf.calc_phi_derivatives(phi_nn, q)
    rho_est = d2phi_dq2.numpy() / (4.*np.pi)
    rho_theory = rho_Miyamoto_Nagai(R, z, 1., 0.1)
    ax_rhosc.scatter(
        rho_theory, rho_est,
        alpha=0.08,
        s=3,
        edgecolors='none'
    )
    xlim = np.min(rho_theory), np.max(rho_theory)
    ax_rhosc.plot(xlim, xlim, c='k', alpha=0.25, lw=1.)
    ax_rhosc.set_xlim(xlim)
    ax_rhosc.set_ylim(xlim)
    ax_rhosc.set_xlabel(r'$\rho$', labelpad=-1)
    ax_rhosc.set_ylabel(r'$\rho_{\theta^{\ast}}$', labelpad=0)

    # phi in (x,y)-plane
    x = np.linspace(-x_max, x_max, grid_size)
    y = np.linspace(-x_max, x_max, grid_size)
    xlim = (x[0], x[-1])
    ylim = (y[0], y[-1])
    x,y = np.meshgrid(x, y)
    s = x.shape
    x.shape = (x.size,)
    y.shape = (y.size,)
    xyz = np.stack([x,y,np.zeros_like(x)], axis=1)
    q_grid = tf.constant(xyz.astype('f4'))
    phi_img = phi_nn(q_grid).numpy()
    phi_img = np.reshape(phi_img, s)
    ax2.imshow(phi_img, extent=xlim+ylim)
    ax2.set_xlabel(r'$x$', labelpad=-1)
    ax2.set_ylabel(r'$y$', labelpad=-2)
    ax2.set_title(r'$\Phi$')

    #ax2.scatter(
    #    xy_data[:,0], xy_data[:,1],
    #    edgecolors='none',
    #    c='k',
    #    s=2,
    #    alpha=0.1
    #)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    # log(rho) in (x,y)-plane
    #p_grid = tf.random.normal(q_grid.shape)
    _,rho_img = potential_tf.calc_phi_derivatives(phi_nn, q_grid)
    rho_img = np.reshape(rho_img.numpy(), s)
    ax3.imshow(np.log(rho_img), extent=xlim+ylim)
    # rho_img[rho_img < 0] = np.nan
    # ax3.imshow(np.sqrt(rho_img), extent=xlim+ylim)
    ax3.set_xlabel(r'$x$', labelpad=-1)
    ax3.set_yticklabels([])
    ax3.set_title(r'$\ln \rho$')

    #ax3.scatter(
    #    xy_data[:,0], xy_data[:,1],
    #    edgecolors='none',
    #    c='k',
    #    s=2,
    #    alpha=0.1
    #)
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)

    for a in (ax2,ax3):
        a.xaxis.set_major_locator(MultipleLocator(4.))
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_major_locator(MultipleLocator(4.))
        a.yaxis.set_minor_locator(AutoMinorLocator())

    fig.savefig(fname, dpi=200)
    plt.close(fig)


def load_df_data(fname):
    with open(fname, 'r') as f:
        o = json.load(f)

    d = {}
    for key in o:
        d[key] = np.array(o[key], dtype='f4')

    return d


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


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Deep Potential: Plot results for Plummer sphere.',
        add_help=True
    )
    parser.add_argument(
        '-i', '--input',
        type=str, required=True,
        help='Input DF data.'
    )
    parser.add_argument(
        '--flows',
        type=str, nargs='+',
        help='Flow model filename pattern.'
    )
    parser.add_argument(
        '--potential',
        type=str, required=True,
        help='Potential model filename.'
    )
    parser.add_argument(
        '--grad',
        type=str,
        default='plots/flow_gradients_comparison.png',
        help='Gradient plot filename.'
    )
    parser.add_argument(
        '--traj',
        type=str,
        default='plots/flow_trajectory_{:02d}.png',
        help='Trajectory plot filename.'
    )
    parser.add_argument(
        '--slice',
        type=str,
        default='plots/flow_slices.png',
        help='Slice plot filename.'
    )
    parser.add_argument(
        '--proj',
        type=str,
        default='plots/flow_proj.png',
        help='Projection plot filename.'
    )
    parser.add_argument(
        '--hist',
        type=str,
        default='plots/flow_hist.png',
        help='Histogram plot filename.'
    )
    parser.add_argument(
        '--Phi',
        type=str,
        default='plots/potential_vs_ideal.png',
        help='Potential plot filename.'
    )
    parser.add_argument(
        '--Phi-slices',
        type=str,
        default='plots/potential_slices.png',
        help='Potential slices plot filename.'
    )
    parser.add_argument(
        '--flows-only',
        action='store_true',
        help='Only plot results for flows. Ignore potential.'
    )
    parser.add_argument(
        '--potential-only',
        action='store_true',
        help='Only plot results for the potential. Ignore the flows.'
    )
    parser.add_argument(
        '--r-max',
        type=float,
        help='True distribution function truncated at this radius.'
    )
    args = parser.parse_args()

    print('Loading DF data ...')
    df_data = load_df_data(args.input)

    print('z statistics:')
    z90,z99 = np.percentile(np.abs(df_data['eta'][:,2]), [90., 99.])
    z_max = np.max(np.abs(df_data['eta'][:,2]))
    z_std = np.std(df_data['eta'][:,2])
    print(rf'  max: {z_max:.3f}')
    print(rf'  99%: {z99:.3f}')
    print(rf'  90%: {z90:.3f}')
    print(rf'  std: {z_std:.3f}')
    print('')

    if not args.flows_only:
        print('Loading potential model ...')
        phi_model = potential_tf.PhiNN.load(args.potential)

        print('Plotting force residuals ...')
        #plot_force_residuals(phi_model, df_data, 'plots/force_residuals.png')
        plot_force_residuals_slices(phi_model, 'plots/force_residuals_slices.png')

        #print('Plotting potential ...')
        #plot_phi(phi_model, df_data, args.Phi, r_trunc=args.r_max)

        #print('Plotting slices of potential ...')
        #plot_phi_slices(
        #    phi_model, args.Phi_slices,
        #    grid_size=80, z_max=1.0
        #)

    if not args.potential_only:
        print('Loading flow models ...')
        flows = load_flows(args.flows)

        print('Plotting flow trajectories ...')
        plot_flow_trajectories_multiple(flows, args.traj)

        print('Plotting slices through flows ...')
        #plot_flow_slices(flows, args.slice)

        print('Plotting projections and histograms of flows ...')
        plot_flow_projections_ensemble(flows, args.proj, args.hist)

    return 0

if __name__ == '__main__':
    main()

