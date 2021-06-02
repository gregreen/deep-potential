#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import scipy.stats

import matplotlib
matplotlib.use('Agg')
#matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
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


def df_ideal(eta, r_max=None):
    q,p = tf.split(eta, 2, axis=1)

    r2 = tf.math.reduce_sum(q**2, axis=1)
    v2 = tf.math.reduce_sum(p**2, axis=1)

    Phi = -(1+r2)**(-1/2)
    E = v2/2 + Phi

    f = tf.clip_by_value(-E, 0, np.inf)**(7/2)

    A = 24 * np.sqrt(2.) / (7. * np.pi**3)
    
    if r_max is not None:
        A /= plummer_mass(r_max)

    return A * f


def plummer_mass(r_max):
    return r_max**3 / (1. + r_max)**1.5


def plot_gradients(df_data, fname, batch_size=1024, r_max=None):
    eta = df_data['eta']

    df_trunc = lambda eta: df_ideal(eta, r_max=r_max)

    # Calculate ideal gradients
    df_deta_ideal = batch_calc_df_deta(
        df_trunc, eta,
        batch_size=batch_size
    )
    print(f'df/deta (ideal): {df_deta_ideal.shape} {type(df_deta_ideal)}')

    #
    # Plot the true vs. estimated gradients
    #

    df_deta_est = [df_data['df_deta']]

    if 'df_deta_indiv' in df_data:
        df_deta_est = [np.median(df_data['df_deta_indiv'], axis=0)]
        df_deta_est += [x for x in df_data['df_deta_indiv']]

    #if r_max is not None:
    #    r2 = np.sum(eta[:,:3]**2, axis=1)
    #    idx = (r2 < (0.8*r_max)**2)
    #    print(f'Keeping {np.count_nonzero(idx)} of {idx.size} points.')
    #    eta = eta[idx]
    #    df_deta_ideal = df_deta_ideal[idx]
    #    for k,v in enumerate(df_deta_est):
    #        df_deta_est[k] = v[idx]

    xlim_list = []
    nlim_list = []

    df_dq_lim = np.array([-0.15, 0.15])
    df_dp_lim = np.array([-0.22, 0.22])

    if r_max is not None:
        M = plummer_mass(r_max)
        df_dq_lim /= M
        df_dp_lim /= M
    
    n_sc = 2**14

    fname_base, ext = os.path.splitext(fname)
    ext = ext.lstrip('.')

    for k,df_deta in enumerate(df_deta_est):
        suffix = (k-1) if k else 'ensemble'

        print(f'Plotting flow: {suffix} ...')

        fig,ax_arr = plt.subplots(2,3, figsize=(16,9))

        for i,ax in enumerate(ax_arr.flat):
            ax.set_aspect('equal')
            ax.scatter(
                df_deta_ideal[:n_sc,i],
                df_deta[:n_sc,i],
                alpha=0.1, s=2,
                edgecolors='none'
            )

            if i < 3:
                xlim = df_dq_lim
            else:
                xlim = df_dp_lim

            ax.set_xlim(xlim)
            ax.set_ylim(xlim)

            ax.plot([xlim[0],xlim[1]], [xlim[0],xlim[1]], c='k', alpha=0.25)

            ax.set_xlabel(r'true')
            ax.set_ylabel(r'normalizing flow')

            ax.grid('on', which='major', alpha=0.20)
            ax.grid('on', which='minor', alpha=0.05)

            ax.set_title(rf'$\mathrm{{d}}f / \mathrm{{d}}\eta_{i}$')

        fig.subplots_adjust(
            hspace=0.25, wspace=0.3,
            top=0.91, bottom=0.06
        )
        fig.suptitle('Performance of normalizing flow gradients', fontsize=20)

        fig.savefig(f'{fname_base}_scatter_{suffix}.{ext}', dpi=100)
        plt.close(fig)

        #
        # Plot histogram of gradient residuals along each dimension in phase space
        #

        fig,ax_arr = plt.subplots(2,3, figsize=(16,9))

        for i,ax in enumerate(ax_arr.flat):
            ax.set_aspect('auto')
            resid = df_deta[:,i] - df_deta_ideal[:,i]
            
            ax.hist(
                resid,
                range=(-0.05, 0.05),
                bins=51,
                log=True
            )

            ax.set_xlabel(r'(normalizing flow) - (true)')
            ax.set_title(rf'$\mathrm{{d}}f / \mathrm{{d}}\eta_{i}$')

            if k == 0:
                nlim = ax.get_ylim()
                nlim_list.append(nlim)
            else:
                nlim = nlim_list[i]
            ax.set_ylim(nlim)

            sigma = np.std(resid)
            kurt = scipy.stats.kurtosis(resid)
            ax.text(
                0.95, 0.95,
                rf'$\sigma = {sigma:.4f}$'+'\n'+rf'$\kappa = {kurt:.2f}$',
                ha='right',
                va='top',
                transform=ax.transAxes
            )

            ax.grid('on', which='major', alpha=0.20)
            ax.grid('on', which='minor', alpha=0.05)

        fig.subplots_adjust(
            hspace=0.25, wspace=0.3,
            top=0.91, bottom=0.06
        )
        fig.suptitle('Performance of normalizing flow gradients', fontsize=20)

        fig.savefig(f'{fname_base}_hist_{suffix}.{ext}', dpi=100)
        plt.close(fig)


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

    log_df_fns = (
        [lambda eta: tf.math.log(df_ideal(eta))]
        + [flow.log_prob for flow in flows]
    )

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


def plot_phi(phi_nn, df_data, fname,
             r_max=13., x_max=5., r_trunc=None,
             n_samples=1024, grid_size=51):
    plummer_sphere = toy_systems.PlummerSphere()
    q,_ = plummer_sphere.sample_df(n_samples)
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

    # phi vs. r
    r = tf.sqrt(tf.reduce_sum(q**2, axis=1))
    phi_r = phi_nn(q).numpy()
    phi_theory_r = plummer_sphere.phi(r.numpy())
    if r_trunc is None:
        phi_0 = np.median(phi_r - phi_theory_r)
    else:
        idx = (r.numpy() < r_trunc)
        phi_0 = np.median(phi_r[idx] - phi_theory_r[idx])

    # rho vs. r
    # rho_theory_r = plummer_sphere.rho(r.numpy())
    _, d2phi_dq2 = potential_tf.calc_phi_derivatives(phi_nn, q)
    rho_r = d2phi_dq2.numpy() / (4.*np.pi)

    r_range = np.logspace(-1.1, np.log10(r_max), 100, base=10.)
    # r_range = np.linspace(0.0, r_max, 100)
    phi_theory_r = plummer_sphere.phi(r_range)
    ax_phisc.scatter(
        r, phi_r-phi_0,
        alpha=0.08,
        s=3,
        label=r'$\mathrm{Approximation}$'
    )
    ax_phisc.semilogx(
        r_range,
        phi_theory_r,
        c='g',
        alpha=0.5,
        label=r'$\mathrm{Theory}$'
    )
    ax_phisc.set_xlabel(r'$r$', labelpad=-1)
    ax_phisc.set_ylabel(r'$\Phi$', labelpad=0)
    ax_phisc.set_xticks([0.01, 0.1, 1., 10.])
    ax_phisc.set_xlim(10**(-1.1), r_max)
    ax_phisc.set_ylim(-1.1, 0.1)
    ax_phisc.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_phisc.yaxis.set_minor_locator(AutoMinorLocator())

    leg = ax_phisc.legend(loc='upper left', fontsize=8)
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    rho_theory_r = plummer_sphere.rho(r_range)
    ax_rhosc.scatter(
        r, rho_r,
        alpha=0.08,
        s=3,
        label=r'$\mathrm{Approximation}$'
    )
    ax_rhosc.semilogx(
        r_range,
        rho_theory_r,
        c='g',
        alpha=0.5,
        label=r'$\mathrm{Theory}$'
    )
    ax_rhosc.set_xlabel(r'$r$', labelpad=-1)
    ax_rhosc.set_ylabel(r'$\rho$', labelpad=1)
    ax_rhosc.set_xticks([0.01, 0.1, 1., 10.])
    ax_rhosc.set_xlim(10**(-1.1), r_max)
    ax_rhosc.set_ylim(-0.1*rho_theory_r[0], 1.2*rho_theory_r[0])
    ax_rhosc.axhline(0., c='k', alpha=0.5)
    ax_rhosc.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_rhosc.yaxis.set_minor_locator(AutoMinorLocator())

    leg = ax_rhosc.legend(loc='upper right', fontsize=8)
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    # Data near the xy-plane
    #idx = (np.abs(df_data['eta'][:,2]) < 0.1)
    #xy_data = df_data['eta'][idx,:2]
    #r2_data = np.sum(xy_data**2, axis=1)
    #idx = (r2_data > 9.)
    #xy_data = xy_data[idx]

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
    p_grid = tf.random.normal(q_grid.shape)
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

    if not args.flows_only:
        print('Loading potential model ...')
        phi_model = potential_tf.PhiNN.load(args.potential)

        print('Plotting potential ...')
        plot_phi(phi_model, df_data, args.Phi, r_trunc=args.r_max)

    if not args.potential_only:
        print('Loading flow models ...')
        flows = load_flows(args.flows)

        print('Plotting DF gradients ...')
        plot_gradients(df_data, args.grad, r_max=args.r_max)

        print('Plotting flow trajectories ...')
        plot_flow_trajectories_multiple(flows, args.traj)

        print('Plotting slices through flows ...')
        plot_flow_slices(flows, args.slice)

        print('Plotting projections and histograms of flows ...')
        plot_flow_projections_ensemble(flows, args.proj, args.hist)

    return 0

if __name__ == '__main__':
    main()

