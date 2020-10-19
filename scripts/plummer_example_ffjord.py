import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
print(f'Tensorflow version {tf.__version__}')
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_probability as tfp
print(f'Tensorflow Probability version {tfp.__version__}')
tfb = tfp.bijectors
tfd = tfp.distributions

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.gridspec import GridSpec

from time import time, sleep
import re
import json
import progressbar
from glob import glob
import gc
import cerberus

import serializers_tf
import potential_tf
import toy_systems
import flow_ffjord_tf
import utils


def sample_df(n_samples, max_dist=None):
    """
    Returns phase-space locations sampled from the Plummer sphere
    distribution function. The shape of the output is
    (n_samples, 6).
    """
    # Instantiate Plummer sphere class
    plummer_sphere = toy_systems.PlummerSphere()
    x,v = plummer_sphere.sample_df(n_samples)
    if max_dist is not None:
        r2 = np.sum(x**2, axis=1)
        idx = (r2 < max_dist**2)
        x = x[idx]
        v = v[idx]
    return tf.concat([x.astype('f4'), v.astype('f4')], axis=1)


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


def plot_flows(flows):
    n_flows = len(flows)

    # Plot flow trajectories
    for i,flow in enumerate(flows):
        print(f'Plotting trajectories of flow {i+1} of {len(flows)} ...')
        fig = plot_flow_trajectories(flow)
        fig.savefig(f'plots/plummer_flow_traj_{i:02d}.png', dpi=200)
        plt.close(fig)

    # Plot slices of flows
    print('Plotting slices through flows ...')
    plot_flow_slices(flows, fname_prefix='plots/plummer_flow_slices')

    # Plot projections of flows along each axis
    n_samples = 1024*1024
    batch_size = 1024
    n_batches = n_samples // batch_size
    eta_ensemble = []

    for i,flow in enumerate(flows):
        print(f'Plotting projections of flow {i+1} of {len(flows)} ...')

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
        #eta = np.concatenate(
        #    [sample_batch().numpy() for k in range(n_batches)],
        #    axis=0
        #)
        #print('Sampled.')

        eta_ensemble.append(eta[:n_samples//n_flows])

        fig = plot_flow_histograms(eta)
        fig.savefig(f'plots/plummer_flow_hist_{i:02d}.png', dpi=150)
        plt.close(fig)

        fig = plot_flow_projections(eta)
        fig.savefig(f'plots/plummer_flow_proj_{i:02d}.png', dpi=100)
        plt.close(fig)

    print('Plotting projections of ensemble of flows ...')
    eta_ensemble = np.concatenate(eta_ensemble, axis=0)

    fig = plot_flow_histograms(eta_ensemble)
    fig.savefig(f'plots/plummer_flow_hist_ensemble.png', dpi=150)
    plt.close(fig)

    fig = plot_flow_projections(eta_ensemble)
    fig.savefig(f'plots/plummer_flow_proj_ensemble.png', dpi=100)
    plt.close(fig)


def plot_flow_slices(flows, fname_prefix='plots/flow_slices'):
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
            fn = f'{fname_prefix}_{{linlog}}_ideal.png'
        else:
            fn = f'{fname_prefix}_{{linlog}}_{k-1:02d}.png'

        fig = gen_figure(img_stack, True)
        fig.savefig(fn.format(linlog='log'))
        plt.close(fig)

        fig = gen_figure([np.exp(img) for img in img_stack], False)
        fig.savefig(fn.format(linlog='lin'))
        plt.close(fig)

    fig = gen_figure(img_avg_stack, True)
    fig.savefig(f'{fname_prefix}_log_ensemble.png')
    plt.close(fig)

    fig = gen_figure([np.exp(img) for img in img_avg_stack], False)
    fig.savefig(f'{fname_prefix}_lin_ensemble.png')
    plt.close(fig)

    dimg_stack = [img-img0 for img,img0 in zip(img_avg_stack,img_ideal_stack)]
    fig = gen_figure(dimg_stack, True, isdiff=True)
    fig.savefig(f'{fname_prefix}_log_diff.png')
    plt.close(fig)


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


def gen_data(n_samples):
    data = sample_df(int(1.2 * n_samples), max_dist=10.0)
    data = tf.constant(data[:n_samples,:])
    return data


def save_data(data, fname):
    o = {'eta': data.numpy().tolist()}
    with open(fname, 'w') as f:
        json.dump(o, f)


def load_data(fname):
    with open(fname, 'r') as f:
        o = json.load(f)
    d = tf.constant(np.array(o['eta'], dtype='f4'))
    return d


def train_flows(data, n_flows=1, n_hidden=4, hidden_size=32,
                n_epochs=128, batch_size=1024, reg={},
                lr_init=2.e-2, lr_final=1.e-4):
    n_samples = data.shape[0]
    n_steps = n_samples * n_epochs // batch_size
    print(f'n_steps = {n_steps}')

    flow_list = []

    for i in range(n_flows):
        print(f'Training flow {i+1} of {n_flows} ...')

        flow = flow_ffjord_tf.FFJORDFlow(6, n_hidden, hidden_size, reg_kw=reg)
        flow_list.append(flow)
        
        loss_history = flow_ffjord_tf.train_flow(
            flow, data,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr_init=lr_init,
            lr_final=lr_final,
            checkpoint_every=None
        )

        flow.save(f'models/plummer_sphere/df/flow_{i:02d}')

        fig = utils.plot_loss(loss_history)
        fig.savefig(f'plots/plummer_flow_loss_history_{i:02d}.png', dpi=200)
        plt.close(fig)

    return flow_list


def train_potential(df_data, n_hidden=3, hidden_size=256, lam=1.,
                    n_epochs=4096, batch_size=1024,
                    lr_init=1.e-3, lr_final=1.e-6):
    # Create model
    phi_model = potential_tf.PhiNN(
        n_dim=3,
        n_hidden=n_hidden,
        hidden_size=hidden_size
    )

    loss_history = potential_tf.train_potential(
        df_data, phi_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr_init=lr_init,
        lr_final=lr_final,
        checkpoint_every=None,
        lam=lam
    )

    phi_model.save('models/plummer_sphere/Phi/Phi')
    #phi_model = potential_tf.PhiNN.load('models/plummer_sphere/Phi/Phi-1')

    fig = utils.plot_loss(loss_history)
    fig.savefig('plots/plummer_potential_loss_history.png', dpi=200)
    plt.close(fig)

    fig = plot_phi(phi_model, df_data, n_samples=1024, grid_size=51, r_max=13.)
    fig.savefig('plots/plummer_potential_vs_ideal.png', dpi=200)
    plt.close(fig)

    return phi_model


def df_ideal(eta):
    q,p = tf.split(eta, 2, axis=1)

    r2 = tf.math.reduce_sum(q**2, axis=1)
    v2 = tf.math.reduce_sum(p**2, axis=1)

    Phi = -(1+r2)**(-1/2)
    E = v2/2 + Phi

    f = tf.clip_by_value(-E, 0, np.inf)**(7/2)

    A = 24 * np.sqrt(2.) / (7. * np.pi**3)

    return A * f


def plot_gradients(df_data, batch_size=1024):
    #plummer_sphere = toy_systems.PlummerSphere()
    #q,p = plummer_sphere.sample_df(n_points)
    #q = tf.constant(q.astype('f4'))
    #p = tf.constant(p.astype('f4'))

    eta = df_data['eta']

    # Calculate ideal gradients
    df_deta_ideal = batch_calc_df_deta(
        df_ideal, eta,
        batch_size=batch_size
    )
    print(f'df/deta (ideal): {df_deta_ideal.shape} {type(df_deta_ideal)}')
    #idx = np.arange(df_deta_ideal.shape[0])
    #np.random.shuffle(idx)
    #df_deta_ideal = df_deta_ideal[idx]

    #
    # Plot the true vs. estimated gradients
    #

    df_deta_est = [df_data['df_deta']]

    if 'df_deta_indiv' in df_data:
        df_deta_est = [np.median(df_data['df_deta_indiv'], axis=0)]
        df_deta_est += [x for x in df_data['df_deta_indiv']]

    xlim_list = []
    nlim_list = []
    
    n_sc = 2**14

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
                xlim = [-0.15, 0.15]
            else:
                xlim = [-0.22, 0.22]
            #if k == 0:
            #    xlim = ax.get_xlim()
            #    ylim = ax.get_ylim()
            #    xlim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
            #    xlim = max(xlim)
            #    xlim = (-xlim, xlim)
            #    xlim_list.append(xlim)
            #else:
            #    xlim = xlim_list[i]

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

        fig.savefig(
            f'plots/flow_gradients_comparison_scatter_{suffix}.png',
            dpi=100
        )
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

        fig.savefig(
            f'plots/flow_gradients_comparison_hist_{suffix}.png',
            dpi=100
        )
        plt.close(fig)


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


def compare_flow_with_ideal(flows):
    r_lim = (0., 5.)
    v_lim = (0., 1.5)
    bins = (50, 50)

    n, r, v = calc_prob_rv_grid(r_lim, v_lim, bins)

    fig,ax_arr = plt.subplots(
        1,4,
        figsize=(7,2.5),
        gridspec_kw=dict(width_ratios=[1,1,1,0.1]),
        dpi=200
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.17,
        top=0.90,
        wspace=0.1
    )

    ax_ideal,ax_flow,ax_diff,cax_diff = ax_arr.flat

    # Ideal distribution
    dr = r[1] - r[0]
    dv = v[1] - v[0]
    N = np.sum(n) * dr * dv
    print(f'\int f(x,v) d^3x d^3v = {N:.5f}')

    # vmax = np.nanmax(N)
    im = ax_ideal.imshow(
        n,
        extent=r_lim+v_lim,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        # rasterized=True
    )

    # 2D histogram of samples
    n_samples = 1024*1024*4 # Increase this number to increase quality of figure

    # Sample from ensemble of flows
    eta_samp = []
    batch_size = 1024*32
    n_flows = len(flows)
    #samples_per_flow = n_samples // n_flows
    for batch in range(n_samples // batch_size):
        flow = flows[batch % n_flows]
        eta_samp.append(flow.sample([batch_size]).numpy())
    eta_samp = np.concatenate(eta_samp, axis=0)

    x_samp,v_samp = np.split(eta_samp, 2, axis=1)
    r_samp = np.sqrt(np.sum(x_samp**2, axis=1))
    v_samp = np.sqrt(np.sum(v_samp**2, axis=1))

    n_samp,_,_,_ = ax_flow.hist2d(
        r_samp,
        v_samp,
        bins=bins,
        range=[r_lim,v_lim],
        density=False,
        rasterized=True
    )

    # Residuals (samples - ideal)
    n_0 = n*dr*dv * n_samples
    # img = (n_samp.T - n_0) / n_0
    img = np.log10(n_samp.T) - np.log10(n_0)
    im_diff = ax_diff.imshow(
        img,
        extent=r_lim+v_lim,
        vmax=0.4,
        vmin=-0.4,
        origin='lower',
        aspect='auto',
        cmap='coolwarm_r',
        interpolation='nearest',
        # rasterized=True
    )

    # Zero-energy line
    for a,c in ((ax_ideal,'w'),(ax_flow,'w'),(ax_diff,'k')):
        a.plot(r, np.sqrt(2.) * (1+r**2)**(-1/4), c=c)
        a.set_xlabel(r'$r$')
        a.text(
            0.95, 0.95, r'$E > 0$',
            ha='right', va='top',
            fontsize=16, c=c,
            transform=a.transAxes
        )
        a.yaxis.set_major_locator(MultipleLocator(2.))
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_major_locator(MultipleLocator(0.5))
        a.yaxis.set_minor_locator(AutoMinorLocator())

    # Colorbars
    # cb = fig.colorbar(im, cax=cax, label=r'$N$')
    cb_diff = fig.colorbar(
        im_diff,
        cax=cax_diff,
        label=r'$\log_{10} \left( f_{\varphi^{\ast}} / f \right)$',
        extend='both'
    )

    # Axes labels
    ax_flow.set_yticklabels([])
    ax_diff.set_yticklabels([])
    ax_ideal.set_ylabel(r'$v$')

    # Labels
    ax_ideal.set_title(r'$\mathrm{Ideal\ DF}$')
    ax_flow.set_title(r'$\mathrm{Flow}$')
    # ax_diff.set_title(r'$\left( \mathrm{Flow} - \mathrm{Ideal} \right) \, / \, \mathrm{Ideal}$')
    ax_diff.set_title(r'$\log_{10} \left( \mathrm{Flow} / \mathrm{Ideal} \right)$')

    fig.savefig(
        f'plots/plummer_df_vs_ideal.png',
        dpi=200
    )
    #fig.savefig(
    #    f'plots/plummer_df_vs_ideal.pdf',
    #    dpi=200
    #)
    plt.close(fig)


def batch_calc_df_deta(f, eta, batch_size):
    df_deta = np.empty_like(eta)
    n_data = eta.shape[0]

    @tf.function
    def calc_grads(batch):
        print(f'Tracing calc_grads with shape = {batch.shape}')
        return flow_ffjord_tf.calc_f_gradients(f, batch)

    for k in range(0,n_data,batch_size):
        if k != 0:
            bar = progressbar.ProgressBar(max_value=n_data)
            bar.update(k)
        #print(f'{k} to {k+batch_size-1} of {n_data}')
        b0,b1 = k, k+batch_size
        eta_k = tf.constant(eta[b0:b1])
        #res = flow_ffjord_tf.calc_f_gradients(f, eta_k).numpy()
        df_deta[b0:b1] = calc_grads(eta_k).numpy()

    bar.update(n_data)

    return df_deta


def sample_from_flows(flow_list, n_samples, return_indiv=False, batch_size=1024):
    n_flows = len(flow_list)

    # Sample from ensemble of flows
    eta = []
    n_batches = n_samples // (n_flows * batch_size)

    for i,flow in enumerate(flow_list):
        print(f'Sampling from flow {i+1} of {n_flows} ...')

        @tf.function
        def sample_batch():
            print('Tracing sample_batch ...')
            return flow.sample([batch_size])

        bar = progressbar.ProgressBar(max_value=n_batches)
        for k in range(n_batches):
            eta.append(sample_batch().numpy().astype('f4'))
            bar.update(k+1)

    eta = np.concatenate(eta, axis=0)

    # Calculate gradients
    df_deta = np.zeros_like(eta)
    if return_indiv:
        df_deta_indiv = np.zeros((n_flows,)+eta.shape, dtype='f4')

    for i,flow in enumerate(flow_list):
        print(f'Calculating gradients of flow {i+1} of {n_flows} ...')

        df_deta_i = batch_calc_df_deta(
            flow.prob, eta,
            batch_size=batch_size
        )
        df_deta += df_deta_i / n_flows

        if return_indiv:
            df_deta_indiv[i] = df_deta_i

        #print('Cache:')
        #print(f._stateful_fn._function_cache._garbage_collectors[0]._cache)
        #get_f_star._stateful_fn._function_cache._garbage_collectors[0]._cache.popitem()
        #gc.collect()

    ret = {
        'eta': eta,
        'df_deta': df_deta,
    }
    if return_indiv:
        ret['df_deta_indiv'] = df_deta_indiv
        ret['df_deta'] = np.median(df_deta_indiv, axis=0)

    return ret


#def plot_phi_model(phi_model):
#    # Instantiate Plummer sphere class
#    plummer_sphere = toy_systems.PlummerSphere()
#
#    x,v = plummer_sphere.sample_df(1024)
#
#    q = tf.constant(x.astype('f4'))
#    Phi = phi_model(q).numpy()
#
#    r = np.sqrt(np.sum(x**2, axis=1))
#    Phi_ideal = plummer_sphere.phi(r)
#
#    Phi_0 = np.median(Phi_ideal - Phi)
#    Phi += Phi_0
#
#    fig,ax = plt.subplots(1,1, figsize=(8,6))
#
#    r_range = np.linspace(0.05, 50., 1000)
#    ax.semilogx(
#        r_range,
#        plummer_sphere.phi(r_range),
#        c='g', alpha=0.2,
#        label='ideal'
#    )
#    ax.scatter(r, Phi, alpha=0.2, s=3, label='NN model')
#    ax.legend(loc='upper left')
#
#    ax.set_xlim(0.05, 50.)
#    ax.set_ylim(-1.4, 0.4)
#
#    ax.set_xlabel(r'$r$')
#    ax.set_ylabel(r'$\Phi$')
#
#    return fig


def plot_phi(phi_nn, df_data, r_max=13., x_max=5., n_samples=1024, grid_size=51):
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
    phi_0 = np.median(phi_r - phi_theory_r)

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
    idx = (np.abs(df_data['eta'][:,2]) < 0.1)
    xy_data = df_data['eta'][idx,:2]
    r2_data = np.sum(xy_data**2, axis=1)
    idx = (r2_data > 9.)
    xy_data = xy_data[idx]

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

    ax2.scatter(
        xy_data[:,0], xy_data[:,1],
        edgecolors='none',
        c='k',
        s=2,
        alpha=0.1
    )
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

    ax3.scatter(
        xy_data[:,0], xy_data[:,1],
        edgecolors='none',
        c='k',
        s=2,
        alpha=0.1
    )
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)

    for a in (ax2,ax3):
        a.xaxis.set_major_locator(MultipleLocator(4.))
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_major_locator(MultipleLocator(4.))
        a.yaxis.set_minor_locator(AutoMinorLocator())

    return fig



def load_flows():
    flow_list = []

    fnames = glob('models/plummer_sphere/df/flow_*-1.index')
    fnames = sorted(fnames)
    fnames = [fn[:-6] for fn in fnames]

    print(f'Found {len(fnames)} flows.')

    for i,fn in enumerate(fnames):
        print(f'Loading flow {i+1} of {len(fnames)} ...')
        print(fn)
        flow = flow_ffjord_tf.FFJORDFlow.load(fname=fn)
        flow_list.append(flow)

    return flow_list


def save_df_data(df_data, fname):
    o = {}
    for key in df_data:
        d = df_data[key]
        o[key] = d.tolist()
        #if isinstance(d, list):
        #    o[key] = [dd.tolist() for dd in d]
        #else:
        #    o[key] = d.tolist()

    with open(fname, 'w') as f:
        json.dump(o, f)


def load_df_data(fname):
    with open(fname, 'r') as f:
        o = json.load(f)

    d = {}
    #for key in ['eta','df_deta']:
    for key in o:
        d[key] = np.array(o[key], dtype='f4')
    #d['df_deta_indiv'] = [np.array(oo,dtype='f4') for oo in o['df_deta_indiv']]

    return d


def load_params(fname):
    d = {}
    if fname is not None:
        with open(fname, 'r') as f:
            d = json.load(f)
    schema = {
        "df": {
            'type': 'dict',
            'schema': {
                "n_flows": {'type':'integer', 'default':1},
                "n_hidden": {'type':'integer', 'default':4},
                "hidden_size": {'type':'integer', 'default':32},
                "reg": {
                    'type': 'dict',
                    'schema': {
                        "dv_dt_reg": {'type':'float'},
                        "kinetic_reg": {'type':'float'},
                        "jacobian_reg": {'type':'float'}
                    }
                },
                "n_epochs": {'type':'integer', 'default':64},
                "batch_size": {'type':'integer', 'default':512},
                "lr_init": {'type':'float', 'default':0.02},
                "lr_final": {'type':'float', 'default':0.0001}
            }
        },
        "Phi": {
            'type': 'dict',
            'schema': {
                "n_samples": {'type':'integer', 'default':524288},
                "n_hidden": {'type':'integer', 'default':3},
                "hidden_size": {'type':'integer', 'default':256},
                "lam": {'type':'float', 'default':1.0},
                "n_epochs": {'type':'integer', 'default':64},
                "batch_size": {'type':'integer', 'default':1024},
                "lr_init": {'type':'float', 'default':0.001},
                "lr_final": {'type':'float', 'default':0.000001}
            }
        }
    }
    validator = cerberus.Validator(schema)
    params = validator.normalized(d)
    return params


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Deep Potential: Plummer sphere example.',
        add_help=True
    )
    parser.add_argument('--params', type=str, help='JSON with kwargs.')
    args = parser.parse_args()
    params = load_params(args.params)

    print('Options:')
    print(json.dumps(params, indent=2))

    n_samples = 1024 * 128
    data = gen_data(n_samples)
    save_data(data, 'data/plummer_observations.json')

    data = load_data('data/plummer_observations.json')
    flows = train_flows(data, **params['df'])

    flows = load_flows()
    plot_flows(flows)
    compare_flow_with_ideal(flows)
    n_samples = params['Phi'].pop('n_samples')
    df_data = sample_from_flows(
        flows, n_samples,
        return_indiv=True,
        batch_size=1024
    )
    save_df_data(df_data, 'data/plummer_df_data.json')
    plot_gradients(df_data)

    df_data = load_df_data('data/plummer_df_data.json')
    phi_model = train_potential(df_data, **params['Phi'])
    
    return 0


if __name__ == '__main__':
    main()
