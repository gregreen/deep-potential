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
from matplotlib.ticker import AutoMinorLocator

from time import time
import re
import json
import progressbar
from glob import glob
import gc

import serializers_tf
import potential_tf
import flow_tf
import toy_systems


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
    phi = np.arctan2(x[:,1], x[:,0])
    theta = np.arctan2(x[:,2], np.sqrt(x[:,0]**2+x[:,1]**2))
    return theta, phi


def plot_samples(eta):
    fig,ax_arr = plt.subplots(
        3,3,
        figsize=(13,12),
        subplot_kw=dict(aspect='equal')
    )
    fig.subplots_adjust(wspace=0.30, hspace=0.25)

    xlim = (-3., 3.)
    vlim = (-1.5, 1.5)

    for k,(i,j) in enumerate([(0,1), (0,2), (1,2)]):
        ax_arr[0,k].hist2d(eta[:,i], eta[:,j], bins=31, range=[xlim,xlim])
        ax_arr[1,k].hist2d(eta[:,i+3], eta[:,j+3], bins=31, range=[vlim,vlim])

        ax_arr[0,k].set_xlabel(rf'$x_{i}$')
        ax_arr[0,k].set_ylabel(rf'$x_{j}$', labelpad=-5)
        ax_arr[1,k].set_xlabel(rf'$v_{i}$')
        ax_arr[1,k].set_ylabel(rf'$v_{j}$', labelpad=-5)
    r = np.sqrt(np.sum(eta[:,:3]**2, axis=1))
    v = np.sqrt(np.sum(eta[:,3:]**2, axis=1))
    ax_arr[2,0].hist2d(r, v, bins=31, range=[(0.,5.),(0.,1.5)])
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


def get_training_callback(n_steps):
    widgets = [
        progressbar.Bar(),
        progressbar.Percentage(), ' |',
        progressbar.Timer(), '|',
        progressbar.ETA(), '|',
        progressbar.Variable('loss', width=6, precision=4), ', ',
        progressbar.Variable('lr', width=8, precision=3)
    ]
    bar = progressbar.ProgressBar(max_value=n_steps, widgets=widgets)
    def update_progressbar(i, n_steps, loss_history, opt):
        loss = np.mean(loss_history[-50:])
        lr = float(opt._decayed_lr(tf.float32))
        bar.update(i+1, loss=loss, lr=lr)
    return update_progressbar


def train_flows(data, n_flows, n_samples, n_epochs=128, batch_size=1024):
    n_dim = 6
    n_units = 4

    n_steps = n_samples * n_epochs // batch_size
    print(f'n_steps = {n_steps}')

    flow_list = []

    for i in range(n_flows):
        print(f'Training flow {i+1} of {n_flows} ...')

        flow = flow_tf.NormalizingFlow(n_dim, n_units)
        flow_list.append(flow)
        
        # Initialize flow by running arbitrary coordinates through it
        flow.nvp.log_prob(tf.random.normal([1,n_dim]))

        n_var = sum([int(tf.size(v)) for v in flow.bij.trainable_variables])
        print(f'Flow has {n_var} trainable variables.')

        # def plt_fn():
        #     return plot_samples(flow.nvp.sample([100000]))
        loss_history = flow_tf.train_flow(
            flow, data,
            n_epochs=n_epochs,
            batch_size=batch_size,
            callback=get_training_callback(n_steps)
            # optimizer=opt,
            #callback=flow_tf.get_training_callback(
            #    flow,
            #    plt_fn=None,
            #    every=1024,
            #    # fname='plummer_flow_training_{i:05d}.png'
            #)
        )

        with open(f'plummer_flow_{i:02d}.json', 'w') as f:
            json.dump(flow.serialize(), f)
        
        fig = plot_samples(flow.nvp.sample([128*1024]).numpy())
        fig.savefig(f'plummer_flow_{i:02d}.png', dpi=100)
        plt.close(fig)

    return flow_list


@tf.function
def df_ideal(q, p):
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

    q = df_data['q']
    p = df_data['p']

    # Calculate ideal gradients
    df_dq_ideal, df_dp_ideal = batch_calc_df_deta(
        df_ideal, q, p,
        batch_size=batch_size
    )
    #f_ideal, df_dq, df_dp = potential_tf.calc_df_deta(
    #    df_ideal,
    #    tf.constant(q.astype('f4')),
    #    tf.constant(p.astype('f4'))
    #)
    #df_dq_ideal = df_dq.numpy()
    #df_dp_ideal = df_dp.numpy()
    print(f'df/dq (ideal): {df_dq_ideal.shape} {type(df_dq_ideal)}')
    print(f'df/dp (ideal): {df_dp_ideal.shape} {type(df_dp_ideal)}')

    #
    # Plot the true vs. estimated gradients
    #

    df_dq_est = [df_data['df_dq']]
    df_dp_est = [df_data['df_dp']]

    if 'df_dq_indiv' in df_data:
        df_dq_est += df_data['df_dq_indiv']
        df_dp_est += df_data['df_dp_indiv']

    xlim_list = []
    nlim_list = []
    
    n_sc = 2**14

    for k,(df_dq,df_dp) in enumerate(zip(df_dq_est, df_dp_est)):
        suffix = (k-1) if k else 'ensemble'

        print(f'Plotting flow: {suffix} ...')

        fig,ax_arr = plt.subplots(2,3, figsize=(16,9))

        for i,ax in enumerate(ax_arr.flat):
            ax.set_aspect('equal')
            if i < 3:
                ax.scatter(
                    df_dq_ideal[:n_sc,i],
                    df_dq[:n_sc,i],
                    alpha=0.1, s=2,
                    edgecolors='none'
                )
            else:
                ax.scatter(
                    df_dp_ideal[:n_sc,i-3],
                    df_dp[:n_sc,i-3],
                    alpha=0.1, s=2,
                    edgecolors='none'
                )

            if k == 0:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xlim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
                xlim = max(xlim)
                xlim = (-xlim, xlim)
                xlim_list.append(xlim)
            else:
                xlim = xlim_list[i]

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

        fig.savefig(f'flow_gradients_comparison_scatter_{suffix}.png', dpi=100)
        plt.close(fig)

        #
        # Plot histogram of gradient residuals along each dimension in phase space
        #

        fig,ax_arr = plt.subplots(2,3, figsize=(16,9))

        for i,ax in enumerate(ax_arr.flat):
            ax.set_aspect('auto')
            if i < 3:
                resid = df_dq[:,i] - df_dq_ideal[:,i]
            else:
                resid = df_dp[:,i-3] - df_dp_ideal[:,i-3]
            
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

        fig.savefig(f'flow_gradients_comparison_hist_{suffix}.png', dpi=100)
        plt.close(fig)


def batch_calc_df_deta(f_func, q, p, batch_size):
    df_dq = np.empty_like(q)
    df_dp = np.empty_like(p)
    n_data = q.shape[0]

    for k in range(0,n_data,batch_size):
        print(f'{k+batch_size} of {n_data}')
        b0,b1 = k, k+batch_size
        qq = tf.constant(q[b0:b1])
        pp = tf.constant(p[b0:b1])
        res = potential_tf.calc_df_deta(f_func, qq, pp)
        df_dq[b0:b1] = res[1].numpy()
        df_dp[b0:b1] = res[2].numpy()

    return df_dq, df_dp



def sample_from_flows(flow_list, n_samples, return_indiv=False, batch_size=1024):
    n_flows = len(flow_list)

    # Sample from ensemble of flows
    eta = []
    for i,flow in enumerate(flow_list):
        print(f'Sampling from flow {i+1} of {n_flows} ...')
        eta.append(flow.nvp.sample([n_samples//n_flows]).numpy())
    eta = np.concatenate(eta, axis=0)
    q = eta[:,:3].astype('f4')
    p = eta[:,3:].astype('f4')

    # Calculate gradients
    df_dq = np.zeros_like(q)
    df_dp = np.zeros_like(p)

    df_dq_indiv, df_dp_indiv = [], []

    for i,flow in enumerate(flow_list):
        print(f'Calculating gradients of flow {i+1} of {n_flows} ...')

        @tf.function
        def get_f_star(q, p):
            print('Tracing get_f_star ...')
            eta = tf.concat([q,p], axis=1)
            return flow.nvp.prob(eta)
        
        df_dq_i, df_dp_i = batch_calc_df_deta(
            get_f_star, q, p,
            batch_size=batch_size
        )
        df_dq += df_dq_i / n_flows
        df_dp += df_dp_i / n_flows

        if return_indiv:
            df_dq_indiv.append(df_dq_i)
            df_dp_indiv.append(df_dp_i)

        print('Cache:')
        print(f._stateful_fn._function_cache._garbage_collectors[0]._cache)
        get_f_star._stateful_fn._function_cache._garbage_collectors[0]._cache.popitem()

        gc.collect()

    ret = {
        'q': q,
        'p': p,
        'df_dq': df_dq,
        'df_dp': df_dp
    }
    if return_indiv:
        ret['df_dq_indiv'] = df_dq_indiv
        ret['df_dp_indiv'] = df_dp_indiv
    return ret


def plot_phi_model(phi_model):
    # Instantiate Plummer sphere class
    plummer_sphere = toy_systems.PlummerSphere()

    x,v = plummer_sphere.sample_df(1024)

    q = tf.constant(x.astype('f4'))
    Phi = phi_model(q).numpy()

    r = np.sqrt(np.sum(x**2, axis=1))
    Phi_ideal = plummer_sphere.phi(r)

    Phi_0 = np.median(Phi_ideal - Phi)
    Phi += Phi_0

    fig,ax = plt.subplots(1,1, figsize=(8,6))

    r_range = np.linspace(0.05, 50., 1000)
    ax.semilogx(
        r_range,
        plummer_sphere.phi(r_range),
        c='g', alpha=0.2,
        label='ideal'
    )
    ax.scatter(r, Phi, alpha=0.2, s=3, label='NN model')
    ax.legend(loc='upper left')

    ax.set_xlim(0.05, 50.)
    ax.set_ylim(-1.4, 0.4)

    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\Phi$')

    return fig


def train_potential(df_data, n_epochs=4096, batch_size=1024):
    n_samples = df_data['q'].shape[0]

    data = tf.stack(
        [
            df_data['q'], df_data['p'],
            df_data['df_dq'], df_data['df_dp']
        ],
        axis=1
    )
    data = tf.data.Dataset.from_tensor_slices(data)

    phi_model = potential_tf.PhiNN(n_dim=3, n_hidden=3, n_features=128)
    phi_param = phi_model.trainable_variables
    n_variables = sum([int(tf.size(param)) for param in phi_param])
    print(f'{n_variables} variables in the gravitational potential model.')

    # How much to weight Laplacian in loss function
    lam = tf.constant(1.0)  # Penalty for negative matter densities
    mu = tf.constant(0.0)   # Penalty for positive matter densities

    # Optimizer
    n_steps = n_epochs * (n_samples // batch_size)
    print(f'{n_steps} steps planned.')
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        5.e-2,
        n_steps,
        0.0001,
        staircase=False
    )
    opt = tfa.optimizers.RectifiedAdam(
        lr_schedule,
        total_steps=n_steps,
        warmup_proportion=0.1
    )

    # Set up batches of data
    batches = data.repeat(n_epochs)
    batches = batches.shuffle(n_samples, reshuffle_each_iteration=True)
    batches = batches.batch(batch_size, drop_remainder=True)

    loss_history = []

    t0 = time()

    callback = get_training_callback(n_steps)

    for i,b in enumerate(batches):
        # Unpack the data from the batch
        q_b, p_b, df_dq_b, df_dp_b = [
            tf.squeeze(x) for x in tf.split(b, 4, axis=1)
        ]

        # Calculate the loss and its gradients w.r.t. the parameters
        loss, dloss_dparam = potential_tf.get_phi_loss_gradients(
            phi_model, phi_param,
            q_b, p_b,
            df_dq=df_dq_b,
            df_dp=df_dp_b,
            lam=lam,
            mu=mu
        )

        # Take step using optimizer
        opt.apply_gradients(zip(dloss_dparam, phi_param))

        # Logging
        loss_history.append(loss)

        callback(i, n_steps, loss_history, opt)

        if (i % 128 == 0) or (i == n_steps - 1):
            #loss_avg = np.mean(loss_history[-128:])
            #lr = float(opt._decayed_lr(tf.float32))
            #print(
            #    f'Step {i+1} of {n_steps} : '
            #    f'<loss> = {loss_avg:.5g} '
            #    f'lr = {lr:.5g}'
            #)
            fig = plot_phi_model(phi_model)
            fig.savefig(f'phi_training_{i:05d}.png', dpi=150)
            plt.close(fig)

    t1 = time()
    print(f'Elapsed time: {t1-t0:.1f} s')

    d = phi_model.serialize()
    with open('plummer_phi_nn.json', 'w') as f:
        json.dump(d, f)

    return phi_model


def load_flows():
    flow_list = []

    fnames = glob('plummer_flow_??.json')
    fnames = sorted(fnames)

    for i,fn in enumerate(fnames):
        print(f'Loading flow {i+1} of {len(fnames)} ...')
        with open(fn, 'r') as f:
            d = json.load(f)
        flow_list.append(flow_tf.NormalizingFlow.deserialize(d))

    return flow_list


def save_df_data(df_data, fname):
    o = {}
    for key in df_data:
        d = df_data[key]
        if isinstance(d, list):
            o[key] = [dd.tolist() for dd in d]
        else:
            o[key] = d.tolist()

    with open(fname, 'w') as f:
        json.dump(o, f)


def load_df_data(fname):
    with open(fname, 'r') as f:
        o = json.load(f)

    d = {}
    for key in ['q','p','df_dq','df_dp']:
        d[key] = np.array(o[key], dtype='f4')
    for key in ['df_dq_indiv','df_dp_indiv']:
        d[key] = [np.array(oo,dtype='f4') for oo in o[key]]

    return d


def main():
    #n_samples = 1024 * 128
    #data = gen_data(n_samples)
    #save_data(data, 'plummer_observations.json')

    #n_flows = 16
    #data = load_data('plummer_observations.json')
    #flows = train_flows(data, n_flows, n_samples, n_epochs=256, batch_size=8192)

    #n_samples = 1024 * 32
    #flows = load_flows()
    #df_data = sample_from_flows(flows, n_samples, return_indiv=True, batch_size=4)
    #save_df_data(df_data, 'plummer_df_data.json')

    df_data = load_df_data('plummer_df_data.json')
    #plot_gradients(df_data)
    phi_model = train_potential(df_data, n_epochs=1024, batch_size=2048)
    
    return 0


if __name__ == '__main__':
    main()
