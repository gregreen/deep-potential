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


def load_data(fname):
    with open(fname, 'r') as f:
        o = json.load(f)
    d = tf.constant(np.array(o['eta'], dtype='f4'))
    return d


def train_flows(data, fname_pattern, plot_fname_pattern,
                n_flows=1, n_hidden=4, hidden_size=32,
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

        flow.save(fname_pattern.format(i))

        fig = utils.plot_loss(loss_history)
        fig.savefig(plot_fname_pattern.format(i), dpi=200)
        plt.close(fig)

    return flow_list


def train_potential(df_data, fname, plot_fname,
                    n_hidden=3, hidden_size=256, lam=1.,
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

    phi_model.save(fname)

    fig = utils.plot_loss(loss_history)
    fig.savefig('plots/plummer_potential_loss_history.png', dpi=200)
    plt.close(fig)

    return phi_model


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

        df_deta_indiv[i] = batch_calc_df_deta(
            flow.prob, eta,
            batch_size=batch_size
        )
        #df_deta += df_deta_i / n_flows

        #if return_indiv:
        #    df_deta_indiv[i] = df_deta_i

    # Average gradients
    df_deta = np.median(df_deta_indiv, axis=0)

    ret = {
        'eta': eta,
        'df_deta': df_deta,
    }
    if return_indiv:
        ret['df_deta_indiv'] = df_deta_indiv
        ret['df_deta'] = np.median(df_deta_indiv, axis=0)

    return ret


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
                "grad_batch_size": {'type':'integer', 'default':1024},
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
    parser.add_argument(
        '--input', '-i',
        type=str, required=True,
        help='Input data.'
    )
    parser.add_argument(
        '--df-grads-fname',
        type=str, default='data/df_gradients.json',
        help='Directory in which to store data.'
    )
    parser.add_argument(
        '--flow-fname',
        type=str, default='models/df/flow_{:02d}')
        help='Filename pattern to store flows in.'
    )
    parser.add_argument(
        '--flow-loss',
        type=str, default='plots/flow_loss_history_{:02d}.png',
        help='Filename pattern for flow loss history plots.'
    )
    parser.add_argument(
        '--potential-fname',
        type=str, default='models/Phi/Phi_{:02d}')
        help='Filename to store potential in.'
    )
    parser.add_argument(
        '--potential-loss',
        type=str, default='plots/potential_loss_history.png',
        help='Filename for potential loss history plots.'
    )
    parser.add_argument('--params', type=str, help='JSON with kwargs.')
    args = parser.parse_args()
    params = load_params(args.params)

    print('Options:')
    print(json.dumps(params, indent=2))

    # Load input phase-space positions
    data = load_data(parser.input)
    print(f'Loaded {d.shape[0]} phase-space positions.')

    # Train normalizing flows
    flows = train_flows(
        data,
        args.flow_fname,
        args.flow_loss,
        **params['df']
    )

    # Re-load the flows (this removes the regularization terms)
    flows = load_flows()

    # Sample from the flows and calculate gradients
    n_samples = params['Phi'].pop('n_samples')
    batch_size = params['Phi'].pop('grad_batch_size')
    df_data = sample_from_flows(
        flows, n_samples,
        return_indiv=True,
        batch_size=grad_batch_size
    )
    save_df_data(df_data, args.df_grads_fname)

    # Fit the potential
    phi_model = train_potential(
        df_data,
        args.potential_fname,
        args.potential_loss,
        **params['Phi']
    )
    
    return 0


if __name__ == '__main__':
    main()
