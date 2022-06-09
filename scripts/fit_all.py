import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
print(f'Tensorflow version {tf.__version__}')
#tf.debugging.set_log_device_placement(True)
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
import h5py
import progressbar
from glob import glob
import gc
import cerberus
import os.path

import serializers_tf
import potential_tf
import toy_systems
import flow_ffjord_tf
import utils


def load_data(fname):
    _,ext = os.path.splitext(fname)
    if ext == '.json':
        with open(fname, 'r') as f:
            o = json.load(f)
        d = tf.constant(np.array(o['eta'], dtype='f4'))
    elif ext in ('.h5', '.hdf5'):
        with h5py.File(fname, 'r') as f:
            o = f['eta'][:].astype('f4')
        d = tf.constant(o)
    else:
        raise ValueError(f'Unrecognized input file extension: "{ext}"')
    return d


def train_flows(data, fname_pattern, plot_fname_pattern, loss_fname,
                n_flows=1, n_hidden=4, hidden_size=32, n_bij=1,
                n_epochs=128, batch_size=1024, validation_frac=0.25,
                reg={}, lr={}, optimizer='RAdam', warmup_proportion=0.1,
                checkpoint_every=None, checkpoint_hours=None,
                max_checkpoints=None):
    n_samples = data.shape[0]
    n_steps = n_samples * n_epochs // batch_size
    print(f'n_steps = {n_steps}')

    flow_list = []

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    print(f'Using mean: {data_mean}')
    print(f'       std: {data_std}')

    for i in range(n_flows):
        print(f'Training flow {i+1} of {n_flows} ...')

        flow = flow_ffjord_tf.FFJORDFlow(
            6, n_hidden, hidden_size, n_bij,
            reg_kw=reg,
            base_mean=data_mean, base_std=data_std
        )
        flow_list.append(flow)

        flow_fname = fname_pattern.format(i)

        checkpoint_dir, checkpoint_name = os.path.split(flow_fname)
        checkpoint_name += '_chkpt'

        lr_kw = {f'lr_{k}':lr[k] for k in lr}

        loss_history, val_loss_history, lr_history = flow_ffjord_tf.train_flow(
            flow, data,
            n_epochs=n_epochs,
            batch_size=batch_size,
            validation_frac=validation_frac,
            optimizer=optimizer,
            warmup_proportion=warmup_proportion,
            checkpoint_every=checkpoint_every,
            checkpoint_hours=checkpoint_hours,
            max_checkpoints=max_checkpoints,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            **lr_kw
        )

        fn = flow.save(flow_fname)
        utils.save_loss_history(
            f'{fn}_loss.txt',
            loss_history,
            val_loss_history=val_loss_history,
            lr_history=lr_history
        )

        fig = utils.plot_loss(
            loss_history,
            val_loss_hist=val_loss_history,
            lr_hist=lr_history
        )
        fig.savefig(plot_fname_pattern.format(i), dpi=200)
        plt.close(fig)

    return flow_list


def train_potential(df_data, fname, plot_fname, loss_fname,
                    n_hidden=3, hidden_size=256, xi=1., lam=1., l2=0,
                    n_epochs=4096, batch_size=1024, validation_frac=0.25,
                    lr={}, optimizer='RAdam', warmup_proportion=0.1,
                    checkpoint_every=None, checkpoint_hours=None,
                    max_checkpoints=None):
    # Estimate typical spatial scale of DF data along each dimension
    q_scale = np.std(df_data['eta'][:,:3], axis=0)

    # Create model
    phi_model = potential_tf.PhiNN(
        n_dim=3,
        n_hidden=n_hidden,
        hidden_size=hidden_size,
        scale=q_scale
    )

    lr_kw = {f'lr_{k}':lr[k] for k in lr}

    checkpoint_dir, checkpoint_name = os.path.split(fname)
    checkpoint_name += '_chkpt'

    loss_history = potential_tf.train_potential(
        df_data, phi_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        xi=xi,
        lam=lam,
        l2=l2,
        validation_frac=validation_frac,
        optimizer=optimizer,
        warmup_proportion=warmup_proportion,
        checkpoint_every=checkpoint_every,
        checkpoint_hours=checkpoint_hours,
        max_checkpoints=max_checkpoints,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        **lr_kw
    )

    fn = phi_model.save(fname)

    utils.save_loss_history(f'{fn}_loss.txt', loss_history)

    fig = utils.plot_loss(loss_history)
    fig.savefig(plot_fname, dpi=200)
    plt.close(fig)

    return phi_model


def batch_calc_df_deta(flow, eta, batch_size):
    n_data = eta.shape[0]

    @tf.function
    def calc_grads(batch):
        print(f'Tracing calc_grads with shape = {batch.shape}')
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(batch)
            f = flow.prob(batch)
        df_deta = g.gradient(f, batch)
        return df_deta

    eta_dataset = tf.data.Dataset.from_tensor_slices(eta).batch(batch_size)

    df_deta = []
    bar = None
    n_generated = 0
    for k,b in enumerate(eta_dataset):
        if k != 0:
            if bar is None:
                bar = progressbar.ProgressBar(max_value=n_data)
            bar.update(n_generated)
        df_deta.append(calc_grads(b))
        n_generated += int(b.shape[0])

    bar.update(n_data)

    df_deta = np.concatenate([b.numpy() for b in df_deta])

    return df_deta


def clipped_vector_mean(v_samp, clip_threshold=5, rounds=5, **kwargs):
    n_samp, n_point, n_dim = v_samp.shape
    
    # Mean vector: shape = (point, dim)
    v_mean = np.mean(v_samp, axis=0)

    for i in range(rounds):
        # Difference from mean: shape = (sample, point)
        dv_samp = np.linalg.norm(v_samp - v_mean[None], axis=2)
        # Identify outliers: shape = (sample, point)
        idx = (dv_samp > clip_threshold * np.median(dv_samp, axis=0)[None])
        # Construct masked array with outliers masked
        mask_bad = np.repeat(np.reshape(idx, idx.shape+(1,)), n_dim, axis=2)
        v_samp_ma = np.ma.masked_array(v_samp, mask=mask_bad)
        # Take mean of masked array
        v_mean = np.ma.mean(v_samp_ma, axis=0)
    
    return v_mean


def sample_from_flows(flow_list, n_samples,
                      return_indiv=False,
                      grad_batch_size=1024,
                      sample_batch_size=1024,
                      f_reduce=np.median):
    n_flows = len(flow_list)

    # Sample from ensemble of flows
    eta = []
    n_batches = n_samples // (n_flows * sample_batch_size)

    for i,flow in enumerate(flow_list):
        print(f'Sampling from flow {i+1} of {n_flows} ...')

        @tf.function
        def sample_batch():
            print('Tracing sample_batch ...')
            return flow.sample([sample_batch_size])

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
            flow, eta,
            batch_size=grad_batch_size
        )
        #df_deta += df_deta_i / n_flows

        #if return_indiv:
        #    df_deta_indiv[i] = df_deta_i

    # Average gradients
    df_deta = f_reduce(df_deta_indiv, axis=0)

    ret = {
        'eta': eta,
        'df_deta': df_deta,
    }
    if return_indiv:
        ret['df_deta_indiv'] = df_deta_indiv
        #ret['df_deta'] = df_deta#np.median(df_deta_indiv, axis=0)

    return ret


def load_flows(fname_patterns, is_fstring=True):
    # Determine filenames
    fnames = []

    if is_fstring: # Filename pattern is f-string
        n_max = 9999
        for i in range(n_max):
            fn = glob(fname_patterns.format(i)+'-1.index')
            if len(fn):
                fnames.append(fn[0][:-6])
            else:
                break
    else: # Multiple shell globbing patterns
        for fn in fname_patterns:
            fnames += glob(fn)
        fnames = sorted(fnames)
        fnames = [fn[:-6] for fn in fnames]

    print(f'Found {len(fnames)} flows.')

    # Load flows
    flow_list = []

    for i,fn in enumerate(fnames):
        print(f'Loading flow {i+1} of {len(fnames)} ...')
        print(fn)
        flow = flow_ffjord_tf.FFJORDFlow.load(fname=fn)
        flow_list.append(flow)

    return flow_list


def save_df_data(df_data, fname):
    kw = dict(compression='lzf', chunks=True)
    with h5py.File(fname, 'w') as f:
        for key in df_data:
            f.create_dataset(key, data=df_data[key], **kw)


def load_df_data(fname, recalc_avg=None):
    d = {}
    with h5py.File(fname, 'r') as f:
        for k in f.keys():
            d[k] = f[k][:].astype('f4')
    
    if recalc_avg == 'mean':
        d['df_deta'] = clipped_vector_mean(d['df_deta_indiv'])
    elif recalc_avg == 'median':
        d['df_deta'] = np.median(d['df_deta_indiv'], axis=0)

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
                "lr": {
                    'type': 'dict',
                    'schema': {
                        "type": {'type':'string', 'default':'step'},
                        "init": {'type':'float', 'default':0.02},
                        "final": {'type':'float', 'default':0.0001},
                        "patience": {'type':'integer', 'default':32},
                        "min_delta": {'type':'float', 'default':0.01}
                    }
                },
                "n_epochs": {'type':'integer', 'default':64},
                "batch_size": {'type':'integer', 'default':512},
                "validation_frac": {'type':'float', 'default':0.25},
                "optimizer": {'type':'string', 'default':'RAdam'},
                "warmup_proportion": {'type':'float', 'default':0.1},
                "checkpoint_every": {'type':'integer'},
                "checkpoint_hours": {'type':'float'},
                "max_checkpoints": {'type':'integer'}
            }
        },
        "Phi": {
            'type': 'dict',
            'schema': {
                "n_samples": {'type':'integer', 'default':524288},
                "grad_batch_size": {'type':'integer', 'default':512},
                "sample_batch_size": {'type':'integer', 'default':1024},
                "n_hidden": {'type':'integer', 'default':3},
                "hidden_size": {'type':'integer', 'default':256},
                "xi": {'type':'float', 'default':1.0},
                "lam": {'type':'float', 'default':1.0},
                "l2": {'type':'float', 'default':0.01},
                "n_epochs": {'type':'integer', 'default':64},
                "batch_size": {'type':'integer', 'default':1024},
                "lr": {
                    'type': 'dict',
                    'schema': {
                        "type": {'type':'string', 'default':'step'},
                        "init": {'type':'float', 'default':0.001},
                        "final": {'type':'float', 'default':0.0001},
                        "patience": {'type':'integer', 'default':32},
                        "min_delta": {'type':'float', 'default':0.01}
                    }
                },
                "validation_frac": {'type':'float', 'default':0.25},
                "optimizer": {'type':'string', 'default':'RAdam'},
                "warmup_proportion": {'type':'float', 'default':0.1},
                "checkpoint_every": {'type':'integer'},
                "checkpoint_hours": {'type':'float'},
                "max_checkpoints": {'type':'integer'}
            }
        }
    }
    validator = cerberus.Validator(schema, allow_unknown=False)
    params = validator.normalized(d)
    return params


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Deep Potential: Fit potential from phase-space samples.',
        add_help=True
    )
    parser.add_argument(
        '--input', '-i',
        type=str, required=True,
        help='Input data.'
    )
    parser.add_argument(
        '--df-grads-fname',
        type=str, default='data/df_gradients.h5',
        help='Directory in which to store data.'
    )
    parser.add_argument(
        '--flow-save-fname',
        type=str, default='models/df/flow_{:02d}',
        help='Filename pattern to store flows in.'
    )
    parser.add_argument(
        '--use-existing-flows',
        type=str, nargs='+',
        help='Assume that flows are already trained.'
    )
    parser.add_argument(
        '--flow-loss',
        type=str, default='plots/flow_loss_history_{:02d}.png',
        help='Filename pattern for flow loss history plots.'
    )
    parser.add_argument(
        '--potential-fname',
        type=str, default='models/Phi/Phi',
        help='Filename to store potential in.'
    )
    parser.add_argument(
        '--potential-loss',
        type=str, default='plots/potential_loss_history.png',
        help='Filename for potential loss history plot.'
    )
    parser.add_argument(
        '--potential-only',
        action='store_true',
        help='Skip fitting of distribution function. Assume DF model exists.'
    )
    parser.add_argument(
        '--flows-only',
        action='store_true',
        help='Train only the normalizing flows. Do not fit the potential.'
    )
    parser.add_argument(
        '--flow-median',
        action='store_true',
        help='Use the median of the flow gradients (default: use the mean).'
    )
    parser.add_argument(
        '--loss-history',
        type=str, default='data/loss_history_{:02d}.txt',
        help='Filename for loss history data.'
    )
    parser.add_argument('--params', type=str, help='JSON with kwargs.')
    args = parser.parse_args()

    if args.potential_only and args.flows_only:
        print('--potential-only and --flows-only are incompatible.')
        return 1

    params = load_params(args.params)
    print('Options:')
    print(json.dumps(params, indent=2))

    if args.potential_only:
        print('Loading DF gradients ...')
        df_data = load_df_data(args.df_grads_fname)
        params['Phi'].pop('n_samples')
        params['Phi'].pop('grad_batch_size')
        params['Phi'].pop('sample_batch_size')
    else:
        if args.use_existing_flows is None:
            # Load input phase-space positions
            data = load_data(args.input)
            print(f'Loaded {data.shape[0]} phase-space positions.')

            # Train normalizing flows
            print('Training normalizing flows ...')
            flows = train_flows(
                data,
                args.flow_save_fname,
                args.flow_loss,
                args.loss_history,
                **params['df']
            )

        if not args.flows_only:
            # Re-load the flows (this removes the regularization terms)
            if args.use_existing_flows is None:
                flows = load_flows(args.flow_save_fname, is_fstring=True)
            else:
                flows = load_flows(args.use_existing_flows, is_fstring=False)

            if not len(flows):
                print('No trained flows were found! Aborting.')
                return 1

            # Sample from the flows and calculate gradients
            print('Sampling from flows ...')
            n_samples = params['Phi'].pop('n_samples')
            grad_batch_size = params['Phi'].pop('grad_batch_size')
            sample_batch_size = params['Phi'].pop('sample_batch_size')
            df_data = sample_from_flows(
                flows, n_samples,
                return_indiv=True,
                grad_batch_size=grad_batch_size,
                sample_batch_size=sample_batch_size,
                f_reduce=np.median if args.flow_median else clipped_vector_mean
            )
            save_df_data(df_data, args.df_grads_fname)

    # Fit the potential
    if not args.flows_only:
        print(params['Phi'])
        print('Fitting the potential ...')
        phi_model = train_potential(
            df_data,
            args.potential_fname,
            args.potential_loss,
            args.loss_history,
            **params['Phi']
        )
    
    return 0


if __name__ == '__main__':
    main()
