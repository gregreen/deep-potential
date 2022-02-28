#!/usr/bin/env python

from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import scipy.ndimage
import progressbar

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_training_progressbar_fn(n_steps, loss_history, opt):
    widgets = [
        progressbar.Bar(),
        progressbar.Percentage(), ' |',
        progressbar.Timer(format='Elapsed: %(elapsed)s'), '|',
        progressbar.AdaptiveETA(), '|',
        progressbar.Variable('loss', width=6, precision=4), ', ',
        progressbar.Variable('lr', width=8, precision=3)
    ]
    bar = progressbar.ProgressBar(max_value=n_steps, widgets=widgets)

    def update_progressbar(i):
        loss = np.mean(loss_history[-50:])
        lr = float(opt._decayed_lr(tf.float32))
        bar.update(i+1, loss=loss, lr=lr)

    return update_progressbar


def plot_loss(loss_hist, val_loss_hist=None, lr_hist=None, smoothing='auto'):
    if smoothing == 'auto':
        n_smooth = np.clip(len(loss_hist)//16, 4, 128)
    else:
        n_smooth = smoothing

    def smooth_time_series(x):
        w = np.kaiser(2*n_smooth,5)
        w /= np.sum(w)
        x_conv = scipy.ndimage.convolve(x, w, mode='reflect')
        return x_conv

    loss_conv = smooth_time_series(loss_hist)
    if val_loss_hist is not None:
        val_loss_conv = smooth_time_series(val_loss_hist)

    n = np.arange(len(loss_hist))

    # Detect discrete drops in learning rate
    if lr_hist is not None:
        lr_hist = np.array(lr_hist)
        lr_ratio = lr_hist[1:] / lr_hist[:-1]
        n_drop = np.where(lr_ratio < 0.95)[0]

    fig,ax_arr = plt.subplots(1,2, figsize=(8,4))
    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        wspace=0.25
    )

    for i,ax in enumerate(ax_arr):
        if i == 1:
            i0 = len(loss_hist) // 2
            loss_hist = loss_hist[i0:]
            loss_conv = loss_conv[i0:]
            if val_loss_hist is not None:
                val_loss_conv = val_loss_conv[i0:]
            if lr_hist is not None:
                lr_hist = lr_hist[i0:]
            n = n[i0:]

        if lr_hist is not None:
            for k in n_drop:
                ax.axvline(k, c='k', alpha=0.1, ls='--')

        l, = ax.plot(n, loss_hist, alpha=0.1, label=r'loss')
        ax.plot(
            n, loss_conv,
            alpha=0.8,
            color=l.get_color(),
            label=r'$\mathrm{loss\ (smoothed)}$'
        )
        if val_loss_hist is not None:
            ax.plot(
                n, val_loss_conv,
                alpha=0.8,
                label=r'$\mathrm{validation\ loss\ (smoothed)}$'
            )

        ax.set_xlim(n[0], n[-1])

        ax.grid('on', which='major', alpha=0.25)
        ax.grid('on', which='minor', alpha=0.05)
        ax.set_ylabel(r'$\mathrm{loss}$')
        ax.set_xlabel(r'$\mathrm{training\ step}$')
        if i == 0:
            ax.legend(loc='upper right')

    return fig


def batch_function(f, batch_size, base_library=tf):
    def g(x, *args, **kwargs):
        o = []
        n_data = x.shape[0]
        for batch in base_library.split(x, range(0,n_data,batch_size)):
            o.append(f(batch, *args, **kwargs))
        #for k in range(0, n_data, batch_size):
        #    b0,b1 = k, k+batch_size
        #    o.append(f(x[b0:b1], *args, **kwargs))
        return base_library.concatenate(o)


def append_to_loss_history(fname, key, loss_history):
    s = f'# {key}\n'
    s += ' '.join(f'{x}' for x in loss_history) + '\n'
    with open(fname, 'a') as f:
        f.write(s)
    ## Read existing data from JSON
    #if os.path.isfile(fname):
    #    with open(fname, 'r') as f:
    #        d = json.load(f)
    #else:
    #    d = {}
    ## Append new data
    #d[key] = list(loss_history)
    ## Re-write data to JSON
    #with open(fname, 'w') as f:
    #    f.dump(d, f)


def save_loss_history(fname, loss_history, val_loss_history=None, lr_history=None):
    data = [loss_history]
    header = f'{"loss": >16s}'
    if val_loss_history is not None:
        data.append(val_loss_history)
        header += f' {"validation_loss": >18s}'
    if val_loss_history is not None:
        data.append(lr_history)
        header += f' {"learning_rate": >18s}'
    data = np.stack(data, axis=1)
    np.savetxt(fname, data, header=header, fmt='%.12e')


def load_loss_history(fname):
    data = np.loadtxt(fname)
    loss_history = data[:,0].tolist()
    val_loss_history = data[:,1].tolist()
    lr_history = data[:,2].tolist()
    return loss_history, val_loss_history, lr_history


def plot_corr(ax, x, y, x_lim=None, d_max=None, bins=(50,31), pct=(16,50,84)):
    if x_lim is None:
        x_min, x_max = np.min(x), np.max(x)
        # w = x_max - x_min
        xlim = (x_min, x_max)
    else:
        xlim = x_lim

    if d_max is None:
        dmax = 1.2 * np.percentile(np.abs(y-x), 99.9)
    else:
        dmax = d_max
    dlim = (-dmax, dmax)

    d = y - x
    n,x_edges,_ = np.histogram2d(x, d, range=(xlim, dlim), bins=bins)

    norm = np.sum(n, axis=1) + 1.e-10
    n /= norm[:,None]

    ax.imshow(
      n.T,
      origin='lower',
      interpolation='nearest',
      aspect='auto',
      extent=xlim+dlim,
      cmap='binary'
    )
    ax.plot(xlim, [0.,0.], c='b', alpha=0.2, lw=1)

    if len(pct):
        x_pct = np.empty((3, len(x_edges)-1))
        for i,(x0,x1) in enumerate(zip(x_edges[:-1],x_edges[1:])):
            idx = (x > x0) & (x < x1)
            if np.any(idx):
                x_pct[:,i] = np.percentile(d[idx], pct)
            else:
                x_pct[:,i] = np.nan
        
        for i,x_env in enumerate(x_pct):
            ax.step(
                x_edges,
                np.hstack([x_env[0], x_env]),
                c='cyan',
                alpha=0.5
            )

    ax.set_xlim(xlim)
    ax.set_ylim(dlim)


def main():
    return 0

if __name__ == '__main__':
    main()

