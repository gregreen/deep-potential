#!/usr/bin/env python

from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import progressbar

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_training_progressbar_fn(n_steps, loss_history, opt):
    widgets = [
        progressbar.Bar(),
        progressbar.Percentage(), ' |',
        progressbar.Timer(), '|',
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


def plot_loss(loss_history, smoothing=100):
    w = np.kaiser(2*smoothing,5)
    w /= np.sum(w)
    loss_conv = np.hstack([
        np.repeat(loss_history[0], smoothing),
        np.array(loss_history),
        np.array(loss_history)[-smoothing:][::-1]
    ])
    loss_conv = np.convolve(loss_conv, w, mode='valid')

    fig,(ax0,ax1) = plt.subplots(1,2, figsize=(8,4))
    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        wspace=0.25
    )

    ax0.plot(
        np.arange(len(loss_history)),
        loss_history,
        alpha=0.1,
        label=r'$\mathrm{raw}$'
    )
    ax0.plot(
        np.arange(len(loss_conv)),
        loss_conv,
        label=r'$\mathrm{smoothed}$'
    )
    ax0.grid('on', which='major', alpha=0.25)
    ax0.grid('on', which='minor', alpha=0.05)
    ax0.set_ylabel(r'$\mathrm{loss}$')
    ax0.set_xlabel(r'$\mathrm{training\ step}$')
    ax0.legend(loc='upper right')

    i0 = len(loss_history) // 2
    ax1.plot(
        np.arange(i0,len(loss_history)),
        loss_history[i0:],
        alpha=0.1
    )
    ax1.plot(
        np.arange(i0,len(loss_conv)),
        loss_conv[i0:]
    )
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.grid('on', which='major', alpha=0.25)
    ax1.grid('on', which='minor', alpha=0.05)
    ax1.set_xlabel(r'$\mathrm{training\ step}$')

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

