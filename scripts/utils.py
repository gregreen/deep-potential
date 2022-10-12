#!/usr/bin/env python

from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import scipy.ndimage
import progressbar
import pandas as pd

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


def plot_loss(train_loss_hist, val_loss_hist=None, lr_hist=None, smoothing='auto'):
    """
    Plots the loss history for the training set (train_loss_hist) and validation set 
    (val_loss_hist) and marks where the learning rate dropped (based on lr_hist)
    'significantly'.
    Draws two views, one for the whole history, the other for the last 50%.
    """
    if smoothing == 'auto':
        n_smooth = np.clip(len(train_loss_hist)//16, 4, 128)
    else:
        n_smooth = smoothing

    def smooth_time_series(x):
        w = np.kaiser(2*n_smooth,5)
        w /= np.sum(w)
        x_conv = scipy.ndimage.convolve(x, w, mode='reflect')
        return x_conv

    train_loss_conv = smooth_time_series(train_loss_hist)
    if val_loss_hist is not None:
        val_loss_conv = smooth_time_series(val_loss_hist)

    n = np.arange(len(train_loss_hist))

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
            i0 = len(train_loss_hist) // 2
            train_loss_hist = train_loss_hist[i0:]
            train_loss_conv = train_loss_conv[i0:]
            if val_loss_hist is not None:
                val_loss_hist = val_loss_hist[i0:]
                val_loss_conv = val_loss_conv[i0:]
            if lr_hist is not None:
                lr_hist = lr_hist[i0:]
            n = n[i0:]

        if lr_hist is not None:
            for k in n_drop:
                ax.axvline(k, c='k', alpha=0.1, ls='--')

        l, = ax.plot(n, train_loss_hist, alpha=0.1, label=r'$\mathrm{training\ loss}$')
        ax.plot(
            n, train_loss_conv,
            alpha=0.8,
            color=l.get_color(),
            label=r'$\mathrm{training\ loss\ (smoothed)}$'
        )
        if val_loss_hist is not None:
            ax.plot(
                n, val_loss_conv,
                alpha=0.8,
                label=r'$\mathrm{validation\ loss\ (smoothed)}$'
            )

        ax.set_xlim(n[0], n[-1])
        if i == 1:
            # Choose the y-limit as the 2nd and 98th percentile of the training and validation smoothed loss, with 10% padding
            limit_percent = 2, 98
            ylim = np.percentile(train_loss_conv, limit_percent)
            if val_loss_hist is not None:
                ylim_val = np.percentile(val_loss_conv, limit_percent)
                ylim = (min(ylim[0], ylim_val[0]), max(ylim[1], ylim_val[1]))
            ylim = (ylim[0] - 0.1*(ylim[1] - ylim[0]), ylim[1] + 0.1*(ylim[1] - ylim[0]))
            ax.set_ylim(*ylim)

        ax.grid('on', which='major', alpha=0.25)
        ax.grid('on', which='minor', alpha=0.05)
        ax.set_ylabel(r'$\mathrm{training\ loss}$')
        ax.set_xlabel(r'$\mathrm{training\ step}$')
        if i == 0:
            if val_loss_hist is not None:
                # Rearrange the legend so validation is above training loss.
                # This is because validation lines in general are above training in the plot.
                handles, labels = ax.get_legend_handles_labels()
                ax.legend([handles[0], handles[2], handles[1]], [labels[0], labels[2], labels[1]], loc='upper right')
            else:
                ax.legend(loc='upper right')
        else:
            kw = dict(fontsize=8, transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.2, facecolor='white'))
            if val_loss_hist is not None:
                ax.text(0.95, 0.95, f'$\mathrm{{validation\ loss\ final\ (smoothed)}} = {val_loss_conv[-1]:.4f}$\n$\mathrm{{training\ loss\ final\ (smoothed)}} = {train_loss_conv[-1]:.4f}$', **kw)
            else:
                ax.text(0.95, 0.95, f'$\mathrm{{training\ loss\ final\ (smoothed)}} = {train_loss_conv[-1]:.4f}$', **kw)

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


def save_loss_history(fname, loss_history, val_loss_history=None, lr_history=None, loss_noreg_history=None, val_loss_noreg_history=None):
    data = {'loss': loss_history}
    if val_loss_history is not None:
        data['validation_loss'] = val_loss_history
    if lr_history is not None:
        data['learning_rate'] = lr_history
    if loss_noreg_history is not None:
        data['loss_noreg'] = loss_noreg_history
    if val_loss_noreg_history is not None:
        data['validation_loss_noreg'] = val_loss_noreg_history

    pd.DataFrame(data).to_csv(fname, index=False, float_format='%.12e')


def load_loss_history(fname):
    df = pd.read_csv(fname)
    loss_history = list(df['loss'].values)
    val_loss_history = list(df['validation_loss'].values) if 'validation_loss' in df.columns else None
    lr_history = list(df['learning_rate'].values) if 'learning_rate' in df.columns else None
    loss_noreg_history = list(df['loss_noreg'].values) if 'loss_noreg' in df.columns else None
    val_loss_noreg_history = list(df['validation_loss_noreg'].values) if 'validation_loss_noreg' in df.columns else None

    return loss_history, val_loss_history, lr_history, loss_noreg_history, val_loss_noreg_history


def plot_corr(ax, x, y,
              x_lim=None, d_max=None,
              bins=(50,31), pct=(16,50,84),
              normalization='balanced'):
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

    if normalization == None:
        norm = np.ones(n.shape[0])
    elif normalization == 'sum':
        norm = np.sum(n, axis=1) + 1.e-10
    elif normalization == 'max':
        norm = np.max(n, axis=1) + 1.e-10
    elif normalization == 'balanced':
        norm0 = np.sum(n, axis=1)
        norm1 = np.max(n, axis=1)
        norm = np.sqrt(norm0*norm1) + 1.e-10
    else:
        raise ValueError(f'Unrecognized normalization: "{normalization}"')
    n /= norm[:,None]

    #n = n**gamma

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


def hist2d_mean(ax, x, y, c,
                vmin=None, vmax=None, cmap=None,
                bins=10, range=None):
    kw = dict(bins=bins, range=range, density=False)
    nc,xedges,yedges = np.histogram2d(x, y, weights=c, **kw)
    n,_,_ = np.histogram2d(x, y, **kw)
    img = nc / n

    extent = (
        xedges[0], xedges[-1],
        yedges[0], yedges[-1]
    )

    im = ax.imshow(
        img.T,
        extent=extent,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )

    return im


def main():
    rng = np.random.default_rng()

    x = [rng.uniform(0., 0.1, 1000), rng.uniform(0., 1.0, 1000)]
    y = [rng.uniform(0., 1.0, 1000), rng.uniform(0., 0.1, 1000)]
    c = [np.ones(1000), -1 * np.ones(1000)]

    x = np.hstack(x)
    y = np.hstack(y)
    c = np.hstack(c)

    fig,ax = plt.subplots(1,1, figsize=(4,3), dpi=200)

    im = hist2d_mean(
        ax, x, y, c,
        vmin=-1, vmax=1,
        cmap='coolwarm_r',
        bins=10, range=[(0,1),(0,1)]
    )

    fig.colorbar(im, ax=ax)

    fig.savefig('hist2d_mean_example.png', dpi=200)

    return 0

if __name__ == '__main__':
    main()

