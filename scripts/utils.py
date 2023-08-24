#!/usr/bin/env python

from __future__ import print_function, division

import json
import glob
import tensorflow as tf
import numpy as np
import scipy.ndimage
import progressbar
import pandas as pd
import os
import h5py

import matplotlib.pyplot as plt

import potential_tf
import potential_analytic_tf
import flow_ffjord_tf


def load_training_data(fname, cut_attrs=False):
    """
    Loads in the training data, in the form of a (n, d)-dimensional numpy array
    where n is the number of datapoints and d the dimensionality. The attributes,
    specifying relevant metadata for the training data are also loaded in.

    In the case where a training and validation split have already been done,
    those are loaded in as well. (this is necessary when there are duplicate
    datapoints due to upsampling, usually in order to account for the non-
    uniformity of the selection function).

    Inputs:
        fnanme (str): File path to be loaded in. Specifies a hdf5 group
        cut_attrs (bool): Whether the data should be cut to obey the spatial
            extent specified by attrs. This is relevant when handling padded
            data (padding is done for helping flow training by avoiding sharp
            cut-offs).
    """
    def cut(eta, attrs):
        # Don't perform a cut if there are no attrs
        if attrs == {}:
            return eta

        # Cuts eta based on attrs
        r2 = np.sum(eta[:, :3] ** 2, axis=1)
        R2 = np.sum(eta[:, :2] ** 2, axis=1)
        r = r2**0.5
        R = R2**0.5
        z = eta[:, 2]

        if "volume_type" not in attrs or attrs["volume_type"] == "sphere":
            r_in, r_out = 1 / attrs["parallax_max"], 1 / attrs["parallax_min"]
            idx = (r2 > r_in**2) & (r2 < r_out**2)
        elif attrs["volume_type"] == "cylinder":
            R_out, H_out = attrs["R_out"], attrs["H_out"]
            if "r_in" in attrs:
                r_in = attrs["r_in"]
                idx = (r >= r_in) & (R <= R_out) & (np.abs(z) <= H_out)
            if ("R_in" in attrs) and ("H_in" in attrs):
                R_in, H_in = attrs["R_in"], attrs["H_in"]
                idx = (
                    ((R >= R_in) | (np.abs(z) >= H_in)) &
                    (R <= R_out) &
                    (np.abs(z) <= H_out)
                )

        return eta[idx]

    _, ext = os.path.splitext(fname)
    attrs = None
    data = {}
    if ext in (".h5", ".hdf5"):
        with h5py.File(fname, "r") as f:
            attrs = dict(f["eta"].attrs.items())

            data["eta"] = f["eta"][:].astype("f4")

            if cut_attrs:
                data["eta"] = cut(data["eta"], attrs)

            # Check if the train-validation split has been passed
            if "eta_train" in f.keys() and "eta_val" in f.keys():
                print("Loaded in training and validation data")
                data["eta_train"] = f["eta_train"][:].astype("f4")
                data["eta_val"] = f["eta_val"][:].astype("f4")

                if cut_attrs:
                    data["eta_train"] = cut(data["eta_train"], attrs)
                    data["eta_val"] = cut(data["eta_val"], attrs)

            attrs["has_spatial_cut"] = True if attrs != {} else False
            attrs["n"] = len(data["eta"])
    else:
        raise ValueError(f'Unrecognized input file extension: "{ext}"')

    return data, attrs


def load_flow_samples(fname, recalc_avg=None, attrs_to_cut_by=None):
    """
    Loads in the flow samples from a file, and optionally recalculates the
    average flow from several different realizations of the flow. If
    attrs_to_cut_by is specified, eta is cut to only include points inside the
    volume of validity specified by attrs_to_cut_by.
    """
    d = {}
    with h5py.File(fname, "r") as f:
        for k in f.keys():
            d[k] = f[k][:].astype("f4")

    if recalc_avg == "mean":
        d["df_deta"] = clipped_vector_mean(d["df_deta_indiv"])
        if "probs_indiv" in d:
            d["probs"] = clipped_vector_mean(d["probs_indiv"])
    elif recalc_avg == "median":
        d["df_deta"] = np.median(d["df_deta_indiv"], axis=0)
        if "probs_indiv" in d:
            d["probs"] = np.median(d["probs_indiv"])

    if attrs_to_cut_by is not None:
        # Cuts eta based on attrs
        idx = get_index_of_points_inside_attrs(d["eta"], attrs_to_cut_by)

        d["df_deta"] = d["df_deta"][idx]
        d["eta"] = d["eta"][idx]
        if "probs" in d:
            d["probs"] = d["probs"][idx]

    return d


def load_potential(fname):
    """
    Returns a potential, automatically handling FrameShift (by checking if
    fspec exists). If fname is a directory, load the latest one from there,
    otherwise expects an .index of the checkpoint
    """
    # Make sure that the trailing slash is there for a directory
    if os.path.isdir(fname):
        fname = os.path.join(fname, "")

    directory, tail = os.path.split(fname)

    has_frameshift = False
    if len(glob.glob(directory + "/*_fspec.json")) > 0:
        has_frameshift = True

    is_guided = False
    spec_fname = glob.glob(directory + "/*_spec.json")[0]
    with open(spec_fname, "r") as f:
        is_guided = True if "Guided" in json.load(f)["name"] else False

    is_analytic_potential = False
    with open(spec_fname, "r") as f:
        is_analytic_potential = (
            True if "PhiNNAnalytic" in json.load(f)["name"] else False
        )

    if os.path.isdir(fname):
        if is_analytic_potential:
            phi = potential_analytic_tf.PhiNNAnalytic.load_latest(fname)
        elif is_guided:
            phi = potential_tf.PhiNNGuided.load_latest(fname)
        else:
            phi = potential_tf.PhiNN.load_latest(fname)
        fs = potential_tf.FrameShift.load_latest(fname) if has_frameshift else None
    else:
        if is_analytic_potential:
            phi = potential_analytic_tf.PhiNNAnalytic.load(fname[:-6])
        elif is_guided:
            phi = potential_tf.PhiNNGuided.load(fname[:-6])
        else:
            phi = potential_tf.PhiNN.load(fname[:-6])
        fs = potential_tf.FrameShift.load(fname[:-6]) if has_frameshift else None

    return {"phi": phi, "fs": fs}


def load_flows(fname_patterns):
    """
    Loads in a list of flows from a list of file patterns. If the pattern
    matches a directory, the latest checkpoint is loaded from there, otherwise
    the .index file is expected.
    """
    flow_list = []

    fnames = []
    print(fname_patterns)
    for fn in fname_patterns:
        fnames += glob.glob(fn)
    fnames = sorted(fnames)
    if len(fnames) == 0:
        raise ValueError("Can't find any flows!")

    for i, fn in enumerate(fnames):
        print(f"Loading flow {i+1} of {len(fnames)} ...")
        if os.path.isdir(fn):
            print(f"  Loading latest checkpoint from directory {fn} ...")
            flow = flow_ffjord_tf.FFJORDFlow.load_latest(fn)
        else:
            print(f"  Loading {fn} ...")
            flow = flow_ffjord_tf.FFJORDFlow.load(fn[:-6])
        flow_list.append(flow)

    return flow_list


def clipped_vector_mean(v_samp, clip_threshold=5, rounds=5, **kwargs):
    """
    Antiquated function, this is used for multiple flow samples
    """
    n_samp, n_point, n_dim = v_samp.shape

    # Mean vector: shape = (point, dim)
    v_mean = np.mean(v_samp, axis=0)

    for i in range(rounds):
        # Difference from mean: shape = (sample, point)
        dv_samp = np.linalg.norm(v_samp - v_mean[None], axis=2)
        # Identify outliers: shape = (sample, point)
        idx = dv_samp > clip_threshold * np.median(dv_samp, axis=0)[None]
        # Construct masked array with outliers masked
        mask_bad = np.repeat(np.reshape(idx, idx.shape + (1,)), n_dim, axis=2)
        v_samp_ma = np.ma.masked_array(v_samp, mask=mask_bad)
        # Take mean of masked array
        v_mean = np.ma.mean(v_samp_ma, axis=0)

    return v_mean


def get_training_progressbar_fn(n_steps, loss_history, opt):
    """
    Returns a function which can be called to update the progressbar for
    training. The function takes a single argument, the current training step,
    and updates the progressbar accordingly.
    """
    widgets = [
        progressbar.Bar(),
        progressbar.Percentage(),
        " |",
        progressbar.Timer(format="Elapsed: %(elapsed)s"),
        "|",
        progressbar.AdaptiveETA(),
        "|",
        progressbar.Variable("loss", width=6, precision=4),
        ", ",
        progressbar.Variable("lr", width=8, precision=3),
    ]
    bar = progressbar.ProgressBar(max_value=n_steps, widgets=widgets)

    def update_progressbar(i):
        loss = np.mean(loss_history[-50:])
        lr = float(opt._decayed_lr(tf.float32))
        bar.update(i + 1, loss=loss, lr=lr)

    return update_progressbar


def plot_loss(train_loss_hist, val_loss_hist=None, lr_hist=None, smoothing="auto"):
    """
    Plots the loss history for the training set (train_loss_hist) and validation set
    (val_loss_hist) and marks where the learning rate dropped (based on lr_hist)
    'significantly'. Draws two views, one for the whole history, the other
    for the last 50%.
    """
    if smoothing == "auto":
        n_smooth = np.clip(len(train_loss_hist) // 16, 4, 128)
    else:
        n_smooth = smoothing

    def smooth_time_series(x):
        w = np.kaiser(2 * n_smooth, 5)
        w /= np.sum(w)
        x_conv = scipy.ndimage.convolve(x, w, mode="reflect")
        return x_conv

    train_loss_conv = smooth_time_series(train_loss_hist)
    if val_loss_hist is not None:
        val_loss_conv = smooth_time_series(val_loss_hist)

    n = np.arange(len(train_loss_hist))

    # Detect discrete drops in learning rate
    if lr_hist is not None:
        lr_hist = np.array(lr_hist)
        lr_ratio = lr_hist[lr_hist > 0][1:] / lr_hist[lr_hist > 0][:-1]
        n_drop = np.where(lr_ratio < 0.95)[0]

    fig, ax_arr = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(left=0.14, right=0.98, wspace=0.25)

    for i, ax in enumerate(ax_arr):
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
                ax.axvline(k, c="k", alpha=0.1, ls="--")

        (l,) = ax.plot(
            n, train_loss_hist, alpha=0.1, label=r"$\mathrm{training\ loss}$"
        )
        ax.plot(
            n,
            train_loss_conv,
            alpha=0.8,
            color=l.get_color(),
            label=r"$\mathrm{training\ loss\ (smoothed)}$",
        )
        if val_loss_hist is not None:
            ax.plot(
                n,
                val_loss_conv,
                alpha=0.8,
                label=r"$\mathrm{validation\ loss\ (smoothed)}$",
            )

        ax.set_xlim(n[0], n[-1])
        if i == 1:
            # Choose the y-limit as the 2nd and 98th percentile of the training
            # and validation smoothed loss, with 10% padding
            limit_percent = 2, 98
            ylim = np.percentile(train_loss_conv, limit_percent)
            if val_loss_hist is not None:
                ylim_val = np.percentile(val_loss_conv, limit_percent)
                ylim = (min(ylim[0], ylim_val[0]), max(ylim[1], ylim_val[1]))
            ylim = (
                ylim[0] - 0.1 * (ylim[1] - ylim[0]),
                ylim[1] + 0.1 * (ylim[1] - ylim[0]),
            )
            ax.set_ylim(*ylim)

        ax.grid("on", which="major", alpha=0.25)
        ax.grid("on", which="minor", alpha=0.05)
        ax.set_ylabel(r"$\mathrm{training\ loss}$")
        ax.set_xlabel(r"$\mathrm{training\ step}$")
        if i == 0:
            if val_loss_hist is not None:
                # Rearrange the legend so validation is above training loss.
                # This is because validation lines in general are above training
                # in the plot.
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    [handles[0], handles[2], handles[1]],
                    [labels[0], labels[2], labels[1]],
                    loc="upper right",
                )
            else:
                ax.legend(loc="upper right")
        else:
            kw = dict(
                fontsize=8,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", alpha=0.2, facecolor="white"),
            )
            if val_loss_hist is not None:
                ax.text(
                    0.95,
                    0.95,
                    f"$\mathrm{{validation\ loss\ final\ (smoothed)}} = \
                        {val_loss_conv[-1]:.4f}$\n$\mathrm{{training\ loss\ \
                        final\ (smoothed)}} = {train_loss_conv[-1]:.4f}$",
                    **kw,
                )
            else:
                ax.text(
                    0.95,
                    0.95,
                    f"$\mathrm{{training\ loss\ final\ (smoothed)}} = \
                        {train_loss_conv[-1]:.4f}$",
                    **kw,
                )

    return fig


def save_loss_history(
    fname,
    loss_history,
    val_loss_history=None,
    lr_history=None,
    loss_noreg_history=None,
    val_loss_noreg_history=None,
):
    data = {"loss": loss_history}
    if val_loss_history is not None:
        data["validation_loss"] = val_loss_history
    if lr_history is not None:
        data["learning_rate"] = lr_history
    if loss_noreg_history is not None:
        data["loss_noreg"] = loss_noreg_history
    if val_loss_noreg_history is not None:
        data["validation_loss_noreg"] = val_loss_noreg_history

    pd.DataFrame(data).to_csv(fname, index=False, float_format="%.12e")


def append_to_potential_params_history(phi, fs, potential_params_hist={}):
    """
    Append the current values of the potential parameters to the history.
    """
    parameters_to_check = [
        ("omega", "_omega"),
        ("u_y0", "_u_y"),
        ("u_x0", "_u_x"),
        ("u_z0", "_u_z"),
        ("mn_amp", "_mn_logamp"),
        ("mn_a", "_mn_loga"),
        ("mn_b", "_mn_logb"),
        ("halo_amp", "_halo_logamp"),
        ("halo_a", "_halo_loga"),
        ("bulge_amp", "_bulge_logamp"),
        ("bulge_rcut", "_bulge_logrcut"),
        ("bulge_alpha", "_bulge_alpha"),
        ("dz", "_dz"),
        ("fourier_A0", "_A"),
        ("fourier_B0", "_B"),
        ("fourier_C", "_C"),
    ]

    for param_name, attr in parameters_to_check:
        value = None
        if hasattr(fs, attr):
            value = getattr(fs, attr).numpy()
        if hasattr(phi, attr):
            value = getattr(phi, attr).numpy()
        if value is not None:
            if param_name not in potential_params_hist:
                potential_params_hist[param_name] = []
            if "log" in attr:
                value = np.exp(value)
            if type(value) == np.ndarray:
                potential_params_hist[param_name].append(value[0])
            else:
                potential_params_hist[param_name].append(value)

    return potential_params_hist


def save_potential_params_history(fname, potential_params_hist, lr_history=None):
    """
    Save the history of the potential parameters to a CSV file for later analysis.
    """
    df = pd.DataFrame(dict(potential_params_hist, lr_hist=lr_history))
    df.to_csv(fname, index=False, float_format="%.8e")


def load_potential_params(fname, remove_lr=False):
    """
    Load the history of the potential parameters from a CSV file.
    """
    df = pd.read_csv(fname)
    if remove_lr:
        df = df.drop(columns=["lr_hist"])
    return df.to_dict("list")


def load_loss_history(fname):
    df = pd.read_csv(fname)
    train_loss_history = list(df["loss"].values)
    val_loss_history = (
        list(df["validation_loss"].values) if "validation_loss" in df.columns else None
    )
    lr_history = (
        list(df["learning_rate"].values) if "learning_rate" in df.columns else None
    )
    train_loss_noreg_history = (
        list(df["loss_noreg"].values) if "loss_noreg" in df.columns else None
    )
    val_loss_noreg_history = (
        list(df["validation_loss_noreg"].values)
        if "validation_loss_noreg" in df.columns
        else None
    )

    return (
        train_loss_history,
        val_loss_history,
        lr_history,
        train_loss_noreg_history,
        val_loss_noreg_history,
    )


def plot_corr(
    ax,
    x,
    y,
    x_lim=None,
    d_max=None,
    bins=(50, 31),
    pct=(16, 50, 84),
    normalization="balanced",
):
    if x_lim is None:
        x_min, x_max = np.min(x), np.max(x)
        # w = x_max - x_min
        xlim = (x_min, x_max)
    else:
        xlim = x_lim

    if d_max is None:
        dmax = 1.2 * np.percentile(np.abs(y - x), 99.9)
    else:
        dmax = d_max
    dlim = (-dmax, dmax)

    d = y - x
    n, x_edges, _ = np.histogram2d(x, d, range=(xlim, dlim), bins=bins)

    if normalization == None:
        norm = np.ones(n.shape[0])
    elif normalization == "sum":
        norm = np.sum(n, axis=1) + 1.0e-10
    elif normalization == "max":
        norm = np.max(n, axis=1) + 1.0e-10
    elif normalization == "balanced":
        norm0 = np.sum(n, axis=1)
        norm1 = np.max(n, axis=1)
        norm = np.sqrt(norm0 * norm1) + 1.0e-10
    else:
        raise ValueError(f'Unrecognized normalization: "{normalization}"')
    n /= norm[:, None]

    # n = n**gamma

    ax.imshow(
        n.T,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=xlim + dlim,
        cmap="binary",
    )
    ax.plot(xlim, [0.0, 0.0], c="b", alpha=0.2, lw=1)

    if len(pct):
        x_pct = np.empty((3, len(x_edges) - 1))
        for i, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            idx = (x > x0) & (x < x1)
            if np.any(idx):
                x_pct[:, i] = np.percentile(d[idx], pct)
            else:
                x_pct[:, i] = np.nan

        for i, x_env in enumerate(x_pct):
            ax.step(x_edges, np.hstack([x_env[0], x_env]), c="cyan", alpha=0.5)

    ax.set_xlim(xlim)
    ax.set_ylim(dlim)


def hist2d_mean(ax, x, y, c, vmin=None, vmax=None, cmap=None, bins=10, range=None):
    kw = dict(bins=bins, range=range, density=False)
    nc, xedges, yedges = np.histogram2d(x, y, weights=c, **kw)
    n, _, _ = np.histogram2d(x, y, **kw)
    img = nc / n

    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

    im = ax.imshow(
        img.T,
        extent=extent,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    return im


def get_index_of_points_inside_attrs(eta, attrs, r=None, R=None, z=None):
    """
    Attrs defines a volume, the inside of which is considered fully complete and is used for training the potential.
    This function returns the index of points which lie inside the volume defined by attrs.
    """
    if r is None or R is None or z is None:
        r = np.sum(eta[:, :3] ** 2, axis=1) ** 0.5
        R = np.sum(eta[:, :2] ** 2, axis=1) ** 0.5
        z = eta[:, 2]
    if "volume_type" not in attrs or attrs["volume_type"] == "sphere":
        r_in, r_out = 1 / attrs["parallax_max"], 1 / attrs["parallax_min"]
        idx = (r >= r_in) & (r <= r_out)
    elif attrs["volume_type"] == "cylinder":
        R_out, H_out = attrs["R_out"], attrs["H_out"]
        if "r_in" in attrs:
            r_in = attrs["r_in"]
            idx = (r >= r_in) & (R <= R_out) & (np.abs(z) <= H_out)
        if ("R_in" in attrs) and ("H_in" in attrs):
            R_in, H_in = attrs["R_in"], attrs["H_in"]
            idx = (
                ((R >= R_in) | (np.abs(z) >= H_in))
                & (R <= R_out)
                & (np.abs(z) <= H_out)
            )
    return idx


def main():
    rng = np.random.default_rng()

    x = [rng.uniform(0.0, 0.1, 1000), rng.uniform(0.0, 1.0, 1000)]
    y = [rng.uniform(0.0, 1.0, 1000), rng.uniform(0.0, 0.1, 1000)]
    c = [np.ones(1000), -1 * np.ones(1000)]

    x = np.hstack(x)
    y = np.hstack(y)
    c = np.hstack(c)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=200)

    im = hist2d_mean(
        ax, x, y, c, vmin=-1, vmax=1, cmap="coolwarm_r", bins=10, range=[(0, 1), (0, 1)]
    )

    fig.colorbar(im, ax=ax)

    fig.savefig("hist2d_mean_example.png", dpi=200)

    return 0


if __name__ == "__main__":
    main()
