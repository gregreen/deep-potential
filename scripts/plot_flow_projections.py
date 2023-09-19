#!/usr/bin/env python

from __future__ import print_function, division
from genericpath import isfile
from multiprocessing.sharedctypes import Value

import numpy as np

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

import h5py
import progressbar
import os
from glob import glob
from pathlib import Path
import shutil
from scipy.stats import binned_statistic_dd, binned_statistic_2d

import tensorflow as tf

print(f"Tensorflow version {tf.__version__}")

import utils
import fit_all


dpi = 200


def calc_coords(eta, spherical_origin, cylindrical_origin):
    """Calculate coordinates in different coordinate systems.
    Both Cartesian and spherical share the same origin, cylindrical is separate.

    Cartesian coordinates: x, y, z, vx, vy, vz
    Spherical coordiantes: r, cos(theta), phi, v_radial, v_theta (v_phi is missing)
    Cylindrical coordinates: cyl_R, cyl_z, cyl_phi, cyl_vR, cyl_vz, cyl_vT
    """

    sph_x0 = np.array(spherical_origin)
    cyl_x0 = np.array(cylindrical_origin)

    # Cylindrical
    cyl_R = np.linalg.norm(eta[:, :2] - cyl_x0[:2], axis=1)
    cyl_z = eta[:, 2] - cyl_x0[2]
    cyl_phi = np.arctan2(eta[:, 1] - cyl_x0[1], eta[:, 0] - cyl_x0[0])
    cyl_cos_phi = (eta[:, 0] - cyl_x0[0]) / cyl_R
    cyl_sin_phi = (eta[:, 1] - cyl_x0[1]) / cyl_R
    cyl_vR = eta[:, 3] * cyl_cos_phi + eta[:, 4] * cyl_sin_phi
    cyl_vT = -eta[:, 3] * cyl_sin_phi + eta[:, 4] * cyl_cos_phi
    vz = eta[:, 5]

    cyl = {
        "cylR": cyl_R,
        "cylz": cyl_z,
        "cylphi": cyl_phi,
        "cylvR": cyl_vR,
        "cylvz": vz,
        "cylvT": cyl_vT,
    }

    # Cartesian (vz is already in cylindrical)
    x = eta[:, 0] - sph_x0[0]
    y = eta[:, 1] - sph_x0[1]
    z = eta[:, 2] - sph_x0[2]

    cart = {"x": x, "y": y, "z": z, "vx": eta[:, 3], "vy": eta[:, 4], "vz": vz}

    # Spherical
    r = np.linalg.norm(eta[:, :3] - sph_x0, axis=1)
    vr = np.sum((eta[:, :3] - sph_x0) * eta[:, 3:], axis=1) / r
    costheta = z / r
    sph_R = np.linalg.norm(eta[:, :2] - sph_x0[:2], axis=1)
    phi = np.arctan2(eta[:, 1] - sph_x0[1], eta[:, 0] - sph_x0[0])
    vth = (z * vr - r * vz) / sph_R
    cos_phi = (eta[:, 0] - sph_x0[0]) / cyl_R
    sin_phi = (eta[:, 1] - sph_x0[1]) / cyl_R
    vT = -eta[:, 3] * sin_phi + eta[:, 4] * cos_phi

    sph = {"r": r, "cth": costheta, "phi": phi, "vr": vr, "vth": vth, "vT": vT}

    return dict(**cart, **cyl, **sph)


def sample_from_flows(flow_list, n_samples, attrs, batch_size=1024):
    n_flows = len(flow_list)

    # Sample from ensemble of flows
    n_batches = n_samples // (n_flows * batch_size)
    if attrs["has_spatial_cut"]:
        # If there is a spatial cut, oversample by some factor, and then filter
        # out the datapoints outside the spatial limits to n_samples
        n_batches = int(1.5 * n_batches)
    n_samples_rounded = n_flows * n_batches * batch_size
    print(f"Rounding down # of samples: {n_samples} -> {n_samples_rounded}")
    eta = np.empty((n_samples_rounded, 6), dtype="f4")
    eta[:] = np.nan  # Make it obvious if there are unfilled values at the end

    bar = progressbar.ProgressBar(max_value=n_batches * n_flows)

    batch_idx = 0

    for i, flow in enumerate(flow_list):
        # print(f'Sampling from flow {i+1} of {n_flows} ...')

        @tf.function
        def sample_batch():
            print(f"Tracing sample_batch for flow {i+1} of {n_flows} ...")
            return flow.sample([batch_size])

        for k in range(n_batches):
            j0 = batch_idx * batch_size
            eta[j0: j0 + batch_size] = sample_batch().numpy()
            batch_idx += 1
            bar.update(batch_idx)

    # if attrs['has_spatial_cut']:
    #    idx = utils.get_index_of_points_inside_attrs(eta, attrs)
    #    eta = eta[idx][:n_samples]

    return eta


def plot_1d_marginals(
    coords_train, coords_sample, fig_dir, loss=None, coordsys="cart", fig_fmt=("svg",)
):
    if coordsys == "cart":
        labels = ["$x$", "$y$", r"$z$", "$v_x$", "$v_y$", "$v_z$"]
        keys = ["x", "y", "z", "vx", "vy", "vz"]
    elif coordsys == "cyl":
        labels = ["$R$", "$z$", r"$\phi$", "$v_R$", "$v_z$", r"$v_{\phi}$"]
        keys = ["cylR", "cylz", "cylphi", "cylvR", "cylvz", "cylvT"]
    elif coordsys == "sph":
        labels = [
            "$r$", r"$\cos \theta$", r"$\phi$",
            "$v_r$", r"$v_{\theta}$", r"$v_{\phi}$",
        ]
        keys = ["r", "cth", "phi", "vr", "vth", "vT"]
    else:
        raise ValueError(f"Unknown coordsys: {coordsys}.")

    fig, ax_arr = plt.subplots(2, 3, figsize=(6, 4), dpi=120)

    for i, (ax, l, k) in enumerate(zip(ax_arr.flat, labels, keys)):
        xlim = np.percentile(coords_train[k], [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - 0.2 * w, xlim[1] + 0.2 * w]
        if k == "cylR":
            xlim[0] = max(xlim[0], 0.0)
        elif k == "phi":
            xlim = [np.pi, -np.pi]
        elif k == "cth":
            xlim = [-1, 1]

        kw = dict(range=(np.min(xlim), np.max(xlim)), bins=101, density=True)

        ax.hist(coords_train[k], label=r"$\mathrm{train}$", alpha=0.7, **kw)
        ax.hist(
            coords_sample[k], histtype="step", alpha=0.8, label=r"$\mathrm{NF}$", **kw
        )
        ax.set_xlim(xlim)

        ax.set_xlabel(l, labelpad=0)
        ax.set_yticklabels([])

    ax_arr.flat[0].legend()

    if loss is not None:
        ax = ax_arr.flat[1]
        ax.text(
            0.02,
            0.98,
            rf"$\left< \ln p \right> = {-loss:.3f}$",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    fig.subplots_adjust(
        wspace=0.1, hspace=0.3, left=0.03, right=0.97, bottom=0.12, top=0.97
    )

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"DF_marginals_{coordsys}.{fmt}")
        fig.savefig(fname, dpi=dpi)
    if len(fig_fmt) == 0:
        plt.show()
    plt.close(fig)


def plot_2d_marginal(
    coords_train, coords_sample,
    eta_train, eta_sample,
    fig_dir,
    dim1, dim2,
    logscale=False,
    fig_fmt=("svg",),
):
    labels = [
        "$R$", "$z$", r"$\phi$",
        "$v_R$", "$v_z$", r"$v_{\phi}$",
        "$x$", "$y$", "$z$",
        "$v_x$", "$v_y$", "$v_z$",
        "$r$", r"$\phi$", r"$\cos \theta$",
        "$v_r$", r"$v_{\theta}$", r"$v_{\phi}$",
    ]
    keys = [
        "cylR", "cylz", "cylphi",
        "cylvR", "cylvz", "cylvT",
        "x", "y", "z",
        "vx", "vy", "vz",
        "r", "phi", "cth",
        "vr", "vth", "vT",
    ]

    def extract_dims(dim):
        return coords_train[dim], coords_sample[dim]
        # if dim in keys[:-4]:
        #    return coords_train[dim], coords_sample[dim]
        # elif dim in keys[-4:]:
        #    d = {'x':0, 'y':1, 'vx':3, 'vy':4}[dim]
        #    return eta_train[:,d], eta_sample[:,d]

    x_train, x_sample = extract_dims(dim1)
    y_train, y_sample = extract_dims(dim2)

    labels = {k: l for k, l in zip(keys, labels)}

    fig, (ax_t, ax_s, ax_d, cax_d) = plt.subplots(
        1, 4, figsize=(6, 2), dpi=200, gridspec_kw=dict(width_ratios=[1, 1, 1, 0.05])
    )

    lims = []
    for i, (k, z) in enumerate([(dim1, x_train), (dim2, y_train)]):
        xlim = np.percentile(z, [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - 0.2 * w, xlim[1] + 0.2 * w]
        if k == "cylR":
            xlim[0] = max(xlim[0], 0.0)
        elif k == "phi":
            xlim = [np.pi, -np.pi]
        elif k == "cth":
            xlim = [-1, 1]
        lims.append(xlim)

    kw = dict(
        range=[[np.min(lims[0]), np.max(lims[0])], [np.min(lims[1]), np.max(lims[1])]],
        bins=128,
        rasterized=True,
    )

    n_train = len(x_train)
    n_sample = len(x_sample)

    if logscale:
        kw_col = dict(cmap="cubehelix", norm=LogNorm(vmin=1))
    else:
        kw_col = dict(cmap="viridis")
    nt, _, _, _ = ax_t.hist2d(x_train, y_train, **kw, **kw_col)

    if logscale:
        kw_col["norm"] = LogNorm(
            vmin=n_sample / n_train, vmax=np.max(nt) * n_sample / n_train
        )
    else:
        kw_col["norm"] = Normalize(vmin=0, vmax=np.max(nt) * n_sample / n_train)
    ns, _, _, _ = ax_s.hist2d(x_sample, y_sample, **kw, **kw_col)

    dn = ns / n_sample - nt / n_train
    with np.errstate(divide="ignore", invalid="ignore"):
        dn /= np.sqrt(ns * (n_train / n_sample)) / n_train
    vmax = 5.0
    # dn /= np.max(nt)/n_train
    # vmax = 0.2
    im = ax_d.imshow(
        dn.T,
        extent=lims[0] + lims[1],
        cmap="coolwarm_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        aspect="auto",
        rasterized=True,
    )

    cb = fig.colorbar(
        im,
        cax=cax_d,
        label=r"$\mathrm{Poisson\ significance} \ \left( \sigma \right)$"
        # label=r'$\mathrm{fraction\ of\ max\ density}$'
    )

    ax_s.set_yticklabels([])
    ax_d.set_yticklabels([])

    for ax in (ax_s, ax_t, ax_d):
        ax.set_xlabel(labels[dim1], labelpad=0)

    ax_t.set_ylabel(labels[dim2], labelpad=2)

    ax_t.set_title(r"$\mathrm{training\ data}$", fontsize=10)
    ax_s.set_title(r"$\mathrm{normalizing\ flow\ (NF)}$", fontsize=10)
    ax_d.set_title(r"$\mathrm{NF - training}$", fontsize=10)

    fig.subplots_adjust(left=0.11, right=0.88, bottom=0.22, top=0.88, wspace=0.16)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"DF_marginal_{dim1}_{dim2}.{fmt}")
        fig.savefig(fname, dpi=dpi)
    if len(fig_fmt) == 0:
        plt.show()
    plt.close(fig)


def add_1dpopulation_boundaries(axs, dim1, attrs):
    kw = dict(linestyle=(0, (5, 3)), lw=1.0, color="black", zorder=0)
    # Visualise the boundaries of the population

    valid_keys = ["x", "y", "z", "cylR"]
    plot_sph, plot_cyl = [], []
    if "volume_type" not in attrs or attrs["volume_type"] == "sphere":
        if "r_out" in attrs:
            r_out = attrs["r_out"]
        else:
            r_out = 1 / attrs["parallax_min"]
        if "r_in" in attrs:
            r_in = attrs["r_in"]
        else:
            r_in = 1 / attrs["parallax_max"]
        plot_sph = [r_in, r_out]
    elif attrs["volume_type"] == "cylinder":
        if "r_in" in attrs:
            plot_sph = [attrs["r_in"]]
        else:
            plot_cyl = [(attrs["R_in"], attrs["H_in"])]
        plot_cyl.append((attrs["R_out"], attrs["H_out"]))

    for r in plot_sph:
        if dim1 in valid_keys:
            for ax in axs:
                ax.axvline(r, **kw)
                if dim1 != "cylR":
                    ax.axvline(-r, **kw)
    for R, H in plot_cyl:
        if dim1 in ["x", "y", "cylR"]:
            for ax in axs:
                ax.axvline(R, **kw)
                if dim1 != "cylR":
                    ax.axvline(-R, **kw)
        if dim1 in ["z"]:
            for ax in axs:
                ax.axvline(H, **kw)
                ax.axvline(-H, **kw)


def add_2dpopulation_boundaries(axs, dim1, dim2, attrs, color="white"):
    # Visualise the boundaries of the population
    cartesian_keys = ["x", "y", "z"]
    kw = dict(linestyle=(0, (5, 3)), lw=0.5, color=color)

    plot_sph, plot_cyl = [], []
    if "volume_type" not in attrs or attrs["volume_type"] == "sphere":
        if "r_out" in attrs:
            r_out = attrs["r_out"]
        else:
            r_out = 1 / attrs["parallax_min"]
        if "r_in" in attrs:
            r_in = attrs["r_in"]
        else:
            r_in = 1 / attrs["parallax_max"]
        plot_sph = [r_in, r_out]
    elif attrs["volume_type"] == "cylinder":
        if "r_in" in attrs:
            plot_sph = [attrs["r_in"]]
        else:
            plot_cyl = [(attrs["R_in"], attrs["H_in"])]
        plot_cyl.append((attrs["R_out"], attrs["H_out"]))
    for r in plot_sph:
        if (dim1 in cartesian_keys) and (dim2 in cartesian_keys):
            # Plot circles
            for ax in axs:
                circ = plt.Circle((0, 0), r, fill=False, **kw)
                ax.add_patch(circ)
                circ = plt.Circle((0, 0), r, fill=False, **kw)
                ax.add_patch(circ)
        if dim1 in ["cylR"]:
            for ax in axs:
                ax.axvline(r, **kw)
                ax.axvline(r, **kw)
        if dim2 in ["cylR"]:
            for ax in axs:
                ax.axhline(r, **kw)
                ax.axhline(r, **kw)
    for R, H in plot_cyl:
        if (dim1 in ["x", "y"]) and (dim2 in ["x", "y"]):
            # Plot circles
            for ax in axs:
                circ = plt.Circle((0, 0), R, fill=False, **kw)
                ax.add_patch(circ)
                circ = plt.Circle((0, 0), R, fill=False, **kw)
                ax.add_patch(circ)
        else:
            if dim1 in ["cylR"]:
                for ax in axs:
                    ax.axvline(R, **kw)
                    ax.axvline(R, **kw)
            if dim2 in ["cylR"]:
                for ax in axs:
                    ax.axhline(R, **kw)
                    ax.axhline(R, **kw)
            if dim1 in ["z"]:
                for ax in axs:
                    ax.axvline(H, **kw)
                    ax.axvline(H, **kw)
                    ax.axvline(-H, **kw)
                    ax.axvline(-H, **kw)
            if dim2 in ["z"]:
                for ax in axs:
                    ax.axhline(H, **kw)
                    ax.axhline(H, **kw)
                    ax.axhline(-H, **kw)
                    ax.axhline(-H, **kw)
            if dim1 in ["x", "y"]:
                for ax in axs:
                    ax.axvline(R, **kw)
                    ax.axvline(R, **kw)
                    ax.axvline(-R, **kw)
                    ax.axvline(-R, **kw)
            if dim2 in ["x", "y"]:
                for ax in axs:
                    ax.axhline(R, **kw)
                    ax.axhline(R, **kw)
                    ax.axhline(-R, **kw)
                    ax.axhline(-R, **kw)


def plot_2d_slice(
    coords_train, coords_sample,
    fig_dir,
    dim1, dim2, dimz,
    z, dz,
    attrs,
    fig_fmt=("svg",),
    verbose=False,
):
    labels = [
        "$R$", "$z$", r"$\phi$",
        "$v_R$", "$v_z$", r"$v_{\phi}$",
        "$x$", "$y$", "$z$",
        "$v_x$", "$v_y$", "$v_z$",
        "$r$", r"$\phi$", r"$\cos \theta$",
        "$v_r$", r"$v_{\theta}$", r"$v_{\phi}$",
    ]
    keys = [
        "cylR", "cylz", "cylphi",
        "cylvR", "cylvz", "cylvT",
        "x", "y", "z",
        "vx", "vy", "vz",
        "r", "phi", "cth",
        "vr", "vth", "vT",
    ]

    idx_train = (coords_train[dimz] > z - dz) & (coords_train[dimz] < z + dz)
    idx_sample = (coords_sample[dimz] > z - dz) & (coords_sample[dimz] < z + dz)
    x_train, x_sample = coords_train[dim1][idx_train], coords_sample[dim1][idx_sample]
    y_train, y_sample = coords_train[dim2][idx_train], coords_sample[dim2][idx_sample]

    labels = {k: l for k, l in zip(keys, labels)}

    fig, (ax_t, ax_s, ax_d, cax_d) = plt.subplots(
        1, 4, figsize=(6, 2), dpi=200, gridspec_kw=dict(width_ratios=[1, 1, 1, 0.05])
    )
    axs = [ax_t, ax_s, ax_d]

    lims = []
    for i, (k, val) in enumerate([(dim1, x_train), (dim2, y_train)]):
        xlim = np.percentile(val, [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - 0.2 * w, xlim[1] + 0.2 * w]
        if k == "cylR":
            xlim[0] = max(xlim[0], 0.0)
        elif k == "phi":
            xlim = [np.pi, -np.pi]
        elif k == "cth":
            xlim = [-1, 1]
        lims.append(xlim)

    kw = dict(
        range=[[np.min(lims[0]), np.max(lims[0])], [np.min(lims[1]), np.max(lims[1])]],
        bins=64,
        rasterized=True,
    )

    n_train = len(x_train)
    n_sample = len(x_sample)

    nt, _, _, _ = ax_t.hist2d(x_train, y_train, **kw)
    norm = Normalize(vmin=0, vmax=np.max(nt) * n_sample / n_train)
    ns, _, _, _ = ax_s.hist2d(x_sample, y_sample, norm=norm, **kw)

    if attrs["has_spatial_cut"]:
        add_2dpopulation_boundaries(axs, dim1, dim2, attrs)

    dn = ns / n_sample - nt / n_train
    with np.errstate(divide="ignore", invalid="ignore"):
        dn /= np.sqrt(ns * (n_train / n_sample)) / n_train
    vmax = 5.0
    # dn /= np.max(nt)/n_train
    # vmax = 0.2
    im = ax_d.imshow(
        dn.T,
        extent=lims[0] + lims[1],
        cmap="coolwarm_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        aspect="auto",
        rasterized=True,
    )

    cb = fig.colorbar(
        im,
        cax=cax_d,
        label=r"$\mathrm{Poisson\ significance} \ \left( \sigma \right)$"
        # label=r'$\mathrm{fraction\ of\ max\ density}$'
    )

    ax_s.set_yticklabels([])
    ax_d.set_yticklabels([])

    for ax in (ax_s, ax_t, ax_d):
        ax.set_xlabel(labels[dim1], labelpad=0)

    ax_t.set_ylabel(labels[dim2], labelpad=2)

    ax_d.set_title(r"$\mathrm{NF - training}$", fontsize=10)
    if verbose:
        ax_t.set_title(f"$\mathrm{{training\ data}}$\n$n={len(x_train)}$", fontsize=10)
        ax_s.set_title(
            f"$\mathrm{{normalizing\ flow\ (NF)}}$\n$n={len(x_sample)}$", fontsize=10
        )
        # Print additional info on the subplots
        fig.suptitle(f"${z-dz:.2f}\leq${labels[dimz]}$\leq{z+dz:.2f}$", fontsize=10)

        fig.subplots_adjust(left=0.16, right=0.83, bottom=0.18, top=0.74, wspace=0.16)
    else:
        ax_t.set_title(r"$\mathrm{training\ data}$", fontsize=10)
        ax_s.set_title(r"$\mathrm{normalizing\ flow\ (NF)}$", fontsize=10)

        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.22, top=0.88, wspace=0.16)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"DF_slice_{dim1}_{dim2}.{fmt}")
        fig.savefig(fname, dpi=dpi)
    if len(fig_fmt) == 0:
        plt.show()
    plt.close(fig)


def evaluate_loss(flow_list, eta_train, batch_size=1024):
    n_flows = len(flow_list)
    n_samples = eta_train.shape[0]

    # Sample from ensemble of flows
    # n_batches = n_samples // batch_size
    # eta_batches = np.reshape(eta_train, (n_batches,batch_size,6)).astype('f4')
    # eta_batches = [
    #    eta_train[i0:i0+batch_size].astype('f4')
    #    for i0 in range(0,n_samples,batch_size)
    # ]
    # n_batches = len(eta_batches)
    n_batches = n_samples // batch_size
    if np.mod(n_samples, batch_size) > 0:
        n_batches += 1

    loss = []
    bar = progressbar.ProgressBar(max_value=n_batches * n_flows)

    for i, flow in enumerate(flow_list):
        loss_i = []
        weight_i = []

        @tf.function
        def logp_batch(eta):
            print("Tracing logp_batch ...")
            return -tf.math.reduce_mean(flow.log_prob(eta))

        for k in range(0, n_samples, batch_size):
            eta = eta_train[k: k + batch_size].astype("f4")
            loss_i.append(logp_batch(tf.constant(eta)).numpy())
            weight_i.append(eta.shape[0])
            bar.update(i * n_batches + k // batch_size + 1)

        loss.append(np.average(loss_i, weights=weight_i))

    loss_std = np.std(loss)
    loss_mean = np.mean(loss)

    return loss_mean, loss_std


def get_flow_leastsq_dfdt(
    df_data, omega=0.0, v_0=np.array([0.0, 0.0, 0.0]), r_c=8.3, grid_size=32
):
    """Returns a least-squares based estimate on the optimal value of
    \partial f / \partial t corresponding to the flow.
    This is done by binning the spatial space and optimizing for CBE
    """
    eta, df_deta = df_data["eta"], df_data["df_deta"]

    internal_grid_lims = np.percentile(eta[:, :3], [1.0, 99.0], axis=0)
    w = internal_grid_lims[1, :] - internal_grid_lims[0, :]
    internal_grid_lims[0, :] -= 0.2 * w
    internal_grid_lims[1, :] += 0.2 * w

    internal_grid_bins = np.linspace(
        internal_grid_lims[0, :], internal_grid_lims[1, :], grid_size + 1
    )
    internal_grid_bins = [
        internal_grid_bins[:, i] for i in range(internal_grid_bins.shape[1])
    ]

    # bin_indices is of shape (2, N), where row 0 contains the x-coordinate, and row 1 the y-coordinate
    # Numbering is within [1, len(bins_x))
    internal_bin_indices = binned_statistic_dd(
        eta[:, :3], np.zeros(len(eta)), bins=internal_grid_bins, expand_binnumbers=False
    )[2]

    sort_idx = internal_bin_indices.argsort()
    values_to_groupby, sort_eta, sort_df_deta = (
        internal_bin_indices[sort_idx],
        eta[sort_idx],
        df_deta[sort_idx],
    )

    u1, u2 = np.unique(values_to_groupby, return_index=True)
    group_etas = np.split(sort_eta, u2[1:])
    group_df_etas = np.split(sort_df_deta, u2[1:])

    ix, iy, ivx, ivy = 0, 1, 3, 4
    ndim = 3

    group_results = np.zeros((internal_bin_indices.max() + 1, ndim))
    for i in range(len(u1)):
        bin_eta, bin_df_deta = group_etas[i], group_df_etas[i]

        if len(bin_eta) == 0:
            group_results[u1[i]] = 0.0
        else:
            X = bin_df_deta[:, ndim:]

            y = (
                np.sum((bin_eta[:, ndim:] - v_0) * bin_df_deta[:, :ndim], axis=1)
                + omega * bin_eta[:, iy] * bin_df_deta[:, ix]
                - omega * (bin_eta[:, ix] - r_c) * bin_df_deta[:, iy]
                + omega * (bin_eta[:, ivy] - v_0[iy]) * bin_df_deta[:, ivx]
                - omega * (bin_eta[:, ivx] - v_0[ix]) * bin_df_deta[:, ivy]
            )

            theta = np.linalg.lstsq(X, y, rcond=None)[0]

            # res = y - X@theta
            # res = leastsq_residue(X, y)
            group_results[u1[i], :] = theta  # np.mean(res)

    # TODO: Calculating y twice is inefficient, can be sped up by doing it once in the beginning
    all_y = (
        np.sum((eta[:, ndim:] - v_0) * df_deta[:, :ndim], axis=1)
        + omega * eta[:, iy] * df_deta[:, ix]
        - omega * (eta[:, ix] - r_c) * df_deta[:, iy]
        + omega * (eta[:, ivy] - v_0[iy]) * df_deta[:, ivx]
        - omega * (eta[:, ivx] - v_0[ix]) * df_deta[:, ivy]
    )

    theoretical_dfdt_flow = all_y - np.sum(
        df_deta[:, ndim:] * group_results[internal_bin_indices], axis=1
    )

    return theoretical_dfdt_flow


def plot_flow_dfdt(
    df_data, dim1, dim2, omega, v_0, r_c, attrs, fig_dir, fig_fmt=("svg",)
):
    """Plots the \partial f/\partial t corresponding to the flow via solving
    the Collisionless Boltzmann Equation (CBE) in the rotating frame specified
    by omega, v_0 and r_c using the least squares method in cubic spatial bins.
    """

    def amplify(min_val, max_val, k=0.2):
        if min_val * max_val > 0:
            w = max_val - min_val
            return min_val - k * w, max_val + k * w
        max_val = np.max([-min_val, max_val])
        return -max_val * (1 + k), max_val * (1 + k)

    eta = df_data["eta"]

    fig, (all_axs) = plt.subplots(
        2, 1, figsize=(2, 2.2), dpi=200, gridspec_kw=dict(height_ratios=[0.2, 2])
    )
    cax = all_axs[0]
    ax = all_axs[1]

    labels = ["$x$", "$y$", "$z$", "$v_x$", "$v_y$", "$v_z$"]
    keys = ["x", "y", "z", "vx", "vy", "vz"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}
    ix, iy = ikeys[dim1], ikeys[dim2]

    ax.set_xlabel(labels[dim1])
    ax.set_ylabel(labels[dim2])

    # Get the plot limits
    lims = []
    k = 0.2
    for i in [ix, iy]:
        xlim = np.percentile(eta[:, i], [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - k * w, xlim[1] + k * w]
        lims.append(xlim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    if attrs["has_spatial_cut"]:
        # Visualise the boundaries
        add_2dpopulation_boundaries([ax], dim1, dim2, attrs, color="black")

    x_bins = np.linspace(xmin, xmax, 32)
    y_bins = np.linspace(ymin, ymax, 32)

    theoretical_flow_dfdt = get_flow_leastsq_dfdt(
        df_data, omega=omega, v_0=v_0, r_c=r_c, grid_size=32
    )

    # Flow ideal dfdt
    ret = binned_statistic_2d(
        eta[:, ix],
        eta[:, iy],
        theoretical_flow_dfdt,
        statistic=np.mean,
        bins=[x_bins, y_bins],
    )
    vmin, vmax = amplify(*np.nanpercentile(ret.statistic, [1, 99]), k=0.4)
    divnorm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    im = ax.imshow(
        ret.statistic.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="seismic",
        norm=divnorm,
        aspect="auto",
    )
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.locator_params(nbins=5)
    title = (
        "$(\partial f/\partial t)_\mathrm{flow\_leastsq}$"
        + f"\n$\Omega={omega:.3f},$\n$\\vec v_0=({v_0[0]:.2f}, {v_0[1]:.2f}, {v_0[2]:.2f})$"
    )
    cax.set_title(title)

    # plt.tight_layout()
    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"DF_dfdt_{dim1}_{dim2}_omega={omega:.2f}.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    if len(fig_fmt) == 0:
        plt.show()
    plt.close(fig)


def plot_1d_slice(
    coords_train, coords_sample,
    fig_dir,
    dim1, dimy, dimz,
    y, dy, z, dz,
    attrs,
    fig_fmt=("svg",),
    verbose=False,
):
    labels = [
        "$R$", "$z$", r"$\phi$",
        "$v_R$", "$v_z$", r"$v_{\phi}$",
        "$x$", "$y$", "$z$",
        "$v_x$", "$v_y$", "$v_z$",
        "$r$", r"$\phi$", r"$\cos \theta$",
        "$v_r$", r"$v_{\theta}$", r"$v_{\phi}$",
    ]
    keys = [
        "cylR", "cylz", "cylphi",
        "cylvR", "cylvz", "cylvT",
        "x", "y", "z",
        "vx", "vy", "vz",
        "r", "phi", "cth",
        "vr", "vth", "vT",
    ]

    idx_train = (
        (coords_train[dimz] > z - dz)
        & (coords_train[dimz] < z + dz)
        & (coords_train[dimy] > y - dy)
        & (coords_train[dimy] < y + dy)
    )
    idx_sample = (
        (coords_sample[dimz] > z - dz)
        & (coords_sample[dimz] < z + dz)
        & (coords_sample[dimy] > y - dy)
        & (coords_sample[dimy] < y + dy)
    )
    x_train, x_sample = coords_train[dim1][idx_train], coords_sample[dim1][idx_sample]

    labels = {k: l for k, l in zip(keys, labels)}

    fig, (ax_h, ax_r) = plt.subplots(
        1, 2, figsize=(6, 3), dpi=200, gridspec_kw=dict(width_ratios=[1, 1])
    )

    lim_min, lim_max = 99999.0, -99999.0
    for i, (k, val) in enumerate([(dim1, x_train)]):
        xlim = np.percentile(val, [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - 0.05 * w, xlim[1] + 0.05 * w]
        if k == "cylR":
            xlim[0] = max(xlim[0], 0.0)
        elif k == "phi":
            xlim = [np.pi, -np.pi]
        elif k == "cth":
            xlim = [-1, 1]
        lim_min = min(lim_min, xlim[0])
        lim_max = max(lim_max, xlim[1])

    kw = dict(range=(lim_min, lim_max), bins=64)

    n_train = len(x_train)
    n_sample = len(x_sample)

    nt, bins, _ = ax_h.hist(x_train, histtype="step", **kw, label="train")
    ns, *_ = ax_h.hist(
        x_sample,
        histtype="step",
        **kw,
        weights=np.ones_like(x_sample) * n_train / n_sample,
        label="sample",
    )
    ns *= n_sample / n_train
    ax_h.legend(loc="lower right", frameon=False, fontsize=8)
    ax_h.set_ylabel("frequency")

    if attrs["has_spatial_cut"]:
        add_1dpopulation_boundaries([ax_h, ax_r], dim1, attrs)

    dn = ns / n_sample - nt / n_train
    with np.errstate(divide="ignore", invalid="ignore"):
        dn /= np.sqrt(ns * (n_train / n_sample)) / n_train
    ax_r.plot(
        bins[:-1], dn, label=r"gaia_vr, $\varpi$ > 0.2 mas", drawstyle="steps-post"
    )
    ax_r.yaxis.tick_right()
    ax_r.yaxis.set_label_position("right")
    ax_r.set_ylabel(r"$\mathrm{Poisson\ significance} \ \left( \sigma \right)$")
    ax_r.axhline(0, ls="--", lw=1.0, color="black", zorder=0)

    for ax in (ax_h, ax_r):
        ax.set_xlabel(labels[dim1], labelpad=0)

    ax_r.set_title(r"$\mathrm{NF - training}$", fontsize=10)

    if verbose:
        ax_h.set_title(
            f"$n_\mathrm{{train}}={len(x_train)},\quad n_\mathrm{{sample}}={len(x_sample)}$",
            fontsize=10,
        )
        # Print additional info on the subplots
        fig.suptitle(
            f"${y-dy:.2f}\leq${labels[dimy]}$\leq{y+dy:.2f},\quad\
                       {z-dz:.2f}\leq${labels[dimz]}$\leq{z+dz:.2f}$",
            fontsize=10,
        )

        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.18, top=0.84, wspace=0.16)
    else:
        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.22, top=0.88, wspace=0.16)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"DF_slice_{dim1}.{fmt}")
        fig.savefig(fname, dpi=dpi)
    if len(fig_fmt) == 0:
        plt.show()
    plt.close(fig)


def main():
    """
    Plots different diagnostics for the flow and the training data.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Deep Potential: Plot different diagnostics for a normalizing flow.",
        add_help=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Filename of input training data (particle phase-space coords).",
    )
    parser.add_argument(
        "--flows",
        type=str,
        nargs="+",
        required=False,
        help="Flow model filename pattern(s). Can be either checkpoint dir \
            or *.index in that checkpoint dir.",
    )
    parser.add_argument(
        "--store-samples",
        type=str,
        metavar="*.h5",
        help="Save generated samples or load them from this filename.",
    )
    parser.add_argument(
        "--fig-dir", type=str, default="plots", help="Directory to put figures in."
    )
    parser.add_argument(
        "--oversample",
        type=float,
        default=1,
        help="Draw oversample*(# of training samples) samples from flows.",
    )
    parser.add_argument(
        "--spherical-origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Origin of coordinate system for spherical coordinates in (x,y,z) \
            to subtract from coords.",
    )
    parser.add_argument(
        "--cylindrical-origin",
        type=float,
        nargs=3,
        default=(8.3, 0.0, 0.0),
        help="Origin of coordinate system for cylindrical coordinates in (x,y,z) \
            to subtract from coords.",
    )
    parser.add_argument(
        "--fig-fmt",
        type=str,
        nargs="+",
        default=("svg",),
        help="Formats in which to save figures (svg, png, pdf, etc.).",
    )
    parser.add_argument(
        "--dark", action="store_true", help="Use dark background for figures."
    )
    parser.add_argument(
        "--logscale", action="store_true", help="Plot densities with log scale."
    )
    parser.add_argument(
        "--plot-leastsq",
        action="store_true",
        help="Whether to plot the least squares estimate of \partial f/\partial t. \
            Requires computing gradients from the sample which is \
            computationally expensive.",
    )
    parser.add_argument(
        "--plot-leastsq-other",
        action="store_true",
        help="Whether to plot the least squares estimate of \partial f/\partial t \
            based on the values from the options file. Requires computing \
            gradients from the sample which is computationally expensive.",
    )
    parser.add_argument(
        "--autosave",
        action="store_true",
        help="Automatically saves/loads samples and chooses fig dir. \
            Incompatible with save samples, load samples, and fig-dir. The \
            saving location is in plots/ in a subdir deduced from the flows \
            directory. Currently only supports one flow.",
    )
    args = parser.parse_args()

    # Load in the custom style sheet for scientific plotting
    plt.style.use("scientific")
    if args.dark:
        plt.style.use("dark_background")

    if args.autosave:
        # Infer the place to store the samples and figures
        fname_flow = args.flows[0]
        fname_loss_pdf = ""
        if os.path.isdir(fname_flow):
            fname_index = tf.train.latest_checkpoint(fname_flow)
            fname_loss_pdf = fname_index + "_loss.pdf"
            fig_dir = "plots/" + fname_index[fname_index.find("df/") + 3:] + "/"
        else:
            fname_loss_pdf = fname_flow[:-6] + "_loss.pdf"
            fig_dir = (
                "plots/"
                + fname_flow[fname_flow.find("df/") + 3: fname_flow.rfind(".index")]
                + "/"
            )

        sample_fname = fig_dir + "samples.h5"
        print(fname_loss_pdf, os.path.isfile(fname_loss_pdf))
        if os.path.isfile(fname_loss_pdf):
            # Copy the latest loss over to the plots dir
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(fname_loss_pdf, fig_dir)

        args.store_samples = sample_fname
        args.fig_dir = fig_dir

    print("Loading training data ...")
    data_train, attrs_train = utils.load_training_data(args.input, cut_attrs=True)
    eta_train = data_train["eta"]
    n_train = eta_train.shape[0]

    print(attrs_train)
    print(f"  --> Training data shape = {eta_train.shape}")

    flows = None

    df_deta_sample = None
    if args.store_samples is not None and os.path.isfile(args.store_samples):
        print("Loading pre-generated samples ...")
        with h5py.File(args.store_samples, "r") as f:
            eta_sample = f["eta"][:]
            loss_mean = (
                f["eta"].attrs["loss_training"]
                if "loss_training" in f["eta"].attrs
                else 0
            )
            loss_std = (
                f["eta"].attrs["loss_std_training"]
                if "loss_std_training" in f["eta"].attrs
                else 0
            )
            if "df_deta" in f and args.plot_leastsq:
                print("Loading in pre-generated sample gradients ...")
                df_deta_sample = f["df_deta"][:]

        print(f"  --> loss = {loss_mean:.5f} +- {loss_std:.5f}")
        print(f"  --> {len(eta_sample)} samples")
    else:
        if flows is None:
            print("Loading flows ...")
            flows = utils.load_flows(args.flows)

        print("Evaluating loss ...")
        loss_mean, loss_std = evaluate_loss(flows, eta_train)
        print(f"  --> loss = {loss_mean:.5f} +- {loss_std:.5f}")
        print("Sampling from flows ...")
        eta_sample = sample_from_flows(
            flows, int(args.oversample * n_train), attrs_train
        )
        print("  --> Saving samples ...")
        if args.store_samples is not None:
            Path(os.path.split(args.store_samples)[0]).mkdir(
                parents=True, exist_ok=True
            )
            with h5py.File(args.store_samples, "w") as f:
                dset = f.create_dataset(
                    "eta", data=eta_sample, chunks=True, compression="lzf"
                )
                dset.attrs["loss_training"] = loss_mean
                dset.attrs["loss_std_training"] = loss_std

    if args.plot_leastsq:
        if df_deta_sample is None:
            if flows is None:
                print("Loading flows ...")
                flows = utils.load_flows(args.flows)

            # Calculate the gradients of eta_sample!

            from flow_sampling import get_sampling_progressbar_fn

            # Do ceiling divide
            # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
            grad_batch_size = 2048
            n_batches = -(-len(eta_sample) // grad_batch_size)
            bar = get_sampling_progressbar_fn(n_batches, len(eta_sample))
            iteration = 0
            print("Sampling gradients of eta..")

            for i, flow in enumerate(flows):

                @tf.function
                def calc_grads(batch):
                    # print(f'Tracing calc_grads with shape = {batch.shape}')
                    with tf.GradientTape(watch_accessed_variables=False) as g:
                        g.watch(batch)
                        f = flow.prob(batch)
                    df_deta_sample = g.gradient(f, batch)
                    return df_deta_sample

                eta_dataset = tf.data.Dataset.from_tensor_slices(eta_sample).batch(
                    grad_batch_size
                )
                df_deta_sample = []
                for k, b in enumerate(eta_dataset):
                    df_deta_sample.append(calc_grads(b))
                    bar(iteration)
                    iteration += 1

                df_deta_sample = np.concatenate([b.numpy() for b in df_deta_sample])

            # Save the sample if needed
            print("  --> Adding gradients to the saved samples ...")
            if args.store_samples is not None:
                Path(os.path.split(args.store_samples)[0]).mkdir(
                    parents=True, exist_ok=True
                )
                with h5py.File(args.store_samples, "w") as f:
                    dset = f.create_dataset(
                        "eta", data=eta_sample, chunks=True, compression="lzf"
                    )
                    dset.attrs["loss_training"] = loss_mean
                    dset.attrs["loss_std_training"] = loss_std
                    f.create_dataset(
                        "df_deta", data=df_deta_sample, chunks=True, compression="lzf"
                    )
        df_data_sample = {"eta": eta_sample, "df_deta": df_deta_sample}

    if attrs_train["has_spatial_cut"]:
        idx = utils.get_index_of_points_inside_attrs(eta_sample, attrs_train)
        eta_sample = eta_sample[idx]
        if args.plot_leastsq:
            df_deta_sample = df_deta_sample[idx]
            df_data_sample = {"eta": eta_sample, "df_deta": df_deta_sample}

    print(f"  --> {np.count_nonzero(np.isnan(eta_sample))} NaN values")

    # Make sure fig_dir exists
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    print("Calculating cylindrical & spherical coordinates ...")
    coords_train = calc_coords(
        eta_train, args.spherical_origin, args.cylindrical_origin
    )
    coords_sample = calc_coords(
        eta_sample, args.spherical_origin, args.cylindrical_origin
    )

    print("Plotting 1D marginal distributions ...")
    for coordsys in ["cart", "cyl", "sph"]:
        plot_1d_marginals(
            coords_train, coords_sample,
            args.fig_dir,
            loss=loss_mean,
            coordsys=coordsys,
            fig_fmt=args.fig_fmt,
        )

    print("Plotting 2D marginal distributions ...")

    dims = [
        ("r", "vr"),
        ("phi", "cth"),
        ("vT", "vth"),
        ("cylR", "cylz"),
        ("cylR", "cylvz"),
        ("cylR", "cylvR"),
        ("cylR", "cylvT"),
        ("z", "vz"),
        ("cylvz", "cylvT"),
        ("cylvR", "cylvz"),
        ("cylvR", "cylvT"),
        ("x", "y"),
        ("x", "z"),
        ("y", "z"),
        ("vx", "vy"),
        ("vx", "vz"),
        ("vy", "vz"),
    ]

    for dim1, dim2 in dims:
        print(f"  --> ({dim1}, {dim2})")
        plot_2d_marginal(
            coords_train, coords_sample,
            eta_train, eta_sample,
            args.fig_dir,
            dim1, dim2,
            logscale=args.logscale,
            fig_fmt=args.fig_fmt,
        )

    print("Plotting 2d slices of the flow ...")

    dims = [
        ("phi", "cth", "r", np.mean(coords_train["r"]), 0.05),
        ("cylR", "cylphi", "cylz", 0.0, 0.05),
        ("x", "y", "z", 0.0, 0.05),
        ("y", "z", "x", 0.0, 0.05),
        ("x", "z", "y", 0.0, 0.05),
    ]

    for dim1, dim2, dimz, z, dz in dims:
        print(f"  --> ({dim1}, {dim2}, {dimz}={z:.2f}+-{dz:.2f})")
        plot_2d_slice(
            coords_train, coords_sample,
            args.fig_dir,
            dim1, dim2, dimz,
            z, dz,
            attrs=attrs_train,
            fig_fmt=args.fig_fmt,
            verbose=True,
        )

    print("Plotting 1d slices of the flow ...")

    dims = [
        ("x", "y", "z", 0.0, 0.05, 0.0, 0.05),
        ("y", "x", "z", 0.0, 0.05, 0.0, 0.05),
        ("z", "x", "y", 0.0, 0.05, 0.0, 0.05),
    ]

    for dim1, dimy, dimz, y, dy, z, dz in dims:
        print(f"  --> ({dim1}, {dimy}={y}+-{dy}, {dimz}={z}+-{dz})")
        try:
            plot_1d_slice(
                coords_train, coords_sample,
                args.fig_dir,
                dim1, dimy, dimz,
                y, dy, z, dz,
                attrs=attrs_train,
                fig_fmt=args.fig_fmt,
                verbose=True,
            )
        except Exception:
            print("  --> Failed to plot 1d slice! Probably because the slice is empty.")

    if args.plot_leastsq:
        print(
            "Plotting 2D least square estimates of \partial f/\partial t with omega=v_0=0 ..."
        )

        dims = [
            ("x", "y"),
            ("x", "z"),
            ("y", "z"),
            ("vx", "vy"),
            ("vx", "vz"),
            ("vy", "vz"),
            ("z", "vz"),
        ]

        if not args.plot_leastsq_other:
            omega = 0
            v_0 = np.array([0, 0, 0])

            for dim1, dim2 in dims:
                print(f"  --> ({dim1}, {dim2})")
                plot_flow_dfdt(
                    df_data_sample,
                    dim1, dim2,
                    omega, v_0, r_c=8.3,
                    attrs=attrs_train,
                    fig_dir=args.fig_dir,
                    fig_fmt=args.fig_fmt,
                )
        else:
            # Check for an options for file in the active directory
            fname_options = glob("*.json")[0]
            params = fit_all.load_params(fname_options)
            fs_params = params["Phi"]["frameshift"]

            omega = fs_params["omega0"]
            v_0 = np.array([fs_params["u_x0"], fs_params["u_y0"], fs_params["u_z0"]])
            r_c = fs_params["r_c0"]

            for dim1, dim2 in dims:
                print(f"  --> ({dim1}, {dim2})")
                plot_flow_dfdt(
                    df_data_sample,
                    dim1, dim2,
                    omega, v_0, r_c=r_c,
                    attrs=attrs_train,
                    fig_dir=args.fig_dir,
                    fig_fmt=args.fig_fmt,
                )

    return 0


if __name__ == "__main__":
    main()
