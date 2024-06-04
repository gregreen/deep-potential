#!/usr/bin/env python

from __future__ import print_function, division
from binascii import a2b_hqx

import numpy as np

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib import colors

import h5py
import progressbar
import os
from glob import glob
from pathlib import Path
import shutil
from scipy.stats import binned_statistic, binned_statistic_2d

import tensorflow as tf

print(f"Tensorflow version {tf.__version__}")

import potential_tf
import plot_flow_projections
import plot_potential
import fit_all
import utils


dpi = 200


def plot_rho(
    phi_model,
    coords_train,
    fig_dir,
    dim1, dim2, dimz,
    z,
    attrs,
    padding=0.95,
    fig_fmt=("svg",),
    save=True,
):
    labels = [
        "$x\mathrm{\ [kpc]}$",
        "$y\mathrm{\ [kpc]}$",
        "$z\mathrm{\ [kpc]}$",
    ]
    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    fig, axs = plot_potential.plot_rho(
        phi_model,
        coords_train,
        fig_dir,
        dim1, dim2, dimz,
        z,
        padding=padding,
        attrs=attrs,
        fig_fmt=fig_fmt,
        save=False,
    )

    ax_p, cax_p = axs[1, 0], axs[0, 0]
    ax_r, cax_r = axs[1, 1], axs[0, 1]
    ax_e, cax_e = axs[1, 2], axs[0, 2]
    main_axs = [ax_p, ax_r, ax_e]

    cax_p.set_title(r"$\Phi^*$", fontsize=10)
    cax_r.set_title(r"$\rho^*\mathrm{\ [M_\odot/pc^3]}$", fontsize=10)
    cax_e.set_title(r"$\rho_\mathrm{train}\mathrm{\ [1/pc^3]}$", fontsize=10)

    for ax in main_axs:
        ax.set_xlabel(labels[dim1], labelpad=0)

    ax_p.set_ylabel(labels[dim2], labelpad=2)

    if save:
        for fmt in fig_fmt:
            fname = os.path.join(fig_dir, f"phi_rho_{dim1}_{dim2}.{fmt}")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axs


def plot_force_2d_slice(
    phi_model,
    coords_train,
    fig_dir,
    dim1, dim2, dimz,
    z,
    attrs,
    padding=0.95,
    fig_fmt=("svg",),
    save=True,
):
    labels = [
        "$x\mathrm{\ [kpc]}$",
        "$y\mathrm{\ [kpc]}$",
        "$z\mathrm{\ [kpc]}$",
    ]
    titles = [
        "$F_x^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$",
        "$F_y^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$",
        "$F_z^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$",
    ]
    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    fig, axs = plot_potential.plot_force_2d_slice(
        phi_model,
        coords_train,
        fig_dir,
        dim1, dim2, dimz,
        z,
        padding=padding,
        attrs=attrs,
        fig_fmt=fig_fmt,
        save=False,
    )

    ax_x, cax_x = axs[1, 0], axs[0, 0]
    ax_y, cax_y = axs[1, 1], axs[0, 1]
    ax_z, cax_z = axs[1, 2], axs[0, 2]
    main_axs = [ax_x, ax_y, ax_z]
    main_caxs = [cax_x, cax_y, cax_z]

    for i, ax in enumerate(main_axs):
        cax = main_caxs[i]
        cax.set_title(titles[i])

    for ax in main_axs:
        ax.set_xlabel(labels[dim1], labelpad=0)

    ax_x.set_ylabel(labels[dim2], labelpad=2)

    if save:
        for fmt in fig_fmt:
            fname = os.path.join(fig_dir, f"phi_force_slice_{dim1}_{dim2}.{fmt}")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axs


def plot_force_1d_slice(
    phi_model,
    coords_train,
    fig_dir,
    dim1, dimy,
    y, z,
    dimforce,
    attrs,
    fig_fmt=("svg",),
    save=True,
):
    labels = [
        "$x\mathrm{\ [kpc]}$",
        "$y\mathrm{\ [kpc]}$",
        "$z\mathrm{\ [kpc]}$",
    ]
    force_labels = [
        "$F_x^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$",
        "$F_y^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$",
        "$\Sigma_z^*\:\mathrm{[M_\odot/pc^2]}$",
    ]

    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    force_labels = {k: l for k, l in zip(keys, force_labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)

    # Get the plot limits
    k = 0.2
    xlim = np.percentile(coords_train[dim1], [1.0, 99.0])
    w = xlim[1] - xlim[0]
    xlim = [xlim[0] - k * w, xlim[1] + k * w]
    xmin, xmax = xlim

    x_plot = np.linspace(xmin, xmax, 512)
    eta_plot = np.full(shape=(len(x_plot), 3), fill_value=z, dtype="f4")
    eta_plot[:, ikeys[dimy]] = y
    eta_plot[:, ikeys[dim1]] = x_plot

    _, dphi_dq, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], eta_plot, return_phi=True
    )
    Z_plot = -dphi_dq[:, ikeys[dimforce]].numpy()

    if dim1 == "z" and dimforce == "z":
        # with F_z - z, plot the implied surface density instead
        Z_plot *= -367.5

    if attrs["has_spatial_cut"]:
        # Visualise the boundaries
        plot_flow_projections.add_1dpopulation_boundaries([ax], dim1, attrs)

        # Mask for the area for which phi and rho are plotted
        r2 = x_plot**2 + y**2 + z**2
        if dim1 == "z":
            actual_z = x_plot
        elif dimy == "z":
            actual_z = y
        else:
            actual_z = z
        R2 = r2 - actual_z**2

        mask_ = utils.get_index_of_points_inside_attrs(
            None, attrs, r2**0.5, R2**0.5, actual_z
        )

        ax.plot(
            x_plot[~mask_ & (x_plot > 0)],
            Z_plot[~mask_ & (x_plot > 0)],
            color="tab:blue",
        )
        ax.plot(
            x_plot[~mask_ & (x_plot < 0)],
            Z_plot[~mask_ & (x_plot < 0)],
            color="tab:blue",
        )
    else:
        ax.plot(x_plot, Z_plot, color="tab:blue")

    # ax.set_xlim(-r_max, r_max)
    # ax.set_ylim(-r_max, r_max)
    ax.set_xlabel(labels[dim1])
    ax.set_ylabel(f"{force_labels[dimforce]}")

    # Plot the ideal curves
    if dim1 == "z" and dimforce == "z":
        z0 = 255.6  # [pc]
        rho0 = 0.0474  # [1/(M_sun*pc^3)]
        sigma_z = 68 * x_plot / 1.1
        # sigma_z = 2*xs/np.abs(xs)*rho0*z0*(1 - np.exp(-np.abs(xs)*1000/z0))
        # print(az)
        ax.plot(
            x_plot,
            sigma_z,
            color="tab:red",
            label=r"Bovy\&Rix 2013, $\Sigma_{z=1.1\mathrm{kpc}}=68\pm 4 M_\odot/\mathrm{pc}^2$",
        )
        ax.fill_between(
            x_plot, 64 * x_plot / 1.1, 72 * x_plot / 1.1, alpha=0.3, color="tab:red"
        )
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
    if dim1 == "x" and dimforce == "x":
        u = 2.2  # Approximate rotation curve in MW [100 km/s]
        r0 = 8.3  # Distance to MW centre [kpc]
        # Plot the forces assuming constant velocity profile
        ax.plot(
            x_plot,
            u**2 / r0 * (1 + x_plot / r0),
            color="tab:red",
            label="ideal constant rotation curve",
        )
    if dim1 == "y" and dimforce == "y":
        u = 2.2  # Approximate rotation curve in MW [100 km/s]
        r0 = 8.3  # Distance to MW centre [kpc]
        # Plot the forces assuming constant velocity profile
        ax.plot(
            x_plot,
            -(u**2) / r0**2 * x_plot,
            color="tab:red",
            label="ideal constant rotation curve",
        )

    ax.legend()
    if save:
        for fmt in fig_fmt:
            fname = os.path.join(fig_dir, f"phi_force_slice_{dim1}.{fmt}")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax


def plot_dfdt_comparison(
    phi_model, df_data, dphi_dq, fig_dir, dim1, dim2, attrs, fig_fmt=("svg",), save=True
):
    labels = [
        "$x\mathrm{\ [kpc]}$",
        "$y\mathrm{\ [kpc]}$",
        "$z\mathrm{\ [kpc]}$",
        "$v_x\mathrm{\ [100km/s]}$",
        "$v_y\mathrm{\ [100km/s]}$",
        "$v_z\mathrm{\ [100km/s]}$",
    ]
    keys = ["x", "y", "z", "vx", "vy", "vz"]

    labels = {k: l for k, l in zip(keys, labels)}

    fig, axs = plot_potential.plot_dfdt_comparison(
        phi_model, df_data, dphi_dq, fig_dir, dim1, dim2, attrs, save=False
    )

    for i in range(2):
        axs[i].set_xlabel(labels[dim1])
    axs[0].set_ylabel(labels[dim2])
    axs[1].set_yticklabels([])

    if save:
        omega = phi_model["fs"]._omega.numpy()
        for fmt in fig_fmt:
            fname = os.path.join(
                fig_dir, f"phi_dfdt_comparison_{dim1}_{dim2}_omega={omega:.2f}.{fmt}"
            )
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axs


def plot_vcirc_marginals(
    phi_model, coords_train, coords_sample, fig_dir, attrs, fig_fmt=("svg",), save=True
):
    x_train, x_sample = coords_train["cylR"], coords_sample["cylR"]
    y_train, y_sample = coords_train["cylvT"], coords_sample["cylvT"]

    fig, (ax_m, ax_t, ax_s) = plt.subplots(
        1, 3, figsize=(9, 3), dpi=200, gridspec_kw=dict(width_ratios=[1, 1, 1])
    )
    axs = (ax_m, ax_t, ax_s)

    xlim = np.percentile(x_train, [1.0, 99.0])
    nbins = 64
    xbins = np.linspace(*xlim, nbins)

    # Plot vcirc deduced from Phi
    r_c = phi_model["fs"]._r_c.numpy()
    x_model = r_c - np.linspace(
        *xlim, 512
    )  # This assumes that galactic centre is at +x
    eta_plot = np.full(shape=(len(x_model), 3), fill_value=0, dtype="f4")
    eta_plot[:, 0] = x_model
    _, dphi_dq, _ = potential_tf.calc_phi_derivatives(
        phi_model["phi"], eta_plot, return_phi=True
    )
    y_model = (-dphi_dq[:, 0].numpy() * (r_c - x_model)) ** 0.5
    ax_m.plot(r_c - x_model, y_model)
    ax_m.set_title("model")

    # Plot vcric for the training data
    y_train_mean = binned_statistic(
        x_train, y_train, statistic=np.mean, bins=[xbins]
    ).statistic
    ax_t.plot(xbins[:-1], np.abs(y_train_mean), drawstyle="steps-post")
    ax_t.set_title("training data")

    # Plot vcirc for the normalizing flow (for its sample)
    y_sample_mean = binned_statistic(
        x_sample, y_sample, statistic=np.mean, bins=[xbins]
    ).statistic
    ax_s.plot(xbins[:-1], np.abs(y_sample_mean), drawstyle="steps-post")
    ax_s.set_title("normalizing flow")

    for ax in (ax_m, ax_t, ax_s):
        ax.set_xlabel("$R\mathrm{\ [kpc]}$", labelpad=0)
    ax_m.set_ylabel("$v_\mathrm{circ}\mathrm{\ [100 km/s]}$")

    if save:
        for fmt in fig_fmt:
            fname = os.path.join(fig_dir, f"phi_flow_vcirc_marginals.{fmt}")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axs

    return


def plot_vcirc_2d_slice(
    phi_model,
    coords_train, coords_sample,
    fig_dir,
    dim1, dim2, dimz,
    z,
    attrs,
    fig_fmt=("svg",),
    save=True,
):
    labels = [
        "$x\mathrm{\ [kpc]}$",
        "$y\mathrm{\ [kpc]}$",
        "$z\mathrm{\ [kpc]}$",
    ]
    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    for dim in [dim1, dim2]:
        if dim not in keys:
            raise ValueError(f"dimension {dim} not supported")

    fig, all_axs = plt.subplots(
        2, 3,
        figsize=(6, 2.2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[2, 2, 2], height_ratios=[0.2, 2]),
    )
    axs = all_axs[1, :]
    caxs = all_axs[0, :]

    # Get the plot limits
    lims = []
    k = 0.2
    for x in coords_train[dim1], coords_train[dim2]:
        xlim = np.percentile(x, [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - k * w, xlim[1] + k * w]
        lims.append(xlim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    grid_size = 256
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    X, Y = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))

    q_grid = np.full(shape=(X.size, 3), fill_value=z, dtype="f4")
    q_grid[:, ikeys[dim1]] = X.ravel()
    q_grid[:, ikeys[dim2]] = Y.ravel()

    _, dphi_dq, _ = potential_tf.calc_phi_derivatives(
        phi_model["phi"], q_grid, return_phi=True
    )
    offset = np.array((phi_model["fs"]._r_c.numpy(), 0, 0))
    vcirc_img = np.reshape(
        np.sum(dphi_dq * (q_grid - offset), axis=1) ** 0.5, X.shape
    )  # 100 km/s

    if attrs["has_spatial_cut"]:
        # Visualise the boundaries
        plot_flow_projections.add_2dpopulation_boundaries(
            axs, dim1, dim2, attrs, color="black"
        )

        # Mask for the area for which phi and rho are plotted
        r2 = X * X + Y * Y + z**2
        if dim1 == "z":
            actual_z = X
        elif dim2 == "z":
            actual_z = Y
        else:
            actual_z = z

        R2 = r2 - actual_z**2

        mask = ~utils.get_index_of_points_inside_attrs(
            None, attrs, r2**0.5, R2**0.5, actual_z
        )

        vcirc_img = np.ma.masked_where(mask, vcirc_img)

    # Plot v_circ
    min_val, max_val = vcirc_img.min(), vcirc_img.max()

    kw = dict(
        cmap="viridis",
        vmin=min_val,
        vmax=max_val,
        shading="flat",
        lw=0,
        rasterized=True,
    )
    hh = axs[0].pcolormesh(x, y, vcirc_img, **kw)
    # Set the colorbar
    cb_p = fig.colorbar(hh, cax=caxs[0], orientation="horizontal")
    cb_p.ax.xaxis.set_ticks_position("top")
    cb_p.ax.locator_params(nbins=3)
    caxs[0].set_title(r"$v_\mathrm{circ, model}\mathrm{\ [100km/s]}$")

    x_bins = np.linspace(xmin, xmax, 32)
    y_bins = np.linspace(ymin, ymax, 32)

    # Plot a slice of training data
    dz = 0.05
    idx_train = (coords_train[dimz] > z - dz) & (coords_train[dimz] < z + dz)
    dim1_train = coords_train[dim1][idx_train]
    dim2_train = coords_train[dim2][idx_train]
    vcirc_train = coords_train["cylvT"][idx_train]
    ret = binned_statistic_2d(
        dim1_train, dim2_train, vcirc_train, statistic=np.mean, bins=[x_bins, y_bins]
    )
    im = axs[1].imshow(
        np.abs(ret.statistic.T),
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="viridis",
    )
    cb_t = fig.colorbar(im, cax=caxs[1], orientation="horizontal")
    cb_t.ax.xaxis.set_ticks_position("top")
    cb_t.ax.locator_params(nbins=5)
    caxs[1].set_title(r"$v_\mathrm{circ, train}\mathrm{\ [100km/s]}$")

    # Plot a slice of the normalizing flow sample
    dz = 0.05
    idx_train = (coords_sample[dimz] > z - dz) & (coords_sample[dimz] < z + dz)
    dim1_train = coords_sample[dim1][idx_train]
    dim2_train = coords_sample[dim2][idx_train]
    vcirc_train = coords_sample["cylvT"][idx_train]
    ret = binned_statistic_2d(
        dim1_train, dim2_train, vcirc_train, statistic=np.mean, bins=[x_bins, y_bins]
    )
    im = axs[2].imshow(
        np.abs(ret.statistic.T),
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="viridis",
    )
    cb_t = fig.colorbar(im, cax=caxs[2], orientation="horizontal")
    cb_t.ax.xaxis.set_ticks_position("top")
    cb_t.ax.locator_params(nbins=5)
    caxs[2].set_title(r"$v_\mathrm{circ, sample}\mathrm{\ [100km/s]}$")

    for ax in axs:
        ax.set_xlabel(labels[dim1], labelpad=0)
    axs[0].set_ylabel(labels[dim2], labelpad=2)
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])

    if save:
        for fmt in fig_fmt:
            fname = os.path.join(
                fig_dir, f"phi_flow_vcirc_2d_slice_{dim1}_{dim2}.{fmt}"
            )
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axs

    return


def plot_custom_potential_marginal(
    phi_model, fig_dir, dim1, quantity, attrs, padding=0.95, fig_fmt=("svg",), save=True
):
    # Draws samples in the volume of validity, and bins them along a chosen axis
    # (TODO: Currently only z works). Works for the implied surface density
    # TODO: Only works with attributes
    # TODO: Doesn't support attrs with inner cylinder
    if not attrs["has_spatial_cut"]:
        return

    labels = [
        "$x\mathrm{\ [kpc]}$",
        "$y\mathrm{\ [kpc]}$",
        "$z\mathrm{\ [kpc]}$",
    ]

    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    for dim in [dim1]:
        if dim not in keys:
            raise ValueError(f"dimension {dim} not supported")

    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)

    n = 256000
    nbins = 64
    if "volume_type" not in attrs or attrs["volume_type"] == "sphere":
        r_in, r_out = 1 / attrs["parallax_max"], 1 / attrs["parallax_min"]
        xbins = np.linspace(-r_out * padding, r_out * padding, nbins)
    elif attrs["volume_type"] == "cylinder":
        r_in = attrs["r_in"]
        R_out, H_out = attrs["R_out"], attrs["H_out"]
        xbins = np.linspace(-H_out * padding, H_out * padding, nbins)

    xcenters = 0.5 * (xbins[1:] + xbins[:-1])
    q = []
    for x in xcenters:
        n_x = n // len(xcenters)

        if "volume_type" not in attrs or attrs["volume_type"] == "sphere":
            y = (r_out**2 * padding**2 - x**2) ** 0.5 * (
                2 * np.random.random_sample(n_x) - 1
            ).astype("f4")
            z = (r_out**2 * padding**2 - x**2) ** 0.5 * (
                2 * np.random.random_sample(n_x) - 1
            ).astype("f4")
            idx = (x * x + y * y + z * z < r_out**2 * padding**2) & (
                (x * x + y * y + z * z > r_in**2 / padding**2)
            )
        elif attrs["volume_type"] == "cylinder":
            y = R_out * padding * (2 * np.random.random_sample(n_x) - 1).astype("f4")
            z = R_out * padding * (2 * np.random.random_sample(n_x) - 1).astype("f4")
            idx = (y * y + z * z < R_out**2 * padding**2) & (
                (x * x + y * y + z * z > r_in**2 / padding**2)
            )

        new_q = np.full((np.sum(idx), 3), x, dtype="f4")
        iy = (ikeys[dim1] + 1) % 3
        iz = (ikeys[dim1] + 2) % 3
        new_q[:, iy] = y[idx]
        new_q[:, iz] = z[idx]
        q.append(new_q)
    q = np.concatenate(q, axis=0)

    if attrs is not None:
        # Visualise the boundaries
        plot_flow_projections.add_1dpopulation_boundaries([ax], dim1, attrs)

    phi, dphi_dq, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], q, return_phi=True
    )

    if quantity == "rho":
        y = 2.3250854 * d2phi_dq2.numpy() / (4 * np.pi)
        ax.set_ylabel(r"$\rho^*\mathrm{\ [M_\odot/pc^3]}$")
    elif quantity == "sigmaz":
        y = 367.5 * dphi_dq[:, 2].numpy()
        ax.set_ylabel("$\Sigma_z^*\:\mathrm{[M_\odot/pc^2]}$")

        z0 = 255.6  # [pc]
        rho0 = 0.0474  # [1/(M_sun*pc^3)]
        if "volume_type" not in attrs or attrs["volume_type"] == "sphere":
            x_plot = np.linspace(-r_out, r_out, 512)
        elif attrs["volume_type"] == "cylinder":
            x_plot = np.linspace(-H_out, H_out, 512)
        sigma_z = 68 * x_plot / 1.1
        ax.plot(
            x_plot[x_plot > 0],
            sigma_z[x_plot > 0],
            color="tab:red",
            label=r"Bovy\&Rix 2013, $\Sigma_{z=1.1\mathrm{kpc}}=68\pm 4 M_\odot/\mathrm{pc}^2$",
        )
        ax.plot(x_plot[x_plot < 0], sigma_z[x_plot < 0], color="tab:red")
        ax.fill_between(
            x_plot, 64 * x_plot / 1.1, 72 * x_plot / 1.1, alpha=0.3, color="tab:red"
        )
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)

    y_mean = binned_statistic(
        q[:, ikeys[dim1]], y, statistic=np.mean, bins=[xbins]
    ).statistic
    y_lower = binned_statistic(
        q[:, ikeys[dim1]], y, statistic=lambda x: np.percentile(x, 15.865), bins=[xbins]
    ).statistic
    y_upper = binned_statistic(
        q[:, ikeys[dim1]],
        y,
        statistic=lambda x: np.percentile(x, 100 - 15.865),
        bins=[xbins],
    ).statistic

    x_plot = 0.5 * (xbins[1:] + xbins[:-1])
    ax.plot(xcenters, y_mean)
    ax.fill_between(xcenters, y_lower, y_upper, alpha=0.2, label=r"$\pm 1\sigma$")

    ax.set_xlabel(labels[dim1])
    ax.legend(fontsize=8)

    if save:
        for fmt in fig_fmt:
            fname = os.path.join(fig_dir, f"phi_marginal_{dim1}_{quantity}.{fmt}")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

    return


def main():
    """
    Plots different diagnostics for the potential (with frameshift) and flows
    for Gaia populations.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Deep Potential: Plot different diagnostics for a potential derived from Gaia DR3.",
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
        "--potential",
        type=str,
        required=True,
        help="Potential model filename. Can be either checkpoint dir or *.index in that checkpoint dir.",
    )
    parser.add_argument(
        "--df-grads-fname",
        type=str,
        default="data/df_gradients.h5",
        help="Directory in which to store the flow samples (positions and f gradients).",
    )
    parser.add_argument(
        "--fig-dir", type=str, default="plots", help="Directory to put figures in."
    )
    parser.add_argument(
        "--spherical-origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Origin of coordinate system for spherical coordinates in (x,y,z) to subtract from coords.",
    )
    parser.add_argument(
        "--cylindrical-origin",
        type=float,
        nargs=3,
        default=(8.3, 0.0, 0.0),
        help="Origin of coordinate system for cylindrical coordinates in (x,y,z) to subtract from coords.",
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
        "--autosave",
        action="store_true",
        help="Automatically saves/loads samples and chooses fig dir. Incompatible with fig-dir.\
            The saving location is in plots/ in a subdir deduced from the potential directory.",
    )
    args = parser.parse_args()

    # Load in the custom style sheet for scientific plotting
    plt.style.use("scientific")
    if args.dark:
        plt.style.use("dark_background")

    if args.autosave:
        # Infer the place to store the samples and figures
        fname_pot = args.potential
        if os.path.isdir(fname_pot):
            fname_index = tf.train.latest_checkpoint(fname_pot)
            fname_loss_pdf = fname_index + "_loss.pdf"
            fig_dir = "plots/" + fname_index[fname_index.find("models/") + 7:] + "/"
        else:
            fname_loss_pdf = fname_pot[:-6] + "_loss.pdf"
            fig_dir = (
                "plots/"
                + fname_pot[fname_pot.find("models/") + 7: fname_pot.rfind(".index")]
                + "/"
            )

        print(fname_loss_pdf, os.path.isfile(fname_loss_pdf))
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(fname_loss_pdf):
            # Copy the latest loss over to the plots dir
            shutil.copy(fname_loss_pdf, fig_dir)
            shutil.copy(fname_loss_pdf[:-4] + "_noreg.pdf", fig_dir)

        args.fig_dir = fig_dir

    print("Loading training data ...")
    data_train, attrs_train = utils.load_training_data(args.input, cut_attrs=True)
    eta_train = data_train["eta"]

    n_train = eta_train.shape[0]
    print(f"  --> Training data shape = {eta_train.shape}")

    print("Loading potential")
    phi_model = utils.load_potential(args.potential)
    if phi_model["fs"] is not None:
        phi_model["fs"].debug()

    print("Calculating cylindrical & spherical coordinates ...")
    coords_train = plot_flow_projections.calc_coords(
        eta_train, args.spherical_origin, args.cylindrical_origin
    )

    print("Plotting potential parameter evolution ...")
    fname_params_hist = (
        os.path.dirname(args.potential)
        if not os.path.isdir(args.potential)
        else args.potential
    )
    fname_params_hist = glob(os.path.join(fname_params_hist, "*_params.csv"))
    if len(fname_params_hist) > 0:
        potential_params_hist = utils.load_potential_params(fname_params_hist[0])
        plot_potential.plot_potential_param_evolution(
            potential_params_hist, args.fig_dir, fig_fmt=args.fig_fmt
        )

    print("Plotting 2D slices of matter density ...")
    dims = [
        ("x", "y", "z", 0.0),
        ("x", "z", "y", 0.0),
        ("y", "z", "x", 0.0),
    ]
    for dim1, dim2, dimz, z in dims:
        print(f"  --> ({dim1}, {dim2})")
        plot_rho(
            phi_model,
            coords_train,
            args.fig_dir,
            dim1, dim2, dimz,
            z,
            attrs=attrs_train,
            padding=0.95,
            fig_fmt=args.fig_fmt,
        )

    print("Plotting 2D slices of forces ...")
    dims = [
        ("x", "y", "z", 0.0),
        ("x", "z", "y", 0.0),
        ("y", "z", "x", 0.0),
    ]
    for dim1, dim2, dimz, z in dims:
        print(f"  --> ({dim1}, {dim2})")
        plot_force_2d_slice(
            phi_model,
            coords_train,
            args.fig_dir,
            dim1, dim2, dimz,
            z,
            attrs=attrs_train,
            padding=0.95,
            fig_fmt=args.fig_fmt,
        )

    print("Plotting 1D slices of forces ...")
    dims = [
        ("x", "y", 0, 0, "x"),
        ("y", "z", 0, 0, "y"),
        ("z", "x", 0, 0, "z"),
    ]
    for dim1, dimy, y, z, dimforce in dims:
        print(f"  --> ({dim1}, {dimy}={y}, {z})")
        plot_force_1d_slice(
            phi_model,
            coords_train,
            args.fig_dir,
            dim1, dimy,
            y, z,
            dimforce,
            attrs=attrs_train,
            fig_fmt=args.fig_fmt,
        )

    print("Plotting 1D marginals of the potential ...")
    dims = [
        ("z", "rho"),
        ("z", "sigmaz"),
    ]
    for dim1, quantity in dims:
        print(f"  --> ({dim1}: {quantity})")
        plot_custom_potential_marginal(
            phi_model,
            args.fig_dir,
            dim1,
            quantity,
            attrs=attrs_train,
            padding=0.95,
            fig_fmt=args.fig_fmt,
        )

    print("Saving Potential parameter values in a text file ...")
    plot_potential.save_phi_variables(phi_model, args.fig_dir)

    # Extra diagnostics if flow samples are also passed
    if os.path.isfile(args.df_grads_fname) and phi_model["fs"] is not None:
        df_data = utils.load_flow_samples(
            args.df_grads_fname, attrs_to_cut_by=attrs_train
        )
        # coords_sample = plot_flow_projections.calc_coords(df_data['eta'], args.spherical_origin, args.cylindrical_origin)

        print("Plotting marginals of v_circ ...")
        model_u0 = np.array(
            (
                phi_model["fs"]._u_x.numpy(),
                phi_model["fs"]._u_y.numpy(),
                phi_model["fs"]._u_z.numpy(),
            )
        )
        model_r_c = phi_model["fs"]._r_c.numpy()
        eta_gc_train = eta_train.copy()
        eta_gc_train[:, 3:] -= model_u0
        eta_gc_sample = df_data["eta"].copy()
        eta_gc_sample[:, 3:] -= model_u0
        coords_gc_sample = plot_flow_projections.calc_coords(
            eta_gc_sample, args.spherical_origin, args.cylindrical_origin
        )
        coords_gc_train = plot_flow_projections.calc_coords(
            eta_gc_train, args.spherical_origin, args.cylindrical_origin
        )
        plot_vcirc_marginals(
            phi_model,
            coords_gc_train,
            coords_gc_sample,
            fig_dir,
            attrs=attrs_train,
            fig_fmt=args.fig_fmt,
        )

        print("Plotting 2D slices of v_circ ...")
        dims = [
            ("x", "y", "z", 0),
            ("x", "z", "y", 0),
            ("y", "z", "x", 0),
        ]
        for dim1, dim2, dimz, z in dims:
            print(f"  --> ({dim1}, {dim2}, {dimz}={z})")
            plot_vcirc_2d_slice(
                phi_model,
                coords_gc_train, coords_gc_sample,
                fig_dir,
                dim1, dim2, dimz,
                z,
                attrs=attrs_train,
                padding=0.95,
                fig_fmt=args.fig_fmt,
            )

        print("Plotting 2D marginals of \partial f/\partial t ...")
        dims = [
            ("x", "y"),
            ("x", "z"),
            ("y", "z"),
            ("vx", "vy"),
            ("vx", "vz"),
            ("vy", "vz"),
        ]
        print("  Calculating Phi gradients (might take a while) ...")
        _, dphi_dq, _ = potential_tf.calc_phi_derivatives(
            phi_model["phi"], df_data["eta"][:, :3], return_phi=True
        )
        for dim1, dim2 in dims:
            print(f"  --> ({dim1}, {dim2})")
            plot_dfdt_comparison(
                phi_model,
                df_data,
                dphi_dq,
                args.fig_dir,
                dim1, dim2,
                attrs=attrs_train,
                fig_fmt=args.fig_fmt,
            )
    else:
        print("Couldn't find df gradients.")

    return 0


if __name__ == "__main__":
    main()
