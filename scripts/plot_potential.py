#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter

import h5py
import progressbar
import os
from contextlib import suppress
from glob import glob
from pathlib import Path
import shutil
import json
from scipy.stats import binned_statistic_dd, binned_statistic_2d

import tensorflow as tf

print(f"Tensorflow version {tf.__version__}")

import potential_tf
import plot_flow_projections
import utils


dpi = 200


def get_sampling_progressbar_fn(n_batches, n_samples):
    widgets = [
        progressbar.Bar(),
        progressbar.Percentage(),
        " | ",
        progressbar.Timer(format="Elapsed: %(elapsed)s"),
        " | ",
        progressbar.AdaptiveETA(),
        " | ",
        progressbar.Variable("batches_done", width=6, precision=0),
        ", ",
        progressbar.Variable("n_batches", width=6, precision=0),
        ", ",
        progressbar.Variable("n_samples", width=8, precision=0),
    ]
    bar = progressbar.ProgressBar(max_value=n_batches, widgets=widgets)
    # n_batches = n_batches
    # n_samples = n_samples

    def update_progressbar(i):
        bar.update(i + 1, batches_done=i + 1, n_batches=n_batches,
                   n_samples=n_samples)

    return update_progressbar


def get_model_values(phi_model, q_eval, fig_dir=None, fname=None, save=True):
    # Calculate the model potential values
    # In practice, up to 1e6 densities can be calculated at once..
    batch_size = 131072
    n0 = len(q_eval)
    q_eval = tf.data.Dataset.from_tensor_slices(q_eval).batch(batch_size)

    if save:
        fname = os.path.join(fig_dir, f'data/{fname}_{n0}.npz')
    if (not save) or not os.path.exists(fname):
        rhos = []
        dphi_dqs = []
        phis = []

        bar, iteration = get_sampling_progressbar_fn(len(q_eval), n0), 0
        for i, b in enumerate(q_eval):
            phi,dphi_dq,d2phi_dq2 = potential_tf.calc_phi_derivatives(
                phi_model['phi'], b, return_phi=True
            )
            rhos.append(2.325*d2phi_dq2.numpy()/(4*np.pi)) # [M_Sun/pc^3]
            dphi_dqs.append(dphi_dq)
            phis.append(phi)
            bar(iteration)
            iteration += 1
        rhos = np.concatenate(rhos)
        dphi_dqs = np.concatenate(dphi_dqs)
        phis = np.concatenate(phis)
        if save:
            Path(os.path.join(fig_dir), 'data').mkdir(parents=True, exist_ok=True)
            np.savez(fname, phi=phis, dphi_dq=dphi_dqs, rho=rhos)
    else:
        npzfile = np.load(fname)
        rhos = npzfile['rho']
        dphi_dqs = npzfile['dphi_dq']
        phis = npzfile['phi']
    
    return phis, dphi_dqs, rhos


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
    """Only support cartesian."""
    labels = [
        "$x$",
        "$y$",
        "$z$",
    ]

    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    for dim in [dim1, dim2]:
        if dim not in keys:
            raise ValueError(f"dimension {dim} not supported")

    fig, (axs) = plt.subplots(
        2,
        3,
        figsize=(6, 2.2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[2, 2, 2], height_ratios=[0.2, 2]),
    )
    ax_p, cax_p = axs[1, 0], axs[0, 0]
    ax_r, cax_r = axs[1, 1], axs[0, 1]
    ax_e, cax_e = axs[1, 2], axs[0, 2]
    main_axs = [ax_p, ax_r, ax_e]

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

    phi, _, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], q_grid, return_phi=True
    )

    phi_img = np.reshape(phi.numpy(), X.shape)
    rho_img = np.reshape(
        2.309 * d2phi_dq2.numpy() / (4 * np.pi), X.shape
    )  # [M_Sun/pc^3]

    if attrs["has_spatial_cut"]:
        # Visualise the boundaries
        plot_flow_projections.add_2dpopulation_boundaries(
            main_axs, dim1, dim2, attrs, color="black"
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

        phi_img = np.ma.masked_where(mask, phi_img)
        rho_img = np.ma.masked_where(mask, rho_img)

    phi_img = phi_img - np.mean(phi_img)

    # Plot phi
    min_val, max_val = phi_img.min(), phi_img.max()
    if min_val * max_val < 0:
        divnorm = colors.TwoSlopeNorm(vcenter=0.0)
        kw = dict(cmap="seismic", norm=divnorm, shading="flat", lw=0, rasterized=True)
    else:
        kw = dict(
            cmap="viridis",
            vmin=min_val, vmax=max_val,
            shading="flat",
            lw=0,
            rasterized=True,
        )
    hh = ax_p.pcolormesh(x, y, phi_img, **kw)
    # Set the colorbar
    cb_p = fig.colorbar(hh, cax=cax_p, orientation="horizontal")
    cb_p.ax.xaxis.set_ticks_position("top")
    cb_p.ax.locator_params(nbins=3)
    cax_p.set_title(r"$\Phi^*$", fontsize=10)

    # Plot rho
    min_val, max_val = rho_img.min(), rho_img.max()
    if min_val * max_val < 0:
        divnorm = colors.TwoSlopeNorm(vcenter=0.0)
        kw = dict(cmap="seismic", norm=divnorm, shading="flat", lw=0, rasterized=True)
    else:
        kw = dict(
            cmap="viridis",
            vmin=min_val, vmax=max_val,
            shading="flat",
            lw=0,
            rasterized=True,
        )
    hh = ax_r.pcolormesh(x, y, rho_img, **kw)
    # Set the colorbar
    cb_r = fig.colorbar(hh, cax=cax_r, orientation="horizontal")
    cb_r.ax.xaxis.set_ticks_position("top")
    cb_r.ax.locator_params(nbins=3)
    cax_r.set_title(r"$\rho^*$", fontsize=10)

    # Plot a slice of data
    dz = 0.05
    idx_train = (coords_train[dimz] > z - dz) & (coords_train[dimz] < z + dz)
    x_train = coords_train[dim1][idx_train]
    y_train = coords_train[dim2][idx_train]
    nbins = 64
    weights = np.full_like(
        x_train, 1 / ((xmax - xmin) * (xmax - xmin) / nbins**2 * 2 * dz) / 10**9
    )   
    grid_size = 32
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    h = ax_e.hist2d(
        x_train, y_train, range=lims, weights=weights, bins=(x, y), rasterized=True
    )  # , norm=matplotlib.colors.LogNorm(vmin=1))
    cb_e = fig.colorbar(h[3], cax=cax_e, orientation="horizontal")
    cb_e.ax.xaxis.set_ticks_position("top")
    cb_e.ax.locator_params(nbins=3)
    cax_e.set_title(r"$\rho_\mathrm{train}$", fontsize=10)

    ax_r.set_yticklabels([])
    ax_e.set_yticklabels([])

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
    """Only support cartesian."""
    labels = [
        "$x$",
        "$y$",
        "$z$",
    ]
    titles = ["$F_x^*$", "$F_y^*$", "$F_z^*$"]

    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    for dim in [dim1, dim2]:
        if dim not in keys:
            raise ValueError(f"dimension {dim} not supported")

    fig, (axs) = plt.subplots(
        2, 3,
        figsize=(6, 2.2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[2, 2, 2], height_ratios=[0.2, 2]),
    )
    ax_x, cax_x = axs[1, 0], axs[0, 0]
    ax_y, cax_y = axs[1, 1], axs[0, 1]
    ax_z, cax_z = axs[1, 2], axs[0, 2]
    main_axs = [ax_x, ax_y, ax_z]
    main_caxs = [cax_x, cax_y, cax_z]

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

    phi, dphi_dq, _ = potential_tf.calc_phi_derivatives(
        phi_model["phi"], q_grid, return_phi=True
    )

    if attrs["has_spatial_cut"]:
        # Visualise the boundaries
        plot_flow_projections.add_2dpopulation_boundaries(
            main_axs, dim1, dim2, attrs, color="black"
        )

        # Mask for the area for which forces are plotted
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

    for i, ax in enumerate(main_axs):
        F_i = -dphi_dq[:, i].numpy().ravel().reshape(X.shape)
        if attrs["has_spatial_cut"]:
            F_i = np.ma.masked_where(mask, F_i)

        # Plot the force
        min_val, max_val = F_i.min(), F_i.max()
        if min_val * max_val < 0:
            divnorm = colors.TwoSlopeNorm(vcenter=0.0)
            kw = dict(
                cmap="seismic", norm=divnorm, shading="flat", lw=0, rasterized=True
            )
        else:
            kw = dict(
                cmap="viridis",
                vmin=min_val, vmax=max_val,
                shading="flat",
                lw=0,
                rasterized=True,
            )
        hh = ax.pcolormesh(x, y, F_i, **kw)

        # Set the colorbar
        cax = main_caxs[i]
        cb = fig.colorbar(hh, cax=cax, orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.locator_params(nbins=3)
        cax.set_title(titles[i])

    ax_y.set_yticklabels([])
    ax_z.set_yticklabels([])

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


def get_potential_dfdt(phi_model, df_data, dphi_dq):
    """
    Returns the \partial f/\partial t predicted by the potential in the
    associated rotating frame.
    """

    eta, df_deta = df_data["eta"], df_data["df_deta"]

    model_omega = phi_model["fs"]._omega.numpy()
    model_u0 = np.array(
        (
            phi_model["fs"]._u_x.numpy(),
            phi_model["fs"]._u_y.numpy(),
            phi_model["fs"]._u_z.numpy(),
        )
    )
    model_r_c = phi_model["fs"]._r_c.numpy()

    try:
        dphi_dq = dphi_dq.numpy()
    except Exception:
        pass
    pdf_dt_CBE = -np.sum(eta[:, 3:] * df_deta[:, :3], axis=1) + np.sum(
        dphi_dq * df_deta[:, 3:], axis=1
    )

    ix, iy, ivx, ivy = 0, 1, 3, 4
    pdf_dt_stat = -model_omega * (
        (eta[:, ix] - model_r_c) * df_deta[:, iy]
        - eta[:, iy] * df_deta[:, ix]
        + (eta[:, ivx] - model_u0[ix]) * df_deta[:, ivy]
        - (eta[:, ivy] - model_u0[iy]) * df_deta[:, ivx]
    ) - np.sum(model_u0 * df_deta[:, :3], axis=1)

    return pdf_dt_CBE - pdf_dt_stat


def plot_dfdt_comparison(
    phi_model,
    df_data, dphi_dq,
    fig_dir,
    dim1, dim2,
    attrs,
    fig_fmt=("svg",),
    save=True,
    grid_size=32,
):
    """Plots the \partial f/\partial t corresponding to the flow via solving the Collisionless Boltzmann Equation (CBE)
    in the rotating frame specified by omega, v_0 and r_c using the least squares method in cubic spatial bins.
    """

    def amplify(min_val, max_val, k=0.2):
        if min_val * max_val > 0:
            w = max_val - min_val
            return min_val - k * w, max_val + k * w
        return min_val * (1 + k), max_val * (1 + k)

    eta = df_data["eta"]

    fig, (all_axs) = plt.subplots(
        2, 2,
        figsize=(4, 2.2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[2, 2], height_ratios=[0.2, 2]),
    )
    axs = all_axs[1, :]
    caxs = all_axs[0, :]

    labels = ["$x$", "$y$", "$z$", "$v_x$", "$v_y$", "$v_z$"]
    keys = ["x", "y", "z", "vx", "vy", "vz"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}
    ix, iy = ikeys[dim1], ikeys[dim2]

    for i in range(2):
        axs[i].set_xlabel(labels[dim1])
    axs[0].set_ylabel(labels[dim2])
    axs[1].set_yticklabels([])

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
        plot_flow_projections.add_2dpopulation_boundaries(
            axs, dim1, dim2, attrs, color="black"
        )

    x_bins = np.linspace(xmin, xmax, grid_size)
    y_bins = np.linspace(ymin, ymax, grid_size)

    omega = phi_model["fs"]._omega.numpy()
    v0 = np.array(
        (
            phi_model["fs"]._u_x.numpy(),
            phi_model["fs"]._u_y.numpy(),
            phi_model["fs"]._u_z.numpy(),
        )
    )
    r_c = phi_model["fs"]._r_c.numpy()
    # omega = -0.0
    # v0 = np.array([0., 0., 0.])
    # df_data_temp = {'eta': df_data['eta'] - np.array([0, 0, 0, -0.128, -2.222, -0.077]), 'df_deta':df_data['df_deta']}
    theoretical_flow_dfdt = plot_flow_projections.get_flow_leastsq_dfdt(
        df_data, omega=omega, v_0=v0, r_c=r_c, grid_size=32
    )
    potential_dfdt = get_potential_dfdt(phi_model, df_data, dphi_dq)

    # Flow ideal dfdt
    ret = binned_statistic_2d(
        eta[:, ix],
        eta[:, iy],
        theoretical_flow_dfdt,
        statistic=np.mean,
        bins=[x_bins, y_bins],
    )
    vmin, vmax = amplify(*np.nanpercentile(ret.statistic, [1, 99]), k=0.4)
    divnorm = colors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    im = axs[0].imshow(
        ret.statistic.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="seismic",
        norm=divnorm,
        aspect="auto",
    )
    cb = fig.colorbar(im, cax=caxs[0], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.locator_params(nbins=5)
    title = (
        "$(\partial f/\partial t)_\mathrm{flow\_leastsq}$"
        + f"\n$\Omega={omega:.3f},$\n$\\vec v_0=({v0[0]:.2f}, {v0[1]:.2f}, {v0[2]:.2f})$"
    )
    caxs[0].set_title(title)

    # CBE+rotating stationarity discrepancy
    ret = binned_statistic_2d(
        eta[:, ix],
        eta[:, iy],
        -potential_dfdt,
        statistic=np.mean,
        bins=[x_bins, y_bins],
    )
    vmin, vmax = amplify(*np.nanpercentile(ret.statistic, [1, 99]), k=0.4)
    divnorm = colors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    im = axs[1].imshow(
        ret.statistic.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="seismic",
        norm=divnorm,
        aspect="auto",
    )
    cb = fig.colorbar(im, cax=caxs[1], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.locator_params(nbins=5)
    caxs[1].set_title("$(\partial f/\partial t)_\mathrm{ potential}$")

    if save:
        for fmt in fig_fmt:
            fname = os.path.join(
                fig_dir, f"phi_dfdt_comparison_{dim1}_{dim2}_omega={omega:.2f}.{fmt}"
            )
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axs


def plot_potential_param_evolution(potential_params_hist, fig_dir, fig_fmt=("svg",)):
    """Plots the evolution of different potential params. Generates two plots,
    one combining all params in one plot, the other plotting each param separately.

    Args:
        potential_params_hist (dict): A dictionary of lists of training
            step-by-step values for each parameter
        fig_dir (str): The directory to save the figure to
        fig_fmt (tuple): The formats to save the figure in
    """

    # Plot the params combined
    label_converter = {
        "omega": r"$\omega$",
        "u_x0": r"$v_{0x}$",
        "u_y0": r"$v_{0y}$",
        "u_z0": r"$v_{0z}$",
        "mn_amp": r"mn$_\mathrm{amp}$",
        "mn_a": r"mn$_\mathrm{a}$",
        "mn_b": r"mn$_\mathrm{b}$",
        "mn1_amp": r"mn$_\mathrm{amp,1}$",
        "mn1_a": r"mn$_\mathrm{a,1}$",
        "mn1_b": r"mn$_\mathrm{b,1}$",
        "mn2_amp": r"mn$_\mathrm{amp,2}$",
        "mn2_a": r"mn$_\mathrm{a,2}$",
        "mn2_b": r"mn$_\mathrm{b,2}$",
        "mn3_amp": r"mn$_\mathrm{amp,3}$",
        "mn3_a": r"mn$_\mathrm{a,3}$",
        "mn3_b": r"mn$_\mathrm{b,3}$",
        "halo_amp": r"$\mathrm{halo}_\mathrm{amp}$",
        "halo_a": r"$\mathrm{halo}_\mathrm{a}$",
        "bulge_amp": r"$\mathrm{bulge}_\mathrm{amp}$",
        "bulge_rcut": r"$\mathrm{bulge}_\mathrm{rcut}$",
        "bulge_alpha": r"$\mathrm{bulge}_\mathrm{alpha}$",
        "dz": r"$\Delta z$",
        "lr_hist": r"lr",
    }

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    non_constant_params = []
    for i, (label, values) in enumerate(potential_params_hist.items()):
        if label not in label_converter:
            print(f"{label} not in label converter")
        else:
            label = label_converter[label]
        values = np.array(values)
        w = values.max() - values.min()
        if w < 1e-9:
            values *= 0
        else:
            non_constant_params.append((label, values))

            new_label = f"{label} = {values[-1]:.2e}"
            if np.abs(np.log10(np.abs(values[-1]))) < 3:
                new_label = f"{label} = {values[-1]:.3f}"

            values = (values - values.min()) / w

            ax.plot(values, label=new_label, lw=1.0, alpha=0.8)
            ax.set_xlim(0, len(values) - 1)
    ax.legend()
    ax.set_ylim(-0.01, 1.01)
    ax.set_title("Normalized potential parameter evolution vs training step")

    """print(len(non_constant_params), non_constant_params)
    if len(non_constant_params) == 0:
        plt.close()
        return"""

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"potential_param_evolution_combined.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    if len(fig_fmt) == 0:
        plt.show()
    plt.close(fig)

    # Plot the params separately
    # Squeeze makes the returned array always 2d
    nrows = (len(non_constant_params) + 2) // 3
    fig, (axs) = plt.subplots(
        nrows,
        3,
        figsize=(6, nrows * 2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[2, 2, 2], height_ratios=[2] * nrows),
        squeeze=False,
    )

    # Check if learning rate is in the params. If so, then add learning rate drops to the plots
    if "lr_hist" in potential_params_hist:
        lr_hist = np.array(potential_params_hist["lr_hist"])
        lr_ratio = lr_hist[lr_hist > 0][1:] / lr_hist[lr_hist > 0][:-1]
        n_drop = np.where(lr_ratio < 0.95)[0]

        for i in range(len(non_constant_params)):
            ax = axs[i // 3, i % 3]
            for k in n_drop:
                ax.axvline(k, c="k", alpha=0.1, ls="--", lw=1.0)

    for i, (label, values) in enumerate(non_constant_params):
        values = np.array(values)
        ax = axs[i // 3, i % 3]
        new_label = f"{label} = {values[-1]:.2e}"
        if np.abs(np.log10(np.abs(values[-1]))) < 3:
            new_label = f"{label} = {values[-1]:.3f}"
        ax.plot(values, label=new_label, lw=1.0)
        #ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlim(0, len(values) - 1)
        ax.legend(handlelength=0, handletextpad=0)

    for i in range(len(non_constant_params), 3 * nrows):
        axs[i // 3, i % 3].set_axis_off()

    fig.subplots_adjust(
        wspace=0.4, hspace=0.2, left=0.10, right=0.90, bottom=0.20, top=0.90
    )

    fig.suptitle("Potential parameter evolution vs training step")
    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"potential_param_evolution_separate.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    if len(fig_fmt) == 0:
        plt.show()
    plt.close(fig)


def save_phi_variables(phi_model, fig_dir):
    variables = {}
    with suppress(Exception):
        variables["mn_amp"] = phi_model["phi"]._mn_amp.numpy()
    with suppress(Exception):
        variables["mn_a"] = np.exp(phi_model["phi"]._mn_loga.numpy())
    with suppress(Exception):
        variables["mn_b"] = np.exp(phi_model["phi"]._mn_logb.numpy())
    with suppress(Exception):
        variables["halo_amp"] = phi_model["phi"]._halo_amp.numpy()
    with suppress(Exception):
        variables["halo_a"] = np.exp(phi_model["phi"]._halo_loga.numpy())
    with suppress(Exception):
        variables["bulge_amp"] = phi_model["phi"]._bulge_amp.numpy()
    with suppress(Exception):
        variables["bulge_rcut"] = np.exp(phi_model["phi"]._bulge_logrcut.numpy())
    with suppress(Exception):
        variables["bulge_alpha"] = phi_model["phi"]._bulge_alpha.numpy()
    with suppress(Exception):
        variables["fs_v0x"] = phi_model["fs"]._u_x.numpy()
    with suppress(Exception):
        variables["fs_v0y"] = phi_model["fs"]._u_y.numpy()
    with suppress(Exception):
        variables["fs_v0z"] = phi_model["fs"]._u_z.numpy()
    with suppress(Exception):
        variables["fs_omega"] = phi_model["fs"]._omega.numpy()
    with suppress(Exception):
        variables["fs_rc"] = phi_model["fs"]._r_c.numpy()

    np.savetxt(
        os.path.join(fig_dir, "phi_params.txt"),
        [p for p in zip(variables.keys(), variables.values())],
        delimiter=",", fmt="%s",
    )


def plot_force_1d_slice(
    phi_model,
    coords_train,
    fig_dir,
    dim1, dimy,
    y, z,
    dimforce,
    attrs,
    padding=0.95,
    fig_fmt=("svg",),
    save=True,
):
    """Only supports cartesian."""
    labels = [
        "$x$",
        "$y$",
        "$z$",
    ]
    force_labels = ["$F_x^*$", "$F_y^*$", "$F_z^*$"]

    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    force_labels = {k: l for k, l in zip(keys, force_labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    if dim1 not in keys:
        raise ValueError(f"dimension {dim1} not supported")

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
    # print(eta_eval)

    _, dphi_dq, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], eta_plot, return_phi=True
    )
    Z_plot = -dphi_dq[:, ikeys[dimforce]].numpy()

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

    ax.set_xlabel(labels[dim1])
    ax.set_ylabel(f"{force_labels[dimforce]}")

    """fig.subplots_adjust(
        left=0.16,
        right=0.83,
        bottom=0.18,
        top=0.74,
        wspace=0.16
    )"""

    if save:
        for fmt in fig_fmt:
            fname = os.path.join(fig_dir, f"phi_force_slice_{dim1}.{fmt}")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Deep Potential: Plot different diagnostics for a potential.",
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
        default=("pdf",),
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
            fig_dir = "plots/" + fname_pot[fname_pot.find("models/") + 7:] + "/"
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
        plot_potential_param_evolution(
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
            dim1,
            dim2,
            dimz,
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
            dim1,
            dim2,
            dimz,
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
        print(f"  --> ({dim1})")
        plot_force_1d_slice(
            phi_model,
            coords_train,
            args.fig_dir,
            dim1,
            dimy,
            y,
            z,
            dimforce,
            attrs=attrs_train,
            padding=0.95,
            fig_fmt=args.fig_fmt,
        )

    print("Saving Potential parameter values in a text file ...")
    save_phi_variables(phi_model, args.fig_dir)

    # Extra diagnostics if flow samples are also passed
    if os.path.isfile(args.df_grads_fname) and phi_model["fs"] is not None:
        df_data = utils.load_flow_samples(
            args.df_grads_fname, attrs_to_cut_by=attrs_train
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
        _, dphi_dq, _ = get_model_values(
            phi_model, df_data["eta"][:, :3], save=False
        )
        for dim1, dim2 in dims:
            print(f"  --> ({dim1}, {dim2})")
            plot_dfdt_comparison(
                phi_model,
                df_data,
                dphi_dq,
                args.fig_dir,
                dim1,
                dim2,
                attrs=attrs_train,
                fig_fmt=args.fig_fmt,
            )
    else:
        print("Couldn't find df gradients.")

    return 0


if __name__ == "__main__":
    main()
