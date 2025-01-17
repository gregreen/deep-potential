#!/usr/bin/env python

from __future__ import print_function, division
from binascii import a2b_hqx

import numpy as np

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib import colors
import cmasher as cmr

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


def plot_2d_slice_pot(phi_model, coords_ref, dim1, dim2, dim_plot, fig_dir, z_fill=0, attrs=None, fig_fmt=('pdf',), logscale=True, lims=None, fname_mask=None):
    """
    Currently only supports cartesian.
    """
    labels = [
        '$x\mathrm{\ (kpc)}$', '$y\mathrm{\ (kpc)}$', '$z\mathrm{\ (kpc)}$',
    ]
    keys = ['x', 'y', 'z']
    labels = {k:l for k,l in zip(keys,labels)}
    ikeys = {k:i for i, k in enumerate(keys)}

    titles = [
        r'$\rho_\mathrm{model}\mathrm{\ (M_\odot/pc^3)}$',
        r'$\Phi_\mathrm{model}\mathrm{\ (km^2/s^2)}$',
        r'$a_x^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$',
        r'$a_y^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$',
        r'$a_z^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$',
    ]
    plot_keys = ['rho', 'phi', 'ax', 'ay', 'az']
    titles = {k:t for k,t in zip(plot_keys, titles)}
    iplot_keys = {k:i for i, k in enumerate(plot_keys)}


    for dim in [dim1, dim2]:
        if dim not in keys:
            raise ValueError(f'dimension {dim} not supported')

    fig, axs = plt.subplots(
        2, 1,
        figsize=(3, 3*1.1),
        dpi=140,
        gridspec_kw=dict(height_ratios=[0.2, 2])
    )
    cax, ax = axs

    # Determine the limits
    if lims is None:
        lims = []
        k = 0.20
        for x in coords_ref[dim1], coords_ref[dim2]:
            xlim = (np.min(x), np.max(x))
            lims.append(xlim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    # Generate the grid
    grid_size = 128
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    X, Y = np.meshgrid(0.5*(x[1:]+x[:-1]), 0.5*(y[1:]+y[:-1]))

    q_grid = np.full(shape=(X.size, 3), fill_value=z_fill, dtype='f4')
    q_grid[:,ikeys[dim1]] = X.ravel()
    q_grid[:,ikeys[dim2]] = Y.ravel()
    if fname_mask is not None:
        mask = ~utils.get_mask_eta(q_grid, fname_mask, r_min=attrs['r_in'])[0]
    else:
        r_grid = np.sum(q_grid**2, axis=1)**0.5
        mask = ~((r_grid < attrs['r_out']) & (r_grid > attrs['r_in']))

    # Calculate the model rho
    phi, acc, rho = utils.get_model_values(phi_model, q_grid)
    phi = phi.flatten()
    if dim_plot == 'phi':
        values = phi
    elif dim_plot == 'rho':
        values = rho
    elif dim_plot in ['ax', 'ay', 'az']:
        values = acc[:,iplot_keys[dim_plot] - 2]

    values = np.ma.masked_where(mask, values)
    values = np.reshape(values, X.shape)

    # Apply a spatial cut, if passed
    if attrs is not None:
        # Visualise the boundaries
        plot_flow_projections.add_2dpopulation_boundaries(axs, dim1, dim2, attrs, color='black')

    if logscale:
        kw = dict(cmap='cubehelix', norm=colors.LogNorm(), rasterized=True)
    else:
        vmin, vmax = np.nanpercentile(values.compressed(), [1, 99])
        w = vmax - vmin
        kw = dict(cmap='cmr.rainforest', vmin=vmin - 0.1*w, vmax=vmax + 0.1*w, rasterized=True)
    # Plot model values
    hh = ax.pcolormesh(x, y, values, **kw)
    # Set the colorbar
    cb = fig.colorbar(hh, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')

    cax.set_title(titles[dim_plot], fontsize=10)

    ax.set_xlabel(labels[dim1], labelpad=0)
    ax.set_ylabel(labels[dim2], labelpad=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'2d_slice_{dim1}_{dim2}_{dim_plot}.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


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


def generate_grid(r_out, r_in, n_bins=200, fname_mask=None):
    a_spacing = 2 * r_out / n_bins

    print(f'Generating a grid with spacing {a_spacing*1000:.1f} pc')
    volume_total = 4/3*np.pi*(r_out**3 - r_in**3)

    n_grid = int(volume_total / a_spacing**3)

    def centered_spacing(minv, maxv, step):
        # Makes it so 0 is always included as one of the center-points
        return np.concatenate([np.arange(-step, minv - 1e-9, step=-step)[::-1], np.arange(0, maxv + 1e-9, step=step)], axis=0)

    x_grid = centered_spacing(-r_out, r_out, step=a_spacing)
    y_grid = centered_spacing(-r_out, r_out, step=a_spacing)
    z_grid = centered_spacing(-r_out, r_out, step=a_spacing)

    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    q_grid = np.zeros(shape=(X_grid.size, 3), dtype='f4')
    q_grid[:,0] = X_grid.ravel()
    q_grid[:,1] = Y_grid.ravel()
    q_grid[:,2] = Z_grid.ravel()

    r_grid = np.sum(q_grid**2, axis=1)**0.5
    idx = (r_grid < r_out) & (r_grid > r_in)
    print(f'Grid total volume: {a_spacing**3*np.sum(idx):.4f} kpc^3')
    if fname_mask is not None:
        mask, hp = utils.get_mask_eta(q_grid, fname_mask, r_min=r_in, r_max=r_out)
        print(np.sum(idx))
        idx = idx & mask
        print(np.sum(idx))
        print(f'Masked grid total volume: {a_spacing**3*np.sum(idx):.4f} kpc^3')
    q_grid = q_grid[idx]

    return q_grid


def plot_2dhist(
    x, y, weights, dim1, dim2, fig_dir, fig_fmt, bins=(128, 128), xlim=None, ylim=None, cmap='cmr.rainforest', **kwargs
):
    labels = [
        '$x\mathrm{\ (kpc)}$', '$y\mathrm{\ (kpc)}$', '$z\mathrm{\ (kpc)}$', '$R\mathrm{\ (kpc)}$', r'$\varphi\mathrm{\ (rad)}$',
        r'$\rho_\mathrm{model}\mathrm{\ (M_\odot/pc^3)}$', r'$\Phi_\mathrm{model}\mathrm{\ (km^2/s^2)}$', r'$a_x^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$',
        r'$a_y^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$', r'$a_z^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$',
        r'$a_\varphi^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$', r'$a_R^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$'
    ]
    keys = [
        'x', 'y', 'z', 'cylR', 'cylphi',
        'rho', 'phi', 'ax',
        'ay', 'az',
        'acylphi', 'acylR'
    ]
    labels = {k:l for k,l in zip(keys,labels)}
    ikeys = {k:i for i, k in enumerate(keys)}

    fig, axs = plt.subplots(
        2, 1,
        figsize=(4, 4.2),
        gridspec_kw=dict(width_ratios=[3], height_ratios=[0.2, 4]),
        layout='compressed'
    )
    cax, ax = axs

    # Get the plot limits
    lims = []
    for i, lim in enumerate([xlim, ylim]):
        if lim is None:
            lim = np.percentile([x, y][i], [1.0, 99.0])
            w = lim[1] - lim[0]
            lim = [lim[0] - 0.1 * w, lim[1] + 0.1 * w]
        lims.append(lim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    x_bins = np.linspace(xmin, xmax, bins[0])
    y_bins = np.linspace(ymin, ymax, bins[1])

    n = np.histogram2d(x, y, bins=[x_bins, y_bins])[0]
    # # Normalize n along the y-axis
    #n = n / np.max(n, axis=1, keepdims=True)
    val_avg = binned_statistic(x, y, np.median, bins=x_bins).statistic
    val_std = binned_statistic(x, y, lambda x: (np.percentile(x, 84.135) - np.percentile(x, 15.865))/2, bins=x_bins).statistic
    x_bins_c = 0.5 * (x_bins[1:] + x_bins[:-1])
    l = len(x_bins_c)
    idx = (np.arange(l) > l / 4) & (np.arange(l) < 3 * l / 4)
    peak_x = x_bins_c[idx][np.argmax(val_avg[idx], axis=0)]

    im = ax.imshow(
        n.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        **kwargs,
    )
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")

    ax.plot(x_bins_c, val_avg, color='tab:red', lw=0.5, label=f'median, peak={peak_x:.3f}')
    ax.fill_between(x_bins_c, val_avg - val_std, val_avg + val_std, lw=0.5, alpha=0.15, color='tab:red')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend()

    ax.set_xlabel(labels[dim1])
    ax.set_ylabel(labels[dim2])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.locator_params(nbins=5)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'2d_hist_{dim1}_{dim2}.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_2dhist_wrapper(
    x, y, values=None, operation="count",
    bins=(64, 64),
    xlabel="$x$",
    ylabel="$y$",
    lims=None,
    title=None,
    cmap='cmr.rainforest',
    fig=None, ax=None, cax=None,
    normalize_along_axis=None,
    **kwargs
):
    if fig is None:
        fig, axs = plt.subplots(
            2, 1,
            figsize=(4, 4.2),
            gridspec_kw=dict(width_ratios=[3], height_ratios=[0.1, 3]),
            layout='compressed'
        )
    cax, ax = axs

    # Get the plot limits
    if lims is None:
        lims = []
        k = 1.1
        for x_ in [x, y]:
            xlim = np.percentile(x_, [1.0, 99.0])
            w = xlim[1] - xlim[0]
            xlim = [xlim[0] - 0.2 * w, xlim[1] + 0.2 * w]
            lims.append(xlim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    x_bins = np.linspace(xmin, xmax, bins[0])
    y_bins = np.linspace(ymin, ymax, bins[1])

    ret = binned_statistic_2d(
        x, y, values, statistic=operation, bins=[x_bins, y_bins]
    ).statistic.T
    if normalize_along_axis == 0:
        ret /= (np.max(ret, axis=normalize_along_axis) + 1e-10)[None, :]
    elif normalize_along_axis == 1:
        ret /= (np.max(ret, axis=normalize_along_axis) + 1e-10)[:, None]

    # Choose a suitable colormap
    if operation == "count":
        kwargs["vmin"] = 0

    im = ax.imshow(
        ret,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        aspect="auto",
        cmap=cmap,
        **kwargs,
    )
    ret = binned_statistic(x, y, statistic=np.median, bins=x_bins).statistic # TODO not a proper median to weighing with f
    n = binned_statistic(x, y, statistic='count', bins=x_bins).statistic
    #print(x_bins, n)
    #ax.plot(0.5*(x_bins[1:] + x_bins[:-1]), ret, label="median", lw=1, color='tab:red', alpha=1)
    if cax is not None:
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.locator_params(nbins=5)
        # TODO: Title not appearing without cax
        if title is not None:
            cax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, axs


'''def plot_fourier_spectrum(q_grid, rho, fig_dir, fig_fmt):
    """
    Computes the fourier components along x-, y- and z-axes for the density field rho.
    """
    labels = ['$x\mathrm{\ (kpc)}$', '$y\mathrm{\ (kpc)}$', '$z\mathrm{\ (kpc)}$',]
    keys = ['x', 'y', 'z']
    labels = {k:l for k,l in zip(keys, labels)}
    ikeys = {k:i for i,k in enumerate(keys)}

    fig, axs = plt.subplots(
        3, 1,
        figsize=(6, 3),
        dpi=200,
        gridspec_kw=dict(height_ratios=[1, 1, 1])
    )

    for i, dim in enumerate(keys):
        ax = axs[i]
        ax.set_title(labels[dim])

        # Compute the fourier spectrum
        x = q_grid[:, i]


        rho_grid = np.sum(rho_grid, axis=ikeys[dim])
        rho_grid = np.fft.fftshift(rho_grid)
        rho_grid = np.fft.ifftn(rho_grid)

        # Plot the spectrum
        ax.plot(rho_grid)
        ax.set_yscale('log')
    return'''


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
        required=True,
        help="Directory in which to store the flow samples (positions and f gradients).",
    )
    parser.add_argument(
        "--fname-mask",
        type=str,
        required=False,
        default=None,
        help="Filename for the mask for the samples. The mask is in max distance - healpix format.",
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
        default=("png",),
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
    parser.add_argument(
        "--potential-mask",
        type=str,
        required=False,
        default=None,
        help="Filename for the mask for the potential. The mask is in distance - healpix format.",
    )
    args = parser.parse_args()

    params = {
        'text.usetex': True,
        'font.size': 10,
        'font.family': 'lmodern',
        'figure.dpi': 250
    }
    plt.rcParams.update(params)

    if args.dark:
        plt.style.use("dark_background")

    if args.autosave:
        # Infer the place to store the figures
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
        print(fig_dir)
        os.makedirs(fig_dir, exist_ok=True)
        if os.path.isfile(fname_loss_pdf):
            # Copy the latest loss over to the plots dir
            shutil.copy(fname_loss_pdf, fig_dir)
            shutil.copy(fname_loss_pdf[:-4] + "_noreg.pdf", fig_dir)
        args.fig_dir = fig_dir
    else:
        fig_dir = args.fig_dir

    # Loading in data
    print("Loading training data ...")
    data_train, attrs_train = utils.load_training_data(args.input, cut_attrs=True)
    print(f"  --> Training data shape = {data_train['eta'].shape}")

    print("Loading in distribution function gradients ...")
    df_data = utils.load_flow_samples(args.df_grads_fname, attrs_to_cut_by=attrs_train)

    if args.fname_mask is not None:
        # Update df_data to include only the data within the mask
        print("Applying mask to the samples ...")
        mask = utils.get_mask_eta(data_train['eta'], args.fname_mask, r_min=attrs_train['r_in'])[0]
        data_train['eta'] = data_train['eta'][mask]
        if 'weights' in data_train:
            data_train['weights'] = data_train['weights'][mask]

        mask = utils.get_mask_eta(df_data['eta'], args.fname_mask, r_min=attrs_train['r_in'])[0]
        df_data['eta'] = df_data['eta'][mask]
        df_data['df_deta'] = df_data['df_deta'][mask]
        if 'f' in df_data:
            df_data['f'] = df_data['f'][mask]

    q_grid = generate_grid(attrs_train['r_out'], attrs_train['r_in'], fname_mask=args.fname_mask)

    print("Loading potential")
    phi_model = utils.load_potential(args.potential)
    if phi_model["fs"] is not None:
        phi_model["fs"].debug()

    print("Calculating/loading potential values ...")
    fname = 'potential_values'
    phi, acc, rho = utils.get_model_values(phi_model, df_data['eta'][:,:3], fig_dir=fig_dir, fname=fname)
    phi_grid, acc_grid, rho_grid = utils.get_model_values(phi_model, q_grid, fig_dir=fig_dir, fname=fname+'_grid')

    print("Calculating cylindrical & spherical coordinates ...")
    acc_components = utils.calc_coords(df_data['eta'][:,:3], args.spherical_origin, args.cylindrical_origin, acc)
    acc_components_grid = utils.calc_coords(q_grid, args.spherical_origin, args.cylindrical_origin, acc_grid)
    coords_train_eta = utils.calc_coords(data_train['eta'], args.spherical_origin, args.cylindrical_origin)
    coords_df_eta = utils.calc_coords(df_data['eta'], args.spherical_origin, args.cylindrical_origin)
    coords_grid = utils.calc_coords(q_grid, args.spherical_origin, args.cylindrical_origin)

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
    for dim_val in ['rho', 'phi', 'ax', 'ay', 'az']:
        jobs = [
            ("x", "y", 0.0),
            ("x", "z", 0.0),
            ("y", "z", 0.0),
        ]
        for dim1, dim2, z in jobs:
            print(f"  --> ({dim1}, {dim2})")
            plot_2d_slice_pot(
                phi_model, coords_train_eta, dim1, dim2, dim_val, args.fig_dir, z_fill=z, attrs=attrs_train, fig_fmt=args.fig_fmt, logscale=False, lims=None, fname_mask=args.fname_mask
            )

    jobs = [
        ('cylphi', 'acylphi', acc_components['cylphi'], None),
        ('cylR', 'acylR', acc_components['cylR'], None),
        ('z', 'az', acc_components['z'], None),
        ('cylR', 'rho', rho, (0., 0.15)),
        ('z', 'rho', rho, (0., 0.15)),
        ('cylphi', 'rho', rho, (0., 0.15)),
    ]
    for dim1, dim2, val, ylim in jobs:
        plot_2dhist(
            coords_df_eta[dim1], val, df_data['f'], dim1, dim2, fig_dir, fig_fmt=args.fig_fmt, ylim=ylim
        )

    # Plot implied v_circ
    R, a_R = coords_grid['cylR'], -acc_components_grid['cylR']
    omega = phi_model['fs']._omega.numpy()
    fig, axs = plot_2dhist_wrapper(
        R, np.sqrt(R*a_R)*100, np.ones(len(rho_grid)), operation=np.sum,
        xlabel="$R\mathrm{{\>(kpc)}}$", ylabel="$v_\mathrm{circ}\mathrm{{\>(km/s))}}$",
        vmin=0, bins=(111, 128),
        lims=None,
        normalize_along_axis=0,
        title=f'whole volume $\\Omega={omega}$'
    )
    for fmt in args.fig_fmt:
        fname = os.path.join(fig_dir, f'2d_hist_vcirc_cylR.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print("Saving Potential parameter values in a text file ...")
    plot_potential.save_phi_variables(phi_model, args.fig_dir)

    return 0


if __name__ == "__main__":
    main()
