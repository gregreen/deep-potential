#!/usr/bin/env python
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib import colors
import numpy as np
import tensorflow as tf
import math
import sys
import importlib
import h5py
import os
from pathlib import Path
import shutil
from scipy.stats import binned_statistic_2d
from scipy.ndimage.filters import gaussian_filter
from glob import glob

sys.path.append("../../scripts/")
import plot_flow_projections
import plot_potential
import utils
import potential_tf

from astropy import units as u
from astropy import constants as const

import corner

dpi = 200


from tqdm import tqdm


def get_true_phi(q_eval):
    mass_dm = 3.2e5
    mass_star = 9.2e4
    r0 = 0.15

    mass_tot = len(eta_star) * mass_star + len(eta_dm) * mass_dm

    phis = np.zeros(len(q_eval))
    for i, p in tqdm(enumerate(q_eval), total=len(q_eval)):
        for q, m in [(q_star, mass_star), (q_dm, mass_dm)]:
            r2 = np.sum((q - p) ** 2, axis=1)
            phis[i] += -m * np.sum(1 / np.sqrt(r2 + r0**2))
    phis = (phis * u.M_sun / u.kpc * const.G).to(100**2 * u.km**2 / u.s**2)

    return phis


def get_true_phi_parallel(q_eval, n_processes=1):
    if n_processes == -1:
        # Choose maximum number of pools
        import os

        n_processes = int(os.environ["OMP_NUM_THREADS"])
    if n_processes > 1:
        from multiprocessing import Pool

        q_subgrids = np.split(
            q_eval,
            np.arange(0, len(q_eval), (len(q_eval) + n_processes - 1) // n_processes)[
                1:
            ],
            axis=0,
        )
        with Pool(n_processes) as p:
            result = p.map(get_true_phi, q_subgrids)
        return np.concatenate(result)
    else:
        return get_true_phi(q_eval)


def get_true_force(q_eval):
    mass_dm = 3.2e5
    mass_star = 9.2e4
    r0 = 0.15

    forces = np.zeros((len(q_eval), 3))
    for i, p in tqdm(enumerate(q_eval), total=len(q_eval)):
        for q, m in [(q_star, mass_star), (q_dm, mass_dm)]:
            r2 = np.sum((q - p) ** 2, axis=1)
            forces[i, :] += m * np.sum(
                (q - p) / (r2[:, None] + r0**2) ** (3 / 2), axis=0
            )
    forces = (forces * u.M_sun / u.kpc**2 * const.G).decompose()

    return forces


def get_true_force_parallel(q_eval, n_processes=1):
    if n_processes == -1:
        # Choose maximum number of pools
        import os

        n_processes = int(os.environ["OMP_NUM_THREADS"])
    if n_processes > 1:
        from multiprocessing import Pool

        # Look up a shared queue between cores
        q_subgrids = np.split(
            q_eval,
            np.arange(0, len(q_eval), (len(q_eval) + n_processes - 1) // n_processes)[
                1:
            ],
            axis=0,
        )
        with Pool(n_processes) as p:
            result = p.map(get_true_force, q_subgrids)
        return np.concatenate(result)
    else:
        return get_true_force(q_eval)


def get_true_rho(q_eval):
    mass_dm = 3.2e5
    mass_star = 9.2e4
    r0 = 0.15

    mass_tot = len(eta_star) * mass_star + len(eta_dm) * mass_dm

    rhos = np.zeros(len(q_eval))
    for i, p in tqdm(enumerate(q_eval), total=len(q_eval)):
        for q, m in [(q_star, mass_star), (q_dm, mass_dm)]:
            r2 = np.sum((q - p) ** 2, axis=1)
            v0 = np.min(r2**0.5)
            rhos[i] += (
                3 * m / 4 / np.pi / r0**3 * np.sum(1 / (1 + r2 / r0**2) ** (5 / 2))
            )

    rhos = (rhos * u.M_sun / u.kpc**3).to(u.M_sun / u.pc**3)
    return rhos


def get_true_rho_parallel(q_eval, n_processes=1):
    if n_processes == -1:
        # Choose maximum number of pools
        import os

        n_processes = int(os.environ["OMP_NUM_THREADS"])
    if n_processes > 1:
        from multiprocessing import Pool

        q_subgrids = np.split(
            q_eval,
            np.arange(0, len(q_eval), (len(q_eval) + n_processes - 1) // n_processes)[
                1:
            ],
            axis=0,
        )
        with Pool(n_processes) as p:
            result = p.map(get_true_rho, q_subgrids)
        return np.concatenate(result)
    else:
        return get_true_rho(q_eval)


def plot_2dhist(
    eta_star, eta_dm,
    fig_dir,
    dim1, dim2,
    values=None,
    bins=(128, 128),
    operation="count",
    fig_fmt=("pdf",),
):
    # Plot the total densities of both the stellar and dark matter components

    def get_colormap(values, k=0.2):
        # Return cmap and norm with somewhat smart color scaling
        min_val, max_val = np.nanpercentile(values.reshape(-1), [1, 99])
        w = max_val - min_val
        min_val, max_val = min_val - k * w, max_val + k * w

        kw = {}
        if operation == "count":
            min_val = 0
            kw["cmap"] = "cubehelix"
            kw["norm"] = colors.LogNorm(vmin=10, vmax=max_val)
        elif min_val * max_val < 0:
            kw["cmap"] = "seismic"
            kw["norm"] = colors.TwoSlopeNorm(vcenter=0.0, vmin=min_val, vmax=max_val)
        else:
            kw["cmap"] = "viridis"
            kw["vmin"] = min_val
            kw["vmax"] = max_val
        return kw

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
    ikeys = {k: i for i, k in enumerate(keys)}
    ix, iy = ikeys[dim1], ikeys[dim2]

    fig, all_axs = plt.subplots(
        2, 2,
        figsize=(4, 2.2),
        gridspec_kw=dict(width_ratios=[2, 2], height_ratios=[0.2, 2]),
    )
    caxs, axs = all_axs

    axs[0].set_xlabel(labels[dim1])
    axs[1].set_xlabel(labels[dim1])
    axs[0].set_ylabel(labels[dim2])
    axs[1].set_yticklabels([])

    # Get the plot limits
    lims = []
    k = 0.2
    for x in [eta_star[:, ix], eta_star[:, iy]]:
        xlim = np.percentile(x, [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - k * w, xlim[1] + k * w]
        lims.append(xlim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    x_bins = np.linspace(xmin, xmax, bins[0])
    y_bins = np.linspace(ymin, ymax, bins[1])

    ret = binned_statistic_2d(
        eta_star[:, ix], eta_star[:, iy],
        values,
        statistic=operation,
        bins=[x_bins, y_bins],
    )
    # Choose a suitable colormap
    kw = get_colormap(ret.statistic, k=9)
    im = axs[0].imshow(
        ret.statistic.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        **kw,
        aspect="auto",
    )

    # Add contours
    n_bins_cont = 128
    x_bins_cont = np.linspace(xmin, xmax, n_bins_cont)
    y_bins_cont = np.linspace(ymin, ymax, n_bins_cont)
    X_cont, Y_cont = np.meshgrid(
        0.5 * (x_bins_cont[1:] + x_bins_cont[:-1]),
        0.5 * (y_bins_cont[1:] + y_bins_cont[:-1]),
    )
    hist = np.histogram2d(
        eta_star[:, ix], eta_star[:, iy], bins=[x_bins_cont, y_bins_cont], density=False
    )[0]
    axs[0].contour(
        X_cont, Y_cont,
        gaussian_filter(hist.T, 1.0),
        levels=[60, 100, 300, 1000, 4000],
        linewidths=0.5,
        colors="black",
        linestyles="--",
    )

    cb = fig.colorbar(im, cax=caxs[0], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    # cb.ax.locator_params(nbins=5)
    caxs[0].set_title("stellar histogram")

    ret = binned_statistic_2d(
        eta_dm[:, ix], eta_dm[:, iy], values, statistic=operation, bins=[x_bins, y_bins]
    )
    # Choose a suitable colormap
    im = axs[1].imshow(
        ret.statistic.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        **kw,
        aspect="auto",
    )
    cb = fig.colorbar(im, cax=caxs[1], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    # cb.ax.locator_params(nbins=5)
    caxs[1].set_title("dm histogram")

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"rho_true_{dim1}_{dim2}.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_2d_phi(
    phi_model, coords_ref, fig_dir, dim1, dim2, z_fill=0, attrs=None, fig_fmt=("pdf",)
):
    labels = ["$x$", "$y$", "$z$",]

    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    for dim in [dim1, dim2]:
        if dim not in keys:
            raise ValueError(f"dimension {dim} not supported")

    fig, (all_axs) = plt.subplots(
        2, 3,
        figsize=(6, 2.2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[2, 2, 2], height_ratios=[0.2, 2]),
    )
    axs = all_axs[1, :]
    caxs = all_axs[0, :]

    # Determine the limits
    lims = []
    k = 0.20
    for x in coords_ref[dim1], coords_ref[dim2]:
        xlim = np.percentile(x, [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - k * w, xlim[1] + k * w]
        lims.append(xlim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    # Generate the grid
    grid_size = 32
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    X, Y = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))

    q_grid = np.full(shape=(X.size, 3), fill_value=z_fill, dtype="f4")
    q_grid[:, ikeys[dim1]] = X.ravel()
    q_grid[:, ikeys[dim2]] = Y.ravel()

    # Calculate the true phi (can be timeconsuming)
    fname = os.path.join(fig_dir, f"data/phi_{dim1}_{dim2}_{grid_size}.npy")
    if not os.path.exists(fname):
        phi_true = get_true_phi_parallel(q_grid, -1)
        Path(os.path.join(fig_dir), "data").mkdir(parents=True, exist_ok=True)
        np.save(fname, phi_true.value)
    else:
        phi_true = np.load(fname) * 10000 * u.km**2 / u.s**2
    phi_true = np.reshape(phi_true.to(100**2 * u.km**2 / u.s**2).value, X.shape)

    # Calculate the model phi
    phi, _, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], q_grid, return_phi=True
    )

    phi_img = np.reshape(phi.numpy(), X.shape)

    # Apply a spatial cut, if passed
    if attrs["has_spatial_cut"]:
        # Visualise the boundaries
        plot_flow_projections.add_2dpopulation_boundaries(
            axs, dim1, dim2, attrs, color="black"
        )

        # Mask for the area for which phi is plotted
        r2 = X * X + Y * Y + z_fill**2
        if dim1 == "z":
            actual_z = X
        elif dim2 == "z":
            actual_z = Y
        else:
            actual_z = z_fill
        R2 = r2 - actual_z**2
        mask = ~utils.get_index_of_points_inside_attrs(
            None, attrs, r2**0.5, R2**0.5, actual_z
        )

        phi_img = np.ma.masked_where(mask, phi_img)
        phi_true = np.ma.masked_where(mask, phi_true)

    phi_img = phi_img - np.mean(phi_img)
    phi_true = phi_true - np.mean(phi_true)
    phi_res = (phi_img - phi_true) / np.std(phi_true)

    # Plot true phi
    min_val, max_val = phi_true.min(), phi_true.max()
    divnorm = colors.TwoSlopeNorm(vcenter=0.0)
    kw = dict(cmap="viridis", shading="flat", lw=0, rasterized=True)
    hh = axs[0].pcolormesh(x, y, phi_true, **kw)
    # Set the colorbar
    cb = fig.colorbar(hh, cax=caxs[0], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.locator_params(nbins=3)
    caxs[0].set_title(r"$\Phi_\mathrm{truth}$", fontsize=10)

    # Plot model phi
    min_val, max_val = phi_img.min(), phi_img.max()
    ext_val = max(abs(min_val), max_val)
    # divnorm = colors.TwoSlopeNorm(vcenter=0.)
    # kw = dict(cmap='seismic', vmin=-ext_val, vmax=ext_val, shading='flat', lw=0, rasterized=True)
    kw = dict(cmap="viridis", shading="flat", lw=0, rasterized=True)
    hh = axs[1].pcolormesh(x, y, phi_img, **kw)
    # Set the colorbar
    cb = fig.colorbar(hh, cax=caxs[1], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.locator_params(nbins=3)
    caxs[1].set_title(r"$\Phi_\mathrm{model}$", fontsize=10)

    # Plot residual phi
    min_val, max_val = phi_res.min(), phi_res.max()
    ext_val = max(abs(min_val), max_val)
    ext_val = 0.5
    kw = dict(
        cmap="coolwarm_r",
        vmin=-ext_val, vmax=ext_val,
        shading="flat",
        lw=0,
        rasterized=True,
    )
    hh = axs[2].pcolormesh(x, y, phi_res, **kw)
    # Set the colorbar
    cb = fig.colorbar(hh, cax=caxs[2], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.locator_params(nbins=3)
    caxs[2].set_title(
        r"$(\Phi_\mathrm{model} - \Phi_\mathrm{truth})/\sigma_\mathrm{truth}$",
        fontsize=10,
    )

    for i in range(1, len(axs)):
        axs[i].set_yticklabels([])
    axs[0].set_ylabel(labels[dim2], labelpad=2)

    for ax in axs:
        ax.set_xlabel(labels[dim1], labelpad=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"phi_{dim1}_{dim2}.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_2d_force(phi_model, coords_ref, fig_dir, dim1, dim2, z_fill=0, attrs=None):
    labels = ["$x$", "$y$", "$z$"]

    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    for dim in [dim1, dim2]:
        if dim not in keys:
            raise ValueError(f"dimension {dim} not supported")

    # Determine the limits
    lims = []
    k = 0.20
    for x in coords_ref[dim1], coords_ref[dim2]:
        xlim = np.percentile(x, [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - k * w, xlim[1] + k * w]
        lims.append(xlim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    # Generate the grid
    grid_size = 32
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    X, Y = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))

    q_grid = np.full(shape=(X.size, 3), fill_value=z_fill, dtype="f4")
    q_grid[:, ikeys[dim1]] = X.ravel()
    q_grid[:, ikeys[dim2]] = Y.ravel()

    # Calculate the true force (can be timeconsuming)
    fname = os.path.join(fig_dir, f"data/force_{dim1}_{dim2}_{grid_size}.npy")
    if not os.path.exists(fname):
        forces_true = get_true_force_parallel(q_grid, -1).to(
            100**2 * u.km**2 / u.s**2 / u.kpc
        )
        Path(os.path.join(fig_dir), "data").mkdir(parents=True, exist_ok=True)
        np.save(fname, forces_true.value)
    else:
        forces_true = np.load(fname) * 100**2 * u.km**2 / u.s**2 / u.kpc
    forces_true = forces_true.to(100**2 * u.km**2 / u.s**2 / u.kpc).value

    # Calculate the model force
    _, dphi_dq, _ = potential_tf.calc_phi_derivatives(
        phi_model["phi"], q_grid, return_phi=True
    )

    # Combine all the plots into one pdf. This only supports pdf at the moment. For svg, one probably needs to make one big fig
    import matplotlib.backends.backend_pdf

    fname = os.path.join(fig_dir, f"force_{dim1}_{dim2}.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(fname)

    # Loop over all the forces
    for i in range(3):
        fig, (all_axs) = plt.subplots(
            2, 3,
            figsize=(6, 2.2),
            dpi=200,
            gridspec_kw=dict(width_ratios=[2, 2, 2], height_ratios=[0.2, 2]),
        )
        axs = all_axs[1, :]
        caxs = all_axs[0, :]

        force_img = -np.reshape(dphi_dq[:, i].numpy(), X.shape)
        force_true = np.reshape(forces_true[:, i], X.shape)

        if attrs["has_spatial_cut"]:
            # Visualise the boundaries
            plot_flow_projections.add_2dpopulation_boundaries(
                axs, dim1, dim2, attrs, color="black"
            )

            # Mask for the area for which forces are plotted
            r2 = X * X + Y * Y + z_fill**2
            if dim1 == "z":
                actual_z = X
            elif dim2 == "z":
                actual_z = Y
            else:
                actual_z = z_fill
            R2 = r2 - actual_z**2
            mask = ~utils.get_index_of_points_inside_attrs(
                None, attrs, r2**0.5, R2**0.5, actual_z
            )

            force_img = np.ma.masked_where(mask, force_img)
            force_true = np.ma.masked_where(mask, force_true)

        force_res = (force_img - force_true) / force_true

        eps = 1e-9
        # Plot true force
        min_val, max_val = force_img.min(), force_img.max()
        min_val, max_val = force_true.min(), force_true.max()
        ext_val = max(abs(min_val), abs(max_val)) + eps
        divnorm = colors.TwoSlopeNorm(vmin=-ext_val, vcenter=0, vmax=ext_val)
        kw = dict(cmap="seismic", shading="flat", norm=divnorm, lw=0, rasterized=True)
        hh = axs[0].pcolormesh(x, y, force_true, **kw)
        # Set the colorbar
        cb = fig.colorbar(hh, cax=caxs[0], orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.locator_params(nbins=3)
        caxs[0].set_title(f"$F_{{{keys[i]},\mathrm{{truth}}}}$", fontsize=10)

        # Plot model force
        min_val, max_val = force_img.min(), force_img.max()
        ext_val = max(abs(min_val), abs(max_val)) + eps
        divnorm = colors.TwoSlopeNorm(vmin=-ext_val, vcenter=0, vmax=ext_val)
        # kw = dict(cmap='seismic', vmin=-ext_val, vmax=ext_val, shading='flat', lw=0, rasterized=True)
        kw = dict(cmap="seismic", shading="flat", norm=divnorm, lw=0, rasterized=True)
        hh = axs[1].pcolormesh(x, y, force_img, **kw)
        # Set the colorbar
        cb = fig.colorbar(hh, cax=caxs[1], orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.locator_params(nbins=3)
        caxs[1].set_title(f"$F_{{{keys[i]},\mathrm{{model}}}}$", fontsize=10)

        # Plot residual force
        min_val, max_val = force_res.min(), force_res.max()
        ext_val = max(abs(min_val), abs(max_val)) + eps
        ext_val = 1
        kw = dict(
            cmap="coolwarm_r",
            vmin=-ext_val, vmax=ext_val,
            shading="flat",
            lw=0,
            rasterized=True,
        )
        hh = axs[2].pcolormesh(x, y, force_res, **kw)
        # Set the colorbar
        cb = fig.colorbar(hh, cax=caxs[2], orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.locator_params(nbins=3)
        caxs[2].set_title(
            f"$(F_{{{keys[i]},\mathrm{{model}}}}-F_{{{keys[i]},\mathrm{{truth}}}})/F_{{{keys[i]},\mathrm{{truth}}}}$",
            fontsize=10,
        )

        for i in range(1, len(axs)):
            axs[i].set_yticklabels([])
        axs[0].set_ylabel(labels[dim2], labelpad=2)

        for ax in axs:
            ax.set_xlabel(labels[dim1], labelpad=0)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    pdf.close()


def plot_2d_rho(
    phi_model,
    coords_ref,
    fig_dir,
    dim1, dim2,
    z_fill=0,
    attrs=None,
    fig_fmt=("pdf",),
    logscale=True,
):
    labels = ["$x$", "$y$", "$z$"]

    keys = ["x", "y", "z"]

    labels = {k: l for k, l in zip(keys, labels)}
    ikeys = {k: i for i, k in enumerate(keys)}

    for dim in [dim1, dim2]:
        if dim not in keys:
            raise ValueError(f"dimension {dim} not supported")

    fig, (all_axs) = plt.subplots(
        2, 3,
        figsize=(6, 2.2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[2, 2, 2], height_ratios=[0.2, 2]),
    )
    axs = all_axs[1, :]
    caxs = all_axs[0, :]

    # Determine the limits
    lims = []
    k = 0.20
    for x in coords_ref[dim1], coords_ref[dim2]:
        xlim = np.percentile(x, [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - k * w, xlim[1] + k * w]
        lims.append(xlim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    # Generate the grid
    grid_size = 64
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    X, Y = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))

    q_grid = np.full(shape=(X.size, 3), fill_value=z_fill, dtype="f4")
    q_grid[:, ikeys[dim1]] = X.ravel()
    q_grid[:, ikeys[dim2]] = Y.ravel()

    # Calculate the true rho (can be timeconsuming)
    fname = os.path.join(fig_dir, f"data/rho_{dim1}_{dim2}_{grid_size}.npy")
    if not os.path.exists(fname):
        rho_true = get_true_rho_parallel(q_grid, -1)
        Path(os.path.join(fig_dir), "data").mkdir(parents=True, exist_ok=True)
        np.save(fname, rho_true.value)
    else:
        rho_true = np.load(fname) * u.M_sun / u.pc**3
    rho_true = np.reshape(rho_true.to(u.M_sun / u.pc**3).value, X.shape)

    # Calculate the model rho
    phi, _, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], q_grid, return_phi=True
    )

    rho_img = np.reshape(
        2.325 * d2phi_dq2.numpy() / (4 * np.pi), X.shape
    )  # [M_Sun/pc^3]

    # Apply a spatial cut, if passed
    if attrs["has_spatial_cut"]:
        # Visualise the boundaries
        plot_flow_projections.add_2dpopulation_boundaries(
            axs, dim1, dim2, attrs, color="black"
        )

        # Mask for the area for which phi is plotted
        r2 = X * X + Y * Y + z_fill**2
        if dim1 == "z":
            actual_z = X
        elif dim2 == "z":
            actual_z = Y
        else:
            actual_z = z_fill
        R2 = r2 - actual_z**2
        mask = ~utils.get_index_of_points_inside_attrs(
            None, attrs, r2**0.5, R2**0.5, actual_z
        )

        rho_img = np.ma.masked_where(mask, rho_img)
        rho_true = np.ma.masked_where(mask, rho_true)

    rho_res = (rho_img - rho_true) / rho_true

    # Plot true rho
    min_val, max_val = rho_true.min(), rho_true.max()
    divnorm = colors.TwoSlopeNorm(vcenter=0.0)
    if logscale:
        kw = dict(cmap="cubehelix", norm=colors.LogNorm(), rasterized=True)
    else:
        kw = dict(cmap="viridis", vmin=0, rasterized=True)
    hh = axs[0].pcolormesh(x, y, rho_true, **kw)
    # Set the colorbar
    cb = fig.colorbar(hh, cax=caxs[0], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    caxs[0].set_title(r"$\rho_\mathrm{truth}$", fontsize=10)

    # Plot model rho
    min_val, max_val = rho_img.min(), rho_img.max()
    ext_val = max(abs(min_val), max_val)
    # divnorm = colors.TwoSlopeNorm(vcenter=0.)
    # kw = dict(cmap='seismic', vmin=-ext_val, vmax=ext_val, shading='flat', lw=0, rasterized=True)
    hh = axs[1].pcolormesh(x, y, rho_img, **kw)
    # Set the colorbar
    cb = fig.colorbar(hh, cax=caxs[1], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    caxs[1].set_title(r"$\rho_\mathrm{model}$", fontsize=10)

    # Plot residual phi
    min_val, max_val = rho_res.min(), rho_res.max()
    ext_val = max(abs(min_val), max_val)
    ext_val = 1.0
    kw = dict(
        cmap="coolwarm_r",
        vmin=-ext_val, vmax=ext_val,
        shading="flat",
        lw=0,
        rasterized=True,
    )
    hh = axs[2].pcolormesh(x, y, rho_res, **kw)
    # Set the colorbar
    cb = fig.colorbar(hh, cax=caxs[2], orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.locator_params(nbins=3)
    caxs[2].set_title(
        r"$(\rho_\mathrm{model} - \rho_\mathrm{truth})/\sigma_\mathrm{truth}$",
        fontsize=10,
    )

    for i in range(1, len(axs)):
        axs[i].set_yticklabels([])
    axs[0].set_ylabel(labels[dim2], labelpad=2)

    for ax in axs:
        ax.set_xlabel(labels[dim1], labelpad=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"rho_{dim1}_{dim2}.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_force_residuals(phi_model, eta_eval, fig_dir, attrs=None):
    import labellines

    xlabels = [
        r"$\mathrm{d}\Phi/\mathrm{d}R$",
        r"$\mathrm{d}\Phi/\mathrm{d}\phi$",
        r"$\mathrm{d}\Phi/\mathrm{d}z$",
    ]

    ylabels = [
        r"$\mathrm{d}\Phi^*/\mathrm{d}R-\mathrm{d}\Phi/\mathrm{d}R$",
        r"$\mathrm{d}\Phi^*/\mathrm{d}\phi-\mathrm{d}\Phi/\mathrm{d}\phi$",
        r"$\mathrm{d}\Phi^*/\mathrm{d}z-\mathrm{d}\Phi/\mathrm{d}z$",
    ]

    # Calculate the model potential values
    phi, dphi_dq, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], eta_eval[:, :3].astype("f4"), return_phi=True
    )
    force_model = -dphi_dq.numpy()

    # Calculate the true potential forces (can be timeconsuming)
    fname = os.path.join(fig_dir, f"data/forces_res_{len(eta_eval)}.npy")
    if not os.path.exists(fname):
        force_true = (
            get_true_force_parallel(eta_eval[:, :3], -1)
            .to(100**2 * u.km**2 / u.s**2 / u.kpc)
            .value
        )
        Path(os.path.join(fig_dir), "data").mkdir(parents=True, exist_ok=True)
        np.save(fname, force_true)
    else:
        force_true = np.load(fname)

    # Extract the radial component of the gradient
    n_R, n_phi, n_z = (
        np.zeros_like(eta_eval[:, :3]),
        np.zeros_like(eta_eval[:, :3]),
        np.zeros_like(eta_eval[:, :3]),
    )
    n_R[:, 0] = eta_eval[:, 0]
    n_R[:, 1] = eta_eval[:, 1]
    n_R[:, :2] /= (np.sum(n_R[:, :2] ** 2, axis=1) ** 0.5).reshape(-1, 1)

    n_phi[:, 0] = -eta_eval[:, 1]
    n_phi[:, 1] = eta_eval[:, 0]
    n_phi[:, :2] /= (np.sum(n_phi[:, :2] ** 2, axis=1) ** 0.5).reshape(-1, 1)

    n_z[:, 0] = 0
    n_z[:, 1] = 0
    n_z[:, 2] = 1

    dphi_dq_model = -np.concatenate(
        [
            np.reshape(np.sum(force_model * n_R, axis=1), (-1, 1)),
            np.reshape(np.sum(force_model * n_phi, axis=1), (-1, 1)),
            np.reshape(np.sum(force_model * n_z, axis=1), (-1, 1)),
        ],
        axis=1,
    )

    dphi_dq_true = -np.concatenate(
        [
            np.reshape(np.sum(force_true * n_R, axis=1), (-1, 1)),
            np.reshape(np.sum(force_true * n_phi, axis=1), (-1, 1)),
            np.reshape(np.sum(force_true * n_z, axis=1), (-1, 1)),
        ],
        axis=1,
    )

    # Combine all the plots into one pdf. This only supports pdf at the moment. For svg, one probably needs to make one big fig
    import matplotlib.backends.backend_pdf

    fname = os.path.join(fig_dir, "force_residuals.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(fname)

    for i in range(3):
        fig, ax = plt.subplots(
            1, 1,
            figsize=(5, 3),
            dpi=200,
        )

        dphi_dR_res = dphi_dq_model[:, i] - dphi_dq_true[:, i]

        xlim = np.percentile(dphi_dq_true[:, i], [1.0, 99.0])
        ylim = np.percentile(dphi_dR_res, [1.0, 99.0])
        k = 0.5
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - k * w, xlim[1] + k * w]
        w = ylim[1] - ylim[0]
        ylim = [ylim[0] - k * w, ylim[1] + k * w]
        # xlim = (-20, 0.)
        # ylim = (-10, 10)
        bins_x = np.linspace(*xlim, 128)
        bins_y = np.linspace(*ylim, 128)

        ax.hist2d(
            dphi_dq_true[:, i],
            dphi_dR_res,
            bins=(bins_x, bins_y),
            cmap=plt.cm.Greys,
            norm=mpl.colors.LogNorm(),
            rasterized=True,
        )
        # ax.plot(xlim, ylim, label='100\%', color='grey', ls='--')

        slopes = []
        for val in [0, 0.10, 0.20, 1.0]:
            slopes += [-val, val]
        slopes = np.array(slopes[1:])

        for slope in slopes:
            ax.plot(
                [-1000, 1000],
                [-1000 * slope, 1000 * slope],
                color="grey",
                ls="--",
                lw=0.5,
                label=f"{100*slope:.0f}\%",
            )

        x0 = xlim[1] - 0.1 * (xlim[1] - xlim[0])
        y0, y1 = ylim[0] + 0.1 * (ylim[1] - ylim[0]), ylim[1] - 0.1 * (
            ylim[1] - ylim[0]
        )
        x = np.array([x0] * len(slopes))
        x[slopes * x0 > y1] = y1 / slopes[slopes * x0 > y1]
        x[slopes * x0 < y0] = y0 / slopes[slopes * x0 < y0]
        labellines.labelLines(ax.get_lines(), fontsize=8, xvals=x)

        # ax.set_xlim(left=0, right=xlim[1])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])

        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    pdf.close()


def plot_phi_residuals(phi_model, eta_eval, fig_dir, attrs=None, fig_fmt=("pdf",)):
    import labellines

    # Calculate the model potential values
    phi, dphi_dq, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], eta_eval[:, :3].astype("f4"), return_phi=True
    )
    # There is an annoying naming conflict with phi_model... sticking with phi for now
    phi = phi.numpy()[:, 0]

    # Calculate the true potential forces (can be timeconsuming)
    fname = os.path.join(fig_dir, f"data/phi_res_{len(eta_eval)}.npy")
    if not os.path.exists(fname):
        phi_true = get_true_phi_parallel(eta_eval[:, :3], -1)
        Path(os.path.join(fig_dir), "data").mkdir(parents=True, exist_ok=True)
        np.save(fname, phi_true.value)
    else:
        phi_true = np.load(fname) * 10000 * u.km**2 / u.s**2
    phi_true = phi_true.to(100**2 * u.km**2 / u.s**2).value
    phi_true -= np.mean(phi_true)

    idx = utils.get_index_of_points_inside_attrs(eta_eval, attrs)
    print(np.sum(idx), len(eta_eval))

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(5, 3),
        dpi=200,
    )

    phi += np.mean(phi_true) - np.mean(phi)
    phi_res = phi - phi_true

    xlim = np.percentile(phi_true, [1.0, 99.0])
    ylim = np.percentile(phi_res, [1.0, 99.0])
    k = 0.5
    w = xlim[1] - xlim[0]
    xlim = [xlim[0] - k * w, xlim[1] + k * w]
    w = ylim[1] - ylim[0]
    ylim = [ylim[0] - k * w, ylim[1] + k * w]
    # xlim = (-20, 0.)
    # ylim = (-10, 10)
    bins_x = np.linspace(*xlim, 128)
    bins_y = np.linspace(*ylim, 128)

    ax.hist2d(
        phi_true, phi_res,
        bins=(bins_x, bins_y),
        cmap=plt.cm.Greys,
        norm=mpl.colors.LogNorm(),
    )
    # ax.plot(xlim, ylim, label='100\%', color='grey', ls='--')

    slopes = []
    for val in [0, 0.10, 0.20, 1.0]:
        slopes += [-val, val]
    slopes = np.array(slopes[1:])

    for slope in slopes:
        ax.plot(
            [-1000, 1000],
            [-1000 * slope, 1000 * slope],
            color="grey",
            ls="--",
            lw=0.5,
            label=f"{100*slope:.0f}\%",
        )

    x0 = xlim[1] - 0.1 * (xlim[1] - xlim[0])
    y0, y1 = ylim[0] + 0.1 * (ylim[1] - ylim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0])
    x = np.array([x0] * len(slopes))
    x[slopes * x0 > y1] = y1 / slopes[slopes * x0 > y1]
    x[slopes * x0 < y0] = y0 / slopes[slopes * x0 < y0]
    labellines.labelLines(ax.get_lines(), fontsize=8, xvals=x)

    # ax.set_xlim(left=0, right=xlim[1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\Phi$")
    ax.set_ylabel(r"$\Phi^*-\Phi$")

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"phi_residuals.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rho_residuals(phi_model, eta_eval, fig_dir, attrs=None, fig_fmt=("pdf",)):
    import labellines

    # Calculate the model potential values
    phi, dphi_dq, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], eta_eval[:, :3].astype("f4"), return_phi=True
    )
    rho_model = 2.325 * d2phi_dq2.numpy() / (4 * np.pi)  # [M_Sun/pc^3]

    # Calculate the true density (can be timeconsuming)
    fname = os.path.join(fig_dir, f"data/rho_res_{len(eta_eval)}.npy")
    if not os.path.exists(fname):
        rho_true = get_true_rho_parallel(eta_eval[:, :3], -1)
        Path(os.path.join(fig_dir), "data").mkdir(parents=True, exist_ok=True)
        np.save(fname, rho_true.value)
    else:
        rho_true = np.load(fname) * u.M_sun / u.pc**3
    rho_true = rho_true.to(u.M_sun / u.pc**3).value

    idx = utils.get_index_of_points_inside_attrs(eta_eval, attrs)
    print(np.sum(idx), len(eta_eval))

    fig, ax = plt.subplots(
        1, 1,
        figsize=(5, 3),
        dpi=200,
    )

    rho_res = rho_model - rho_true

    xlim = np.percentile(rho_true, [1.0, 99.0])
    ylim = np.percentile(rho_res, [1.0, 99.0])
    k = 0.5
    w = xlim[1] - xlim[0]
    xlim = [xlim[0] - k * w, xlim[1] + k * w]
    w = ylim[1] - ylim[0]
    ylim = [ylim[0] - k * w, ylim[1] + k * w]
    # xlim = (-20, 0.)
    # ylim = (-10, 10)
    bins_x = np.linspace(*xlim, 128)
    bins_y = np.linspace(*ylim, 128)

    ax.hist2d(
        rho_true, rho_res,
        bins=(bins_x, bins_y),
        cmap=plt.cm.Greys,
        norm=mpl.colors.LogNorm(),
    )
    # ax.plot(xlim, ylim, label='100\%', color='grey', ls='--')

    slopes = []
    for val in [0, 0.10, 0.20, 1.0]:
        slopes += [-val, val]
    slopes = np.array(slopes[1:])

    for slope in slopes:
        ax.plot(
            [-1000, 1000],
            [-1000 * slope, 1000 * slope],
            color="grey",
            ls="--",
            lw=0.5,
            label=f"{100*slope:.0f}\%",
        )

    x0 = xlim[1] - 0.1 * (xlim[1] - xlim[0])
    y0, y1 = ylim[0] + 0.1 * (ylim[1] - ylim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0])
    x = np.array([x0] * len(slopes))
    x[slopes * x0 > y1] = y1 / slopes[slopes * x0 > y1]
    x[slopes * x0 < y0] = y0 / slopes[slopes * x0 < y0]
    labellines.labelLines(ax.get_lines(), fontsize=8, xvals=x)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\rho^*-\rho$")

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"rho_residuals.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_to_fits(df_data, fig_dir, spherical_origin, cylindrical_origin):
    import astropy.table as table
    from astropy.table import Table

    # Get the coords
    n = 200000
    eta_eval = df_data["eta"][:n]

    coords = plot_flow_projections.calc_coords(
        eta_eval, spherical_origin, cylindrical_origin
    )

    # Calculate true gradients
    fname = os.path.join(fig_dir, f"data/forces_res_{len(eta_eval)}.npy")
    if not os.path.exists(fname):
        force_true = (
            get_true_force_parallel(eta_eval[:, :3], -1)
            .to(100**2 * u.km**2 / u.s**2 / u.kpc)
            .value
        )
        Path(os.path.join(fig_dir), "data").mkdir(parents=True, exist_ok=True)
        np.save(fname, force_true)
    else:
        force_true = np.load(fname)

    # Calculate model gradients
    phi, dphi_dq, d2phi_dq2 = potential_tf.calc_phi_derivatives(
        phi_model["phi"], eta_eval[:, :3].astype("f4"), return_phi=True
    )
    force_model = -dphi_dq.numpy()

    coords["Ftrue_x"] = force_true[:, 0]
    coords["Ftrue_y"] = force_true[:, 1]
    coords["Ftrue_z"] = force_true[:, 2]
    coords["Fmodel_x"] = force_model[:, 0]
    coords["Fmodel_y"] = force_model[:, 1]
    coords["Fmodel_z"] = force_model[:, 2]

    n_R, n_phi, n_z = (
        np.zeros_like(eta_eval[:, :3]),
        np.zeros_like(eta_eval[:, :3]),
        np.zeros_like(eta_eval[:, :3]),
    )
    n_R[:, 0] = eta_eval[:, 0]
    n_R[:, 1] = eta_eval[:, 1]
    n_R[:, :2] /= (np.sum(n_R[:, :2] ** 2, axis=1) ** 0.5).reshape(-1, 1)

    n_phi[:, 0] = -eta_eval[:, 1]
    n_phi[:, 1] = eta_eval[:, 0]
    n_phi[:, :2] /= (np.sum(n_phi[:, :2] ** 2, axis=1) ** 0.5).reshape(-1, 1)

    n_z[:, 0] = 0
    n_z[:, 1] = 0
    n_z[:, 2] = 1

    force_model = np.concatenate(
        [
            np.reshape(np.sum(force_model * n_R, axis=1), (-1, 1)),
            np.reshape(np.sum(force_model * n_phi, axis=1), (-1, 1)),
            np.reshape(np.sum(force_model * n_z, axis=1), (-1, 1)),
        ],
        axis=1,
    )
    force_true = np.concatenate(
        [
            np.reshape(np.sum(force_true * n_R, axis=1), (-1, 1)),
            np.reshape(np.sum(force_true * n_phi, axis=1), (-1, 1)),
            np.reshape(np.sum(force_true * n_z, axis=1), (-1, 1)),
        ],
        axis=1,
    )
    coords["Ftrue_R"] = force_true[:, 0]
    coords["Ftrue_phi"] = force_true[:, 1]
    coords["Fmodel_R"] = force_model[:, 0]
    coords["Fmodel_phi"] = force_model[:, 1]

    t = Table(coords)

    fname_df_data = os.path.join(fig_dir, "data/data_true_coords.fits")
    t.write(fname_df_data, format="fits", overwrite=True)


if __name__ == "__main__":
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
        "--timestep",
        type=str,
        required=True,
        help="The timestep of the barmodel particles to load in. Is either t80, t170 or t360.",
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
        default=(0.0, 0.0, 0.0),
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

    import multiprocessing

    print(multiprocessing.cpu_count())
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
            fig_dir = (
                "plots/barmodel_" + fname_pot[fname_pot.find("models/") + 7:] + "/"
            )
        else:
            fname_loss_pdf = fname_pot[:-6] + "_loss.pdf"
            fig_dir = (
                "plots/barmodel_"
                + fname_pot[fname_pot.find("models/") + 7: fname_pot.rfind(".index")]
                + "/"
            )

        print(fname_loss_pdf, os.path.isfile(fname_loss_pdf))
        if os.path.isfile(fname_loss_pdf):
            # Copy the latest loss over to the plots dir
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(fname_loss_pdf, fig_dir)
            shutil.copy(fname_loss_pdf[:-4] + "_noreg.pdf", fig_dir)

        args.fig_dir = fig_dir

    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    print("Loading training data ...")
    data_train, attrs_train = utils.load_training_data(args.input, cut_attrs=True)
    eta_train = data_train["eta"]

    print("Loading potential")
    phi_model = utils.load_potential(args.potential)
    if phi_model["fs"] is not None:
        phi_model["fs"].debug()

    df_data = utils.load_flow_samples(args.df_grads_fname)
    coords_ref = plot_flow_projections.calc_coords(
        df_data["eta"], args.spherical_origin, args.cylindrical_origin
    )

    print("Loading in the locations of all test particles...")
    eta_star = np.load(f"../barmodel/preprocessing/barmodel_stars_{args.timestep}.npy")
    eta_dm = np.load(f"../barmodel/preprocessing/barmodel_dm_{args.timestep}.npy")
    q_star = eta_star[:, :3]
    q_dm = eta_dm[:, :3]

    coords_star = plot_flow_projections.calc_coords(
        eta_star, args.spherical_origin, args.cylindrical_origin
    )
    coords_dm = plot_flow_projections.calc_coords(
        eta_dm, args.spherical_origin, args.cylindrical_origin
    )

    # print('Generating a fits dataset for TOPCAT data exploration ...')
    # save_to_fits(df_data, args.fig_dir, args.spherical_origin, args.cylindrical_origin)

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

    print("Plotting model and true densities ...")
    dims = [
        ("x", "y", 0.0),
        ("x", "z", 0.0),
        ("y", "z", 0.0),
    ]
    for dim1, dim2, z_fill in dims:
        print(f"  --> ({dim1}, {dim2})")
        plot_2d_rho(
            phi_model,
            coords_ref,
            args.fig_dir,
            dim1,
            dim2,
            z_fill=z_fill,
            attrs=attrs_train,
            fig_fmt=args.fig_fmt,
        )

    print("Plotting model and true potential ...")
    dims = [("x", "y"), ("x", "z"), ("y", "z")]
    for dim1, dim2 in dims:
        print(f"  --> ({dim1}, {dim2})")
        plot_2d_phi(
            phi_model,
            coords_ref,
            args.fig_dir,
            dim1,
            dim2,
            attrs=attrs_train,
            fig_fmt=args.fig_fmt,
        )

    print("Plotting model and true forces ...")
    for dim1, dim2 in dims:
        print(f"  --> ({dim1}, {dim2})")
        plot_2d_force(
            phi_model, coords_ref, args.fig_dir, dim1, dim2, attrs=attrs_train
        )

    # Plot the overall stellar and dark matter distributions
    print("Plotting overall stellar and dark matter distributions ...")
    dims = [
        ("x", "y"),
        ("x", "z"),
        ("y", "z"),
        ("vx", "vy"),
        ("vx", "vz"),
        ("vy", "vz"),
    ]
    for dim1, dim2 in dims:
        print(f"  --> ({dim1}, {dim2})")
        plot_2dhist(eta_star, eta_dm, args.fig_dir, dim1, dim2, fig_fmt=args.fig_fmt)

    print("Plotting residuals between the model and the ground truth ...")
    eta_eval = df_data["eta"][:30000]
    plot_force_residuals(phi_model, eta_eval, args.fig_dir, attrs=attrs_train)
    plot_phi_residuals(
        phi_model, eta_eval, args.fig_dir, attrs=attrs_train, fig_fmt=args.fig_fmt
    )
    plot_rho_residuals(
        phi_model, eta_eval, args.fig_dir, attrs=attrs_train, fig_fmt=args.fig_fmt
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

    # Limit df_data to 1 million particles, because otherwise this gives OOM
    df_data["eta"] = df_data["eta"][:1048576, :]
    df_data["df_deta"] = df_data["df_deta"][:1048576, :]
    fname = os.path.join(args.fig_dir, "data/dphi_dq.npy")
    if not os.path.exists(fname):
        print("  Calculating Phi gradients (might take a while) ...")
        _, dphi_dq, _ = potential_tf.calc_phi_derivatives(
            phi_model["phi"], df_data["eta"][:, :3], return_phi=True
        )
        Path(os.path.join(args.fig_dir), "data").mkdir(parents=True, exist_ok=True)
        np.save(fname, dphi_dq.numpy())
    else:
        print("  Loading Phi gradients from disk ...")
        dphi_dq = np.load(fname)
        dphi_dq = tf.convert_to_tensor(dphi_dq, np.float32)

    print(
        "Plotting dfdt using least squares method on the flow, and using the potential model ..."
    )
    for dim1, dim2 in dims:
        print(f"  --> ({dim1}, {dim2})")
        plot_potential.plot_dfdt_comparison(
            phi_model,
            df_data,
            dphi_dq,
            args.fig_dir,
            dim1,
            dim2,
            attrs=attrs_train,
            fig_fmt=args.fig_fmt,
            grid_size=256,
        )

    print("Plotting ideal dfdt implied by the true potential and its gradients ...")
    dims = [
        ("x", "y"),
        ("x", "z"),
        ("y", "z"),
        ("vx", "vy"),
        ("vx", "vz"),
        ("vy", "vz"),
    ]
    n = 200000
    df_data_eval = {"df_deta": df_data["df_deta"][:n, :], "eta": df_data["eta"][:n, :]}

    fname = os.path.join(
        args.fig_dir, f'data/forces_res_{len(df_data_eval["eta"])}.npy'
    )
    if not os.path.exists(fname):
        force_true = (
            get_true_force_parallel(df_data_eval["eta"][:, :3], -1)
            .to(100**2 * u.km**2 / u.s**2 / u.kpc)
            .value
        )
        Path(os.path.join(args.fig_dir), "data").mkdir(parents=True, exist_ok=True)
        np.save(fname, force_true)
    else:
        force_true = np.load(fname)

    dphi_dq = -force_true

    for dim1, dim2 in dims:
        print(f"  --> ({dim1}, {dim2})")
        fig, axs = plot_potential.plot_dfdt_comparison(
            phi_model,
            df_data_eval,
            dphi_dq,
            "",
            dim1,
            dim2,
            attrs=attrs_train,
            fig_fmt="",
            save=False,
            grid_size=256,
        )

        for fmt in args.fig_fmt:
            fname = os.path.join(fig_dir, f"phi_dfdt_truth_{dim1}_{dim2}.{fmt}")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
