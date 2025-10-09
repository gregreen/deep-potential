#!/usr/bin/env python

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
import cmasher as cmr

import h5py
import progressbar
import os
from glob import glob
from pathlib import Path
import shutil
from scipy.stats import binned_statistic, binned_statistic_2d

import astropy.units as u

import utils

dpi = 200


def get_lims(values, is_spatial=True, p=0.05, k=0.2, make_symmetric=False):
    """
    Gets the limits of values. Typically when the values represent spatial coordinates,
    the limits are sharp, i.e. defined my the min and max of values. When the values
    represent velocity coordinates, then due to the tails of the distribution, the limits
    are determined based on percentiles instead.
    """
    if is_spatial:
        lim = np.min(values), np.max(values)
    else:
        lim = np.nanpercentile(values, [p * 100, (1 - p) * 100])
        w = lim[1] - lim[0]
        lim = lim[0] - k * w, lim[1] + k * w
    if make_symmetric:
        w = np.max(np.abs(lim))
        lim = -w, w
    return lim

def get_labels_and_keys(is_gaia):
    """
    Returns dictionaries for labels and keys, with or without units based on the is_gaia flag.
    """
    if is_gaia:
        labels_list = [
            '$R\mathrm{\ (kpc)}$', '$z\mathrm{\ (kpc)}$', r'$\phi\mathrm{\ (rad)}$', '$v_R\mathrm{\ (km/s)}$', '$v_z\mathrm{\ (km/s)}$', r'$-v_{\phi}\mathrm{\ (km/s)}$',
            '$x\mathrm{\ (kpc)}$', '$y\mathrm{\ (kpc)}$', '$z\mathrm{\ (kpc)}$', '$v_x\mathrm{\ (km/s)}$', '$v_y\mathrm{\ (km/s)}$', '$v_z\mathrm{\ (km/s)}$',
            '$r\mathrm{\ (kpc)}$', r'$\ell\mathrm{\ (rad)}$', r'$\cos b$', '$v_r\mathrm{\ (km/s)}$', r'$v_{\theta}\mathrm{\ (km/s)}$', r'$-v_{\phi}\mathrm{\ (km/s)}$'
        ]
    else:
        labels_list = [
            '$R$', '$z$', r'$\phi$', '$v_R$', '$v_z$', r'$-v_{\phi}$',
            '$x$', '$y$', '$z$', '$v_x$', '$v_y$', '$v_z$',
            '$r$', r'$\ell$', r'$\cos b$', '$v_r$', r'$v_{\theta}$', r'$-v_{\phi}$'
        ]
    keys = [
        'cylR', 'cylz', 'cylphi', 'cylvR', 'cylvz', 'cylvT',
        'x', 'y', 'z', 'vx', 'vy', 'vz',
        'r', 'phi', 'cth', 'vr', 'vth', 'vT'
    ]
    labels = {k: l for k, l in zip(keys, labels_list)}
    ikeys = {k: i for i, k in enumerate(keys)}
    return labels, ikeys, keys


def plot_2d_slice_pot(phi_model, coords_ref, dim1, dim2, dim_plot, fig_dir, z_fill=0, attrs=None, fig_fmt=('png',), logscale=True, lims=None, fname_mask=None, is_gaia=True):
    """
    Currently only supports cartesian.
    """
    labels, ikeys, keys = get_labels_and_keys(is_gaia)

    if is_gaia:
        titles_list = [
            r'$\rho_\mathrm{model}\mathrm{\ (M_\odot/pc^3)}$',
            r'$\Phi_\mathrm{model}\mathrm{\ (km^2/s^2)}$',
            r'$a_x^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$',
            r'$a_y^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$',
            r'$a_z^*\mathrm{\ (10^4 km^2/(kpc\cdot s^2))}$',
        ]
    else:
        titles_list = [
            r'$\rho_\mathrm{model}$',
            r'$\Phi_\mathrm{model}$',
            r'$a_x^*$',
            r'$a_y^*$',
            r'$a_z^*$',
        ]
    plot_keys = ['rho', 'phi', 'ax', 'ay', 'az']
    titles = {k: t for k, t in zip(plot_keys, titles_list)}
    iplot_keys = {k: i for i, k in enumerate(plot_keys)}

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
    q_grid[:, ikeys[dim1]] = X.ravel()
    q_grid[:, ikeys[dim2]] = Y.ravel()
    if fname_mask is not None:
        mask = ~utils.get_mask_eta(q_grid, fname_mask, r_min=attrs['r_in'])[0]
    else:
        r_grid = np.sum(q_grid**2, axis=1)**0.5
        mask = ~((r_grid < attrs['r_out']) & (r_grid > attrs['r_in']))

    # Calculate the model rho
    phi, acc, rho = utils.get_model_values(phi_model, q_grid, disable_tqdm=True, convert_rho_to_msunpc3=is_gaia)
    phi = phi.flatten()
    if dim_plot == 'phi':
        values = phi
    elif dim_plot == 'rho':
        values = rho
    elif dim_plot in ['ax', 'ay', 'az']:
        values = acc[:, iplot_keys[dim_plot] - 2]

    values = np.ma.masked_where(mask, values)
    values = np.reshape(values, X.shape)

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


def plot_2d_slices_rho(phi_model, coords_grid, dim1, dim2, attrs, fig_dir=None, fig_fmt=('png',), fname_mask=None, is_gaia=True):
    labels, _, keys = get_labels_and_keys(is_gaia)
    
    # dims must be from x,y,z because that simplifies handling the 3rd dimension
    if dim1 not in ['x', 'y', 'z'] or dim2 not in ['x', 'y', 'z']:
        raise ValueError('dim1 and dim2 must be from x,y,z')
    dim3 = [d for d in ['x', 'y', 'z'] if d not in [dim1, dim2]][0]

    fig, all_axs = plt.subplots(
        2, 3,
        figsize=(7, 3.3),
        dpi=140,
        width_ratios=[1, 1, 1],
        height_ratios=[0.1, 2],
        layout='compressed'
    )
    caxs = all_axs[0,:]
    axs = all_axs[1,:]

    lims = [get_lims(coords_grid[dim1]), get_lims(coords_grid[dim2])]
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    # Generate the grid
    grid_size = 128
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    X, Y = np.meshgrid(0.5*(x[1:]+x[:-1]), 0.5*(y[1:]+y[:-1]))

    kw = dict(cmap='cmr.rainforest', vmin=0, rasterized=True)
    if is_gaia:
        kw['vmax'] = 0.15
        
    # The z values are chosen at 37.5%, 50% and 62.5% of the 3rd data dimension
    z_lims = get_lims(coords_grid[dim3])
    w = z_lims[1] - z_lims[0]
    z_values = [z_lims[0] + 0.375*w, z_lims[0] + 0.5*w, z_lims[0] + 0.625*w]
    z_values = [round(z, 3) for z in z_values]

    for i, z_fill in enumerate(z_values):
        q_grid = np.full(shape=(X.size, 3), fill_value=z_fill, dtype='f4')
        q_grid[:,['x', 'y', 'z'].index(dim1)] = X.ravel()
        q_grid[:,['x', 'y', 'z'].index(dim2)] = Y.ravel()
        mask = ~utils.get_mask_eta(q_grid, fname_mask, r_min=attrs['r_in'], r_max=attrs['r_out'])[0]

        # Calculate the model rho
        rho = utils.get_model_values(phi_model, q_grid, disable_tqdm=True, convert_rho_to_msunpc3=is_gaia)[2]
        rho = np.ma.masked_where(mask, rho)
        rho = np.reshape(rho, X.shape)

        # Plot model rho
        ax, cax = axs[i], caxs[i]
        hh = ax.pcolormesh(x, y, rho, **kw)
        # Add small text saying the z value
        z_label = keys[keys.index(dim3)]
        text = f'${z_label}={z_fill:.2f}$'
        if is_gaia:
            text += ' kpc'
        t = ax.text(0.04, 0.92, text, transform=ax.transAxes, fontsize=8, color='black')
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Set the colorbar
        cb = fig.colorbar(hh, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    title = '$\\rho_\\mathrm{{model}}$'
    if is_gaia:
        title = '$\\rho_\\mathrm{{\\ (M_\\odot/pc^3)}}$'
    caxs[1].set_title(title, fontsize=10)

    axs[0].set_ylabel(labels[dim2], labelpad=2)
    for ax in axs:
        ax.set_xlabel(labels[dim1], labelpad=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        # set square aspect
        ax.set_box_aspect(1)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'2d_rho_slices_{dim1}_{dim2}.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')


def plot_2d_slices_selfn(phi_model, coords_grid, dim1, dim2, attrs, fig_dir=None, fig_fmt=('png',), fname_mask=None, is_gaia=True):
    """
    Plots 2D slices of the selection function.
    """
    labels, _, _ = get_labels_and_keys(is_gaia)

    if dim1 not in ['x', 'y', 'z'] or dim2 not in ['x', 'y', 'z']:
        raise ValueError('dim1 and dim2 must be from x,y,z')
    dim3 = [d for d in ['x', 'y', 'z'] if d not in [dim1, dim2]][0]

    fig, all_axs = plt.subplots(
        2, 3,
        figsize=(7, 3.3),
        dpi=140,
        width_ratios=[1, 1, 1],
        height_ratios=[0.1, 2],
        layout='compressed'
    )
    caxs = all_axs[0, :]
    axs = all_axs[1, :]

    lims = [get_lims(coords_grid[dim1]), get_lims(coords_grid[dim2])]
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    grid_size = 128
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    X, Y = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))

    kw = dict(cmap='viridis', vmin=0, vmax=None, rasterized=True)
    z_lims = get_lims(coords_grid[dim3])
    w = z_lims[1] - z_lims[0]
    z_values = [z_lims[0] + 0.375 * w, z_lims[0] + 0.5 * w, z_lims[0] + 0.625 * w]
    z_values = [round(z, 3) for z in z_values]

    for i, z_fill in enumerate(z_values):
        q_grid = np.full(shape=(X.size, 3), fill_value=z_fill, dtype='f4')
        q_grid[:, ['x', 'y', 'z'].index(dim1)] = X.ravel()
        q_grid[:, ['x', 'y', 'z'].index(dim2)] = Y.ravel()
        mask = ~utils.get_mask_eta(q_grid, fname_mask, r_min=attrs['r_in'], r_max=attrs['r_out'])[0]

        selfn = utils.get_selfn_values(phi_model, q_grid, disable_tqdm=True)
        selfn = np.ma.masked_where(mask, selfn)
        selfn = np.reshape(selfn, X.shape)

        ax, cax = axs[i], caxs[i]
        hh = ax.pcolormesh(x, y, selfn, **kw)

        z_label_key = [k for k in ['x', 'y', 'z'] if k not in [dim1, dim2]][0]
        label_char = labels[z_label_key].replace('$', '')
        text = f'${label_char}={z_fill:.2f}$'
        if is_gaia:
            text = f'${label_char[0]}={z_fill:.2f}$ kpc'

        t = ax.text(0.04, 0.92, text, transform=ax.transAxes, fontsize=8, color='black')
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='none'))

        cb = fig.colorbar(hh, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    caxs[1].set_title('Selection Function', fontsize=10)

    axs[0].set_ylabel(labels[dim2], labelpad=2)
    for ax in axs:
        ax.set_xlabel(labels[dim1], labelpad=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.set_box_aspect(1)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'2d_selfn_slices_{dim1}_{dim2}.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_potential_param_evolution(potential_params_hist, fig_dir, fig_fmt=("png",)):
    """Plots the evolution of different potential params. Generates two plots,
    one combining all params in one plot, the other plotting each param separately.

    Args:
        potential_params_hist (dict): A dictionary of lists of training
            step-by-step values for each parameter
        fig_dir (str): The directory to save the figure to
        fig_fmt (tuple): The formats to save the figure in
    """

    # Make a copy of potential_params_hist and remove loss histories if persent
    potential_params_hist = potential_params_hist.copy()
    for key in list(potential_params_hist.keys()):
        if key in ['train', 'val', 'train_noreg', 'val_noreg']:
            potential_params_hist.pop(key)
    # Plot the params combined
    label_converter = {
        "omega": r"$\omega$",
        "v0_x": r"$v_{0x}$",
        "v0_y": r"$v_{0y}$",
        "v0_z": r"$v_{0z}$",
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
        "lr": r"lr",
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

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f"potential_param_evolution_combined.{fmt}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
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
    plt.close(fig)


def plot_2dhist_custom(
    x, y, values=None, operation="count",
    bins=(64, 64),
    xlabel="$x$",
    ylabel="$y$",
    lims=None,
    title=None,
    cmap='Greys',
    fig=None, ax=None, cax=None,
    normalize_along_axis=None,
    cax_label='normalized count', rotate_cax_label=True, cax_orientation='vertical',
    x_reference=None, y_reference=None, reference_label=None, plot_reference_std=False,
    cax_nbins=4, cax_spacing=None, cax_minor_auto_frequency=None,
    **kwargs
):
    if fig is None:
        fig, axs = plt.subplots(
            1, 2,
            figsize=(4.2, 4),
            gridspec_kw=dict(width_ratios=[3, 0.1], height_ratios=[3]),
            layout='compressed'
        )
        ax, cax = axs

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

    if type(bins[0]) is not np.ndarray:
        x_bins = np.linspace(xmin, xmax, bins[0])
    else:
        x_bins = bins[0]
    if type(bins[1]) is not np.ndarray:
        y_bins = np.linspace(ymin, ymax, bins[1])
    else:
        y_bins = bins[1]

    ret = binned_statistic_2d(
        x, y, values, statistic=operation, bins=[x_bins, y_bins]
    ).statistic.T
    if normalize_along_axis == 0:
        ret /= (np.max(ret, axis=normalize_along_axis) + 1e-10)[None, :]
    elif normalize_along_axis == 1:
        ret /= (np.max(ret, axis=normalize_along_axis) + 1e-10)[:, None]
    elif normalize_along_axis == 2:
        ret /= np.max(ret)

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
    if x_reference is not None:
        # Draw median lines for x_reference, y_reference. This is to compare with a different model
        ret = binned_statistic(
            x_reference, y_reference, statistic=np.median, bins=x_bins
        ).statistic.T
        x_bin_centers = 0.5*(x_bins[1:] + x_bins[:-1])
        ax.plot(x_bin_centers, ret, label=reference_label, lw=1, color='tab:red', alpha=1)
        if plot_reference_std:
            y_min = binned_statistic(
                x_reference, y_reference, statistic=lambda x: np.percentile(x, 16), bins=x_bins
            ).statistic.T
            y_max = binned_statistic(
                x_reference, y_reference, statistic=lambda x: np.percentile(x, 84), bins=x_bins
            ).statistic.T
            ax.fill_between(
                x_bin_centers, y_min, y_max, alpha=0.3, color='tab:red'
            )

    if cax is not None:
        cb = fig.colorbar(im, cax=cax, orientation=cax_orientation)
        if cax_orientation == 'horizontal':
            cb.ax.xaxis.set_ticks_position('top')
            if rotate_cax_label:
                cax.set_title(cax_label, rotation=-90, pad=10)
            else:
                cax.set_title(cax_label)
        else:
            if rotate_cax_label:
                cax.set_ylabel(cax_label, rotation=-90, labelpad=10)
            else:
                cax.set_ylabel(cax_label)

        # TODO: The following bit of code currently only works when the cax is horizontal..
        if cax_spacing is not None:
            cb.ax.xaxis.set_major_locator(ticker.MultipleLocator(cax_spacing))
        else:
            cb.ax.locator_params(nbins=cax_nbins)
        if cax_minor_auto_frequency is not None:
            cb.ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(cax_minor_auto_frequency))


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig, (ax, cax)


def plot_1d_acc(coords_grid, acc_components_grid, r0, fig_dir=None, fig_fmt=('png',), gamma=0.7, is_gaia=True):
    coef = 1
    if is_gaia:
        coef = (100**2*u.km**2/u.s**2/u.kpc).to(u.km/u.s/u.Myr).value

    fig, axs = plt.subplots(1, 4, figsize=(9, 4), width_ratios=[1, 1, 1, 0.05], layout='compressed')

    cmap = 'Greys'

    kw_line = dict(lw=1, alpha=2/3, color='tab:red')

    dR = coords_grid['cylR'] - r0
    x_coord = (coords_grid['cylphi'] - np.pi)
    
    xlabel = "$(\phi - \pi)R_0$"
    ylabel = "$a_\phi$"
    if is_gaia:
        xlabel += "$\mathrm{\>(kpc)}$"
        ylabel += "$\mathrm{\>(km/(s\cdot Myr))}$"
        
    plot_2dhist_custom(
        x_coord, coef*acc_components_grid['cylphi'], np.ones(len(coords_grid['cylR'])), operation=np.sum,
        xlabel=xlabel, ylabel=ylabel,
        norm=colors.PowerNorm(vmin=0, gamma=gamma), bins=(81, 128),
        lims=[get_lims(x_coord), get_lims(coef*acc_components_grid['cylphi'], False)],
        normalize_along_axis=2,
        fig=fig, ax=axs[0], cax=axs[-1], cmap=cmap,
    )

    y = np.sqrt(np.abs(-coords_grid['cylR']*acc_components_grid['cylR']))
    if is_gaia:
        y *= 100
    
    xlabel = "$R$"
    ylabel = "$v_\mathrm{circ}=\sqrt{-Ra_R}$"
    if is_gaia:
        xlabel += "$\mathrm{\>(kpc)}$"
        ylabel += "$\mathrm{\>(km/s)}$"
        
    plot_2dhist_custom(
        coords_grid['cylR'], y, np.ones(len(coords_grid['cylR'])), operation=np.sum,
        xlabel=xlabel, ylabel=ylabel,
        norm=colors.PowerNorm(vmin=0, gamma=gamma),
        bins=(81, 128),
        lims=[get_lims(coords_grid['cylR']), get_lims(y, False)],
        normalize_along_axis=2,
        fig=fig, ax=axs[1], cax=axs[-1], cmap=cmap,
    )
    
    xlabel = "$z$"
    ylabel = "$a_z$"
    if is_gaia:
        xlabel += "$\mathrm{\>(kpc)}$"
        ylabel += "$\mathrm{\>(km/(s\cdot Myr))}$"
        
    plot_2dhist_custom(
        coords_grid['z'], coef*acc_components_grid['z'], np.ones(len(coords_grid['cylR'])), operation=np.sum,
        xlabel=xlabel, ylabel=ylabel,
        norm=colors.PowerNorm(vmin=0, gamma=gamma), bins=(81, 128),
        lims=[get_lims(coords_grid['z']), get_lims(coef*acc_components_grid['z'], False)],
        normalize_along_axis=2,
        fig=fig, ax=axs[2], cax=axs[-1], cmap=cmap,
        cax_label='Normalized count',
    )

    for ax in axs[:3]:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.set_box_aspect(1)

    if fig_dir is not None:
        for fmt in fig_fmt:
            fname = os.path.join(fig_dir, f'1d_acc_marginal.{fmt}')
            fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return


def plot_1d_rho(coords_grid, rho_grid, dim, fig_dir, fig_fmt, gamma=0.7, is_gaia=True):
    """
    Makes a 1d plot of the matter density rho as a function of a dimension, marginalizing over the other
    spatial dimensions.
    """
    labels, _, _ = get_labels_and_keys(is_gaia)
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(3.6, 3), width_ratios=[3, 0.1], layout='compressed')

    cmap = 'Greys'
    ylabel = '$\\rho$'
    if is_gaia:
        ylabel += '$\mathrm{\ (M_\odot/pc^3)}$'
        
    plot_2dhist_custom(
        coords_grid[dim], rho_grid, np.ones(len(rho_grid)), operation=np.sum,
        xlabel=labels[dim], ylabel=ylabel,
        norm=colors.PowerNorm(vmin=0, gamma=gamma), bins=(81, 128),
        lims=[get_lims(coords_grid[dim]), get_lims(rho_grid, False)],
        normalize_along_axis=2,
        fig=fig, ax=ax, cax=cax, cmap=cmap,
        cax_label='Normalized count',
    )

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.set_box_aspect(1)

    if fig_dir is not None:
        for fmt in fig_fmt:
            fname = fig_dir / f'1d_rho_{dim}_marginal.{fmt}'
            fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def plot_1d_selfn(coords_grid, selfn_grid, dim, fig_dir, fig_fmt, gamma=0.7, is_gaia=True):
    """
    Makes a 1d plot of the selection function as a function of a dimension,
    marginalizing over the other spatial dimensions.
    """
    labels, _, _ = get_labels_and_keys(is_gaia)
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(3.6, 3), width_ratios=[3, 0.1], layout='compressed')

    cmap = 'Greys'
    plot_2dhist_custom(
        coords_grid[dim], selfn_grid, np.ones(len(selfn_grid)), operation=np.sum,
        xlabel=labels[dim], ylabel='Selection Function',
        norm=colors.PowerNorm(vmin=0, gamma=gamma), bins=(81, 128),
        lims=[get_lims(coords_grid[dim]), [0, 1]],
        normalize_along_axis=2,
        fig=fig, ax=ax, cax=cax, cmap=cmap,
        cax_label='Normalized count',
    )

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.set_box_aspect(1)

    if fig_dir is not None:
        for fmt in fig_fmt:
            fname = fig_dir / f'1d_selfn_{dim}_marginal.{fmt}'
            fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def add_polar_grid(ax, r0, radii=None, kw_overwrite=dict()):
    x0 = r0
    a = 1e2
    # Make the line have a black outline
    kw = dict(lw=1, alpha=0.25, color='black', linestyle=(0, (6, 3)))
    kw.update(kw_overwrite)
    for phi in np.linspace(0, np.pi, 5)[:-1]:
        ax.plot([x0 - a * np.cos(phi), x0 + a * np.cos(phi)], [-a * np.sin(phi), a * np.sin(phi)], **kw)

    if radii is None:
        radii = [r0]
    for r in radii:
        circle = plt.Circle((x0, 0), r, fill=False, **kw)
        ax.add_patch(circle)


def plot_2d_acc(coords_grid, acc_components_grid, r0, fig_dir=None, fig_fmt=('png',), is_gaia=True):
    labels, _, _ = get_labels_and_keys(is_gaia)
    fig, all_axs = plt.subplots(
        2, 3, figsize=(9, 4), width_ratios=[1, 1, 1], height_ratios=[0.2, 3],
        layout='compressed'
    )
    axs = all_axs[1,:]
    caxs = all_axs[0,:]

    for ax in axs[:2]:
        add_polar_grid(ax, r0)
    add_polar_grid(axs[2], 0, [r0])

    cmap = 'RdBu'
    bins = (101, 101)
    
    coef_acc = 1
    if is_gaia:
        coef_acc = (((100*u.km/u.s)**2/u.kpc).to(u.km/u.s/u.Myr)).value
        acc_components_grid['vcirc'] = 100 * np.sqrt(-acc_components_grid['cylR'] * coords_grid['cylR'])
    else:
        acc_components_grid['vcirc'] = np.sqrt(-acc_components_grid['cylR'] * coords_grid['cylR'])


    def get_xyz(dimx, dimy, dimz, dimval):
        # We integrate dimz from 37.5% to 62.5% of the range
        z_lim = get_lims(coords_grid[dimz])
        w = z_lim[1] - z_lim[0]
        z_lim = round(z_lim[0] + 0.375*w, 3), round(z_lim[0] + 0.625*w, 3)
        idx = (coords_grid[dimz] < z_lim[1]) & (coords_grid[dimz] > z_lim[0])
        values = acc_components_grid[dimval][idx]
        if dimval != 'vcirc':
            values *= coef_acc

        return coords_grid[dimx][idx], coords_grid[dimy][idx], values, labels[dimx], labels[dimy], z_lim

    x, y, values, xlabel, ylabel, z_lim = get_xyz('x', 'y', 'z', 'cylphi')
    values_lims = get_lims(values, False, make_symmetric=True)
    cax_label = f"$a_\phi, {z_lim[0]} < z < {z_lim[1]}$"
    if is_gaia:
        cax_label = f"$a_\phi\mathrm{{\>(km/(s\cdot Myr))}}, {z_lim[0]} < z < {z_lim[1]}$"
    plot_2dhist_custom(
        x, y, values, operation=np.mean,
        xlabel=xlabel, ylabel=ylabel,
        bins=bins, lims=[get_lims(x), get_lims(y)],
        fig=fig, ax=axs[0], cax=caxs[0], cmap=cmap, cax_orientation='horizontal', rotate_cax_label=False,
        cax_label=cax_label, vmin=values_lims[0], vmax=values_lims[1],
        cax_minor_auto_frequency=5
    )

    x, y, values, xlabel, ylabel, z_lim = get_xyz('x', 'y', 'z', 'vcirc')
    values_lims = get_lims(values, False)
    cax_label = f"$v_\mathrm{{circ}}=\sqrt{{-Ra_R}}, {z_lim[0]} < z < {z_lim[1]}$"
    if is_gaia:
        cax_label = f"$v_\mathrm{{circ}}=\sqrt{{-Ra_R}}\mathrm{{\>(km/s)}}, {z_lim[0]} < z < {z_lim[1]}$"
    plot_2dhist_custom(
        x, y, values, operation=np.mean,
        xlabel=xlabel, ylabel=ylabel,
        bins=bins, lims=[get_lims(x), get_lims(y)],
        fig=fig, ax=axs[1], cax=caxs[1], cmap='cmr.rainforest', cax_orientation='horizontal', rotate_cax_label=False,
        cax_label=cax_label, vmin=values_lims[0], vmax=values_lims[1],
        cax_minor_auto_frequency=5
    )

    x, y, values, xlabel, ylabel, z_lim = get_xyz('cylR', 'z', 'y', 'z')
    values_lims = get_lims(values, False, make_symmetric=True)
    cax_label = f"$a_z, {z_lim[0]} < y < {z_lim[1]}$"
    if is_gaia:
        cax_label = f"$a_z\mathrm{{\>(km/(s\cdot Myr))}}, {z_lim[0]} < y < {z_lim[1]}$"
    plot_2dhist_custom(
        x, y, values, operation=np.mean,
        xlabel=xlabel, ylabel=ylabel,
        bins=bins, lims=[get_lims(x), get_lims(y)],
        fig=fig, ax=axs[2], cax=caxs[2], cmap=cmap, cax_orientation='horizontal', rotate_cax_label=False,
        cax_label=cax_label, vmin=values_lims[0], vmax=values_lims[1],
        cax_minor_auto_frequency=5
    )

    for ax in axs:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.set_box_aspect(1)

    if fig_dir is not None:
        for fmt in fig_fmt:
            fname = fig_dir / f'2d_acc_integrated.{fmt}'
            fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def plot_2d_rho(coords_grid, rho_grid, r0, fig_dir=None, fig_fmt=('png',), is_gaia=True):
    labels, _, _ = get_labels_and_keys(is_gaia)
    fig, all_axs = plt.subplots(
        2, 3, figsize=(9, 4), width_ratios=[1, 1, 1], height_ratios=[0.15, 3],
        layout='compressed'
    )
    axs = all_axs[1,:]
    caxs = all_axs[0,:]

    cmap = 'cmr.rainforest'

    def get_xyz(dimx, dimy, dimz):
        z_lim = get_lims(coords_grid[dimz])
        w = z_lim[1] - z_lim[0]
        z_lim = round(z_lim[0] + 0.375*w, 3), round(z_lim[0] + 0.625*w, 3)
        idx = (coords_grid[dimz] > z_lim[0]) & (coords_grid[dimz] < z_lim[1])

        return coords_grid[dimx][idx], coords_grid[dimy][idx], rho_grid[idx], labels[dimx], labels[dimy], z_lim

    cax_label = '$\\rho$'
    if is_gaia:
        cax_label += '$\mathrm{\,(M_\odot/pc^3)}$'
    cax_nbins = 4
    vmax = None
    if is_gaia:
        vmax = 0.15
    bins = (161, 161)

    x, y, values, xlabel, ylabel, z_lim = get_xyz('x', 'y', 'z')
    plot_2dhist_custom(
        x, y, values, operation=np.mean,
        xlabel=xlabel, ylabel=ylabel,
        bins=bins, lims=[get_lims(x), get_lims(y)],
        fig=fig, ax=axs[0], cax=caxs[0], cmap=cmap, cax_orientation='horizontal', rotate_cax_label=False,
        cax_label=cax_label + f', ${z_lim[0]} < z < {z_lim[1]}$', vmin=0, vmax=vmax, cax_nbins=cax_nbins, cax_minor_auto_frequency=5
    )

    x, y, values, xlabel, ylabel, z_lim = get_xyz('y', 'z', 'x')
    plot_2dhist_custom(
        x, y, values, operation=np.mean,
        xlabel=xlabel, ylabel=ylabel,
        bins=bins, lims=[get_lims(x), get_lims(y)],
        fig=fig, ax=axs[1], cax=caxs[1], cmap=cmap, cax_orientation='horizontal', rotate_cax_label=False,
        cax_label=cax_label + f', ${z_lim[0]} < x < {z_lim[1]}$', vmin=0, vmax=vmax, cax_nbins=cax_nbins, cax_minor_auto_frequency=5
    )

    x, y, values, xlabel, ylabel, z_lim = get_xyz('x', 'z', 'y')
    plot_2dhist_custom(
        x, y, values, operation=np.mean,
        xlabel=xlabel, ylabel=ylabel,
        bins=bins, lims=[get_lims(x), get_lims(y)],
        fig=fig, ax=axs[2], cax=caxs[2], cmap=cmap, cax_orientation='horizontal', rotate_cax_label=False,
        cax_label=cax_label + f', ${z_lim[0]} < y < {z_lim[1]}$', vmin=0, vmax=vmax, cax_nbins=cax_nbins, cax_minor_auto_frequency=5
    )

    for ax in axs:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.set_box_aspect(1)

    if fig_dir is not None:
        for fmt in fig_fmt:
            fname = fig_dir / f'2d_rho_integrated.{fmt}'
            fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def get_potential_dfdt(phi_model, df_data, dphi_dq):
    """ Returns the \partial f/\partial t predicted by the potential in the associated rotating frame.
    """

    eta, df_deta = df_data['eta'], df_data['df_deta']

    fs = phi_model.frameshift_model
    model_omega = float(fs.omega)
    model_v0 = (float(fs.v0_x), float(fs.v0_y), float(fs.v0_z))
    model_r0 = float(fs.r0)

    pdf_dt_CBE = -np.sum(eta[:, 3:] * df_deta[:, :3], axis=1) +\
        np.sum(dphi_dq * df_deta[:, 3:], axis=1)

    ix, iy, ivx, ivy = 0, 1, 3, 4
    pdf_dt_stat = -model_omega * ((eta[:, ix] - model_r0) * df_deta[:, iy] -
                                  eta[:, iy] * df_deta[:, ix] +
                                  (eta[:, ivx] - model_v0[ix]) * df_deta[:, ivy] -
                                  (eta[:, ivy] - model_v0[iy]) * df_deta[:, ivx]) - \
        np.sum(model_v0 * df_deta[:,:3], axis=1)

    return pdf_dt_CBE - pdf_dt_stat


def plot_dfdt_comparison(phi_model, df_data, coords, acc, dim1, dim2, grid_size=64, idx=None, fig_dir=None, fig_fmt=('png',), is_gaia=True):
    labels, _, _ = get_labels_and_keys(is_gaia)

    if idx is None:
        idx = np.full(len(acc), True)

    fig, (all_axs) = plt.subplots(
        2, 2,
        figsize=(7, 4.6),
        dpi=200,
        gridspec_kw=dict(width_ratios=[2,2], height_ratios=[0.1, 2]),
        layout='compressed'
    )
    axs = all_axs[1,:]
    caxs = all_axs[0,:]

    def get_coef_and_lims(dim):
        coef = 1
        is_spatial = True
        if '(km/s)' in labels[dim] and is_gaia:
            coef = 100
            is_spatial = False
        if 'v_{\phi}' in labels[dim]:
            coef *= -1
        lim = get_lims(coords[dim][idx] * coef, is_spatial)
        return coef, lim

    coef_x, xlim = get_coef_and_lims(dim1)
    coef_y, ylim = get_coef_and_lims(dim2)
    xmin, xmax = xlim
    ymin, ymax = ylim
    x_bins = np.linspace(xmin, xmax, grid_size)
    y_bins = np.linspace(ymin, ymax, grid_size)

    for ax in axs:
        ax.set_xlabel(labels[dim1])
        ax.set_box_aspect(1)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axs[0].set_ylabel(labels[dim2])
    axs[1].set_yticklabels([])

    potential_dfdt = get_potential_dfdt(phi_model, df_data, -acc)
    f_prob = df_data['f']
    
    conversion_coef = 1
    if is_gaia:
        conversion_coef = ((100*u.km/u.s/u.kpc).to(1/u.Myr)).value

    # CBE+rotating stationarity discrepancy
    ret = binned_statistic_2d(
        coef_x * coords[dim1][idx], coef_y * coords[dim2][idx],
        -conversion_coef*potential_dfdt[idx]/f_prob[idx],
        statistic=np.median, bins=[x_bins, y_bins]
    )

    vmin, vmax = get_lims(ret.statistic, False, make_symmetric=True)
    cmap = matplotlib.colormaps.get_cmap('RdYlBu')
    cmap.set_bad(color='#dddcdb')
    im = axs[0].imshow(
        ret.statistic.T, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap=cmap,
        vmin=vmin, vmax=vmax, aspect='auto', rasterized=True, interpolation='none'
    )
    cb = fig.colorbar(im, cax=caxs[0], orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    
    title = 'Inverse timescale of non-stationarities\n$\partial \ln f / \partial t$'
    if is_gaia:
        title += ' ($\mathrm{Myr}^{-1}$)'
    caxs[0].set_title(title)

    # Number counts
    ret = binned_statistic_2d(
        coef_x * coords[dim1][idx], coef_y * coords[dim2][idx], values=np.ones(np.sum(idx)), statistic='count',
        bins=[x_bins, y_bins]
    )
    cmap = matplotlib.colormaps.get_cmap('turbo')
    cmap.set_bad(color='#dddcdb')
    vmin, vmax = get_lims(ret.statistic[ret.statistic > 0], False, k=0.0)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    im = axs[1].imshow(
        ret.statistic.T, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap=cmap, norm=norm,
        aspect='auto', rasterized=True, interpolation='none'
    )
    cb = fig.colorbar(im, cax=caxs[1], orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    caxs[1].set_title('Count')

    if fig_dir is not None:
        for fmt in fig_fmt:
            fname = fig_dir / f'2d_nonstationarities_{dim1}_{dim2}.{fmt}'
            fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def create_grid(attrs_train, fname_mask, a_spacing=0.01):
    r_out = attrs_train['r_out']

    volume_total = 4/3*np.pi*(attrs_train['r_out']**3 - attrs_train['r_in']**3)
    print(f'Total volume: {volume_total:.2f} kpc^3')

    n_grid = int(volume_total / a_spacing**3)
    print(f'grid spacing: {1000*a_spacing:.3f} pc')
    print(f'Expected number of points inside the volume: {n_grid}')

    def centered_spacing(minv, maxv, step):
        # Makes it so 0 is always included as one of the center-points
        return np.concatenate([np.arange(-step, minv - 1e-9, step=-step)[::-1], np.arange(0, maxv + 1e-9, step=step)], axis=0)

    x_grid = centered_spacing(-r_out, r_out, step=a_spacing)
    y_grid = centered_spacing(-r_out, r_out, step=a_spacing)
    z_grid = centered_spacing(-r_out, r_out, step=a_spacing)
    print(len(x_grid), len(y_grid), len(z_grid))


    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    q_grid = np.zeros(shape=(X_grid.size, 3), dtype='f4')
    q_grid[:,0] = X_grid.ravel()
    q_grid[:,1] = Y_grid.ravel()
    q_grid[:,2] = Z_grid.ravel()

    idx = (np.sum(q_grid[:,:3]**2, axis=1)**0.5 < r_out)
    print(f'Points inside the volume: {np.sum(idx)}')

    print('Evaluating the potential on uniformly spaced grid..')


    mask_grid = utils.get_mask_eta(q_grid, fname_mask, r_min=attrs_train['r_in'])[0]
    #mask_grid, _ = utils.mask_eta(q_grid, f'{dir_data}/mask_sph_400pc.h5', r_max=r_out) # Mask is True if a point is within volume of validity
    print(f'Grid volume before masking: {a_spacing**3*np.sum(idx):.4f} kpc^3')
    q_grid = q_grid[mask_grid]
    print(f'Grid volume after masking: {a_spacing**3*len(q_grid):.4f} kpc^3')
    return q_grid


def create_random_grid(attrs_train, fname_mask, n_points=100000):
    r_out = attrs_train['r_out']
    r_in = attrs_train['r_in']

    volume_total = 4/3*np.pi*(attrs_train['r_out']**3 - attrs_train['r_in']**3)
    print(f'Total volume: {volume_total:.2f} kpc^3')

    n_grid = n_points
    print(f'Generating {n_grid} random points inside the volume')

    rng = np.random.default_rng()

    batch_size = 1000000
    q_grids = []
    n_done = 0
    n_all = 0
    while n_done < n_grid:
        u = rng.random(batch_size)
        costheta = 2 * rng.random(batch_size) - 1
        phi = 2 * np.pi * rng.random(batch_size)

        theta = np.arccos(costheta)

        r = ((u * (r_out**3 - r_in**3)) + r_in**3)**(1/3)

        x_grid = r * np.sin(theta) * np.cos(phi)
        y_grid = r * np.sin(theta) * np.sin(phi)
        z_grid = r * np.cos(theta)

        q_grid = np.zeros(shape=(n_grid, 3), dtype='f4')
        q_grid[:,0] = x_grid
        q_grid[:,1] = y_grid
        q_grid[:,2] = z_grid

        if fname_mask is not None:
            mask_grid = utils.get_mask_eta(q_grid, fname_mask, r_min=r_in, r_max=r_out)[0]
            q_grid = q_grid[mask_grid]

        q_grids.append(q_grid)
        n_done += len(q_grid)
        n_all += batch_size

    q_grids = np.concatenate(q_grids, axis=0)
    print(f'Random points generated. Points inside the volume: {len(q_grids)} out of {n_all} ({100*len(q_grids)/n_all:.2f}%)')
    return q_grids[:n_grid]


def benchmark_potential(phi_model, loss_history, fname_mask, data_train, attrs_train, df_data, spherical_origin=(0.0, 0.0, 0.0), cylindrical_origin=(8.277, 0.0, 0.0), fig_fmt=('png',), checkpoint_index=None, is_gaia=True):
    if checkpoint_index is None:
        checkpoint_index = phi_model.checkpoint_index
    save_dir = Path(str(phi_model.model_dir).replace('models', 'plots'))
    # Add the checkpoint number as a subdirectory
    save_dir = save_dir / f'checkpoint_{checkpoint_index}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot the loss
    utils.plot_loss_new(loss_history, save_dir / f'potential-{checkpoint_index}_loss.png', w=1)
    loss_history_noreg = {
        'train': loss_history['train_noreg'],
        'val': loss_history['val_noreg'],
        'lr': loss_history['lr'],
    }
    utils.plot_loss_new(loss_history_noreg, save_dir / f'potential-{checkpoint_index}_loss_noreg.png', w=1)

    # Calculate coords
    coords_train_eta = utils.calc_coords(data_train['eta'], spherical_origin, cylindrical_origin)
    coords_df_eta = utils.calc_coords(df_data['eta'], spherical_origin, cylindrical_origin)

    # Plot of parameter evolution
    plot_potential_param_evolution(loss_history, save_dir, fig_fmt=fig_fmt)
    fs = phi_model.frameshift_model
    print(f"Frameshift parameters:\nomega={float(fs.omega):.4f}, r0={float(fs.r0):.4f}")
    print(f"v0=({float(fs.v0_x):.4f}, {float(fs.v0_y):.4f}, {float(fs.v0_z):.4f})")

    # ------------- Creating a randomized grid of points ----------------
    print("Generating a grid of potential values ...")
    # If attrs_train doesn't have outer and inner radii, then we set them here
    if 'r_out' not in attrs_train:
        attrs_train['r_out'] = np.percentile(np.sum(data_train['eta'][:,:3]**2, axis=1)**0.5, 99)
    if 'r_in' not in attrs_train:
        attrs_train['r_in'] = 0
    print(f"Using r_in={attrs_train['r_in']:.3f} kpc and r_out={attrs_train['r_out']:.3f} kpc for the grid")
    n_points = 1000000
    fname = save_dir / f'potential_grid_points_{n_points}.npy'
    if fname.exists():
        print(f'Loading grid points from {fname}')
        q_grid = np.load(fname)
    else:
        print(f'Creating grid points and saving to {fname}')
        q_grid = create_random_grid(attrs_train, fname_mask, n_points=n_points)
        np.save(fname, q_grid)
    coords_grid = utils.calc_coords(q_grid, spherical_origin, cylindrical_origin)
    phi_grid, acc_grid, rho_grid = utils.get_model_values(phi_model, q_grid, convert_rho_to_msunpc3=is_gaia) # , fname=save_dir / 'potential_grid_values.npz')
    acc_components_grid = utils.calc_coords(q_grid, spherical_origin, cylindrical_origin, vector_field=acc_grid)
    r0 = cylindrical_origin[0]
    if r0 < 1e-8:
        r0 = 1
    has_selfn = phi_model.log_selection_function_model is not None
    if has_selfn:
        fname = save_dir / f'selfn_grid_values{n_points}.npy'
        selfn_grid = utils.get_selfn_values(phi_model, q_grid) # , fname=fname)


    # ------------- Plotting ----------------
    if has_selfn:
        print("Plotting 2D histogram of selection function slices ...")
        for dim1, dim2 in [('x', 'y'), ('x', 'z'), ('y', 'z')]:
            plot_2d_slices_selfn(phi_model, coords_grid, dim1, dim2, attrs_train, fig_dir=save_dir, fig_fmt=fig_fmt, fname_mask=fname_mask, is_gaia=is_gaia)

        print("Plotting 1D histogram of selection function ...")
        for dim in ['x', 'y', 'z', 'r', 'cylR', 'cylphi']:
            plot_1d_selfn(coords_grid, selfn_grid, dim, fig_dir=save_dir, fig_fmt=fig_fmt, gamma=0.7, is_gaia=is_gaia)

    phi, acc, rho = utils.get_model_values(phi_model, df_data['eta'][:,:3], convert_rho_to_msunpc3=is_gaia) # , fname=save_dir / 'potential_train_values.npz')
    print("Plotting 2D marginals of non-stationarities ...")
    for dim1, dim2 in [('x', 'y'), ('z', 'vz'), ('cylvR', 'cylvT')]:
        # Make the plot for the innermost 50% of the radii
        idx = (np.sum(df_data['eta'][:,:3]**2, axis=1)**0.5 < 0.5 * attrs_train['r_out'])
        plot_dfdt_comparison(phi_model, df_data, coords_df_eta, acc, dim1, dim2, grid_size=64, idx=idx, fig_dir=save_dir, fig_fmt=fig_fmt, is_gaia=is_gaia)

    print("Plotting 1D histogram of accelerations ...")
    plot_1d_acc(coords_grid, acc_components_grid, r0, fig_dir=save_dir, fig_fmt=fig_fmt, gamma=0.7, is_gaia=is_gaia)

    print("Plotting 1D histogram of matter density ...")
    for dim in ['x', 'y', 'z', 'r', 'cylR', 'cylphi']:
        plot_1d_rho(coords_grid, rho_grid, dim, fig_dir=save_dir, fig_fmt=fig_fmt, gamma=0.7, is_gaia=is_gaia)

    print("Plotting 2D histogram of matter density slices ...")
    for dim1, dim2 in [('x', 'y'), ('x', 'z'), ('y', 'z')]:
        plot_2d_slices_rho(phi_model, coords_grid, dim1, dim2, attrs_train, fig_dir=save_dir, fig_fmt=fig_fmt, fname_mask=fname_mask, is_gaia=is_gaia)

    print("Plotting 2D integrated histograms of accelerations ...")
    plot_2d_acc(coords_grid, acc_components_grid, r0, fig_dir=save_dir, fig_fmt=fig_fmt, is_gaia=is_gaia)

    print("Plotting 2D integrated histograms of densities ...")
    plot_2d_rho(coords_grid, rho_grid, r0, fig_dir=save_dir, fig_fmt=fig_fmt, is_gaia=is_gaia)

    return