#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True
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
print(f'Tensorflow version {tf.__version__}')

import potential_tf
import plot_flow_projections
import plot_potential
import fit_all


dpi = 200


def plot_rho(phi_model, coords_train, fig_dir, dim1, dim2, dimz, z, padding=0.95, attrs=None, fig_fmt=('svg',)): 
    labels = [
        '$x\mathrm{\ [kpc]}$', '$y\mathrm{\ [kpc]}$', '$z\mathrm{\ [kpc]}$',
    ]
    keys = [
        'x', 'y', 'z'
    ]
    
    labels = {k:l for k,l in zip(keys,labels)}
    ikeys = {k:i for i, k in enumerate(keys)}
    
    
    fig, axs = plot_potential.plot_rho(phi_model, coords_train, fig_dir, dim1, dim2, dimz, z, padding=padding, attrs=attrs, fig_fmt=fig_fmt, save=False)
    
    ax_p, cax_p = axs[1, 0], axs[0, 0]
    ax_r, cax_r = axs[1, 1], axs[0, 1]
    ax_e, cax_e = axs[1, 2], axs[0, 2]
    main_axs = [ax_p, ax_r, ax_e]
    
    cax_p.set_title(r'$\Phi^*$', fontsize=10)
    cax_r.set_title(r'$\rho^*\mathrm{\ [M_\odot/pc^3]}$', fontsize=10)
    cax_e.set_title(r'$\rho_\mathrm{train}\mathrm{\ [M_\odot/pc^3]}$', fontsize=10)
    
    for ax in main_axs:
        ax.set_xlabel(labels[dim1], labelpad=0)

    ax_p.set_ylabel(labels[dim2], labelpad=2)

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'phi_rho_{dim1}_{dim2}.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_force_2d_slice(phi_model, fig_dir, dim1, dim2, dimz, z, padding=0.95, attrs=None, fig_fmt=('svg',)): 
    labels = [
        '$x\mathrm{\ [kpc]}$', '$y\mathrm{\ [kpc]}$', '$z\mathrm{\ [kpc]}$',
    ]
    titles = [
        '$F_x^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$', '$F_y^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$', '$F_z^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$'
    ]
    keys = [
        'x', 'y', 'z'
    ]
    
    labels = {k:l for k,l in zip(keys,labels)}
    ikeys = {k:i for i, k in enumerate(keys)}
    

    fig, axs = plot_potential.plot_force_2d_slice(phi_model, fig_dir, dim1, dim2, dimz, z, padding=padding, attrs=attrs, fig_fmt=fig_fmt, save=False)
    
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

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'phi_force_slice_{dim1}_{dim2}.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    
def plot_force_1d_slice(phi_model, fig_dir, dim1, dimy, y, z, dimforce, padding=0.95, attrs=None, fig_fmt=('svg',)): 
    labels = [
        '$x\mathrm{\ [kpc]}$', '$y\mathrm{\ [kpc]}$', '$z\mathrm{\ [kpc]}$',
    ]
    force_labels = [
        '$F_x^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$', '$F_y^*\:\mathrm{[10^4 km^2/(kpc\cdot s^2)]}$', '$\Sigma_z^*\:\mathrm{[M_\odot/pc^2]}$'
    ]
    
    keys = [
        'x', 'y', 'z'
    ]
    
    labels = {k:l for k,l in zip(keys,labels)}
    force_labels = {k:l for k,l in zip(keys,force_labels)}
    ikeys = {k:i for i, k in enumerate(keys)}
    
    fig,ax = plt.subplots(figsize=(3,3), dpi=200)
    
    # Get the plot limits
    xmin, xmax, ymin, ymax = 0, 0, 0, 0
    if attrs is not None:
        r_inner, r_outer = 1/attrs['parallax_max'], 1/attrs['parallax_min'] # [kpc], [kpc]
        xmin, xmax, ymin, ymax = -r_outer, r_outer, -r_outer, r_outer
        
        # Visualise the boundaries
        cartesian_keys = ['x', 'y', 'z']
        kw = dict(linestyle=(0, (5, 3)), lw=1.0, color='black', zorder=0)
        if dim1 in keys:
            ax.axvline(r_inner, **kw)
            ax.axvline(r_outer, **kw)
            ax.axvline(-r_inner, **kw)
            ax.axvline(-r_outer, **kw)

    xlim = (xmin, xmax)
    
    
    x_plot = np.linspace(xmin, xmax, 512)
    
    # Create mask for the region of interest (a donut stretching from parallax_max to parallax_min)
    mask_ = (((x_plot**2+y**2+z**2)>r_outer**2*padding**2) | ((x_plot**2+y**2+z**2)<r_inner**2/padding**2))
    
    eta_plot = np.full(shape=(len(x_plot), 3), fill_value=z, dtype='f4')
    eta_plot[:,ikeys[dimy]] = y
    eta_plot[:,ikeys[dim1]] = x_plot
    #print(eta_eval)
    
    _,dphi_dq,d2phi_dq2 = potential_tf.calc_phi_derivatives(
            phi_model['phi'], eta_plot, return_phi=True)
    Z_plot = -dphi_dq[:,ikeys[dimforce]].numpy()
    
    if dim1 == 'z' and dimforce == 'z':
        # with F_z - z, plot the implied surface density instead
        Z_plot *= -367.5
    ax.plot(x_plot[~mask_ & (x_plot > 0)], Z_plot[~mask_ & (x_plot > 0)], color='tab:blue')
    ax.plot(x_plot[~mask_ & (x_plot < 0)], Z_plot[~mask_ & (x_plot < 0)], color='tab:blue')
    
    #ax.set_xlim(-r_max, r_max)
    #ax.set_ylim(-r_max, r_max)
    ax.set_xlabel(labels[dim1])
    ax.set_ylabel(f'{force_labels[dimforce]}')

    # Plot the ideal curves
    if dim1 == 'z' and dimforce == 'z':
        z0 = 255.6 # [pc]
        rho0 = 0.0474 # [1/(M_sun*pc^3)]
        sigma_z = 68*x_plot/1.1
        #sigma_z = 2*xs/np.abs(xs)*rho0*z0*(1 - np.exp(-np.abs(xs)*1000/z0))
        #print(az)
        plt.plot(x_plot[~mask_ & (x_plot > 0)], sigma_z[~mask_ & (x_plot > 0)], color="tab:red", label=r"Bovy\&Rix 2013, $\Sigma_{z=1.1\mathrm{kpc}}=68\pm 4 M_\odot/\mathrm{pc}^2$")
        plt.plot(x_plot[~mask_ & (x_plot < 0)], sigma_z[~mask_ & (x_plot < 0)], color="tab:red")
        plt.fill_between(x_plot, 64*x_plot/1.1, 72*x_plot/1.1, alpha=0.3, color="tab:red")
        plt.axhline(0, color="black", lw=0.5)
        plt.axvline(0, color="black", lw=0.5)
    if dim1 == 'x' and dimforce == 'x':
        u = 2.2 # Approximate rotation curve in MW [100 km/s]
        r0 = 8.3 # Distance to MW centre [kpc]
        # Plot the forces assuming constant velocity profile
        plt.plot(x_plot, u**2/r0*(1 + x_plot/r0), color='tab:red', label='ideal constant rotation curve')
    if dim1 == 'y' and dimforce == 'y':
        u = 2.2 # Approximate rotation curve in MW [100 km/s]
        r0 = 8.3 # Distance to MW centre [kpc]
        # Plot the forces assuming constant velocity profile
        plt.plot(x_plot, -u**2/r0**2*x_plot, color='tab:red', label='ideal constant rotation curve')

    ax.legend()
    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'phi_force_slice_{dim1}.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_dfdt_2d_marginal(phi_model, df_data, dphi_dq, fig_dir, dim1, dim2, padding=0.95, attrs=None, fig_fmt=('svg',)):
    fig,(all_axs) = plt.subplots(2, 3,
            figsize=(6,2.2),
            dpi=200,
            gridspec_kw=dict(width_ratios=[2,2,2], height_ratios=[0.2, 2]))
    axs = all_axs[1,:]
    caxs = all_axs[0,:]


    eta, df_deta = df_data['eta'], df_data['df_deta']

    model_omega = phi_model['fs']._omega.numpy()
    model_u0 = np.array((phi_model['fs']._u_x.numpy(), phi_model['fs']._u_y.numpy(), phi_model['fs']._u_z.numpy()))
    model_r_c = phi_model['fs']._r_c.numpy()

    pdf_dt_CBE_ideal = -np.sum(eta[:, 3:]*df_deta[:, :3], axis=1) +\
                        np.sum(dphi_dq.numpy()*df_deta[:, 3:], axis=1)


    ix, iy, ivx, ivy = 0, 1, 3, 4
    pdf_dt_stat_ideal = -model_omega*((eta[:, ix] - model_r_c)*df_deta[:, iy] -\
                        eta[:, iy]*df_deta[:, ix] +\
                        (eta[:, ivx] - model_u0[ix])*df_deta[:, ivy] -\
                        (eta[:, ivy] - model_u0[iy])*df_deta[:, ivx]) -\
                        np.sum(model_u0*df_deta[:,:3], axis=1)

    labels = ['$x\mathrm{\ [kpc]}$', '$y\mathrm{\ [kpc]}$', '$z\mathrm{\ [kpc]}$']
    keys = ['x', 'y', 'z']
    
    labels = {k:l for k,l in zip(keys,labels)}
    ikeys = {k:i for i, k in enumerate(keys)}
    ix, iy = ikeys[dim1], ikeys[dim2]

    for i in range(3):
        axs[i].set_xlabel(labels[dim1])
    axs[0].set_ylabel(labels[dim2])
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])

    if attrs is not None:
        r_inner, r_outer = 1/attrs['parallax_max'], 1/attrs['parallax_min'] # [kpc], [kpc]
        xmin, xmax, ymin, ymax = -r_outer, r_outer, -r_outer, r_outer
        
        # Visualise the boundaries
        cartesian_keys = ['x', 'y', 'z']
        kw = dict(linestyle=(0, (5, 3)), lw=0.5, color='black')
        if (dim1 in cartesian_keys) and (dim2 in cartesian_keys):
            # Plot circles
            for ax in axs:
                circ = plt.Circle((0, 0), r_inner, fill=False, **kw)
                ax.add_patch(circ)
                circ = plt.Circle((0, 0), r_outer, fill=False, **kw)
                ax.add_patch(circ)
        if dim1 in ['R']:
            for ax in axs:
                ax.axvline(r_inner, **kw)
                ax.axvline(r_outer, **kw)
        if dim2 in ['R']:
            for ax in axs:
                ax.axhline(r_inner, **kw)
                ax.axhline(r_outer, **kw)
    x_bins = np.linspace(xmin, xmax, 32)
    y_bins = np.linspace(ymin, ymax, 32)
        

    # Ideal CBE discrepancy
    ret = binned_statistic_2d(eta[:, ix], eta[:, iy], pdf_dt_CBE_ideal, statistic=np.mean, bins=[x_bins, y_bins])
    divnorm = colors.TwoSlopeNorm(vcenter=0., vmin=-5, vmax=5)
    im = axs[0].imshow(ret.statistic.T, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='seismic', norm=divnorm)
    cb = fig.colorbar(im, cax=caxs[0], orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.locator_params(nbins=5)
    caxs[0].set_title('$(\partial f/\partial t)_\mathrm{CBE}$')

    # Ideal CBE stationarity discrepancy
    ret = binned_statistic_2d(eta[:, ix], eta[:, iy], pdf_dt_stat_ideal, statistic=np.mean, bins=[x_bins, y_bins])
    divnorm = colors.TwoSlopeNorm(vcenter=0., vmin=-5, vmax=5)
    im = axs[1].imshow(ret.statistic.T, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='seismic', norm=divnorm)
    cb = fig.colorbar(im, cax=caxs[1], orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.locator_params(nbins=5)
    caxs[1].set_title('$(\partial f/\partial t)_\mathrm{stat}$')

    # CBE+rotating stationarity discrepancy
    ret = binned_statistic_2d(eta[:, ix], eta[:, iy], pdf_dt_CBE_ideal - pdf_dt_stat_ideal, statistic=np.mean, bins=[x_bins, y_bins])
    divnorm = colors.TwoSlopeNorm(vcenter=0., vmin=-3, vmax=3)
    im = axs[2].imshow(ret.statistic.T, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='seismic', norm=divnorm)
    cb = fig.colorbar(im, cax=caxs[2], orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.locator_params(nbins=5)
    caxs[2].set_title('$(\partial f/\partial t)_\mathrm{CBE+stat}$')
    
    #plt.tight_layout()
    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'phi_flow_dfdt_discrepancy_{dim1}_{dim2}.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)  


def plot_vcirc_marginals(phi_model, coords_train, coords_sample, fig_dir, attrs=None, fig_fmt=('svg',)):    
    x_train, x_sample = coords_train['cylR'], coords_sample['cylR']
    y_train, y_sample = coords_train['cylvT'], coords_sample['cylvT']

    fig,(ax_m, ax_t, ax_s) = plt.subplots(
        1,3,
        figsize=(9,3),
        dpi=200,
        gridspec_kw=dict(width_ratios=[1,1,1])
    )

    xlim = np.percentile(x_train, [1., 99.])
    nbins = 64
    xbins = np.linspace(*xlim, nbins)

    # Plot vcirc deduced from Phi
    r_c = phi_model['fs']._r_c.numpy()
    x_model = r_c - np.linspace(*xlim, 512) # This assumes that galactic centre is at +x
    eta_plot = np.full(shape=(len(x_model), 3), fill_value=0, dtype='f4')
    eta_plot[:,0] = x_model
    _,dphi_dq,_ = potential_tf.calc_phi_derivatives(
            phi_model['phi'], eta_plot, return_phi=True)
    y_model = (-dphi_dq[:,0].numpy()*(r_c - x_model))**0.5
    ax_m.plot(r_c - x_model, y_model)
    ax_m.set_title('model')

    # Plot vcric for the training data
    y_train_mean = binned_statistic(x_train, y_train, statistic=np.mean, bins=[xbins]).statistic
    ax_t.plot(xbins[:-1], np.abs(y_train_mean), drawstyle='steps-post')
    ax_t.set_title('training data')

    # Plot vcirc for the normalizing flow (for its sample)
    y_sample_mean = binned_statistic(x_sample, y_sample, statistic=np.mean, bins=[xbins]).statistic
    ax_s.plot(xbins[:-1], np.abs(y_sample_mean), drawstyle='steps-post')
    ax_s.set_title('normalizing flow')

    for ax in (ax_m, ax_t, ax_s):
        ax.set_xlabel('$R\mathrm{\ [kpc]}$', labelpad=0)
    ax_m.set_ylabel('$v_\mathrm{circ}\mathrm{\ [100 km/s]}$')

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'phi_flow_vcirc_marginals.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return


def plot_vcirc_2d_slice(phi_model, coords_train, coords_sample, fig_dir, dim1, dim2, dimz, z, padding=0.95, attrs=None, fig_fmt=('svg',)):
    labels = [
        '$x\mathrm{\ [kpc]}$', '$y\mathrm{\ [kpc]}$', '$z\mathrm{\ [kpc]}$',
    ]
    keys = [
        'x', 'y', 'z'
    ]
    
    labels = {k:l for k,l in zip(keys,labels)}
    ikeys = {k:i for i, k in enumerate(keys)}
    
    for dim in [dim1, dim2]:
        if dim not in keys:
            raise ValueError(f'dimension {dim} not supported')
    
    fig,all_axs = plt.subplots(
        2, 3,
        figsize=(6,2.2),
        dpi=200,
        gridspec_kw=dict(width_ratios=[2,2,2], height_ratios=[0.2, 2])
    )
    axs = all_axs[1,:]
    caxs = all_axs[0,:]
    
    # Get the plot limits
    xmin, xmax, ymin, ymax = 0, 0, 0, 0
    if attrs is not None:
        r_inner, r_outer = 1/attrs['parallax_max'], 1/attrs['parallax_min'] # [kpc], [kpc]
        xmin, xmax, ymin, ymax = -r_outer, r_outer, -r_outer, r_outer
        
        # Visualise the boundaries
        cartesian_keys = ['x', 'y', 'z']
        kw = dict(linestyle=(0, (5, 3)), lw=0.5, color='black')
        if (dim1 in cartesian_keys) and (dim2 in cartesian_keys):
            # Plot circles
            for ax in axs:
                circ = plt.Circle((0, 0), r_inner, fill=False, **kw)
                ax.add_patch(circ)
                circ = plt.Circle((0, 0), r_outer, fill=False, **kw)
                ax.add_patch(circ)
        if dim1 in ['R']:
            for ax in axs:
                ax.axvline(r_inner, **kw)
                ax.axvline(r_outer, **kw)
        if dim2 in ['R']:
            for ax in axs:
                ax.axhline(r_inner, **kw)
                ax.axhline(r_outer, **kw)
    xlim, ylim = (xmin, xmax), (ymin, ymax)
    
    grid_size = 256
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    X, Y = np.meshgrid(0.5*(x[1:]+x[:-1]), 0.5*(y[1:]+y[:-1]))
    
    # Mask for the area for which v_circ is plotted
    R2 = X*X+Y*Y+z**2
    #mask = ((R2 > r_outer**2) | (R2 < r_inner**2))
    mask = (R2 > r_outer**2*padding**2) | (R2 < r_inner**2/padding**2)
    
    q_grid = np.full(shape=(X.size, 3), fill_value=z, dtype='f4')
    q_grid[:,ikeys[dim1]] = X.ravel()
    q_grid[:,ikeys[dim2]] = Y.ravel()
    
    _,dphi_dq,_ = potential_tf.calc_phi_derivatives(
        phi_model['phi'], q_grid, return_phi=True
    )
    offset = np.array((phi_model['fs']._r_c.numpy(), 0, 0))
    vcirc_img = np.reshape(np.sum(dphi_dq*(q_grid - offset), axis=1)**0.5, X.shape) # 100 km/s
    vcirc_img = np.ma.masked_where(mask, vcirc_img)
    
    # Plot v_circ
    min_val, max_val = vcirc_img.min(), vcirc_img.max()

    kw = dict(cmap='viridis', vmin=min_val, vmax=max_val, shading='flat', lw=0, rasterized=True)
    hh = axs[0].pcolormesh(x, y, vcirc_img, **kw)
    # Set the colorbar
    cb_p = fig.colorbar(hh, cax=caxs[0], orientation='horizontal')
    cb_p.ax.xaxis.set_ticks_position('top')
    cb_p.ax.locator_params(nbins=3)
    caxs[0].set_title(r'$v_\mathrm{circ, model}\mathrm{\ [100km/s]}$')


    x_bins = np.linspace(xmin, xmax, 32)
    y_bins = np.linspace(ymin, ymax, 32)

    # Plot a slice of training data
    dz = 0.05
    idx_train = (coords_train[dimz] > z - dz) & (coords_train[dimz] < z + dz)
    dim1_train = coords_train[dim1][idx_train]
    dim2_train = coords_train[dim2][idx_train]
    vcirc_train = coords_train['cylvT'][idx_train]
    ret = binned_statistic_2d(dim1_train, dim2_train, vcirc_train, statistic=np.mean, bins=[x_bins, y_bins])
    im = axs[1].imshow(np.abs(ret.statistic.T), origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='viridis')
    cb_t = fig.colorbar(im, cax=caxs[1], orientation='horizontal')
    cb_t.ax.xaxis.set_ticks_position('top')
    cb_t.ax.locator_params(nbins=5)
    caxs[1].set_title(r'$v_\mathrm{circ, train}\mathrm{\ [100km/s]}$')
    
    # Plot a slice of the normalizing flow sample
    dz = 0.05
    idx_train = (coords_sample[dimz] > z - dz) & (coords_sample[dimz] < z + dz)
    dim1_train = coords_sample[dim1][idx_train]
    dim2_train = coords_sample[dim2][idx_train]
    vcirc_train = coords_sample['cylvT'][idx_train]
    ret = binned_statistic_2d(dim1_train, dim2_train, vcirc_train, statistic=np.mean, bins=[x_bins, y_bins])
    im = axs[2].imshow(np.abs(ret.statistic.T), origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='viridis')
    cb_t = fig.colorbar(im, cax=caxs[2], orientation='horizontal')
    cb_t.ax.xaxis.set_ticks_position('top')
    cb_t.ax.locator_params(nbins=5)
    caxs[2].set_title(r'$v_\mathrm{circ, sample}\mathrm{\ [100km/s]}$')
    
    for ax in axs:
        ax.set_xlabel(labels[dim1], labelpad=0)
    axs[0].set_ylabel(labels[dim2], labelpad=2)
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])

    for fmt in fig_fmt:
        fname = os.path.join(fig_dir, f'phi_flow_vcirc_2d_slice_{dim1}_{dim2}.{fmt}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return


def main():
    """
    Plots different diagnostics for the potential (with frameshift) and flows for Gaia populations.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Deep Potential: Plot different diagnostics for a potential derived from Gaia DR3.',
        add_help=True
    )
    parser.add_argument(
        '-i', '--input',
        type=str, required=True,
        help='Filename of input training data (particle phase-space coords).'
    )
    parser.add_argument(
        '--potential',
        type=str,
        required=True,
        help='Potential model filename. Can be either checkpoint dir or *.index in that checkpoint dir.'
    )
    parser.add_argument(
        '--df-grads-fname',
        type=str, default='data/df_gradients.h5',
        help='Directory in which to store the flow samples (positions and f gradients).'
    )
    parser.add_argument(
        '--fig-dir',
        type=str,
        default='plots',
        help='Directory to put figures in.'
    )
    parser.add_argument(
        '--spherical-origin',
        type=float,
        nargs=3,
        default=(0.,0.,0.),
        help='Origin of coordinate system for spherical coordinates in (x,y,z) to subtract from coords.'
    )
    parser.add_argument(
        '--cylindrical-origin',
        type=float,
        nargs=3,
        default=(8.3,0.,0.),
        help='Origin of coordinate system for cylindrical coordinates in (x,y,z) to subtract from coords.'
    )
    parser.add_argument(
        '--fig-fmt',
        type=str,
        nargs='+',
        default=('svg',),
        help='Formats in which to save figures (svg, png, pdf, etc.).'
    )
    parser.add_argument(
        '--dark',
        action='store_true',
        help='Use dark background for figures.'
    )
    parser.add_argument(
        '--autosave',
        action='store_true',
        help='Automatically saves/loads samples and chooses fig dir. Incompatible with fig-dir.\
            The saving location is in plots/ in a subdir deduced from the potential directory.'
    )
    args = parser.parse_args()

    # Load in the custom style sheet for scientific plotting
    plt.style.use('scientific')
    if args.dark:
        plt.style.use('dark_background')

    if args.autosave:
        # Infer the place to store the samples and figures
        fname_pot = args.potential
        if os.path.isdir(fname_pot):
            fname_index = tf.train.latest_checkpoint(fname_pot)
            fname_loss_pdf = fname_index + '_loss.pdf'
            fig_dir = 'plots/Gaia_' + fname_index[fname_index.find('models/') + 7:] + '/'
        else:
            fname_loss_pdf = fname_pot[:-6] + '_loss.pdf'
            fig_dir = 'plots/Gaia_' + fname_pot[fname_pot.find('models/') + 7:fname_pot.rfind('.index')] + '/'
        
        print(fname_loss_pdf, os.path.isfile(fname_loss_pdf))
        if os.path.isfile(fname_loss_pdf):
            # Copy the latest loss over to the plots dir
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(fname_loss_pdf, fig_dir)
            shutil.copy(fname_loss_pdf[:-4] + '_noreg.pdf', fig_dir)

        args.fig_dir = fig_dir

    print('Loading training data ...')
    eta_train, attrs_train = plot_flow_projections.load_training_data(args.input, True)
    n_train = eta_train.shape[0]
    print(f'  --> Training data shape = {eta_train.shape}')
    
    print('Loading potential')
    phi_model = plot_potential.load_potential(args.potential)
    if phi_model['fs'] is not None:
        phi_model['fs'].debug()


    print('Calculating cylindrical & spherical coordinates ...')
    coords_train = plot_flow_projections.calc_coords(eta_train, args.spherical_origin, args.cylindrical_origin)

    print('Plotting 2D slices of matter density ...')
    dims = [
        ('x', 'y', 'z', 0.),
        ('x', 'z', 'y', 0.),
        ('y', 'z', 'x', 0.),
    ]
    for dim1, dim2, dimz, z in dims:
        print(f'  --> ({dim1}, {dim2})')
        plot_rho(phi_model, coords_train, args.fig_dir, dim1, dim2, dimz, z, padding=0.95, attrs=attrs_train, fig_fmt=args.fig_fmt)

    
    print('Plotting 2D slices of forces ...')
    dims = [
        ('x', 'y', 'z', 0.),
        ('x', 'z', 'y', 0.),
        ('y', 'z', 'x', 0.),
    ]
    for dim1, dim2, dimz, z in dims:
        print(f'  --> ({dim1}, {dim2})')
        plot_force_2d_slice(phi_model, args.fig_dir, dim1, dim2, dimz, z, padding=0.95, attrs=attrs_train, fig_fmt=args.fig_fmt)


    print('Plotting 1D slices of forces ...')
    dims = [
        ('x', 'y', 0, 0, 'x'),
        ('y', 'z', 0, 0, 'y'),
        ('z', 'x', 0, 0, 'z'),
    ]
    for dim1, dimy, y, z, dimforce in dims:
        print(f'  --> ({dim1}, {dimy}={y}, {z})')
        plot_force_1d_slice(phi_model, args.fig_dir, dim1, dimy, y, z, dimforce, padding=0.95, attrs=attrs_train, fig_fmt=args.fig_fmt)


    # Extra diagnostics if flow samples are also passed
    if os.path.isfile(args.df_grads_fname):
        df_data = fit_all.load_df_data(args.df_grads_fname)
        #coords_sample = plot_flow_projections.calc_coords(df_data['eta'], args.spherical_origin, args.cylindrical_origin)

        print('Plotting marginals of v_circ ...')
        model_u0 = np.array((phi_model['fs']._u_x.numpy(), phi_model['fs']._u_y.numpy(), phi_model['fs']._u_z.numpy()))
        model_r_c = phi_model['fs']._r_c.numpy()
        eta_gc_train = eta_train.copy()
        eta_gc_train[:, 3:] -= model_u0
        eta_gc_sample = df_data['eta'].copy()
        eta_gc_sample[:, 3:] -= model_u0
        coords_gc_sample = plot_flow_projections.calc_coords(eta_gc_sample, args.spherical_origin, args.cylindrical_origin)
        coords_gc_train = plot_flow_projections.calc_coords(eta_gc_train, args.spherical_origin, args.cylindrical_origin)
        plot_vcirc_marginals(phi_model, coords_gc_train, coords_gc_sample, fig_dir, attrs=attrs_train, fig_fmt=args.fig_fmt)

        print('Plotting 2D slices of v_circ ...')
        dims = [
            ('x', 'y', 'z', 0),
            ('x', 'z', 'y', 0),
            ('y', 'z', 'x', 0),
        ]
        for dim1, dim2, dimz, z in dims:
            print(f'  --> ({dim1}, {dim2}, {dimz}={z})')
            plot_vcirc_2d_slice(phi_model, coords_gc_train, coords_gc_sample, fig_dir, dim1, dim2, dimz, z, padding=0.95, attrs=attrs_train, fig_fmt=args.fig_fmt)

        print('Plotting 2D slices of \partial f/\partial t ...')
        dims = [
            ('x', 'y'),
            ('x', 'z'),
            ('y', 'z'),
        ]
        print('  Calculating Phi gradients (might take a while) ...')
        _, dphi_dq,_ = potential_tf.calc_phi_derivatives(phi_model['phi'], df_data['eta'][:,:3], return_phi=True)
        for dim1, dim2 in dims:
            print(f'  --> ({dim1}, {dim2})')
            plot_dfdt_2d_marginal(phi_model, df_data, dphi_dq, args.fig_dir, dim1, dim2, padding=0.95, attrs=attrs_train, fig_fmt=args.fig_fmt)
    else:
        print("Couldn't find df gradients.")

    print('Plotting frameshift parameters evolution (might take a while) ...')
    plot_potential.plot_frameshift_params(args.potential, args.fig_dir, args.fig_fmt)
    
    return 0


if __name__ == '__main__':
    main()

