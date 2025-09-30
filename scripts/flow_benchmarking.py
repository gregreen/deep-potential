from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pathlib import Path
import json

import utils


def plot_simple_1d_marginal(coords_sample, coords_train, weights_train, *dims, lims=None, n_rows=1):
    """
    Plots 1D marginal distributions for a given set of dimensions, including a z-score (pull) plot
    for each marginal to compare the 'data' and 'flow' distributions.

    Args:
        coords_sample (dict): Dictionary of coordinate arrays for the sampled data (the "flow").
        coords_train (dict): Dictionary of coordinate arrays for the training data (the "data").
        weights_train (array): Weights corresponding to the training data.
        *dims (str): Variable number of dimension keys to plot (e.g., 'x', 'vy').
        lims (list of tuples, optional): A list of (min, max) tuples for the x-axis of each plot.
                                         If None, limits are determined automatically. Defaults to None.
        n_rows (int, optional): The number of rows to arrange the plots in. Defaults to 1.
    """
    labels = [
        '$R\mathrm{\ (kpc)}$', '$z\mathrm{\ (kpc)}$', r'cylindrical $\phi$ (deg)', '$v_R\mathrm{\ (km/s)}$', '$v_z\mathrm{\ (km/s)}$', r'$v_{\phi}\mathrm{\ (km/s)}$',
        '$x\mathrm{\ (kpc)}$', '$y\mathrm{\ (kpc)}$', '$z\mathrm{\ (kpc)}$', '$v_x\mathrm{\ (km/s)}$', '$v_y\mathrm{\ (km/s)}$', '$v_z\mathrm{\ (km/s)}$',
        '$r\mathrm{\ (kpc)}$', r'$\phi\mathrm{\ (deg)}$', r'$\cos \theta$', '$v_r\mathrm{\ (km/s)}$', r'$v_{\theta}\mathrm{\ (km/s)}$', r'$v_{\phi}\mathrm{\ (km/s)}$'
    ]
    keys = [
        'cylR', 'cylz', 'cylphi', 'cylvR', 'cylvz', 'cylvT',
        'x', 'y', 'z', 'vx', 'vy', 'vz',
        'r', 'phi', 'cth', 'vr', 'vth', 'vT'
    ]
    labels = {k:l for k,l in zip(keys,labels)}

    n_cols = -(-len(dims) // n_rows)  # Ceiling division to get number of columns

    # Create a figure with a main plot and a pull plot for each dimension.
    fig, axs = plt.subplots(
        2 * n_rows, n_cols,
        figsize=(3 * n_cols, 3.5 * n_rows),
        dpi=200,
        layout='compressed',
        gridspec_kw={'height_ratios': [1, 0.4] * n_rows}
    )

    # Ensure axs is always a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axs = np.array(axs).reshape(2, 1)
    elif n_rows == 1:
        axs = axs.reshape(2, n_cols)
    elif n_cols == 1:
        axs = axs.reshape(2 * n_rows, 1)

    # --- Data Extraction and Preparation ---
    def extract_dim(dim):
        # Apply scaling coefficients for velocities and angle conversions
        coef = 1
        if dim in ['cylvR', 'cylvz', 'cylvT', 'vx', 'vy', 'vz', 'vr', 'vth', 'vT']:
            coef = 100
        if dim in ['phi', 'cylphi']:
            coef = 180 / np.pi
        if dim in ['cylvT']:
            coef *= -1
        return coef * coords_train[dim], coef * coords_sample[dim]

    x_trains = []
    x_samples = []
    for dim in dims:
        x_train, x_sample = extract_dim(dim)
        x_trains.append(x_train)
        x_samples.append(x_sample)

    # --- Automatic Limit Calculation ---
    if lims is None:
        lims = []
        for dim, x_train in zip(dims, x_trains):
            low, high = np.percentile(x_train, [1, 99])
            width = np.abs(high - low) * 0.2
            lim = [low - width, high + width]
            # Override with default limits for specific dimensions
            if dim == 'phi':
                lim = [180, -180]
            if dim == 'cth':
                lim = [-1, 1]
            if dim == 'cylR':
                lim[0] = max(0, lim[0]) # Radius cannot be negative
            if dim == 'r':
                # We typically have radius from the center of the Sun.
                # Hence we set r = 0 but this can be changed.
                lim[0] = 0
            lims.append(lim)
    lims_ordered = [[np.min(lim), np.max(lim)] for lim in lims]

    main_axs_flat = [] # To store main axes for the return value

    # --- Plotting Loop ---
    main_ylims = [[] for _ in range(n_rows)] 
    for i, (dim, x_train, x_sample) in enumerate(zip(dims, x_trains, x_samples)):
        row, col = (i // n_cols), (i % n_cols)

        # Select the main axis and the pull axis
        ax = axs[2 * row, col]
        ax_pull = axs[2 * row + 1, col]
        main_axs_flat.append(ax)

        current_lim = lims_ordered[i]
        bins = np.linspace(current_lim[0], current_lim[1], 65)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]

        # Calculate total counts/weights for normalization
        N_sample = len(x_sample)
        N_train = len(x_train) if weights_train is None else np.sum(weights_train)

        # Create histograms and normalize to get fractions per bin
        n_train_counts, _ = np.histogram(x_train, bins=bins, weights=weights_train)
        n_sample_counts, _ = np.histogram(x_sample, bins=bins)

        n_train_frac = n_train_counts / N_train
        n_sample_frac = n_sample_counts / N_sample

        # Convert fractions to density
        density_train = n_train_frac / np.max(n_train_frac)
        density_sample = n_sample_frac / np.max(n_sample_frac)

        # Plot main histograms
        ax.step(bin_centers, density_train, where='mid', color='C0', alpha=0.8, label='data')
        ax.step(bin_centers, density_sample, where='mid', color='C1', label='flow')

        # Calculate z-score (pull)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Difference in fractions
            n_diff = n_sample_frac - n_train_frac
            # Uncertainty on the difference (using Poisson error approximation for fractions)
            n_diff_sigma = np.sqrt(n_sample_frac / N_sample + n_train_frac / N_train)
            z_score = n_diff / n_diff_sigma
            z_score = np.nan_to_num(z_score) # Replace NaN/inf with 0

        # Plot pull
        ax_pull.axhline(0, color='grey', lw=0.5)
        ax_pull.scatter(bin_centers, z_score, s=1.5, marker='o', color='C1')

        # --- Axis Styling and Labeling ---
        #ax.set_yscale('log')
        #main_ylims[row].append(ax.get_ylim())
        ax.set_ylim(0, 1.05)
        ax.set_xlim(lims[i])
        ax_pull.set_xlim(lims[i])
        ax_pull.set_ylim(-5, 5) # Set a standard range for z-score

        ax.set_xticklabels([]) # Remove x-labels from main plot
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax_pull.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax_pull.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax_pull.set_xlabel(labels[dim], labelpad=0, fontsize=10)

        # Special tick locator for phi angle
        if dim == 'phi':
            ax_pull.xaxis.set_major_locator(ticker.MultipleLocator(90))

        # Set y-labels only on the leftmost column
        if col == 0:
            ax.set_ylabel('Normalized count', labelpad=2, fontsize=10)
            ax_pull.set_ylabel('Poisson significance', labelpad=2, fontsize=10)
        else:
            ax.set_yticklabels([])
            ax_pull.set_yticklabels([])

        if i == 0:
            ax.legend(fontsize=8)
    # # Set consistent y-limits for main plots in the same row
    # for row in range(n_rows):
    #     if main_ylims[row]:
    #         common_ylim = [min(ylim[0] for ylim in main_ylims[row]), max(ylim[1] for ylim in main_ylims[row])]
    #         for col in range(n_cols):
    #             axs[2 * row, col].set_ylim(common_ylim)

    # Hide any unused axes in the grid
    for j in range(len(dims), n_rows * n_cols):
        row, col = (j // n_cols), (j % n_cols)
        axs[2 * row, col].set_axis_off()
        axs[2 * row + 1, col].set_axis_off()

    return fig, main_axs_flat


def plot_simple_2d_marginal(coords_sample, coords_train, weights_train, dim1, dim2, lims=None, cmap='viridis', logscale=False):
    labels = [
        '$R$', '$z$', r'$\phi$', '$v_R\mathrm{\ (km/s)}$', '$v_z\mathrm{\ (km/s)}$', r'$v_{\phi}\mathrm{\ (km/s)}$',
        '$x\mathrm{\ (kpc)}$', '$y\mathrm{\ (kpc)}$', '$z\mathrm{\ (kpc)}$', '$v_x$', '$v_y$', '$v_z$',
        '$r$', r'$\phi\mathrm{\ (deg)}$', r'$\cos \theta$', '$v_r\mathrm{\ (km/s)}$', r'$v_{\theta}\mathrm{\ (km/s)}$', r'$v_{\phi}\mathrm{\ (km/s)}$'
    ]
    keys = [
        'cylR', 'cylz', 'cylphi', 'cylvR', 'cylvz', 'cylvT',
        'x', 'y', 'z', 'vx', 'vy', 'vz',
        'r', 'phi', 'cth', 'vr', 'vth', 'vT'
    ]

    fig, all_axs = plt.subplots(
        2, 3,
        figsize=(8, 5),
        dpi=200,
        gridspec_kw=dict(width_ratios=[1,1,1], height_ratios=[0.1,3]),
        layout='compressed'
    )
    axs = all_axs[1,:]
    caxs = all_axs[0,:]

    def extract_dims(dim):
        coef = 100 if dim in ['cylvR', 'cylvz', 'cylvT', 'vx', 'vy', 'vz'] else 1
        if dim in ['phi']:
            coef = 180/np.pi
        if dim in ['cylvT']:
            coef *= -1

        return coef*coords_train[dim], coef*coords_sample[dim]

    x_train, x_sample = extract_dims(dim1)
    y_train, y_sample = extract_dims(dim2)
    labels = {k:l for k,l in zip(keys,labels)}

    if lims is None:
        # Take the 1th and 99th percentile
        lims = [
            [np.percentile(x_train, 1), np.percentile(x_train, 99)],
            [np.percentile(y_train, 1), np.percentile(y_train, 99)]
        ]
        k = 0.2
        for lim in lims:
            w = np.abs(lim[1] - lim[0]) * k
            lim[0] -= w
            lim[1] += w
        # Flip the x-axis if it's phi
        if dim1 == 'phi':
            # lims[0] = [lims[0][1], lims[0][0]]
            lims[0] = [180, -180]
        if dim2 == 'cth':
            lims[1] = [-1, 1]
    lims_ordered = [[np.min(lims[0]), np.max(lims[0])], [np.min(lims[1]), np.max(lims[1])]]
    kw = dict(range=lims_ordered, bins=64)

    n_train = len(x_train) if weights_train is None else np.sum(weights_train)
    n_sample = len(x_sample)

    if logscale:
        kw_col = dict(cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=3000))
    else:
        kw_col = dict(cmap=cmap)

    #vmax = np.max(np.histogram2d(x_train, y_train, **kw)[0])
    vmax = None
    kw['rasterized'] = True
    nt, _, _, im = axs[0].hist2d(x_train, y_train, weights=weights_train, **kw, **kw_col, vmax=vmax, vmin=0)
    cb = fig.colorbar(im, cax=caxs[0], orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')

    ns,_,_,im = axs[1].hist2d(x_sample, y_sample, **kw, **kw_col, weights=np.full(len(x_sample), n_train/n_sample), vmax=vmax, vmin=0)
    ns *= n_sample/n_train
    cb = fig.colorbar(im, cax=caxs[1], orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')

    dn = ns/n_sample - nt/n_train
    with np.errstate(divide='ignore', invalid='ignore'):
        dn /= np.sqrt(ns * (n_train/n_sample)) / n_train
    vmax = 5.

    im = axs[2].imshow(
        dn.T,
        extent=lims_ordered[0]+lims_ordered[1],
        cmap='coolwarm_r',
        vmin=-vmax, vmax=vmax,
        origin='lower', aspect='auto',
        rasterized=True,
        interpolation='none'
    )
    cb = fig.colorbar(
        im, cax=caxs[2],
        orientation='horizontal'
    )
    cb.set_ticks([-5, -2.5, 0, 2.5, 5])
    cb.ax.xaxis.set_ticks_position('top')

    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])

    for ax in axs:
        ax.set_xlabel(labels[dim1], labelpad=0, fontsize=10)
        if dim1 == 'phi':
            ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
        ax.set_box_aspect(1)

    axs[0].set_ylabel(labels[dim2], labelpad=2, fontsize=10)

    caxs[0].set_title(r'$\mathrm{True\ Stellar\ Density}$', fontsize=10)
    caxs[1].set_title(r'$\mathrm{Normalizing\ Flow}$', fontsize=10)
    caxs[2].set_title(r'$\mathrm{Poisson\ significance} \ \left( \sigma \right)$', fontsize=10)

    if lims[0][0] > lims[0][1]:
        for ax in axs:
            ax.invert_xaxis()
    return fig, axs


@eqx.filter_jit
def batch_loss_fn(model, x_batch, weights_batch):
    log_probs = model.log_prob(x_batch)
    return jnp.sum(weights_batch * log_probs)


@eqx.filter_jit
def sample_batch_fn(model, sample_key, batch_size):
    return model.sample(sample_key, (batch_size,))


def value_and_grad_fn(model, eta_batch):
    return jax.vmap(eqx.filter_value_and_grad(model.log_prob))(eta_batch)


def benchmark(flow_model, key, time_logger, train_data, val_data, loss_history,
              spherical_origin=(0.0, 0.0, 0.0), cylindrical_origin=(8.277, 0.0, 0.0),
              fig_fmt=('png',)):
    """
    For benchmarking we do the following, while keeping track how much time each
    thing takes:
    * Compute the loss on val set
    * Compute the loss on train set
    * Draw n_samples number of samples and plot them in a couple of basic projections
      (x-y, vx-vy, z-vz, and phi-cth)
    * Calculate gradients for the first n_samples // 100 samples
    We save the results in a directory parallel to the model directory. We replace 'models'
    with 'plots'.
    """
    # Replace 'models' with 'plots'
    save_dir = Path(str(flow_model.model_dir).replace('models', 'plots'))
    # Add the checkpoint number as a subdirectory
    save_dir = save_dir / f'checkpoint_{flow_model.checkpoint_index}'
    # Make dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot the loss
    utils.plot_loss_new(loss_history, save_dir / f'flow-{flow_model.checkpoint_index}_loss.png')

    batch_size = 5000
    grad_batch_size = 500

    if val_data is not None:
        n_val_batches = (val_data["eta"].shape[0] + batch_size - 1) // batch_size
        print(f"Calculating final validation loss in batches of {batch_size}... (n_batches = {n_val_batches})")

        total_weighted_log_prob = 0.0
        time_logger.start("Benchmarking | Validation Loss Calculation")
        for i in tqdm(range(n_val_batches)):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = val_data["eta"][start_idx:end_idx]
            weights_batch = val_data["weights"][start_idx:end_idx]
            total_weighted_log_prob += batch_loss_fn(flow_model, x_batch, weights_batch)
        time_logger.stop("Benchmarking | Validation Loss Calculation")
        final_val_loss = (-total_weighted_log_prob / jnp.sum(val_data["weights"])).item()
        dt = time_logger.get_duration("Benchmarking | Validation Loss Calculation")
        print(f"Final loss on entire validation set: {final_val_loss:.6f}  (duration: {dt:.2f} s)")
    
    performance_data = {"final_val_loss": round(final_val_loss, 6)}
    fname_performance = save_dir / "performance.json"
    with open(fname_performance, "w") as f:
        json.dump(performance_data, f, indent=4)

    if train_data is not None:
        n_train_batches = (train_data["eta"].shape[0] + batch_size - 1) // batch_size
        print(f"Calculating final train loss in batches of {batch_size}... (n_batches = {n_train_batches})")

        total_weighted_log_prob = 0.0
        time_logger.start("Benchmarking | Train Loss Calculation")
        for i in tqdm(range(n_train_batches)):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = train_data["eta"][start_idx:end_idx]
            weights_batch = train_data["weights"][start_idx:end_idx]
            total_weighted_log_prob += batch_loss_fn(flow_model, x_batch, weights_batch)
        time_logger.stop("Benchmarking | Train Loss Calculation")
        final_train_loss = (-total_weighted_log_prob / jnp.sum(train_data["weights"])).item()
        dt = time_logger.get_duration("Benchmarking | Train Loss Calculation")
        print(f"Final loss on entire train set: {final_train_loss:.6f}  (duration: {dt:.2f} s)")
    
    performance_data["final_train_loss"] = round(final_train_loss, 6)
    with open(fname_performance, "w") as f:
        json.dump(performance_data, f, indent=4)

    n_samples = 100000
    n_batches = n_samples // batch_size
    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, n_batches)
    print(f"Generating {n_samples} samples in {n_batches} batches of {batch_size}...")
    time_logger.start("Benchmarking | Sampling")
    samples = []
    for i in tqdm(range(n_batches)):
        samples.append(sample_batch_fn(flow_model, sample_keys[i], batch_size))
    samples = jnp.concatenate(samples)
    time_logger.stop("Benchmarking | Sampling")
    print(f"Sampling {n_samples} points took: {time_logger.get_duration('Benchmarking | Sampling'):.4f} s")

    coords_sample = utils.calc_coords(samples, spherical_origin, cylindrical_origin)
    coords_train = utils.calc_coords(train_data["eta"], spherical_origin, cylindrical_origin)

    #
    # 1D marginals
    #

    # Cartesian projections
    fig, axs = plot_simple_1d_marginal(
        coords_sample, coords_train, train_data["weights"],
        'x', 'y', 'z', 'vx', 'vy', 'vz',
        n_rows=2
    )
    plt.savefig(save_dir / '1d_sample_density_cartesian.png', dpi=250, bbox_inches="tight")
    plt.close()

    # Spherical projections
    fig, axs = plot_simple_1d_marginal(
        coords_sample, coords_train, train_data["weights"],
        'r', 'phi', 'cth', 'vr', 'vT', 'vth',
        n_rows=2
    )
    plt.savefig(save_dir / '1d_sample_density_spherical.png', dpi=250, bbox_inches="tight")
    plt.close()

    # Cylindrical projections
    fig, axs = plot_simple_1d_marginal(
        coords_sample, coords_train, train_data["weights"],
        'cylR', 'cylz', 'cylphi', 'cylvR', 'cylvz', 'cylvT',
        n_rows=2
    )
    plt.savefig(save_dir / '1d_sample_density_cylindrical.png', dpi=250, bbox_inches="tight")
    plt.close()

    print("Saved 1d sample density plots")

    #
    # 2D marginals
    #

    # Cartesian projections
    fig, axs = plot_simple_2d_marginal(
        coords_sample, coords_train, train_data["weights"],
        dim1='x', dim2='y', cmap='viridis'
    )
    plt.savefig(save_dir / 'sample_density_x_y.png', dpi=250, bbox_inches="tight")
    plt.close()

    fig, axs = plot_simple_2d_marginal(
        coords_sample, coords_train, train_data["weights"],
        dim1='vx', dim2='vy', cmap='viridis'
    )
    plt.savefig(save_dir / 'sample_density_vx_vy.png', dpi=250, bbox_inches="tight")
    plt.close()

    fig, axs = plot_simple_2d_marginal(
        coords_sample, coords_train, train_data["weights"],
        dim1='z', dim2='vz', cmap='viridis'
    )
    plt.savefig(save_dir / 'sample_density_z_vz.png', dpi=250, bbox_inches="tight")
    plt.close()

    # Sky projections
    fig, axs = plot_simple_2d_marginal(
        coords_sample, coords_train, train_data["weights"],
        dim1='phi', dim2='cth', cmap='viridis'
    )
    plt.savefig(save_dir / 'sample_density_phi_cth.png', dpi=250, bbox_inches="tight")
    plt.close()

    # Spherical projections
    fig, axs = plot_simple_2d_marginal(
        coords_sample, coords_train, train_data["weights"],
        dim1='r', dim2='vr', cmap='viridis'
    )
    plt.savefig(save_dir / 'sample_density_r_vr.png', dpi=250, bbox_inches="tight")
    plt.close()

    fig, axs = plot_simple_2d_marginal(
        coords_sample, coords_train, train_data["weights"],
        dim1='r', dim2='phi', cmap='viridis'
    )
    plt.savefig(save_dir / 'sample_density_r_phi.png', dpi=250, bbox_inches="tight")
    plt.close()

    # Cylindrical projections
    fig, axs = plot_simple_2d_marginal(
        coords_sample, coords_train, train_data["weights"],
        dim1='cylR', dim2='cylz', cmap='viridis'
    )
    plt.savefig(save_dir / 'sample_density_R_z.png', dpi=250, bbox_inches="tight")
    plt.close()

    fig, axs = plot_simple_2d_marginal(
        coords_sample, coords_train, train_data["weights"],
        dim1='cylvz', dim2='cylvT', cmap='viridis'
    )
    plt.savefig(save_dir / 'sample_density_vz_vT.png', dpi=250, bbox_inches="tight")
    plt.close()

    print("Saved 2d sample density plots")

    # Calculating gradients
    n_samples = 1000
    n_batches = n_samples // grad_batch_size
    print(f"Calculating gradients in {n_batches} batches of {grad_batch_size}...")
    time_logger.start("Benchmarking | Gradient Calculation")
    for i in tqdm(range(n_batches)):
        eta_batch = samples[i * grad_batch_size: (i + 1) * grad_batch_size]
        logp_batch, dlogp_deta_batch = value_and_grad_fn(flow_model, eta_batch)
    time_logger.stop("Benchmarking | Gradient Calculation")
    dt = time_logger.get_duration("Benchmarking | Gradient Calculation")
    print(f"Gradient calculation took: {dt:.4f} s")

    # Save final losses and how much time everything took
    performance_data["training_time"] = time_logger.get_duration("Flow training", 1e-9)
    # Add extra timings for subcategories of Training
    for key in time_logger.times.keys():
        if key.startswith("Training | ") or key.startswith('Benchmarking | '):
            performance_data[key] = round(time_logger.get_duration(key), 3)
    # for key in time_logger.times.keys():
    #     if key.startswith("Training | "):
    #         performance_data[f'{key} %'] = performance_data[key] / training_time * 100 if training_time > 0 else 0
    # Save it as a json file
    with open(fname_performance, "w") as f:
        json.dump(performance_data, f, indent=4)
    print(f"Saved performance metrics to {fname_performance}")