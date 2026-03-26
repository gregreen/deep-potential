import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import h5py


def load_flow_samples(filename):
    with h5py.File(filename, 'r') as f:
        data = {key: f[key][:] for key in f.keys()}
    return data


def get_bins(data, n_bins=50):
    min_val, max_val = np.nanpercentile(data, [10, 90])
    return np.linspace(min_val, max_val, n_bins)


def std(x, axis=None):
    # return percentile based std
    return 0.5 * (np.nanpercentile(x, 84, axis=axis) - np.nanpercentile(x, 16, axis=axis))


def _block_average_df_data(df_datas, block_size):
    """
    Average consecutive blocks of flow data.

    Args:
        df_datas: List of dictionaries containing 'f' and 'df_deta' arrays
        block_size: Number of consecutive flows to average

    Returns:
        List of block-averaged dictionaries
    """
    df_datas_reduced = []
    for i in range(0, len(df_datas), block_size):
        df_datas_reduced.append({
            'f': np.mean(np.stack([x['f'] for x in df_datas[i:i + block_size]], axis=1), axis=1),
            'df_deta': np.mean(np.stack([x['df_deta'] for x in df_datas[i:i + block_size]], axis=2), axis=2),
        })
    return df_datas_reduced


def _compute_deltas(df_datas_reduced, mode='cyclic', df_data_reference=None):
    """
    Compute differences between flow data.

    Args:
        df_datas_reduced: Block-averaged flow data
        mode: 'cyclic' for consecutive comparisons, 'reference' for reference-based
        df_data_reference: Reference data (required for 'reference' mode)

    Returns:
        List of delta dictionaries
    """
    delta_f = []
    for i in range(len(df_datas_reduced)):
        x = df_datas_reduced[i]

        if mode == 'cyclic':
            i_n = (i + 1) % len(df_datas_reduced)
            x_ref = df_datas_reduced[i_n]
            delta_f.append({
                'f': x_ref['f'] / x['f'] - 1,
                'df_deta': x_ref['df_deta'] / x['df_deta'] - 1,
            })
        else:  # reference mode
            delta_f.append({
                'f': df_data_reference['f'] / x['f'] - 1,
                'df_deta': x['df_deta'] / df_data_reference['df_deta'] - 1,
            })
    return delta_f


def _plot_gradient_histograms(delta_f, df_datas_reduced, block_size, mode, fig_dir, fig_fmt):
    """
    Create and save histogram plots of gradient differences.

    Args:
        delta_f: List of delta dictionaries
        df_datas_reduced: Block-averaged data (for loop count)
        block_size: Block size used for averaging
        mode: 'cyclic' or 'reference'
        fig_dir: Output directory
        fig_fmt: Tuple of output formats
    """
    fig = plt.figure(figsize=(9, 9), layout='compressed')
    gs = GridSpec(3, 3, figure=fig)

    ax_top = fig.add_subplot(gs[0, :])
    axs = [ax_top] + [fig.add_subplot(gs[r, c]) for r in range(1, 3) for c in range(3)]

    # Plot for f
    df = np.stack([x['f'] for x in delta_f], axis=-1)
    std_df = std(df, axis=0)
    ax = axs[0]
    for i in range(df.shape[1]):
        ax.hist(df[:,i], bins=get_bins(df.flatten()), histtype='step', density=True)
    ax.set_title(f'Mean Standard Deviation = {std_df.mean():.3f}')
    ax.set_xlabel(r'$\Delta f / f$')

    # Plot for each dimension of df_deta
    labels = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z']
    for j in range(6):
        df_deta_dim = np.stack([x['df_deta'][:,j] for x in delta_f], axis=1)
        std_df_deta_dim = std(df_deta_dim, axis=0)
        ax = axs[1 + j]
        for i in range(len(df_datas_reduced)):
            ax.hist(df_deta_dim[:,i], bins=get_bins(df_deta_dim.flatten()), histtype='step', density=True)
        ax.set_title(f'Mean Std Dev = {std_df_deta_dim.mean():.3f}')
        ax.set_xlabel(f'$\Delta(\partial f / \partial {labels[j]}) / (\partial f / \partial {labels[j]})$')

    for ax in axs:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.grid(which='both', alpha=0.2)

    if mode == 'cyclic':
        plt.suptitle(f'Cyclic differences of gradients (averaged over k={block_size} flows)')
        fname_prefix = f'k={block_size}_cyclic'
    else:
        plt.suptitle(f'Difference between block-averaged gradients (averaged over k={block_size} flows) and reference flow')
        fname_prefix = f'k={block_size}_reference'

    if fig_dir is not None:
        for fmt in fig_fmt:
            fname = fig_dir / f'{fname_prefix}_df_gradients_comparison.{fmt}'
            fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)


def plot_df_gradients_comparison(df_datas, block_size, fig_dir, fig_fmt=('png',),
                                 mode='reference', df_data_reference=None):
    """
    Generates and saves a plot comparing gradients of the distribution function.

    Args:
        df_datas: List of dictionaries, each containing 'eta', 'df_deta', and 'f' arrays
        block_size: Number of consecutive flows to average before computing differences
        fig_dir: Directory to save the output plot
        fig_fmt: Tuple of output formats (e.g., ('png',))
        mode: 'cyclic' for consecutive comparisons, 'reference' for reference-based
        df_data_reference: Reference data (required for 'reference' mode)
    """
    if mode == 'reference' and df_data_reference is None:
        raise ValueError("df_data_reference is required when mode='reference'")

    df_datas_reduced = _block_average_df_data(df_datas, block_size)
    delta_f = _compute_deltas(df_datas_reduced, mode, df_data_reference)
    _plot_gradient_histograms(delta_f, df_datas_reduced, block_size, mode, fig_dir, fig_fmt)


def plot_df_gradients_cyclic_comparison(df_datas, block_size, fig_dir, fig_fmt=('png',)):
    """
    Generates cyclic comparison plots. Wrapper for backward compatibility.

    Args:
        df_datas: List of dictionaries containing flow data
        block_size: Number of consecutive flows to average
        fig_dir: Output directory
        fig_fmt: Output formats
    """
    plot_df_gradients_comparison(df_datas, block_size, fig_dir, fig_fmt, mode='cyclic')


def expand_inputs(inputs):
    files = []
    for inp in inputs:
        # Case 1: pattern with explicit numeric wildcard
        if "*" in inp:
            files += glob.glob(inp)
        # Case 2: single filename
        else:
            files.append(inp)
    return sorted(set(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot cyclic differences of distribution function gradients from multiple flow files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--df-grads-fname-pattern',
        nargs='+',  # Accepts one or more arguments and gathers them into a list. [1, 2]
        required=True,
        help='Filename pattern(s) to load the individual flow gradients from. Can be a glob pattern.'
    )
    parser.add_argument(
        '--reference-df-grads-fname',
        type=str,
        required=False,
        default=None,
        help='Filename to load a reference flow gradient from.'
    )
    parser.add_argument(
        '--no-cyclic-comparison',
        action='store_true',
        help='If set, do not generate the cyclic comparison plots.'
    )
    parser.add_argument(
        '--fig-dir',
        type=str,
        default='.',
        help='Directory in which to save the output plot.'
    )

    args = parser.parse_args()

    # Expand file patterns and load data
    fnames = expand_inputs(args.df_grads_fname_pattern)
    if not fnames:
        print(f"Error: No files found matching the pattern: {' '.join(args.df_grads_fname_pattern)}")
        exit(1)

    print(f"Loading {len(fnames)} files...")
    df_datas = [load_flow_samples(f) for f in fnames]

    # Prepare output directory and filename
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_grads = len(df_datas)
    # Select powers of two that remain at least two times smaller than n_grads
    block_sizes = [2**i for i in range(int(np.log2(n_grads))) if 2**i * 2 <= n_grads]

    if not args.no_cyclic_comparison:
        for block_size in block_sizes:
            plot_df_gradients_cyclic_comparison(df_datas, block_size, fig_dir, fig_fmt=('png',))

    if args.reference_df_grads_fname is not None:
        ref_data = load_flow_samples(args.reference_df_grads_fname)
        for block_size in block_sizes:
            plot_df_gradients_comparison(df_datas, block_size, fig_dir, fig_fmt=('png',),
                                         mode='reference', df_data_reference=ref_data)
