from tqdm import tqdm
import time
import numpy as np

import utils
import fit_all

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from pathlib import Path
import glob


def value_and_grad_fn(model, eta_batch):
    return jax.vmap(eqx.filter_value_and_grad(model.log_prob))(eta_batch)


def sample_from_different_flows(
    key,
    flow_list,
    attrs_list,
    n_samples,
    sample_batch_size=5000,
):
    """
    Returns a combined sample from different flows, while respecting their own
    spatial boundaries. When getting the averaged differentials at a point,
    only flows are counted whose training data are complete in that volume.
    """
    tot_n = sum([attrs["n"] for attrs in attrs_list])
    nflow_samples = [(attrs["n"] * n_samples) // tot_n for attrs in attrs_list]
    nflow_samples[0] += n_samples - sum(nflow_samples)  # Fix off by one due to rounding

    # Do ceiling divide
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    nflow_batches = [
        -(-nflow_samples[i] // sample_batch_size) for i in range(len(nflow_samples))
    ]
    total_iters = sum(nflow_batches)

    eta = []
    print("Sampling eta..")

    with tqdm(total=total_iters, desc="Sampling flows") as pbar:
        for i, flow in enumerate(flow_list):
            attrs = attrs_list[i]

            n_processed = 0
            while True:
                key, sample_key = jax.random.split(key)
                eta_sample = np.array(flow.sample(sample_key, sample_batch_size))
                # Reject samples that are outside the range of validity

                if attrs["has_spatial_cut"]:
                    idx = utils.get_index_of_points_inside_attrs(eta_sample, attrs)
                    idx_size = eta_sample[idx].shape[0]
                    n_keep = min(nflow_samples[i] - n_processed, idx_size)
                    n_processed += n_keep

                    eta.append(eta_sample[idx][:n_keep])
                else:
                    n_keep = min(nflow_samples[i] - n_processed, eta_sample.shape[0])
                    eta_sample = eta_sample[:n_keep]
                    n_processed += n_keep

                    eta.append(eta_sample)
                # If we have enough samples, we can stop
                if n_processed >= nflow_samples[i]:
                    break

                pbar.update(1)
            jax.clear_caches()

    # All eta will have at least one flow in their volume of validity
    eta = np.concatenate(eta, axis=0)
    print(f'Shape of sampled eta: {eta.shape}')

    return eta


def calculate_gradients(eta, flow_list, attrs_list, grad_batch_size=500):
    # Do ceiling divide
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    print("Sampling gradients of eta..")
    n_batches = -(-len(eta) // grad_batch_size)
    df_deta_list = np.zeros((len(flow_list),) + eta.shape, dtype="f4")
    f_list = np.zeros((len(flow_list), len(eta)), dtype="f4")

    with tqdm(total=n_batches * len(flow_list), desc="Calculating f, df/deta") as pbar:
        for i, flow in enumerate(flow_list):
            df_deta = []
            prob = []
            for k in range(n_batches):
                eta_batch = eta[k * grad_batch_size: (k + 1) * grad_batch_size]
                logp_batch, dlogp_deta_batch = value_and_grad_fn(flow, eta_batch)
                prob.append(np.exp(logp_batch))
                df_deta.append(dlogp_deta_batch * np.exp(logp_batch)[:, None])

                pbar.update(1)

            df_deta = np.concatenate([np.array(b) for b in df_deta])
            prob = np.concatenate([np.array(b) for b in prob])

            if attrs_list[i]["has_spatial_cut"]:
                print(f'Flow {i+1} has spatial cut. Applying it to gradients and probabilities.')
                # Replace all out-of-bounds values with np.nan. That way, np.nanmean ignores them.
                idx = utils.get_index_of_points_inside_attrs(eta, attrs_list[i])
                df_deta[~idx] = np.nan
                prob[~idx] = np.nan

            df_deta_list[i] = df_deta
            f_list[i] = prob

    return f_list, df_deta_list


def std(x, axis=None):
    # return percentile based std
    return 0.5 * (np.nanpercentile(x, 84, axis=axis) - np.nanpercentile(x, 16, axis=axis))


def robust_mean(data, sigma=5.0, max_iter=5, axis=None, use_mad=True):
    """
    Compute a robust mean by sigma-clipping outliers along a given axis.

    Parameters
    ----------
    data : array_like
        Input array.
    sigma : float, optional
        Clipping threshold in standard deviations. Default is 5.
    max_iter : int, optional
        Maximum number of iterations. Default is 5.
    axis : int or None, optional
        Axis along which to compute the mean. If None, flatten the array.
    use_mad : bool, optional
        If True, use median/MAD for robust scale estimation.
        If False, use mean/std.

    Returns
    -------
    clipped_mean : ndarray
        Robust mean computed along the specified axis.
    mask : ndarray (bool)
        Boolean mask of values kept (same shape as input).
    """
    data = np.asanyarray(data, dtype=float)
    mask = np.isfinite(data)

    # Initialize working copy
    clipped = np.where(mask, data, np.nan)

    for _ in range(max_iter):
        if use_mad:
            center = np.nanmedian(clipped, axis=axis, keepdims=True)
            scale = 1.4826 * np.nanmedian(np.abs(clipped - center), axis=axis, keepdims=True)
        else:
            center = np.nanmean(clipped, axis=axis, keepdims=True)
            scale = np.nanstd(clipped, axis=axis, keepdims=True)

        # Avoid division by zero
        scale = np.where(scale == 0, np.nan, scale)

        new_mask = np.abs(clipped - center) <= sigma * scale
        new_mask &= np.isfinite(clipped)

        # Stop if mask is unchanged
        if np.all(new_mask == mask):
            break

        mask = new_mask
        clipped = np.where(mask, data, np.nan)

    # Compute final robust mean
    clipped_mean = np.nanmean(clipped, axis=axis)

    return clipped_mean, mask


def combine_gradients(f_list, df_data_list):
    """
    Returns the combined gradients and probabilities over multiple flows. We do this
    by taking the mean of the gradients and probabilities at each point, ignoring
    out-of-bounds values (which are already set to np.nan).

    Before combining the flows, we perform outlier rejection based on how similar the flows are to
    the, initial, averaged flow. This is done by calculating the standard deviation of the deviation
    of the densities from the averaged density, and rejecting the flows which deviate too much.
    """

    f_best = np.nanmean(f_list, axis=0)
    df_deta_best = np.nanmean(df_data_list, axis=0)

    std_f = std(np.stack(f_list, axis=1) / f_best[:, None] - 1, axis=0)
    deviation = std_f / np.nanmedian(std_f)
    threshold = 1.15  # Reject flows that deviate more than this factor times the median deviation
    idx_rejected = deviation > threshold
    n_rejected = np.sum(idx_rejected)
    print(f'Rejecting {n_rejected} out of {f_list.shape[0]} flows based on deviation threshold {threshold}.')
    print(f'Rejected flows: {np.where(idx_rejected)[0]}')

    f_best = robust_mean(f_list[~idx_rejected], axis=0)[0]
    df_deta_best = robust_mean(df_data_list[~idx_rejected], axis=0)[0]

    return f_best, df_deta_best


def sample_and_differentiate_from_different_flows(
    key,
    flow_list,
    attrs_list,
    n_samples,
    grad_batch_size=500,
    sample_batch_size=5000,
):
    """
    Returns a combined sample from different flows, while respecting their own
    spatial boundaries. When getting the averaged differentials at a point,
    only flows are counted whose training data are complete in that volume.
    """
    eta = sample_from_different_flows(key, flow_list, attrs_list, n_samples,
                                      grad_batch_size, sample_batch_size)

    f_list, df_deta_list = calculate_gradients(eta, flow_list, attrs_list, grad_batch_size)

    f, df_deta = combine_gradients(f_list, df_deta_list)

    ret = {"eta": eta, "df_deta": df_deta, "f": f}
    return ret


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Deep Potential: Fit potential from phase-space samples.",
        add_help=True,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str,
                        required=False, help="Input data.")
    parser.add_argument(
        "--flow-dir-pattern",
        nargs="+",  # allows multiple arguments or one pattern
        default=None,
        help="Filename pattern to load the flows from",
    )
    parser.add_argument(
        "--df-samples-fname",
        type=str,
        default=None,
        help="Filename in which to store the combined samples of all of the flow",
    )
    parser.add_argument(
        "--df-grads-fname-pattern",
        nargs="+",  # allows multiple arguments or one pattern
        type=str,
        default=None,
        help="Filename in which to store the individual gradients of one of the flow",
    )
    parser.add_argument(
        "--df-grads-final-fname",
        type=str,
        default=None,
        help="Filename in which to store the combined gradients of all of the flows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="JSON with kwargs.",
        default="options.json"
    )
    args = parser.parse_args()

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

    n_flows = 1
    if args.flow_dir_pattern is not None:
        n_flows = len(expand_inputs(args.flow_dir_pattern))
    elif args.df_grads_fname_pattern is not None:
        n_flows = len(expand_inputs(args.df_grads_fname_pattern))

    # 1. Load in the data and attributes
    if args.input is not None:
        _, attrs = utils.load_training_data(args.input)
        attrs_list = [attrs for _ in range(n_flows)]

    # 2. Sample from flows. Save the combined sample
    if args.df_samples_fname is not None:
        df_samples_fname = Path(args.df_samples_fname)
        if not df_samples_fname.exists():
            # Load in the flows
            if args.flow_dir_pattern is None:
                raise ValueError("If df_samples_fname is specified, flow_dir_pattern must also be specified.")
            print(f'Loading in flows from {args.flow_dir_pattern}')
            flow_fnames = expand_inputs(args.flow_dir_pattern)
            flow_list = [fit_all.load_flow(fname) for fname in flow_fnames]

            key = jax.random.key(args.seed)
            params = utils.load_params(args.params)

            print('Extracting samples from flows')
            eta = sample_from_different_flows(key, flow_list, attrs_list, params['flow_sampling']['n_samples'], params['flow_sampling']['sample_batch_size'])

            # Save the samples
            print(f'Saving samples to {df_samples_fname}')
            fit_all.save_df_data({'eta': eta}, df_samples_fname)

    # 3. Calculate gradients from flows. Save the individual gradients
    if args.df_grads_fname_pattern is not None and args.flow_dir_pattern is not None and args.df_samples_fname is not None:
        # If flow_list does not exist, create it
        if 'flow_list' not in locals():
            print(f'Loading in flows from {args.flow_dir_pattern}')
            flow_fnames = expand_inputs(args.flow_dir_pattern)
            flow_list = [fit_all.load_flow(fname) for fname in flow_fnames]

        df_grads_fname_list = expand_inputs(args.df_grads_fname_pattern)
        if len(df_grads_fname_list) != n_flows:
            raise ValueError("Number of files matching df_grads_fname_pattern must be equal to number of flows.")

        for i in range(n_flows):
            df_grads_fname = Path(df_grads_fname_list[i])
            if not df_grads_fname.exists():
                # Load in the samples
                if 'eta' not in locals():
                    print(f'Loading in samples from {args.df_samples_fname}')
                    df_sample = utils.load_flow_samples(args.df_samples_fname)
                    eta = df_sample['eta']

                print(f'Calculating gradients for flow {i+1}/{n_flows}')
                params = utils.load_params(args.params)
                grad_batch_size = params['flow_sampling']['grad_batch_size']
                f_list, df_deta_list = calculate_gradients(eta, [flow_list[i]], [attrs_list[i]], grad_batch_size)

                print(f'Saving gradients to {df_grads_fname}')
                fit_all.save_df_data({'eta': eta, 'f': f_list[0], 'df_deta': df_deta_list[0]}, df_grads_fname)

    # 4. Combine gradients from flows. Save the combined gradients
    if args.df_grads_final_fname is not None:
        df_grads_final_fname = Path(args.df_grads_final_fname)
        if not df_grads_final_fname.exists():
            if 'f_list' not in locals() or 'df_deta_list' not in locals():
                if args.df_grads_fname_pattern is None:
                    raise ValueError("If df_grads_final_fname is specified, df_grads_fname_pattern must also be specified.")
                print(f'Loading in individual gradients from {args.df_grads_fname_pattern}')
                df_grads_fname_list = expand_inputs(args.df_grads_fname_pattern)
                if len(df_grads_fname_list) != n_flows:
                    raise ValueError("Number of files matching df_grads_fname_pattern must be equal to number of flows.")

                f_list = []
                df_deta_list = []
                for i in range(n_flows):
                    df_grads_fname = Path(df_grads_fname_list[i])
                    if not df_grads_fname.exists():
                        raise ValueError(f"File {df_grads_fname} does not exist.")
                    df_data = utils.load_flow_samples(df_grads_fname)
                    f_list.append(df_data['f'])
                    df_deta_list.append(df_data['df_deta'])
                    if i == 0:
                        eta = df_data['eta']
                f_list = np.array(f_list)
                df_deta_list = np.array(df_deta_list)

            print('Combining gradients')
            f, df_deta = combine_gradients(f_list, df_deta_list)

            print(f'Saving combined gradients to {df_grads_final_fname}')
            df_grads_final_fname.parent.mkdir(parents=True, exist_ok=True)
            fit_all.save_df_data({'eta': eta, 'f': f, 'df_deta': df_deta}, df_grads_final_fname)
