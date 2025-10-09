from __future__ import print_function, division

import matplotlib.colors as colors
import matplotlib.ticker as ticker

import json
import glob
import numpy as np
import scipy.ndimage
import pandas as pd
import os
import h5py
from pathlib import Path
import time

import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm


def warmup_cosine_restarts_schedule(
    init_value: float = 0.0,
    peak_value: float = 1e-4,
    warmup_steps: int = 1000,
    cycle_length: int = 10000,
    num_cycles: int = 3,
    decay_factor: float = 0.8,
    min_value_ratio: float = 0.1
):
    import jax.numpy as jnp
    """
    Create a learning rate schedule with multiple warmup-cosine cycles.

    Args:
        init_value: Initial learning rate (usually 0.0)
        peak_value: Peak learning rate for the first cycle
        warmup_steps: Steps for warmup in each cycle
        cycle_length: Total steps per cycle
        num_cycles: Number of cycles to run
        decay_factor: Factor to reduce peak_value each cycle (0.8 means 80% of previous)
        min_value_ratio: Minimum LR as ratio of peak_value in each cycle

    Returns:
        Schedule function that takes step and returns learning rate
    """

    def schedule_fn(step: int) -> float:
        # Determine which cycle we're in
        cycle_step = step % cycle_length
        cycle_num = step // cycle_length

        # Clamp cycle number to prevent going beyond num_cycles
        cycle_num = jnp.minimum(cycle_num, num_cycles - 1)

        # Calculate peak value for current cycle
        current_peak = peak_value * (decay_factor ** cycle_num)
        current_min = current_peak * min_value_ratio

        # Within cycle: warmup then cosine decay
        warmup_lr = init_value + (current_peak - init_value) * cycle_step / warmup_steps

        # Cosine decay phase
        decay_steps = cycle_length - warmup_steps
        decay_progress = (cycle_step - warmup_steps) / decay_steps
        cosine_lr = current_min + (current_peak - current_min) * 0.5 * (1 + jnp.cos(jnp.pi * decay_progress))

        # Choose based on position in cycle
        lr = jnp.where(cycle_step < warmup_steps, warmup_lr, cosine_lr)

        # After all cycles, maintain minimum learning rate
        final_lr = current_min
        return jnp.where(step < num_cycles * cycle_length, lr, final_lr)

    return schedule_fn


class TimeLogger:
    def __init__(self):
        self.times = {}
        self.start_times = {}  # Track start time for each key

    def start(self, key):
        if key not in self.times:
            self.times[key] = 0
        self.start_times[key] = time.time()

    def stop(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer for key '{key}' was not started")
        elapsed = time.time() - self.start_times[key]
        self.times[key] += elapsed
        del self.start_times[key]

    def get_duration(self, key, default_value=0.0):
        return self.times.get(key, default_value)


def load_training_data(fname, cut_attrs=False):
    """
    Loads in the training data, in the form of a (n, d)-dimensional numpy array
    where n is the number of datapoints and d the dimensionality. The attributes,
    specifying relevant metadata for the training data are also loaded in.

    In the case where a training and validation split have already been done,
    those are loaded in as well. (this is necessary when there are duplicate
    datapoints due to upsampling, usually in order to account for the non-
    uniformity of the selection function).

    Inputs:
        fnanme (str): File path to be loaded in. Specifies a hdf5 group
        cut_attrs (bool): Whether the data should be cut to obey the spatial
            extent specified by attrs. This is relevant when handling padded
            data (padding is done for helping flow training by avoiding sharp
            cut-offs).
    """
    def get_inside_index(eta, attrs):
        # Don't perform a cut if there are no attrs
        if attrs == {}:
            return np.ones(len(eta), dtype=bool)

        # Cuts eta based on attrs
        r2 = np.sum(eta[:, :3] ** 2, axis=1)
        R2 = np.sum(eta[:, :2] ** 2, axis=1)
        r = r2**0.5
        R = R2**0.5
        z = eta[:, 2]

        if "volume_type" not in attrs or attrs["volume_type"] == "sphere":
            if "r_out" in attrs:
                r_out = attrs["r_out"]
            else:
                r_out = 1 / attrs["parallax_min"]
            if "r_in" in attrs:
                r_in = attrs["r_in"]
            else:
                r_in = 1 / attrs["parallax_max"]
            idx = (r2 > r_in**2) & (r2 < r_out**2)
        elif attrs["volume_type"] == "cylinder":
            R_out, H_out = attrs["R_out"], attrs["H_out"]
            if "r_in" in attrs:
                r_in = attrs["r_in"]
                idx = (r >= r_in) & (R <= R_out) & (np.abs(z) <= H_out)
            if ("R_in" in attrs) and ("H_in" in attrs):
                R_in, H_in = attrs["R_in"], attrs["H_in"]
                idx = (
                    ((R >= R_in) | (np.abs(z) >= H_in)) &
                    (R <= R_out) &
                    (np.abs(z) <= H_out)
                )

        return idx

    _, ext = os.path.splitext(fname)
    attrs = None
    data = {}
    if ext in (".h5", ".hdf5"):
        with h5py.File(fname, "r") as f:
            attrs = dict(f["eta"].attrs.items())

            data["eta"] = f["eta"][:].astype("f4")
            if "weights" in f.keys():
                data["weights"] = f["weights"][:].astype("f4")

            # Check if the train-validation split has been passed
            if "eta_train" in f.keys() and "eta_val" in f.keys():
                print("Loaded in training and validation data")
                data["eta_train"] = f["eta_train"][:].astype("f4")
                data["eta_val"] = f["eta_val"][:].astype("f4")
            if "weights_train" in f.keys() and "weights_val" in f.keys():
                data["weights_train"] = f["weights_train"][:].astype("f4")
                data["weights_val"] = f["weights_val"][:].astype("f4")

            if cut_attrs:
                if "weights" in data:
                    data["weights"] = data["weights"][get_inside_index(data["eta"], attrs)]
                if "weights_train" in data and "weights_val" in data:
                    data["weights_train"] = data["weights_train"][get_inside_index(data["eta_train"], attrs)]
                    data["weights_val"] = data["weights_val"][get_inside_index(data["eta_val"], attrs)]
                if "eta_train" in data and "eta_val" in data:
                    data["eta_train"] = data["eta_train"][get_inside_index(data["eta_train"], attrs)]
                    data["eta_val"] = data["eta_val"][get_inside_index(data["eta_val"], attrs)]
                data["eta"] = data["eta"][get_inside_index(data["eta"], attrs)]

            attrs["has_spatial_cut"] = True if attrs != {} else False
            attrs["n"] = len(data["eta"])
    elif ext == '.npz':
        with np.load(fname) as d:
            data['eta'] = np.array(d["x"])
            data['weights'] = np.array(d["weights"], dtype=np.float32)
        attrs = {'has_spatial_cut': False, 'n': len(data['eta'])}
    else:
        raise ValueError(f'Unrecognized input file extension: "{ext}"')

    return data, attrs


def split_data(data, val_split=0.25):
    """Splits the data into training and validation sets."""
    n_total = data['eta'].shape[0]
    n_val = int(n_total * val_split)

    train_data = {"eta": data["eta"][n_val:], "weights": data["weights"][n_val:]}
    val_data = {"eta": data["eta"][:n_val], "weights": data["weights"][:n_val]}
    return train_data, val_data


# --- Data Loading ---
def load_and_split_data(path: Path, val_split: float = 0.25):
    """Loads data, splits into train/validation, and converts to jax arrays."""
    print(f"Loading data from {path}...")
    data, attrs = load_training_data(path)
    return split_data(data, val_split)


def get_mask(l_, b, r, max_distance, hp=None):
    '''
    Returns if a point with coordinates l, b, r is inside the mask defined by max_distance.
    Max distance is a healpixel map indicating the maximal distance for each healpixel.
    '''
    from astropy_healpix import HEALPix
    from astropy import units as u
    if hp is None:
        nside = int((max_distance.shape[0]/12)**0.5)
        hp = HEALPix(nside=nside, order='nested')
    hpix_idx = hp.lonlat_to_healpix(l_*u.deg, b*u.deg)
    return max_distance[hpix_idx] > r, hp


def load_mask(fname):
    with h5py.File(fname, "r") as f:
        max_distance = f['max_distance'][:].astype('f4')
        nside = f['max_distance'].attrs['nside']
    return max_distance, nside


def get_mask_eta(eta, fname_mask, hp=None, r_min=None, r_max=None):
    '''
    Returns if a point with coordinates l, b, r is inside the mask defined by max_distance.
    Max distance is a healpixel map indicating the maximal distance for each healpixel.
    '''
    r = np.sum(eta[:, :3]**2, axis=1)**0.5
    if fname_mask is None:
        mask, hp = np.ones(len(eta), dtype=bool), None
    else:
        max_distance, nside = load_mask(fname_mask)

        l_ = np.arctan2(eta[:, 1], eta[:, 0]) * 180 / np.pi
        b = np.arcsin(eta[:, 2] / r) * 180 / np.pi
        mask, hp = get_mask(l_, b, r, max_distance, hp)
    if r_min is not None:
        mask = mask & (r > r_min)
    if r_max is not None:
        mask = mask & (r < r_max)
    return mask, hp


def load_flow_samples(fname, recalc_avg=None, attrs_to_cut_by=None):
    """
    Loads in the flow samples from a file, and optionally recalculates the
    average flow from several different realizations of the flow. If
    attrs_to_cut_by is specified, eta is cut to only include points inside the
    volume of validity specified by attrs_to_cut_by.
    """
    d = {}
    with h5py.File(fname, "r") as f:
        for k in f.keys():
            d[k] = f[k][:].astype("f4")

    if recalc_avg == "mean":
        d["df_deta"] = clipped_vector_mean(d["df_deta_indiv"])
        if "probs_indiv" in d:
            d["probs"] = clipped_vector_mean(d["probs_indiv"])
    elif recalc_avg == "median":
        d["df_deta"] = np.median(d["df_deta_indiv"], axis=0)
        if "probs_indiv" in d:
            d["probs"] = np.median(d["probs_indiv"])

    if attrs_to_cut_by is not None:
        # Cuts eta based on attrs
        idx = get_index_of_points_inside_attrs(d["eta"], attrs_to_cut_by)

        d["df_deta"] = d["df_deta"][idx]
        d["eta"] = d["eta"][idx]
        if "probs" in d:
            d["probs"] = d["probs"][idx]

    return d


def get_model_values(phi_model, q_eval,
                     batch_size=131072,
                     fname=None,
                     disable_tqdm=False,
                     convert_rho_to_msunpc3=True):
    """
    Calculate the potential, acceleration and density implied by the
    differentiable model of the potential. If specified, the results are saved to a file
    in a subdirectory of fig_dir, named data.
    Currently, the spatial dimension is expected to be in units of kpc
    and velocity dimension 100 km/s. By default, the conversion factor for rho
    is chosen for it to return density in M_sun/pc^3. Acceleration is in units of
    (100 km/s)^2/kpc

    Parameters:
        phi_model (dict): The model of the potential to be used.
        q_eval (np.ndarray): An array of shape (n, 3) specifying where to evaluate the potential
        fig_dir (str): The directory where the data is to be saved
        fname (str): The name of the file where the data is to be saved
        batch_size (int): The batch size to be used when evaluating the model
        disable_tqdm (bool): Whether to disable the tqdm progress bar
        convert_rho_to_msunpc3 (bool): Whether to convert the density to M_sun/pc^3
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    @eqx.filter_jit
    def calc_phi_derivatives(phi_func, q):
        """Calculates higher order derivatives of the potential at q."""
        phi = phi_func(q)
        dphi_dq = jax.grad(phi_func)(q)

        # Laplacian of phi
        hessian_phi = jax.hessian(phi_func)(q)
        d2phi_dq2 = jnp.trace(hessian_phi)

        return phi, dphi_dq, d2phi_dq2
    # If phi_model is an instance of class PotentialModel, then extract the phi function
    if hasattr(phi_model, 'phi_model'):
        phi_model = phi_model.phi_model
    # vmap the function w.r.t. q
    jit_phi_derivs = jax.vmap(calc_phi_derivatives, in_axes=(None, 0))

    n0 = len(q_eval)

    if fname is not None:
        fname = Path(fname)
        # Append the number of datapoints at the end of the file name
        fname = fname.with_name(f"{fname.stem}_{n0}{'.npz'}")

    if fname is None or (fname is not None and not fname.exists()):
        rhos = []
        accs = []
        phis = []

        if convert_rho_to_msunpc3:
            rho_conv = 2.325  # (100 km/s)^2/kpc^2 in units of M_sun/pc^3
        else:
            rho_conv = 1.0

        for i in (pbar := tqdm(range(0, n0, batch_size), disable=disable_tqdm)):
            b = q_eval[i: i + batch_size]

            phi,dphi_dq,d2phi_dq2 = jit_phi_derivs(
                phi_model, b
            )
            rhos.append(rho_conv*d2phi_dq2/(4*np.pi)) # [M_Sun/pc^3]
            accs.append(-dphi_dq) # [(100 km/s)^2/kpc]
            phis.append(phi)

            pbar.set_description("Calculating model values")
        rhos = np.concatenate(rhos)
        accs = np.concatenate(accs)
        phis = np.concatenate(phis)
        if fname is not None:
            # Make sure save_path directory exists
            fname.parent.mkdir(parents=True, exist_ok=True)
            np.savez(fname, phi=phis, acc=accs, rho=rhos)
    else:
        npzfile = np.load(fname)
        rhos = npzfile['rho']
        accs = npzfile['acc']
        phis = npzfile['phi']

    return phis, accs, rhos


def get_selfn_values(selfn_model, q_eval,
                     batch_size=131072,
                     fname=None,
                     disable_tqdm=False,):
    """
    Calculate the selection function implied by the model.
    If specified, the results are saved to a file in a subdirectory of fig_dir,
    named data.
    Currently, the spatial dimension is expected to be in units of kpc
    and velocity dimension 100 km/s. By default, the conversion factor for rho
    is chosen for it to return density in M_sun/pc^3. Acceleration is in units of
    (100 km/s)^2/kpc

    Parameters:
        selfn_model (eqx.Module): The model of the selection function to be used.
        q_eval (np.ndarray): An array of shape (n, 3) specifying where to evaluate the selection function
        fig_dir (str): The directory where the data is to be saved
        fname (str): The name of the file where the data is to be saved
        batch_size (int): The batch size to be used when evaluating the model
        disable_tqdm (bool): Whether to disable the tqdm progress bar
        convert_rho_to_msunpc3 (bool): Whether to convert the density to M_sun/pc^3
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    @eqx.filter_jit
    def calc_selfn(selfn, q):
        return jnp.exp(selfn(q))

    # If selfn_model is an instance of class PotentialModel, then extract the selfn function
    if hasattr(selfn_model, 'log_selection_function_model'):
        selfn_model = selfn_model.log_selection_function_model
    # vmap the function w.r.t. q
    jit_calc_selfn = jax.vmap(calc_selfn, in_axes=(None, 0))

    n0 = len(q_eval)

    if fname is not None:
        fname = Path(fname)
        # Append the number of datapoints at the end of the file name
        fname = fname.with_name(f"{fname.stem}_{n0}{'.npz'}")

    if fname is None or (fname is not None and not fname.exists()):
        selfns = []

        for i in (pbar := tqdm(range(0, n0, batch_size), disable=disable_tqdm)):
            b = q_eval[i: i + batch_size]

            selfns.append(jit_calc_selfn(selfn_model, b))

            pbar.set_description("Calculating selfn values")
        selfns = np.concatenate(selfns)
        if fname is not None:
            # Make sure save_path directory exists
            fname.parent.mkdir(parents=True, exist_ok=True)
            np.savez(fname, selfn=selfns)
    else:
        npzfile = np.load(fname)
        selfns = npzfile['selfn']

    return selfns


def clipped_vector_mean(v_samp, clip_threshold=5, rounds=5, **kwargs):
    """
    Antiquated function, this is used for multiple flow samples
    """
    n_samp, n_point, n_dim = v_samp.shape

    # Mean vector: shape = (point, dim)
    v_mean = np.mean(v_samp, axis=0)

    for i in range(rounds):
        # Difference from mean: shape = (sample, point)
        dv_samp = np.linalg.norm(v_samp - v_mean[None], axis=2)
        # Identify outliers: shape = (sample, point)
        idx = dv_samp > clip_threshold * np.median(dv_samp, axis=0)[None]
        # Construct masked array with outliers masked
        mask_bad = np.repeat(np.reshape(idx, idx.shape + (1,)), n_dim, axis=2)
        v_samp_ma = np.ma.masked_array(v_samp, mask=mask_bad)
        # Take mean of masked array
        v_mean = np.ma.mean(v_samp_ma, axis=0)

    return v_mean


def plot_loss(train_loss_hist, val_loss_hist=None, lr_hist=None, smoothing="auto"):
    """
    Plots the loss history for the training set (train_loss_hist) and validation set
    (val_loss_hist) and marks where the learning rate dropped (based on lr_hist)
    'significantly'. Draws two views, one for the whole history, the other
    for the last 50%.
    """
    if smoothing == "auto":
        n_smooth = np.clip(len(train_loss_hist) // 16, 4, 128)
    else:
        n_smooth = smoothing

    def smooth_time_series(x):
        w = np.kaiser(2 * n_smooth, 5)
        w /= np.sum(w)
        x_conv = scipy.ndimage.convolve(x, w, mode="reflect")
        return x_conv

    train_loss_conv = smooth_time_series(train_loss_hist)
    if val_loss_hist is not None:
        val_loss_conv = smooth_time_series(val_loss_hist)

    n = np.arange(len(train_loss_hist))

    # Detect discrete drops in learning rate
    if lr_hist is not None:
        lr_hist = np.array(lr_hist)
        lr_ratio = lr_hist[lr_hist > 0][1:] / lr_hist[lr_hist > 0][:-1]
        n_drop = np.where(lr_ratio < 0.95)[0]

    fig, ax_arr = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(left=0.14, right=0.98, wspace=0.25)

    for i, ax in enumerate(ax_arr):
        if i == 1:
            i0 = len(train_loss_hist) // 2
            train_loss_hist = train_loss_hist[i0:]
            train_loss_conv = train_loss_conv[i0:]
            if val_loss_hist is not None:
                val_loss_hist = val_loss_hist[i0:]
                val_loss_conv = val_loss_conv[i0:]
            if lr_hist is not None:
                lr_hist = lr_hist[i0:]
            n = n[i0:]

        if lr_hist is not None:
            for k in n_drop:
                ax.axvline(k, c="k", alpha=0.1, ls="--")

        (l,) = ax.plot(
            n, train_loss_hist, alpha=0.1, label=r"$\mathrm{training\ loss}$"
        )
        ax.plot(
            n,
            train_loss_conv,
            alpha=0.8,
            color=l.get_color(),
            label=r"$\mathrm{training\ loss\ (smoothed)}$",
        )
        if val_loss_hist is not None:
            ax.plot(
                n,
                val_loss_conv,
                alpha=0.8,
                label=r"$\mathrm{validation\ loss\ (smoothed)}$",
            )

        ax.set_xlim(n[0], n[-1])
        if i == 1:
            # Choose the y-limit as the 2nd and 98th percentile of the training
            # and validation smoothed loss, with 10% padding
            limit_percent = 2, 98
            ylim = np.percentile(train_loss_conv, limit_percent)
            if val_loss_hist is not None:
                ylim_val = np.percentile(val_loss_conv, limit_percent)
                ylim = (min(ylim[0], ylim_val[0]), max(ylim[1], ylim_val[1]))
            ylim = (
                ylim[0] - 0.1 * (ylim[1] - ylim[0]),
                ylim[1] + 0.1 * (ylim[1] - ylim[0]),
            )
            ax.set_ylim(*ylim)

        ax.grid("on", which="major", alpha=0.25)
        ax.grid("on", which="minor", alpha=0.05)
        ax.set_ylabel(r"$\mathrm{training\ loss}$")
        ax.set_xlabel(r"$\mathrm{training\ step}$")
        if i == 0:
            if val_loss_hist is not None:
                # Rearrange the legend so validation is above training loss.
                # This is because validation lines in general are above training
                # in the plot.
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    [handles[0], handles[2], handles[1]],
                    [labels[0], labels[2], labels[1]],
                    loc="upper right",
                )
            else:
                ax.legend(loc="upper right")
        else:
            kw = dict(
                fontsize=8,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", alpha=0.2, facecolor="white"),
            )
            if val_loss_hist is not None:
                ax.text(
                    0.95,
                    0.95,
                    f"$\mathrm{{validation\ loss\ final\ (smoothed)}} = \
                        {val_loss_conv[-1]:.4f}$\n$\mathrm{{training\ loss\ \
                        final\ (smoothed)}} = {train_loss_conv[-1]:.4f}$",
                    **kw,
                )
            else:
                ax.text(
                    0.95,
                    0.95,
                    f"$\mathrm{{training\ loss\ final\ (smoothed)}} = \
                        {train_loss_conv[-1]:.4f}$",
                    **kw,
                )

    return fig


def plot_loss_new(loss_history, fname, final_train_loss=None, final_val_loss=None, share_loss_axis=True, w=10):
    # Save loss curve
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), layout='compressed')

    def moving_average(x, w):
        ret = np.convolve(x, np.ones(w), 'valid') / w
        # Pad both sides with zeros
        pad_left = (w - 1) // 2
        pad_right = w - 1 - pad_left
        return np.pad(ret, (pad_left, pad_right), mode='edge')
    if final_train_loss is None:
        final_train_loss = loss_history["train"][-1]
    if final_val_loss is None:
        final_val_loss = loss_history["val"][-1]

    alpha = 0.8
    for i, ax in enumerate(axs):
        x = np.arange(len(loss_history["train"])) + 0.5
        ax.plot(x, moving_average(loss_history["train"], w), label="Train Loss", alpha=alpha)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        # Also plot lr_history on the sample plot but with secondary y-axis
        if "lr" in loss_history:
            ax2 = ax.twinx()
            ax2.plot(x, loss_history["lr"], label="Learning Rate", color='grey', linestyle='--', alpha=0.4)
            #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
            ax2.legend(loc='upper center', frameon=False)
            ax2.tick_params(axis='y', which='both', left=False, right=False)
            ax2.set_yticklabels([])
            #ax2.set_ylabel('Learning rate')
        if not share_loss_axis:
            ax3 = ax.twinx()
            ax3.plot(x, moving_average(loss_history["val"], w), label="Validation Loss", color='tab:orange', alpha=alpha)
            ax3.set_ylabel('Validation loss')
            ax3.legend(loc=(0.7, 0.85), frameon=False)
        else:
            ax.plot(x, moving_average(loss_history["val"], w), label="Validation Loss", color='tab:orange', alpha=alpha)
            ax.set_ylabel("Loss")

        ax.legend(loc=(0.7, 0.75), frameon=False)
        ax.grid(True)

        if i == 1:
            lims_train = np.percentile(loss_history["train"], [5, 95])
            lims_val = np.percentile(loss_history["val"], [5, 95])
            if not share_loss_axis:
                ax.set_ylim(lims_train)
                ax3.set_ylim(lims_val)
            else:
                ax.set_ylim((min(lims_train[0], lims_val[0]), max(lims_train[1], lims_val[1])))

    if final_val_loss is not None:
        axs[0].set_title(f"Final Val Loss: {final_val_loss}. Smoothing w={w} epochs.")
    if final_train_loss is not None:
        axs[1].set_title(f"Final Train Loss: {final_train_loss}")
    # Set ylim for second axis to be 95% percentile of the loss values
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved loss plot to {fname}")


def save_loss_history(
    fname,
    loss_history,
    val_loss_history=None,
    lr_history=None,
    loss_noreg_history=None,
    val_loss_noreg_history=None,
):
    data = {"loss": loss_history}
    if val_loss_history is not None:
        data["validation_loss"] = val_loss_history
    if lr_history is not None:
        data["learning_rate"] = lr_history
    if loss_noreg_history is not None:
        data["loss_noreg"] = loss_noreg_history
    if val_loss_noreg_history is not None:
        data["validation_loss_noreg"] = val_loss_noreg_history

    pd.DataFrame(data).to_csv(fname, index=False, float_format="%.12e")


def append_to_potential_params_history(phi, fs, potential_params_hist={}):
    """
    Append the current values of the potential parameters to the history.
    """
    parameters_to_check = [
        ("omega", "_omega"),
        ("u_y0", "_u_y"),
        ("u_x0", "_u_x"),
        ("u_z0", "_u_z"),
        ("mn1_amp", "_mn1_logamp"),
        ("mn1_a", "_mn1_loga"),
        ("mn1_b", "_mn1_logb"),
        ("mn2_amp", "_mn2_logamp"),
        ("mn2_a", "_mn2_loga"),
        ("mn2_b", "_mn2_logb"),
        ("mn3_amp", "_mn3_logamp"),
        ("mn3_a", "_mn3_loga"),
        ("mn3_b", "_mn3_logb"),
        ("halo_amp", "_halo_logamp"),
        ("halo_a", "_halo_loga"),
        ("bulge_amp", "_bulge_logamp"),
        ("bulge_rcut", "_bulge_logrcut"),
        ("bulge_alpha", "_bulge_alpha"),
        ("dz", "_dz"),
        ("fourier_A0", "_A"),
        ("fourier_B0", "_B"),
        ("fourier_C", "_C"),
    ]

    for param_name, attr in parameters_to_check:
        value = None
        if hasattr(fs, attr):
            value = getattr(fs, attr).numpy()
        if hasattr(phi, attr):
            value = getattr(phi, attr).numpy()
        if value is not None:
            if param_name not in potential_params_hist:
                potential_params_hist[param_name] = []
            if "log" in attr:
                value = np.exp(value)
            if type(value) is np.ndarray:
                potential_params_hist[param_name].append(value[0])
            else:
                potential_params_hist[param_name].append(value)

    return potential_params_hist


def save_potential_params_history(fname, potential_params_hist, lr_history=None):
    """
    Save the history of the potential parameters to a CSV file for later analysis.
    """
    df = pd.DataFrame(dict(potential_params_hist, lr_hist=lr_history))
    df.to_csv(fname, index=False, float_format="%.8e")


def load_potential_params(fname, remove_lr=False):
    """
    Load the history of the potential parameters from a CSV file.
    """
    df = pd.read_csv(fname)
    if remove_lr:
        df = df.drop(columns=["lr_hist"])
    return df.to_dict("list")


def load_params(fname):
    if fname is not None:
        with open(fname, "r") as f:
            return json.load(f)


def load_loss_history(fname):
    df = pd.read_csv(fname)
    train_loss_history = list(df["loss"].values)
    val_loss_history = (
        list(df["validation_loss"].values) if "validation_loss" in df.columns else None
    )
    lr_history = (
        list(df["learning_rate"].values) if "learning_rate" in df.columns else None
    )
    train_loss_noreg_history = (
        list(df["loss_noreg"].values) if "loss_noreg" in df.columns else None
    )
    val_loss_noreg_history = (
        list(df["validation_loss_noreg"].values)
        if "validation_loss_noreg" in df.columns
        else None
    )

    return (
        train_loss_history,
        val_loss_history,
        lr_history,
        train_loss_noreg_history,
        val_loss_noreg_history,
    )


def plot_corr(
    ax,
    x,
    y,
    x_lim=None,
    d_max=None,
    bins=(50, 31),
    pct=(16, 50, 84),
    normalization="balanced",
):
    if x_lim is None:
        x_min, x_max = np.min(x), np.max(x)
        # w = x_max - x_min
        xlim = (x_min, x_max)
    else:
        xlim = x_lim

    if d_max is None:
        dmax = 1.2 * np.percentile(np.abs(y - x), 99.9)
    else:
        dmax = d_max
    dlim = (-dmax, dmax)

    d = y - x
    n, x_edges, _ = np.histogram2d(x, d, range=(xlim, dlim), bins=bins)

    if normalization == None:
        norm = np.ones(n.shape[0])
    elif normalization == "sum":
        norm = np.sum(n, axis=1) + 1.0e-10
    elif normalization == "max":
        norm = np.max(n, axis=1) + 1.0e-10
    elif normalization == "balanced":
        norm0 = np.sum(n, axis=1)
        norm1 = np.max(n, axis=1)
        norm = np.sqrt(norm0 * norm1) + 1.0e-10
    else:
        raise ValueError(f'Unrecognized normalization: "{normalization}"')
    n /= norm[:, None]

    # n = n**gamma

    ax.imshow(
        n.T,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=xlim + dlim,
        cmap="binary",
    )
    ax.plot(xlim, [0.0, 0.0], c="b", alpha=0.2, lw=1)

    if len(pct):
        x_pct = np.empty((3, len(x_edges) - 1))
        for i, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            idx = (x > x0) & (x < x1)
            if np.any(idx):
                x_pct[:, i] = np.percentile(d[idx], pct)
            else:
                x_pct[:, i] = np.nan

        for i, x_env in enumerate(x_pct):
            ax.step(x_edges, np.hstack([x_env[0], x_env]), c="cyan", alpha=0.5)

    ax.set_xlim(xlim)
    ax.set_ylim(dlim)


def hist2d_mean(ax, x, y, c, vmin=None, vmax=None, cmap=None, bins=10, range=None):
    kw = dict(bins=bins, range=range, density=False)
    nc, xedges, yedges = np.histogram2d(x, y, weights=c, **kw)
    n, _, _ = np.histogram2d(x, y, **kw)
    img = nc / n

    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

    im = ax.imshow(
        img.T,
        extent=extent,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    return im


def get_index_of_points_inside_attrs(eta, attrs, r=None, R=None, z=None):
    """
    Attrs defines a volume, the inside of which is considered fully complete and is used for training the potential.
    This function returns the index of points which lie inside the volume defined by attrs.
    """
    if r is None or R is None or z is None:
        r = np.sum(eta[:, :3] ** 2, axis=1) ** 0.5
        R = np.sum(eta[:, :2] ** 2, axis=1) ** 0.5
        z = eta[:, 2]
    if "volume_type" not in attrs or attrs["volume_type"] == "sphere":
        if "r_out" in attrs:
            r_out = attrs["r_out"]
        else:
            r_out = 1 / attrs["parallax_min"]
        if "r_in" in attrs:
            r_in = attrs["r_in"]
        else:
            r_in = 1 / attrs["parallax_max"]
        idx = (r >= r_in) & (r <= r_out)
    elif attrs["volume_type"] == "cylinder":
        R_out, H_out = attrs["R_out"], attrs["H_out"]
        if "r_in" in attrs:
            r_in = attrs["r_in"]
            idx = (r >= r_in) & (R <= R_out) & (np.abs(z) <= H_out)
        if ("R_in" in attrs) and ("H_in" in attrs):
            R_in, H_in = attrs["R_in"], attrs["H_in"]
            idx = (
                ((R >= R_in) | (np.abs(z) >= H_in))
                & (R <= R_out)
                & (np.abs(z) <= H_out)
            )
    return idx


def calc_coords(eta, spherical_origin=(0,0,0), cylindrical_origin=(0,0,0), spherical_vel_origin=(0,0,0), cylindrical_vel_origin=(0,0,0), vector_field=None):
    """Calculate components in different coordinate systems. If a vector field is specified, then the function
    returns the components of the vector field in Cartesian, Spherical, and Cylindrical coordinates. This assumes
    that the positions of the vector fields values are specified by eta (eta is in Cartesian). If vector_field is
    None, then the function returns the coordinates of eta in different coordinates.

    Cartesian coordinates: x, y, z, vx, vy, vz
    Spherical coordiantes: r, cos(theta), phi, v_radial, v_theta (v_phi is missing)
    Cylindrical coordinates: cyl_R, cyl_z, cyl_phi, cyl_vR, cyl_vz, cyl_vT
    """
    def dot(a, b):
        # a and b are of shape (n, 3)
        return np.sum(a*b, axis=1)

    sph_x0 = np.array(spherical_origin)
    cyl_x0 = np.array(cylindrical_origin)
    sph_v0 = np.array(spherical_vel_origin)
    cyl_v0 = np.array(cylindrical_vel_origin)

    if vector_field is not None:
        # Cartesian
        zeros = np.zeros((len(eta), 1))
        cart = {
            "x": vector_field[:,0],
            "y": vector_field[:,1],
            "z": vector_field[:,2],
        }

        # Cylindrical
        x, y, z = np.split(eta[:,:3] - cyl_x0, 3, axis=1)
        R = (x**2 + y**2)**0.5
        e_cyl_R = np.concatenate([x, y, zeros], axis=1)/R
        e_cyl_z = np.concatenate([zeros, zeros, np.ones((len(x), 1))], axis=1)
        e_cyl_phi = np.concatenate([-y, x, zeros], axis=1)/R
        cyl = {
            "cylR": dot(e_cyl_R, vector_field),
            "cylz": dot(e_cyl_z, vector_field),
            "cylphi": dot(e_cyl_phi, vector_field),
        }

        # Spherical
        x, y, z = np.split(eta[:,:3] - sph_x0, 3, axis=1)
        R = (x**2 + y**2)**0.5
        r = (x**2 + y**2 + z**2)**0.5
        e_sph_r = np.concatenate([x, y, z], axis=1)/r
        e_sph_th = np.concatenate([x*z, y*z, -R**2], axis=1)/r/R
        e_sph_phi = np.concatenate([-y, x, zeros], axis=1)/R
        sph = {
            "sphR": dot(e_sph_r, vector_field),
            "sphth": np.nan_to_num(dot(e_sph_th, vector_field)),
            "sphphi": dot(e_sph_phi, vector_field),
        }
    else:
        # Cartesian
        x = eta[:, 0] - sph_x0[0]
        y = eta[:, 1] - sph_x0[1]
        z = eta[:, 2] - sph_x0[2]
        cart = {"x": x, "y": y, "z": z}

        has_velocities = eta.shape[1] == 6

        # Cylindrical
        cyl_R = np.linalg.norm(eta[:, :2] - cyl_x0[:2], axis=1)
        cyl_z = eta[:, 2] - cyl_x0[2]
        cyl_phi = np.arctan2(eta[:, 1] - cyl_x0[1], eta[:, 0] - cyl_x0[0])
        # Convert cyl_phi to be between 0 and 2pi
        cyl_phi = np.mod(cyl_phi, 2*np.pi)
        cyl_cos_phi = (eta[:, 0] - cyl_x0[0]) / cyl_R
        cyl_sin_phi = (eta[:, 1] - cyl_x0[1]) / cyl_R

        cyl = {
            "cylR": cyl_R,
            "cylz": cyl_z,
            "cylphi": cyl_phi,
        }

        # Spherical
        r = np.linalg.norm(eta[:, :3] - sph_x0, axis=1)
        costheta = z / r
        sph_R = np.linalg.norm(eta[:, :2] - sph_x0[:2], axis=1)
        phi = np.arctan2(eta[:, 1] - sph_x0[1], eta[:, 0] - sph_x0[0])

        sph = {"r": r, "cth": np.nan_to_num(costheta), "phi": phi}

        if has_velocities:
            # Cartesian
            vz = eta[:, 5]
            cart = {**cart, **{"vx": eta[:, 3], "vy": eta[:, 4], "vz": vz}}

            # Cylindrical
            vx = eta[:, 3] - cyl_v0[0]
            vy = eta[:, 4] - cyl_v0[1]
            vz = eta[:, 5] - cyl_v0[2]
            cyl_vR = vx * cyl_cos_phi + vy * cyl_sin_phi
            cyl_vT = -vx * cyl_sin_phi + vy * cyl_cos_phi
            cyl_vz = vz
            cyl = {**cyl, **{"cylvR": cyl_vR, "cylvT": cyl_vT, "cylvz": cyl_vz}}

            # Spherical
            vx = eta[:, 3] - sph_v0[0]
            vy = eta[:, 4] - sph_v0[1]
            vz = eta[:, 5] - sph_v0[2]
            vr = np.sum((eta[:, :3] - sph_x0) * (eta[:, 3:] - sph_v0), axis=1) / r
            vth = (z * vr - r * vz) / sph_R
            cos_phi = (eta[:, 0] - sph_x0[0]) / cyl_R
            sin_phi = (eta[:, 1] - sph_x0[1]) / cyl_R
            vT = -vx * sin_phi + vy * cos_phi
            sph = {**sph, **{"vth": vth, "vT": vT, "vr": vr}}

    return dict(**cart, **cyl, **sph)
