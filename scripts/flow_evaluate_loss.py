#!/usr/bin/env python

from __future__ import print_function, division
from genericpath import isfile
from multiprocessing.sharedctypes import Value
from re import X

import numpy as np

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import pandas as pd

import h5py
import progressbar
import os
from glob import glob
from pathlib import Path
import shutil
from scipy.stats import binned_statistic_dd, binned_statistic_2d

import tensorflow as tf

print(f"Tensorflow version {tf.__version__}")

import utils
import fit_all


dpi = 200


def plot_loss(steps, train_loss, val_loss=None):
    """
    Plots the loss history for the training set (train_loss) and validation set
    (val_loss).
    Draws two views, one for the whole history, the other for the last 50%.
    """
    steps, train_loss = np.array(steps), np.array(train_loss)
    if val_loss is not None:
        val_loss = np.array(val_loss)

    fig, ax_arr = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(left=0.14, right=0.98, wspace=0.25)

    for i, ax in enumerate(ax_arr):
        if i == 1:
            train_loss = train_loss[steps > np.max(steps) / 2]
            if val_loss is not None:
                val_loss = val_loss[steps > np.max(steps) / 2]
            steps = steps[steps > np.max(steps) / 2]

        ax.plot(steps, train_loss, alpha=0.8, label=r"$\mathrm{training\ loss}$")
        if val_loss is not None:
            ax.plot(steps, val_loss, alpha=0.8, label=r"$\mathrm{validation\ loss}$")

        ax.set_xlim(np.min(steps), np.max(steps))
        if i == 1:
            # Choose the y-limit as the 5nd and 95th percentile of the training
            # and validation smoothed loss, with 20% padding
            limit_percent = 5, 95
            ylim = np.percentile(train_loss, limit_percent)
            if val_loss is not None:
                ylim_val = np.percentile(val_loss, limit_percent)
                ylim = (min(ylim[0], ylim_val[0]), max(ylim[1], ylim_val[1]))
            ylim = (
                ylim[0] - 0.2 * (ylim[1] - ylim[0]),
                ylim[1] + 0.2 * (ylim[1] - ylim[0]),
            )
            ax.set_ylim(*ylim)

        ax.grid("on", which="major", alpha=0.25)
        ax.grid("on", which="minor", alpha=0.05)
        ax.set_ylabel(r"$\mathrm{training\ loss}$")
        ax.set_xlabel(r"$\mathrm{training\ step}$")
        if i == 0:
            if val_loss is not None:
                # Rearrange the legend so validation is above training loss.
                # This is because validation lines in general are above training in the plot.
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    [handles[1], handles[0]], [labels[1], labels[0]], loc="upper right"
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
            if val_loss is not None:
                ax.text(
                    0.95,
                    0.95,
                    f"$\mathrm{{validation\ loss\ final}} = {val_loss[-1]:.4f}$\n$\mathrm{{training\ loss\ final}} = {train_loss[-1]:.4f}$",
                    **kw,
                )
            else:
                ax.text(
                    0.95,
                    0.95,
                    f"$\mathrm{{training\ loss\ final}} = {train_loss[-1]:.4f}$",
                    **kw,
                )

    return fig


def evaluate_loss(flow_list, eta_train, batch_size=1024):
    n_flows = len(flow_list)
    n_samples = eta_train.shape[0]

    n_batches = n_samples // batch_size
    if np.mod(n_samples, batch_size) > 0:
        n_batches += 1

    loss = []
    bar = progressbar.ProgressBar(max_value=n_batches * n_flows)

    for i, flow in enumerate(flow_list):
        loss_i = []
        weight_i = []

        @tf.function
        def logp_batch(eta):
            print("Tracing logp_batch ...")
            return -tf.math.reduce_mean(flow.log_prob(eta))

        for k in range(0, n_samples, batch_size):
            eta = eta_train[k: k + batch_size].astype("f4")
            loss_i.append(logp_batch(tf.constant(eta)).numpy())
            weight_i.append(eta.shape[0])
            bar.update(i * n_batches + k // batch_size + 1)

        loss.append(np.average(loss_i, weights=weight_i))

    loss_std = np.std(loss)
    loss_mean = np.mean(loss)

    return loss_mean, loss_std


def main():
    """
    Plots different diagnostics for the flow and the training data.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Deep Potential: Plot different diagnostics for a normalizing flow.",
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
        "--fig-dir",
        type=str,
        default="no_reg_loss",
        help="Directory to put figures in.",
    )
    parser.add_argument(
        "--flows", type=str, default="models/df/*", help="Directories of the flows."
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

    args = parser.parse_args()

    # Load in the custom style sheet for scientific plotting
    plt.style.use("scientific")
    if args.dark:
        plt.style.use("dark_background")

    print("Loading training data ...")
    data, attrs = utils.load_training_data(args.input, cut_attrs=True)

    batch_size = 16384
    if "eta_train" in data and "eta_val" in data:
        print("Flow training/validation batches were passed in manually..")
        n_samples = data["eta_train"].shape[0]
        n_val = data["eta_val"].shape[0]
        val_batch_size = int(n_val / (n_samples + n_val) * batch_size)

        eta_train = data["eta_train"]
        eta_val = data["eta_val"]
    else:
        print("Forming flow training/validation batches..")
        n_samples = data["eta"].shape[0]
        n_val = int(0.25 * n_samples)

        val_batch_size = int(0.25 * batch_size)
        n_samples -= n_val

        eta_val = data["eta"][:n_val]
        eta_train = data["eta"][n_val:]

    print(attrs)
    print(f"  --> Training data shape = {eta_train.shape}, val shape = {eta_val.shape}")

    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    for flow_dir in glob(args.flows):
        print(flow_dir)

        if flow_dir[-1] == "/":
            flow_dir = flow_dir[:-1]
        fname_loss = args.fig_dir + flow_dir[flow_dir.rfind("/"):]
        fname_txt = fname_loss + ".txt"
        print("txt fname", fname_txt)

        fnames_index = np.array(glob(f"{flow_dir}/*.index"))
        indices = np.array([int(x[x.rfind("-") + 1: -6]) for x in fnames_index])
        idx = np.argsort(indices)
        fnames_index, indices = fnames_index[idx], indices[idx]

        train_loss, val_loss = [], []

        # Check if losses have been saved, if not, store them
        if os.path.isfile(fname_txt):
            df = pd.read_csv(fname_txt)
            stored_indices = df["index"].values
            stored_train_loss = df["train_loss"].values
            stored_val_loss = df["val_loss"].values

            print(f"Loaded in {len(stored_indices)} values")
        else:
            stored_indices, stored_train_loss, stored_val_loss = [], [], []

        print(f"Found indices: {indices}")
        processed_indices = []
        for i, fname in enumerate(fnames_index):
            if indices[i] not in stored_indices:
                flows = utils.load_flows([fname])

                processed_indices.append(indices[i])
                # train_loss.append(i)
                # val_loss.append(i+1)
                train_loss.append(evaluate_loss(flows, eta_train, batch_size)[0])
                val_loss.append(evaluate_loss(flows, eta_val, batch_size)[0])

        train_loss = list(stored_train_loss) + list(train_loss)
        val_loss = list(stored_val_loss) + list(val_loss)
        indices = list(stored_indices) + list(processed_indices)

        idx = np.argsort(indices)
        indices, train_loss, val_loss = (
            np.array(indices)[idx],
            np.array(train_loss)[idx],
            np.array(val_loss)[idx],
        )

        # Save the losses to a txt
        pd.DataFrame(
            {
                "index": indices,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val-train": val_loss - train_loss,
            }
        ).to_csv(fname_txt, index=False, float_format="%.12e")

        fig = plot_loss(indices, train_loss, val_loss)
        for fmt in args.fig_fmt:
            print(f'Saving to {fname_loss + "." + fmt}')
            fig.savefig(fname_loss + "." + fmt)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    main()
