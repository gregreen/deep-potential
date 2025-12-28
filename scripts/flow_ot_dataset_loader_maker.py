#!/usr/bin/env python
import argparse
import warnings
from functools import partial
from multiprocessing import get_context
from pathlib import Path

# Standard library and third-party libraries (JAX-free section)
import numpy as np
from tqdm import tqdm
import utils


# Suppress warnings from the POT library
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ot as pot


# --- JAX-Free Functions ---
# These functions will be used by the workers and do not depend on JAX.


def np_cdist2(x1, x2):
    """Computes squared Euclidean distances between two numpy arrays."""
    x1_squared = np.sum(x1**2, axis=1, keepdims=True)
    x2_squared = np.sum(x2**2, axis=1, keepdims=True).T
    dot_product = np.dot(x1, x2.T)
    return x1_squared + x2_squared - 2 * dot_product

class OTSampler:
    """Computes and samples from an Optimal Transport plan using only NumPy."""
    def __init__(self, method: str = "exact", reg: float = 0.05, warn: bool = True):
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        else:
            raise ValueError(f"Unknown OT method: {method}")
        self.warn = warn

    def get_map(self, x0, x1, w0, w1):
        """
        Computes the OT plan (cost matrix) between two distributions,
        accounting for their weights.
        """
        M = np_cdist2(x0, x1)
        p = self.ot_fn(w0, w1, M)

        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan; reverting to a uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map_indices(self, rng, x0, x1, w0, w1, replace=True):
        """
        Computes the OT plan with weights and samples paired indices (i, j).
        """
        pi = self.get_map(x0, x1, w0, w1)

        col_sums = pi.sum(axis=0, keepdims=True)
        cond = np.divide(pi, col_sums, out=np.zeros_like(pi), where=col_sums > 0)

        # For each j, sample i ~ P(i|j)
        j_idx = np.arange(x1.shape[0])
        i_idx = [rng.choice(x0.shape[0], p=cond[:, j]) for j in j_idx]
        return np.array(i_idx), j_idx


def get_matched_indices(x0_epoch, x1_epoch_permuted, w1_epoch_permuted, ot_batch_size=256, use_tqdm=False, desc=None):
    """
    Matches given samples (x0_epoch) to a permuted dataset (x1_epoch_permuted)
    using batched Optimal Transport, accounting for weights. This function is JAX-free.
    """
    ot_sampler = OTSampler(method="exact", warn=False)
    n = x1_epoch_permuted.shape[0]
    n_batches = -(-n // ot_batch_size)

    # Use NumPy's default random generator
    rng = np.random.default_rng()

    all_idx_x0, all_idx_x1_permuted = [], []

    batch_iterator = range(n_batches)
    if use_tqdm:
        batch_iterator = tqdm(batch_iterator, desc=desc, leave=False)

    for i in batch_iterator:
        i_min, i_max = i * ot_batch_size, min((i + 1) * ot_batch_size, n)

        x0_batch = x0_epoch[i_min:i_max]
        x1_batch = x1_epoch_permuted[i_min:i_max]

        w0_batch = np.ones(x0_batch.shape[0], dtype=np.float32) / x0_batch.shape[0]
        w1_batch = w1_epoch_permuted[i_min:i_max].copy()
        # Normalize weights for the current batch to sum to 1
        w1_batch /= w1_batch.sum()

        idx_i, idx_j = ot_sampler.sample_map_indices(
            rng, x0_batch, x1_batch, w0_batch, w1_batch, replace=False
        )

        all_idx_x0.append(i_min + idx_i)
        all_idx_x1_permuted.append(i_min + idx_j)

    final_idx_x0 = np.concatenate(all_idx_x0)
    final_idx_x1_permuted = np.concatenate(all_idx_x1_permuted)

    return final_idx_x0, final_idx_x1_permuted

def worker_init_no_gpu():
    """Initializer for multiprocessing workers to disable GPU access."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ["JAX_PLATFORMS"] = "cpu"  # force JAX to ignore CUDA in workers

def _precompute_worker(args):
    """
    A JAX-free worker function to compute OT pairings for a single epoch's data.
    """
    epoch_data, batch_size = args
    x0_epoch, x1_epoch_permuted, w1_epoch_permuted, original_x1_indices = epoch_data

    idx_x0, idx_x1_permuted = get_matched_indices(
        x0_epoch, x1_epoch_permuted, w1_epoch_permuted, ot_batch_size=batch_size
    )

    # Un-shuffle the permuted x1 indices to match the original dataset order
    final_idx_x1 = original_x1_indices[idx_x1_permuted]

    return idx_x0, final_idx_x1

# --- Main Logic with JAX ---
# This section remains in the main process and handles all JAX operations.

def precompute_ot_indices(
    num_epochs: int,
    train_data: dict,
    batch_size: int,
    batch_size_storing_epochs: int,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    fname_prefix: Path,
    num_workers: int = 1,
    key=None,
):
    """
    Generates x0 and x1 for each epoch in the main process, then feeds
    the data to JAX-free workers for OT computation.
    """
    # JAX is imported here, only in the main process
    import jax
    import jax.numpy as jnp

    print("--- Starting Pre-computation of OT Indices ---")
    x1_full_dataset = (train_data['eta'] - norm_mean) / norm_std
    w1_full_dataset_np = np.array(train_data['weights'], dtype=np.float32)
    x1_full_dataset_np = np.array(x1_full_dataset)
    n_samples = x1_full_dataset_np.shape[0]
    n_store_batches = -(-num_epochs // batch_size_storing_epochs)  # Ceiling division

    # 1. Generate all JAX keys and data for all epochs in the main process
    print(f"Generating data for {num_epochs} epochs...")
    # Generate epoch_keys one by one such that the sequence of values is independent of num_epochs
    epoch_keys = []
    for _ in range(num_epochs):
        key, subkey = jax.random.split(key)
        epoch_keys.append(subkey)

    for i_epoch in range(n_store_batches):
        start_idx = i_epoch * batch_size_storing_epochs
        end_idx = min((i_epoch + 1) * batch_size_storing_epochs, num_epochs)
        num_epochs_in_batch = end_idx - start_idx

        epochs_data = []
        epoch_x0_generation_keys = []
        for i in range(num_epochs_in_batch):
            epoch_key = epoch_keys[start_idx + i]
            key_x0, key_perm = jax.random.split(epoch_key, 2)
            epoch_x0_generation_keys.append(np.array(key_x0))

            # Generate x0 (noise) and permute x1 for one epoch
            x0_epoch = jax.random.normal(key_x0, shape=x1_full_dataset.shape)
            perm_x1_indices = jax.random.permutation(key_perm, n_samples)
            x1_epoch_permuted = x1_full_dataset[perm_x1_indices]
            w1_epoch_permuted = w1_full_dataset_np[perm_x1_indices]

            # Convert to NumPy arrays before sending to workers
            epochs_data.append((
                np.array(x0_epoch),
                np.array(x1_epoch_permuted),
                np.array(w1_epoch_permuted),
                np.array(perm_x1_indices) # Keep original indices for un-shuffling
            ))

        all_x0_indices, all_x1_indices = [], []

        # 2. Distribute work to JAX-free workers
        if num_workers > 1 and num_epochs > 0:
            print(f"Distributing {num_epochs_in_batch} epochs across {num_workers} JAX-free workers.")

            args_list = [(epoch_data, batch_size) for epoch_data in epochs_data]

            with get_context('spawn').Pool(processes=num_workers, initializer=worker_init_no_gpu) as pool:
                # The pool.imap is memory-efficient for large numbers of epochs
                results_iterator = pool.imap(_precompute_worker, args_list)

                for res_x0, res_x1 in tqdm(results_iterator, total=len(args_list), desc="Processing Epochs"):
                    if res_x0 is not None:
                        all_x0_indices.append(res_x0)
                        all_x1_indices.append(res_x1)

        # 3. Handle single-worker case or remaining epochs
        elif num_epochs_in_batch > 0:
            print(f"Running {num_epochs_in_batch} epochs on the main process (JAX-free computation)...")
            for i in tqdm(range(num_epochs_in_batch), desc="Processing Epochs"):
                x0_epoch, x1_epoch_permuted, w1_epoch_permuted, perm_x1_indices = epochs_data[i]

                # Call the JAX-free function directly
                idx_x0, idx_x1_permuted = get_matched_indices(
                    x0_epoch, x1_epoch_permuted, w1_epoch_permuted, batch_size, use_tqdm=True, desc=f"Epoch {i_epoch+1}/{num_epochs}"
                )
                # Un-shuffle the permuted x1 indices
                final_idx_x1 = perm_x1_indices[idx_x1_permuted]

                all_x0_indices.append(idx_x0)
                all_x1_indices.append(final_idx_x1)

        # 4. Save results to disk
        if not all_x0_indices:
            print("\nWarning: No indices were generated. Nothing to save.")
            return


        final_x0 = np.concatenate(all_x0_indices, axis=0)
        final_x1 = np.concatenate(all_x1_indices, axis=0)

        fname = fname_prefix.with_name(fname_prefix.name + f"_epochs_{start_idx}_{end_idx}.npz")
        print(fname)
        start_idx *= n_samples
        end_idx *= n_samples

        # Save the indices for this batch of epochs
        np.savez(fname, x0_perm_indices=final_x0, x1_perm_indices=final_x1, x0_generation_keys=np.array(epoch_x0_generation_keys))

    print(f"\n--- Pre-computation complete. Data saved to {fname} ---")
    print(f"Saved index arrays with shape: {final_x0.shape}")


def make_ot_dataloader(data_fname: str, epochs: int, batch_size: int, batch_size_storing_epochs: int,
                       num_workers: int, seed: int, spatial_only: bool):
    """
    Pre-computes Optimal Transport (OT) pairings for generative modeling, see
    https://arxiv.org/abs/2302.00482.

    This function is designed to accelerate the training of Normalizing Flows via Flow
    Matching with minibatch optimal transport by pre-calculating the computationally
    expensive optimal transport plans between a simple noise distribution (x0,
    standard Gaussian) and a target data distribution (x1, the user's dataset).

    The function generates a large, static dataset of paired indices between x0 and x1
    that can later be read in using a training script. Since only the indices of x0
    are stored (this way, a (n, d) array takes (n, ) space), the noise needs to be
    reproduced exactly the same way in the training script, starting from the same
    jax.random.key.

    The OT is computed using the 'POT' (Python Optimal Transport) library which uses
    CPUs only. To avoid any interference with GPUs, the function spawns a pool of 
    workers that purposely avoid loading in JAX or seeing any GPUs.

    The OT is calculated in batches of 256. For example, a dataset of 1e7 samples
    with 500 precomputed epochs takes up around 30GB of storage space.
    """
    import jax

    data_fname = Path(data_fname)

    save_dir = data_fname.parent / "ot_pairings"
    if spatial_only:
        ot_pairings_dir = save_dir / f"pairings_{data_fname.stem}_spatial_seed{seed}"
    else:
        ot_pairings_dir = save_dir / f"pairings_{data_fname.stem}_full_seed{seed}"
    ot_pairings_dir.mkdir(parents=True, exist_ok=True)
    train_output_fname_prefix = ot_pairings_dir / "train"
    val_output_fname_prefix = ot_pairings_dir / "val"

    print(f"Precomputing OT pairing indices for dataset: {data_fname}")
    print(f"  Training pairings will be saved to: {train_output_fname_prefix}_epochs_*.npz")
    print(f"  Validation pairings will be saved to: {val_output_fname_prefix}_epochs_*.npz")
    print(f"  Number of epochs to precompute: {epochs}")
    print(f"  Batch size for OT matching: {batch_size}")
    print(f"  Batch size for storing epochs: {batch_size_storing_epochs}")
    print(f"  Number of parallel workers: {num_workers}")
    print(f"  Random seed: {seed}\n")
    print(f"  Spatial only: {spatial_only}\n")

    # Assuming load_and_split_data might return JAX arrays
    train_data, val_data = utils.load_and_split_data(path=data_fname, val_split=0.25)
    if spatial_only:
        print("Using only spatial dimensions for OT matching (ignoring velocities).")
        train_data['eta'] = train_data['eta'][:, :3]
        val_data['eta'] = val_data['eta'][:, :3]
    #train_data['weights'] = np.ones_like(train_data['weights'])

    # Convert to NumPy for JAX-free processing
    data_mean = np.mean(train_data["eta"], axis=0)
    data_std = np.std(train_data["eta"], axis=0)

    key = jax.random.PRNGKey(seed)
    key_train, key_val = jax.random.split(key, 2)

    precompute_ot_indices(
        num_epochs=epochs,
        train_data=train_data,
        batch_size=batch_size,
        batch_size_storing_epochs=batch_size_storing_epochs,
        norm_mean=data_mean,
        norm_std=data_std,
        fname_prefix=train_output_fname_prefix,
        num_workers=num_workers,
        key=key_train,
    )

    precompute_ot_indices(
        num_epochs=epochs,
        train_data=val_data,
        batch_size=batch_size,
        batch_size_storing_epochs=batch_size_storing_epochs,
        norm_mean=data_mean,
        norm_std=data_std,
        fname_prefix=val_output_fname_prefix,
        num_workers=num_workers,
        key=key_val,
    )


if __name__ == "__main__":
    """
    Example usage:
        python flow_ot_dataset_loader_maker.py --data-fname ../data/simple_datasets/data_400pc_corner.npz --epochs 10 --batch-size-storing-epochs 10 --seed 0
        python flow_ot_dataset_loader_maker.py --data-fname ../data/simple_datasets/data_400pc_corner.npz --epochs 2000 --batch-size-storing-epochs 2000 --num-workers 40 --seed 0
        python flow_ot_dataset_loader_maker.py --data-fname ../data/simple_datasets/data_400pc_lite.npz --epochs 2000 --batch-size-storing-epochs 2000 --num-workers 40 --seed 0
        python flow_ot_dataset_loader_maker.py --data-fname ../runs/plummer_no_selfn/data/plummer_sphere.h5 --epochs 1500 --batch-size-storing-epochs 1500 --num-workers 30 --seed 0
        python flow_ot_dataset_loader_maker.py --data-fname ../data/simple_datasets/data_1000pc_corner.npz --epochs 10 --batch-size-storing-epochs 10 --num-workers 1 --seed 0
        python flow_ot_dataset_loader_maker.py --data-fname ../data/gdr3_pad_sph_v3/sph_1000pc.h5 --epochs 500 --batch-size-storing-epochs 50 --num-workers 25 --seed 0 --spatial-only
        python flow_ot_dataset_loader_maker.py --data-fname data/plummer_sphere_allstars.h5 --epochs 2048 --batch-size-storing-epochs 256 --num-workers 20 --seed 0
        python flow_ot_dataset_loader_maker.py --data-fname ../data/plummer_1m/plummer_sphere_obsstars_v2.h5 --epochs 2048 --batch-size-storing-epochs 256 --num-workers 40 --seed 0
        python flow_ot_dataset_loader_maker.py --data-fname ../data/gdr3_pad_sph_v4/sph_1000pc.h5 --epochs 500 --batch-size-storing-epochs 50 --num-workers 25 --seed 0 --spatial-only
    """
    parser = argparse.ArgumentParser(description="Precompute OT pairings between a noise distribution (x0) and a dataset (x1).")
    parser.add_argument("--data-fname", type=str, default="", help="Dataset file name.")
    parser.add_argument("--epochs", type=int, default=2000, help="Total number of epochs to pre-compute.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for OT matching.")
    parser.add_argument("--batch-size-storing-epochs", type=int, default=1000, help="Batch size the number of epochs to save on disk.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers to use.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for JAX key generation.")
    parser.add_argument("--spatial-only", action="store_true", help="Use only spatial dimensions for OT matching (ignores velocities).")
    args = parser.parse_args()

    make_ot_dataloader(
        data_fname=args.data_fname,
        epochs=args.epochs,
        batch_size=args.batch_size,
        batch_size_storing_epochs=args.batch_size_storing_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        spatial_only=args.spatial_only,
    )
