from tqdm import tqdm
import time
import numpy as np

import utils

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp



@eqx.filter_jit
def sample_batch_fn(model, sample_key, batch_size):
    return model.sample(sample_key, (batch_size,))

def value_and_grad_fn(model, eta_batch):
    return jax.vmap(eqx.filter_value_and_grad(model.log_prob))(eta_batch)

def sample_from_different_flows(
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
                eta_sample = np.array(sample_batch_fn(flow, sample_key, sample_batch_size))
                # Reject samples that are outside the range of validity

                if attrs["has_spatial_cut"]:
                    idx = utils.get_index_of_points_inside_attrs(eta_sample, attrs)
                    idx_size = eta_sample[idx].shape[0]
                    n_keep = min(nflow_samples[i] - n_processed, idx_size)
                    n_processed += n_keep

                    eta.append(eta_sample[idx][:n_keep])
                else:
                    n_processed += eta_sample.shape[0]
                    eta.append(eta_sample)
                # If we have enough samples, we can stop
                if n_processed >= nflow_samples[i]:
                    break

                pbar.update(1)
    # All eta will have at least one flow in their region of validity
    eta = np.concatenate(eta, axis=0)
    print(f'Shape of sampled eta: {eta.shape}')

    # Do ceiling divide
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    print("Sampling gradients of eta..")
    nflow_batches = [
        -(-nflow_samples[i] // grad_batch_size) for i in range(len(nflow_samples))
    ]
    total_iters = sum(nflow_batches)

    df_deta_indiv = np.zeros((len(flow_list),) + eta.shape, dtype="f4")
    with tqdm(total=total_iters, desc="Calculating f, df/deta") as pbar:
        for i, flow in enumerate(flow_list):
            df_deta = []
            prob = []
            for k in range(nflow_batches[i]):
                eta_batch = eta[k * grad_batch_size: (k + 1) * grad_batch_size]
                logp_batch, dlogp_deta_batch = value_and_grad_fn(flow, eta_batch)
                prob.append(np.exp(logp_batch))
                df_deta.append(dlogp_deta_batch * np.exp(logp_batch)[:, None])

                pbar.update(1)

            df_deta = np.concatenate([np.array(b) for b in df_deta])
            prob = np.concatenate([np.array(b) for b in prob])
            df_deta_indiv[i] = df_deta

    '''if len(flow_list) > 1:
        # Collapse!
        probs_indiv = np.zeros((len(flow_list), len(eta)), dtype="f4")
        n_batches = -(-len(eta) // sample_batch_size)
        bar = get_sampling_progressbar_fn(len(flow_list) * n_batches, n_samples)
        iteration = 0
        print("Calculating probs at eta..")
        for i, flow in enumerate(flow_list):

            @tf.function
            def prob_batch(batch):
                # print('Tracing sample_batch ...')
                return flow.prob(batch)

            probs = []
            for k in range(n_batches):
                prob_sample = (
                    prob_batch(eta[k * sample_batch_size: (k + 1) * sample_batch_size])
                    .numpy()
                    .astype("f4")
                )
                probs.append(prob_sample)
                bar(iteration)
                iteration += 1
            probs = np.concatenate(probs, axis=0)
            probs_indiv[i] = probs

        # Combine the gradients by using the local probability density * N as
        # the weight (prob is normalized)
        mask = np.full((len(flow_list), len(eta)), 1.0, dtype="f4")
        for i, flow in enumerate(flow_list):
            attrs = attrs_list[i]

            if attrs["has_spatial_cut"]:
                idx = utils.get_index_of_points_inside_attrs(eta, attrs)

                mask[i] = idx

        df_deta = np.full((len(eta), 6), 0, dtype="f4")
        for i in range(6):
            df_deta[:, i] = np.sum(
                probs_indiv * attrs["n"] * df_deta_indiv[..., i] * mask, axis=0
            ) / np.sum(probs_indiv * attrs["n"] * mask, axis=0)
        ret = {"eta": eta, "df_deta": df_deta}
        # TODO: Doesn't support probabilities
    else:
        ret = {"eta": eta, "df_deta": df_deta_indiv[0], "f": prob}
    if return_indiv:
        ret["df_deta_indiv"] = df_deta_indiv
    '''
    ret = {"eta": eta, "df_deta": df_deta, "f": prob}

    return ret
