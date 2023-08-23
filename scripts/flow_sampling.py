import tensorflow as tf

print(f"Tensorflow version {tf.__version__}")
# tf.debugging.set_log_device_placement(True)
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_probability as tfp

print(f"Tensorflow Probability version {tfp.__version__}")

import progressbar
import time
import numpy as np

import utils


def get_sampling_progressbar_fn(n_batches, n_samples):
    widgets = [
        progressbar.Bar(),
        progressbar.Percentage(),
        " | ",
        progressbar.Timer(format="Elapsed: %(elapsed)s"),
        " | ",
        progressbar.AdaptiveETA(),
        " | ",
        progressbar.Variable("batches_done", width=6, precision=0),
        ", ",
        progressbar.Variable("n_batches", width=6, precision=0),
        ", ",
        progressbar.Variable("n_samples", width=8, precision=0),
    ]
    bar = progressbar.ProgressBar(max_value=n_batches, widgets=widgets)
    # n_batches = n_batches
    # n_samples = n_samples

    def update_progressbar(i):
        bar.update(i + 1, batches_done=i + 1, n_batches=n_batches,
                   n_samples=n_samples)

    return update_progressbar


def sample_from_different_flows(
    flow_list,
    attrs_list,
    n_samples,
    return_indiv=False,
    grad_batch_size=128,
    sample_batch_size=128,
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

    eta = []
    bar = get_sampling_progressbar_fn(sum(nflow_batches), n_samples)
    iteration = 0
    print("Sampling eta..")
    for i, flow in enumerate(flow_list):
        attrs = attrs_list[i]

        @tf.function
        def sample_batch():
            # print('Tracing sample_batch ...')
            return flow.sample([sample_batch_size])

        for k in range(nflow_batches[i]):
            n_sample = min(sample_batch_size, nflow_samples[i] - k * sample_batch_size)
            eta_sample = sample_batch().numpy().astype("f4")[:n_sample]
            # Reject samples that are outside the range of validity

            if attrs["has_spatial_cut"]:
                idx = utils.get_index_of_points_inside_attrs(eta_sample, attrs)

                eta.append(eta_sample[idx])
            else:
                eta.append(eta_sample)
            bar(iteration)
            iteration += 1
    # All eta will have at least one flow in their region of validity
    eta = np.concatenate(eta, axis=0)

    # Do ceiling divide
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    n_batches = -(-len(eta) // grad_batch_size)
    bar = get_sampling_progressbar_fn(len(flow_list) * n_batches, n_samples)
    iteration = 0
    print("Sampling gradients of eta..")

    df_deta_indiv = np.zeros((len(flow_list),) + eta.shape, dtype="f4")
    for i, flow in enumerate(flow_list):

        @tf.function
        def calc_grads(batch):
            # print(f'Tracing calc_grads with shape = {batch.shape}')
            with tf.GradientTape(watch_accessed_variables=False) as g:
                g.watch(batch)
                f = flow.prob(batch)
            df_deta = g.gradient(f, batch)
            return df_deta, f

        eta_dataset = tf.data.Dataset.from_tensor_slices(eta).batch(grad_batch_size)
        df_deta = []
        prob = []
        for k, b in enumerate(eta_dataset):
            df_deta_batch, prob_batch = calc_grads(b)
            df_deta.append(df_deta_batch)
            prob.append(prob_batch)
            bar(iteration)
            iteration += 1

        df_deta = np.concatenate([b.numpy() for b in df_deta])
        prob = np.concatenate([b.numpy() for b in prob])
        df_deta_indiv[i] = df_deta

    if len(flow_list) > 1:
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

    return ret
