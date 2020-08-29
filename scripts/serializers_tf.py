#!/usr/bin/env python

from __future__ import print_function, division

import tensorflow as tf
print(f'Tensorflow version {tf.__version__}')
from tensorflow import keras
import numpy as np
import re


def weights_as_list(layer):
    """
    Returns a (possibly nested) list containing
    the weights in a tf.keras.Layer.
    """
    return [w.tolist() for w in layer.get_weights()]


def set_weights_w_list(layer, weights):
    """
    Sets the weights of a tf.keras.Layer using the provided
    weights. The weights are in a (possibly nested) list, in
    the form provided by `weights_as_list`.
    """
    layer.set_weights([np.array(w, dtype='f4') for w in weights])


def serialize_variable(v):
    """
    Returns a JSON-serializable dictionary representing a
    tf.Variable.
    """
    return dict(
        dtype=v.dtype.name,
        shape=list(v.shape),
        values=v.numpy().tolist(),
        name=re.sub('\:[0-9]+$', '', v.name),
        trainable=v.trainable
    )


def deserialize_variable(d):
    """
    Returns a tf.Variable, constructed using a dictionary
    of the form returned by `serialize_variable`.
    """
    return tf.Variable(
        np.array(d['values'], dtype=d['dtype']),
        name=d['name'],
        trainable=d['trainable']
    )

