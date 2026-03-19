import tensorflow as tf
import numpy as np

def serialize_variable(v):
    """Serializes a tensorflow Variable (or Tensor) to a list."""
    if hasattr(v, 'numpy'):
        return v.numpy().tolist()
    return v

def deserialize_variable(data, dtype=None, name=None, trainable=True):
    """Deserializes a list to a tensorflow Variable."""
    val = np.array(data)
    
    # Infer dtype if not provided
    if dtype is None:
        if val.dtype.kind in ('i', 'u'):
            dtype = tf.int32
        else:
            dtype = tf.float32

    return tf.Variable(
        initial_value=val,
        trainable=trainable,
        name=name,
        dtype=dtype
    )

def weights_as_list(layer):
    """Returns the weights of a layer as a list of lists."""
    return [w.numpy().tolist() for w in layer.weights]

def set_weights_w_list(layer, weights_list):
    """Sets the weights of a layer from a list of lists."""
    weights = [np.array(w) for w in weights_list]
    layer.set_weights(weights)
