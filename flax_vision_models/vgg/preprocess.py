import jax.numpy as jnp

from typing import List, Optional


def preprocess(
    x,
    mean: Optional[List[float]] = None,
    std: Optional[float] = None,
    dtype=jnp.float64,
):
    """ "Reference: https://github.com/keras-team/keras/blob/d8fcb9d4d4dad45080ecfdd575483653028f8eda/keras/applications/imagenet_utils.py#L168"""
    x = x.astype(dtype)
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68] if mean is None else mean
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x
