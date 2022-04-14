import jax
import numpy as np
from typing import Tuple
from clu import parameter_overview

from .vgg16 import VGG16
from .vgg19 import VGG19


def build_vgg16(
    input_shape: Tuple = (224, 224, 3),
    seed: int = 0,
    show_parameter_overview: bool = False,
    **kwargs
):
    key = jax.random.PRNGKey(seed)
    model = VGG16(**kwargs)
    parameters = model.init(key, np.random.randn(1, *input_shape))
    if show_parameter_overview:
        print(parameter_overview.get_parameter_overview(parameters))
    return model, parameters


def build_vgg19(
    input_shape: Tuple = (224, 224, 3),
    seed: int = 0,
    show_parameter_overview: bool = False,
    **kwargs
):
    key = jax.random.PRNGKey(seed)
    model = VGG19(**kwargs)
    parameters = model.init(key, np.random.randn(1, *input_shape))
    if show_parameter_overview:
        print(parameter_overview.get_parameter_overview(parameters))
    return model, parameters
