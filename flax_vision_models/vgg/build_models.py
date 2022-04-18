import os
import wget
import wandb
import deepdish
from typing import Tuple

import jax
import numpy as np
from clu import parameter_overview

from .vgg16 import VGG16
from .vgg19 import VGG19


VGG16_ARTIFACT_ADDRESS = "geekyrakshit/flax-vision-models/vgg16-keras-imagenet:v0"
VGG16_URL = "https://github.com/soumik12345/flax-vision-models/releases/download/0.0.1/vgg16_keras_imagenet.h5"
VGG16_FILE_NAME = "vgg16_keras_imagenet.h5"

VGG19_ARTIFACT_ADDRESS = "geekyrakshit/flax-vision-models/vgg19-keras-imagenet:v0"
VGG19_URL = "https://github.com/soumik12345/flax-vision-models/releases/download/0.0.1/vgg19_keras_imagenet.h5"
VGG19_FILE_NAME = "vgg19_keras_imagenet.h5"


def build_vgg16(
    input_shape: Tuple = (224, 224, 3),
    seed: int = 0,
    pretrained: bool = True,
    show_parameter_overview: bool = False,
    **kwargs
):
    key = jax.random.PRNGKey(seed)
    model = VGG16(**kwargs)
    parameters = model.init(key, np.random.randn(1, *input_shape))

    if pretrained:
        weight_file = None
        
        if wandb.run is not None:
            artifact = wandb.use_artifact(VGG16_ARTIFACT_ADDRESS)
            artifact_dir = artifact.download()
            weight_file = os.path.join(artifact_dir, VGG16_FILE_NAME)
        else:
            weight_file = wget.download(VGG16_URL)
        
        parameters = deepdish.io.load(weight_file)

    if show_parameter_overview:
        print(parameter_overview.get_parameter_overview(parameters))
    return model, parameters


def build_vgg19(
    input_shape: Tuple = (224, 224, 3),
    seed: int = 0,
    pretrained: bool = True,
    show_parameter_overview: bool = False,
    **kwargs
):
    key = jax.random.PRNGKey(seed)
    model = VGG19(**kwargs)
    parameters = model.init(key, np.random.randn(1, *input_shape))

    if pretrained:
        weight_file = None
        
        if wandb.run is not None:
            artifact = wandb.use_artifact(VGG19_ARTIFACT_ADDRESS)
            artifact_dir = artifact.download()
            weight_file = os.path.join(artifact_dir, VGG19_FILE_NAME)
        else:
            weight_file = wget.download(VGG19_URL)
        
        parameters = deepdish.io.load(weight_file)

    if show_parameter_overview:
        print(parameter_overview.get_parameter_overview(parameters))
    return model, parameters
