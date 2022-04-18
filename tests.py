import os
import wget
import wandb
import unittest
from PIL import Image

import jax
import numpy as np
import jax.numpy as jnp

from flax_vision_models.utils import decode_probabilities_imagenet
from flax_vision_models.vgg import build_vgg16, build_vgg19, preprocess as prepocessing_fn


class VGG16TestCase(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        wandb.init(project="flax-vision-models", job_type="unit-test")
        image_file = (
            wget.download("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
            if not os.path.isfile("dog.jpg")
            else "dog.jpg"
        )
        image = Image.open(image_file)
        image = image.resize((224, 224))
        self.input_batch = np.array(image)
        self.input_batch = prepocessing_fn(self.input_batch)
        self.input_batch = jnp.expand_dims(self.input_batch, axis=0)
        self.k = 5

    def test_vgg16_with_top(self):
        model, parameters = build_vgg16(show_parameter_overview=False, pretrained=True)
        out = model.apply(parameters, self.input_batch)
        assert out.shape == (1, 1000)
        topk_probs, topk_classes = jax.lax.top_k(out, k=self.k)
        topk_probs = jnp.squeeze(topk_probs, axis=0)
        topk_classes = jnp.squeeze(topk_classes, axis=0)
        topk_labels, _ = decode_probabilities_imagenet(topk_classes, topk_probs)
        assert "Samoyed" in topk_labels

    def test_vgg16_without_top(self):
        model, parameters = build_vgg16(
            show_parameter_overview=False, include_top=False
        )
        out = model.apply(parameters, self.input_batch)
        assert out.shape == (1, 512)


class VGG19TestCase(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        wandb.init(project="flax-vision-models", job_type="unit-test")
        image_file = (
            wget.download("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
            if not os.path.isfile("dog.jpg")
            else "dog.jpg"
        )
        image = Image.open(image_file)
        image = image.resize((224, 224))
        self.input_batch = np.array(image)
        self.input_batch = prepocessing_fn(self.input_batch)
        self.input_batch = jnp.expand_dims(self.input_batch, axis=0)
        self.k = 5

    def test_vgg16_with_top(self):
        model, parameters = build_vgg19(show_parameter_overview=False, pretrained=True)
        out = model.apply(parameters, self.input_batch)
        assert out.shape == (1, 1000)
        topk_probs, topk_classes = jax.lax.top_k(out, k=self.k)
        topk_probs = jnp.squeeze(topk_probs, axis=0)
        topk_classes = jnp.squeeze(topk_classes, axis=0)
        topk_labels, _ = decode_probabilities_imagenet(topk_classes, topk_probs)
        assert "Samoyed" in topk_labels

    def test_vgg16_without_top(self):
        model, parameters = build_vgg19(
            show_parameter_overview=False, include_top=False
        )
        out = model.apply(parameters, self.input_batch)
        assert out.shape == (1, 512)
