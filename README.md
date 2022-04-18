# Flax Vision Models

A repository of Deep Learning models in [Flax](https://github.com/google/flax) pre-trained on [Imagenet](https://image-net.org/). Most of the models instead of being trained from scratch have been ported from exiting repositories such as [`tf.keras.applications`](https://www.tensorflow.org/api_docs/python/tf/keras/applications) and [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models).

## Usage

### Installation

![](https://i.imgur.com/Jt9jKbS.png)

### Sample Inference Example

![](https://i.imgur.com/1YZSKIV.png)

# Pre-trained Models

## VGG Models

|Model|Original Source|Paper|
|---|---|---|
|`flax_models.vgg.build_vgg16`|[`tf.keras.applications.vgg16.VGG16`](https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/vgg16.py#L43-L227)|[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)|
|`flax_models.vgg.build_vgg19`|[`tf.keras.applications.vgg19.VGG19`](https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/vgg19.py#L43-L231)|[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)|