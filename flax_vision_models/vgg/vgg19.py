import jax.numpy as jnp
import flax.linen as nn

from .blocks import VGGBlock


class VGG19(nn.Module):
    include_top: bool = True
    pooling: str = "avg"
    num_classes: int = 1000
    kernel_initializer: nn.initializers = nn.initializers.lecun_normal()
    bias_initializer: nn.initializers = nn.initializers.zeros
    classifier_activation: nn.Module = nn.softmax

    @nn.compact
    def __call__(self, x):
        # Block 1
        x = VGGBlock(
            num_features=64,
            kernel_initializer=self.bias_initializer,
            bias_initializer=self.bias_initializer,
        )(x)
        # Block 2
        x = VGGBlock(
            num_features=128,
            kernel_initializer=self.bias_initializer,
            bias_initializer=self.bias_initializer,
        )(x)
        # Block 3
        x = VGGBlock(
            num_features=256,
            num_convs=4,
            kernel_initializer=self.bias_initializer,
            bias_initializer=self.bias_initializer,
        )(x)
        # Block 4
        x = VGGBlock(
            num_features=512,
            num_convs=4,
            kernel_initializer=self.bias_initializer,
            bias_initializer=self.bias_initializer,
        )(x)
        # Block 5
        x = VGGBlock(
            num_features=512,
            num_convs=4,
            kernel_initializer=self.bias_initializer,
            bias_initializer=self.bias_initializer,
        )(x)
        if self.include_top:
            x = jnp.reshape(x, (x.shape[0], -1))  # Flatten
            x = nn.Dense(
                4096,
                kernel_init=self.bias_initializer,
                bias_init=self.bias_initializer,
            )(x)
            x = nn.relu(x)
            x = nn.Dense(
                4096,
                kernel_init=self.bias_initializer,
                bias_init=self.bias_initializer,
            )(x)
            x = nn.relu(x)
            x = nn.Dense(
                self.num_classes,
                kernel_init=self.bias_initializer,
                bias_init=self.bias_initializer,
            )(x)
            x = self.classifier_activation(x)
        else:
            if self.pooling == "avg":
                x = jnp.mean(x, axis=(1, 2))  # Global Average Pooling
            elif self.pooling == "max":
                x = jnp.max(x, axis=(1, 2))  # Global Max Pooling
        return x
