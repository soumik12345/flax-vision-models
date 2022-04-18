import flax.linen as nn


class VGGBlock(nn.Module):
    num_features: int = 64
    num_convs: int = 2
    kernel_initializer: nn.initializers = nn.initializers.lecun_normal()
    bias_initializer: nn.initializers = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_convs):
            x = nn.Conv(
                self.num_features,
                (3, 3),
                padding="same",
                kernel_init=self.kernel_initializer,
                bias_init=self.bias_initializer,
            )(x)
            x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x
