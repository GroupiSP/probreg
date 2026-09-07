"""1D-convolutional deterministic mean model for windowed CMAPSS RUL data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class Cnn1DBackbone(nnx.Module):
    """Two 1D-convolutional layers over the time axis, sensors as channels."""

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the stacked 1D-convolutional feature extractor.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_channels: Number of output channels for both
                convolutional layers.
            kernel_size: Convolution kernel width along the time axis.
            rngs: NNX random-number generators used to initialize parameters.
        """
        self.conv1 = nnx.Conv(
            in_features=n_sensors,
            out_features=hidden_channels,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=hidden_channels,
            out_features=hidden_channels,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Extract a fixed-size feature vector per windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            Time-averaged convolutional features, shape ``(batch,
            hidden_channels)``.
        """
        hidden = nnx.relu(self.conv1(inputs))
        hidden = nnx.relu(self.conv2(hidden))
        return jnp.mean(hidden, axis=1)


class Cnn1DMeanModel(nnx.Module):
    """Deterministic point-RUL regressor over windowed CMAPSS sensor data."""

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the convolutional backbone and the mean output head.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_channels: Number of channels produced by the
                convolutional backbone.
            kernel_size: Convolution kernel width along the time axis.
            rngs: NNX random-number generators used to initialize parameters.
        """
        self.backbone = Cnn1DBackbone(
            n_sensors,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            rngs=rngs,
        )
        self.output = nnx.Linear(hidden_channels, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Predict a scalar RUL for each windowed example in the batch.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            Predicted RUL, shape ``(batch, 1)``.
        """
        return self.output(self.backbone(inputs))
