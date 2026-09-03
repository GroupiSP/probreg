from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core.losses import NegativeLogLikelihoodLoss
from probreg.core.types import Batch
from probreg.jax import (
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)
from probreg.jax.distributions import GaussianHead


def make_heteroscedastic_dataset(
    *, num_samples: int, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Sample ``y = 2x + noise`` with noise scale increasing monotonically in ``x``."""
    inputs_key, noise_key = jax.random.split(key)
    inputs = jax.random.uniform(inputs_key, (num_samples, 1), minval=-3.0, maxval=3.0)
    noise_scale = 0.1 + 0.2 * (inputs + 3.0)
    noise = noise_scale * jax.random.normal(noise_key, (num_samples, 1))
    targets = 2.0 * inputs + noise
    return inputs, targets


def make_loader(inputs: jax.Array, targets: jax.Array, *, batch_size: int):
    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split
        permutation = jax.random.permutation(jax.random.key(epoch), inputs.shape[0])
        shuffled_inputs = inputs[permutation]
        shuffled_targets = targets[permutation]
        return [
            Batch(
                inputs=shuffled_inputs[start : start + batch_size],
                targets=shuffled_targets[start : start + batch_size],
            )
            for start in range(0, inputs.shape[0], batch_size)
        ]

    return loader


def test_mve_training_decreases_loss_and_tracks_heteroscedastic_noise() -> None:
    data_key, model_key, rng_key = jax.random.split(jax.random.key(0), 3)
    inputs, targets = make_heteroscedastic_dataset(num_samples=256, key=data_key)

    model = GaussianHead(1, 1, rngs=nnx.Rngs(model_key))
    optimizer = create_optimizer(model, optax.adam(learning_rate=0.05))
    state = initialize_training_state(model, optimizer, rng_key=rng_key)

    mve_loss = make_supervised_loss(NegativeLogLikelihoodLoss())

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=make_loader(inputs, targets, batch_size=32),
        loss=mve_loss,
        state=state,
        epochs=100,
    )

    initial_loss = result.state.metric_history["training_loss"][0]
    final_loss = result.state.metric_history["training_loss"][-1]
    assert final_loss < initial_loss

    low_uncertainty_features = jnp.array([[-3.0]])
    high_uncertainty_features = jnp.array([[3.0]])
    low_prediction = model(low_uncertainty_features)
    high_prediction = model(high_uncertainty_features)

    assert bool(jnp.all(high_prediction.variance() > low_prediction.variance()))
