"""Shared XSin-inspired benchmark utilities for probabilistic regression examples."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core.losses import GaussianNLLLoss
from probreg.core.protocols import LoaderFactory
from probreg.core.types import Batch, TrainingState
from probreg.jax import (
    GammaHead,
    GammaVarianceStage,
    GaussianHead,
    MeanStage,
    SupervisedStageOptions,
    create_optimizer,
    initialize_training_state,
    make_mve_loss,
    run_supervised,
)


@dataclass(frozen=True)
class XSinConfig:
    """Configuration shared by the MVE and two-step XSin runs."""

    train_size: int = 512
    evaluation_size: int = 301
    batch_size: int = 64
    hidden_features: int = 32
    mve_epochs: int = 300
    mean_epochs: int = 300
    variance_epochs: int = 300
    learning_rate: float = 0.01
    seed: int = 0


@dataclass(frozen=True)
class XSinData:
    """Training observations and a noiseless evaluation grid."""

    train_inputs: jax.Array
    train_targets: jax.Array
    evaluation_inputs: jax.Array
    true_mean: jax.Array
    true_variance: jax.Array


@dataclass(frozen=True)
class XSinResult:
    """Predictions and comparable error metrics for one benchmark method."""

    mean: jax.Array
    variance: jax.Array
    mean_rmse: float
    variance_rmse: float


class XSinBackbone(nnx.Module):
    """Two-layer tanh backbone used by all XSin models."""

    def __init__(self, hidden_features: int, *, rngs: nnx.Rngs) -> None:
        self.input_layer = nnx.Linear(1, hidden_features, rngs=rngs)
        self.output_layer = nnx.Linear(hidden_features, hidden_features, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        hidden = jnp.tanh(self.input_layer(inputs))
        return jnp.tanh(self.output_layer(hidden))


class XSinMeanModel(nnx.Module):
    """Deterministic mean regressor for the XSin benchmark."""

    def __init__(self, hidden_features: int, *, rngs: nnx.Rngs) -> None:
        self.backbone = XSinBackbone(hidden_features, rngs=rngs)
        self.output = nnx.Linear(hidden_features, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.output(self.backbone(inputs))


class XSinGaussianModel(nnx.Module):
    """Joint Gaussian mean/scale regressor for the MVE comparison."""

    def __init__(self, hidden_features: int, *, rngs: nnx.Rngs) -> None:
        self.backbone = XSinBackbone(hidden_features, rngs=rngs)
        self.head = GaussianHead(hidden_features, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array):
        return self.head(self.backbone(inputs))


class XSinGammaModel(nnx.Module):
    """Gamma residual regressor for the two-step comparison."""

    def __init__(self, hidden_features: int, *, rngs: nnx.Rngs) -> None:
        self.backbone = XSinBackbone(hidden_features, rngs=rngs)
        self.head = GammaHead(hidden_features, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array):
        return self.head(self.backbone(inputs))


def xsin_mean(inputs: jax.Array) -> jax.Array:
    """Return the nonlinear XSin mean function."""
    return inputs * jnp.sin(inputs)


def xsin_variance(inputs: jax.Array) -> jax.Array:
    """Return the positive heteroscedastic variance function."""
    scale = 0.15 + 0.45 * jax.nn.sigmoid(inputs)
    return jnp.square(scale)


def make_xsin_data(config: XSinConfig) -> XSinData:
    """Generate deterministic XSin training data and an evaluation grid.

    Args:
        config: Benchmark configuration.

    Returns:
        Training observations and exact evaluation functions.
    """
    key = jax.random.key(config.seed)
    inputs_key, noise_key = jax.random.split(key)
    train_inputs = jax.random.uniform(
        inputs_key,
        (config.train_size, 1),
        minval=-6.0,
        maxval=6.0,
    )
    train_targets = xsin_mean(train_inputs) + jnp.sqrt(
        xsin_variance(train_inputs)
    ) * jax.random.normal(noise_key, train_inputs.shape)
    evaluation_inputs = jnp.linspace(
        -6.0,
        6.0,
        config.evaluation_size,
    ).reshape(-1, 1)
    return XSinData(
        train_inputs=train_inputs,
        train_targets=train_targets,
        evaluation_inputs=evaluation_inputs,
        true_mean=xsin_mean(evaluation_inputs),
        true_variance=xsin_variance(evaluation_inputs),
    )


def make_xsin_loader(data: XSinData, config: XSinConfig) -> LoaderFactory:
    """Build a deterministic epoch-shuffled training loader.

    Args:
        data: XSin benchmark data.
        config: Benchmark configuration.

    Returns:
        Loader factory over the training observations.
    """

    def loader(*, split: str, epoch: int) -> list[Batch]:
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown XSin split {split!r}.")
        permutation = jax.random.permutation(
            jax.random.fold_in(jax.random.key(config.seed), epoch),
            config.train_size,
        )
        inputs = data.train_inputs[permutation]
        targets = data.train_targets[permutation]
        return [
            Batch(
                inputs=inputs[start : start + config.batch_size],
                targets=targets[start : start + config.batch_size],
            )
            for start in range(0, config.train_size, config.batch_size)
        ]

    return loader


def run_xsin_mve(data: XSinData, config: XSinConfig) -> XSinResult:
    """Train and evaluate the joint Gaussian MVE baseline.

    Args:
        data: Shared XSin data.
        config: Benchmark configuration.

    Returns:
        MVE predictions and comparison metrics.
    """
    model_key, train_key = jax.random.split(jax.random.key(config.seed + 1))
    model = XSinGaussianModel(
        config.hidden_features,
        rngs=nnx.Rngs(model_key),
    )
    optimizer = create_optimizer(model, optax.adam(config.learning_rate))
    state = initialize_training_state(model, optimizer, rng_key=train_key)
    run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=make_xsin_loader(data, config),
        loss=make_mve_loss(GaussianNLLLoss()),
        state=state,
        epochs=config.mve_epochs,
        stage="xsin_mve",
    )
    prediction = model(data.evaluation_inputs)
    return _summarize(
        prediction.mean(),
        prediction.variance(),
        data,
    )


def run_xsin_two_step(data: XSinData, config: XSinConfig) -> XSinResult:
    """Train and evaluate separate mean and Gamma variance stages.

    Args:
        data: Shared XSin data.
        config: Benchmark configuration.

    Returns:
        Two-step predictions and comparison metrics.
    """
    mean_key, variance_key, train_key = jax.random.split(
        jax.random.key(config.seed + 2),
        3,
    )
    loader = make_xsin_loader(data, config)
    state = TrainingState(rng_state=train_key)

    mean_model = XSinMeanModel(
        config.hidden_features,
        rngs=nnx.Rngs(mean_key),
    )
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(
            mean_model,
            optax.adam(config.learning_rate),
        ),
        train_loader=loader,
        options=SupervisedStageOptions(epochs=config.mean_epochs),
    )
    mean_stage.prepare(state)
    mean_stage.train(state)

    variance_model = XSinGammaModel(
        config.hidden_features,
        rngs=nnx.Rngs(variance_key),
    )
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(
            variance_model,
            optax.adam(config.learning_rate),
        ),
        source_loader=loader,
        options=SupervisedStageOptions(epochs=config.variance_epochs),
        splits=("train",),
    )
    variance_stage.prepare(state)
    variance_stage.train(state)

    return _summarize(
        mean_model(data.evaluation_inputs),
        variance_model(data.evaluation_inputs).mean(),
        data,
    )


def print_xsin_metrics(method: str, result: XSinResult) -> None:
    """Print the common XSin comparison metrics.

    Args:
        method: Human-readable method label.
        result: Benchmark result to report.
    """
    print(f"method={method}")
    print(f"mean_rmse={result.mean_rmse:.6f}")
    print(f"aleatoric_variance_rmse={result.variance_rmse:.6f}")


def plot_xsin_result(method: str, data: XSinData, result: XSinResult) -> None:
    """Show a common interactive mean/aleatoric comparison plot.

    Args:
        method: Human-readable method label used in the figure title.
        data: Shared observations and exact benchmark functions.
        result: Predicted mean and aleatoric variance.
    """
    import matplotlib.pyplot as plt

    inputs = jax.device_get(data.evaluation_inputs).reshape(-1)
    true_mean = jax.device_get(data.true_mean).reshape(-1)
    true_variance = jax.device_get(data.true_variance).reshape(-1)
    predicted_mean = jax.device_get(result.mean).reshape(-1)
    predicted_variance = jax.device_get(result.variance).reshape(-1)
    train_inputs = jax.device_get(data.train_inputs).reshape(-1)
    train_targets = jax.device_get(data.train_targets).reshape(-1)

    figure, (mean_axis, variance_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
    )
    mean_axis.scatter(
        train_inputs,
        train_targets,
        s=8,
        alpha=0.2,
        label="training observations",
    )
    mean_axis.plot(inputs, true_mean, color="black", label="true mean")
    mean_axis.plot(inputs, predicted_mean, color="C1", label="predicted mean")
    mean_axis.fill_between(
        inputs,
        true_mean - 2.0 * jnp.sqrt(true_variance),
        true_mean + 2.0 * jnp.sqrt(true_variance),
        color="black",
        alpha=0.1,
        label="true aleatoric band",
    )
    mean_axis.fill_between(
        inputs,
        predicted_mean - 2.0 * jnp.sqrt(predicted_variance),
        predicted_mean + 2.0 * jnp.sqrt(predicted_variance),
        color="C1",
        alpha=0.2,
        label="predicted aleatoric band",
    )
    mean_axis.set_ylabel("target")
    mean_axis.legend(ncol=2)

    variance_axis.plot(
        inputs,
        true_variance,
        color="black",
        label="true aleatoric variance",
    )
    variance_axis.plot(
        inputs,
        predicted_variance,
        color="C1",
        label="predicted aleatoric variance",
    )
    variance_axis.set_xlabel("x")
    variance_axis.set_ylabel("variance")
    variance_axis.legend()

    figure.suptitle(
        f"{method}: mean RMSE={result.mean_rmse:.4f}, "
        f"variance RMSE={result.variance_rmse:.4f}"
    )
    figure.tight_layout()
    plt.show()


def _summarize(
    mean: jax.Array,
    variance: jax.Array,
    data: XSinData,
) -> XSinResult:
    """Build common RMSE summaries for mean and aleatoric variance."""
    mean_rmse = float(jnp.sqrt(jnp.mean(jnp.square(mean - data.true_mean))))
    variance_rmse = float(jnp.sqrt(jnp.mean(jnp.square(variance - data.true_variance))))
    return XSinResult(
        mean=mean,
        variance=variance,
        mean_rmse=mean_rmse,
        variance_rmse=variance_rmse,
    )
