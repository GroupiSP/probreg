"""Shared XSin-inspired benchmark utilities for probabilistic regression examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core.losses import NegativeLogLikelihoodLoss
from probreg.core.protocols import LoaderFactory
from probreg.core.tracking import EventSink, TrainingEvent
from probreg.core.types import Batch, TrainingState
from probreg.jax import (
    Gamma,
    GammaHead,
    GammaVarianceStage,
    Gaussian,
    GaussianHead,
    MeanStage,
    SupervisedStageOptions,
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

INTERVAL_MULTIPLIER = 1.96
"""Half-width of the 95% predictive interval, in predictive scales."""


@dataclass(frozen=True)
class XSinConfig:
    """Configuration shared by the MVE, two-step and posterior XSin runs.

    Attributes:
        train_size: Number of noisy training observations.
        evaluation_size: Number of points in the noiseless evaluation grid.
        batch_size: Number of training observations per mini-batch.
        hidden_features: Width of each benchmark model's hidden layers.
        mve_epochs: Training epochs for the joint Gaussian MVE model.
        mean_epochs: Training epochs for the deterministic mean stage.
        variance_epochs: Training epochs for the Gamma variance stage.
        posterior_epochs: Training epochs for the posterior stage.
        posterior_num_draws: Draws taken from an unlimited posterior (Bayes by
            Backprop) to form the posterior predictive.
        learning_rate: Adam learning rate used by all benchmark models.
        bbb_learning_rate: Adam learning rate of Bayes by Backprop's
            variational parameters.
        bbb_initial_std: Standard deviation every variational parameter
            starts with.
        psgld_step_size: Langevin step size of pSGLD.
        psgld_burn_in: pSGLD update steps whose positions are discarded.
        psgld_thinning: pSGLD retains one position every this many steps
            after burn-in.
        seed: Base JAX random seed for data, model, and training keys.
        train_min: Exclusive lower bound of the training-input domain.
        train_max: Exclusive upper bound of the training-input domain.
        evaluation_min: Lower endpoint of the evaluation grid.
        evaluation_max: Upper endpoint of the evaluation grid.
    """

    train_size: int = 512
    evaluation_size: int = 301
    batch_size: int = 64
    hidden_features: int = 32
    mve_epochs: int = 300
    mean_epochs: int = 300
    variance_epochs: int = 300
    posterior_epochs: int = 100
    posterior_num_draws: int = 64
    learning_rate: float = 0.01
    bbb_learning_rate: float = 1e-2
    bbb_initial_std: float = 1e-2
    psgld_step_size: float = 3e-3
    psgld_burn_in: int = 0
    psgld_thinning: int = 8
    seed: int = 0
    train_min: float = 0.0
    train_max: float = 10.0
    evaluation_min: float = -5.0
    evaluation_max: float = 15.0

    def __post_init__(self) -> None:
        """Validate benchmark sizes and interpolation/extrapolation domains.

        Raises:
            ValueError: If sizes are not positive, ``psgld_burn_in`` is
                negative, or domain bounds are not strictly ordered around the
                training interval.
        """
        sizes = (
            self.train_size,
            self.evaluation_size,
            self.batch_size,
            self.hidden_features,
            self.mve_epochs,
            self.mean_epochs,
            self.variance_epochs,
            self.posterior_epochs,
            self.posterior_num_draws,
            self.psgld_thinning,
        )
        if any(value <= 0 for value in sizes):
            raise ValueError("benchmark sizes and epoch counts must be positive.")
        if self.psgld_burn_in < 0:
            raise ValueError("psgld_burn_in must not be negative.")
        if not (
            self.evaluation_min < self.train_min < self.train_max < self.evaluation_max
        ):
            raise ValueError(
                "evaluation bounds must strictly contain the training interval."
            )


@dataclass(frozen=True)
class XSinData:
    """Training observations and a noiseless evaluation grid.

    Attributes:
        train_inputs: Inputs used to fit the benchmark models.
        train_targets: Noisy observations used to fit the benchmark models.
        evaluation_inputs: Grid used to compare predictions with truth.
        true_mean: Exact mean function on ``evaluation_inputs``.
        true_variance: Exact aleatoric variance on ``evaluation_inputs``.
        evaluation_targets: One noisy observation per evaluation input, which
            the NLL and CRPS score predictive distributions against.
    """

    train_inputs: jax.Array
    train_targets: jax.Array
    evaluation_inputs: jax.Array
    true_mean: jax.Array
    true_variance: jax.Array
    evaluation_targets: jax.Array


@dataclass(frozen=True)
class XSinResult:
    """Predictions and comparable error metrics for one benchmark method.

    Attributes:
        mean: Predicted mean on the evaluation grid.
        variance: Predicted aleatoric variance on the evaluation grid.
        mean_rmse: Root mean squared error of ``mean`` against truth.
        variance_rmse: Root mean squared error of ``variance`` against truth.
        interpolation_mean_rmse: Mean RMSE inside the training domain.
        interpolation_variance_rmse: Variance RMSE inside the training domain.
        extrapolation_mean_rmse: Mean RMSE outside the training domain.
        extrapolation_variance_rmse: Variance RMSE outside the training domain.
    """

    mean: jax.Array
    variance: jax.Array
    mean_rmse: float
    variance_rmse: float
    interpolation_mean_rmse: float
    interpolation_variance_rmse: float
    extrapolation_mean_rmse: float
    extrapolation_variance_rmse: float


@dataclass(frozen=True)
class StagePrintingEventSink:
    """Print periodic epoch losses from multiple supervised stages.

    Attributes:
        every: Positive epoch interval between printed events.
    """

    every: int = 50

    def __post_init__(self) -> None:
        """Validate the reporting interval.

        Raises:
            ValueError: If ``every`` is not positive.
        """
        if self.every <= 0:
            raise ValueError("every must be positive.")

    def on_event(self, event: TrainingEvent) -> None:
        """Print one stage-qualified epoch loss when reporting is due.

        Args:
            event: Training event emitted by either supervised stage.
        """
        if event.name != "epoch_end" or event.step % self.every != 0:
            return
        print(
            f"stage={event.stage} epoch={event.step} loss={event.metrics['loss']:.6f}"
        )


class XSinBackbone(nnx.Module):
    """Two-layer SiLU backbone used by all XSin models.

    It maps the training domain onto ``[-1, 1]`` before the first layer, and
    its unbounded activation lets outputs keep changing away from that
    domain instead of saturating.
    """

    def __init__(
        self,
        hidden_features: int,
        *,
        train_domain: tuple[float, float],
        rngs: nnx.Rngs,
    ) -> None:
        train_min, train_max = train_domain
        self.input_center = (train_min + train_max) / 2
        self.input_half_width = (train_max - train_min) / 2
        self.input_layer = nnx.Linear(1, hidden_features, rngs=rngs)
        self.output_layer = nnx.Linear(hidden_features, hidden_features, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        scaled = (inputs - self.input_center) / self.input_half_width
        hidden = nnx.silu(self.input_layer(scaled))
        return nnx.silu(self.output_layer(hidden))


class XSinMeanModel(nnx.Module):
    """Deterministic mean regressor for the XSin benchmark."""

    def __init__(
        self,
        hidden_features: int,
        *,
        train_domain: tuple[float, float],
        rngs: nnx.Rngs,
    ) -> None:
        self.backbone = XSinBackbone(
            hidden_features, train_domain=train_domain, rngs=rngs
        )
        self.output = nnx.Linear(hidden_features, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.output(self.backbone(inputs))


class XSinGaussianModel(nnx.Module):
    """Joint Gaussian mean/scale regressor for the MVE comparison."""

    def __init__(
        self,
        hidden_features: int,
        *,
        train_domain: tuple[float, float],
        rngs: nnx.Rngs,
    ) -> None:
        self.backbone = XSinBackbone(
            hidden_features, train_domain=train_domain, rngs=rngs
        )
        self.head = GaussianHead(hidden_features, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> Gaussian:
        return self.head(self.backbone(inputs))


class XSinGammaModel(nnx.Module):
    """Gamma residual regressor for the two-step comparison."""

    def __init__(
        self,
        hidden_features: int,
        *,
        train_domain: tuple[float, float],
        rngs: nnx.Rngs,
    ) -> None:
        self.backbone = XSinBackbone(
            hidden_features, train_domain=train_domain, rngs=rngs
        )
        self.head = GammaHead(hidden_features, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> Gamma:
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
        minval=config.train_min,
        maxval=config.train_max,
    )
    train_targets = xsin_mean(train_inputs) + jnp.sqrt(
        xsin_variance(train_inputs)
    ) * jax.random.normal(noise_key, train_inputs.shape)
    evaluation_inputs = jnp.linspace(
        config.evaluation_min,
        config.evaluation_max,
        config.evaluation_size,
    ).reshape(-1, 1)
    true_mean = xsin_mean(evaluation_inputs)
    true_variance = xsin_variance(evaluation_inputs)
    evaluation_noise = jax.random.normal(
        jax.random.fold_in(key, 2), evaluation_inputs.shape
    )
    return XSinData(
        train_inputs=train_inputs,
        train_targets=train_targets,
        evaluation_inputs=evaluation_inputs,
        true_mean=true_mean,
        true_variance=true_variance,
        evaluation_targets=true_mean + jnp.sqrt(true_variance) * evaluation_noise,
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
        train_domain=(config.train_min, config.train_max),
        rngs=nnx.Rngs(model_key),
    )
    optimizer = create_optimizer(model, optax.adam(config.learning_rate))
    state = initialize_training_state(model, optimizer, rng_key=train_key)
    run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=make_xsin_loader(data, config),
        loss=make_supervised_loss(NegativeLogLikelihoodLoss()),
        state=state,
        epochs=config.mve_epochs,
        stage="xsin_mve",
    )
    prediction = model(data.evaluation_inputs)
    return summarize_xsin(
        prediction.mean(),
        prediction.variance(),
        data,
        config,
    )


def run_xsin_two_step(
    data: XSinData,
    config: XSinConfig,
    *,
    event_sinks: Sequence[EventSink] = (),
) -> XSinResult:
    """Train and evaluate separate mean and Gamma variance stages.

    Args:
        data: Shared XSin data.
        config: Benchmark configuration.
        event_sinks: Sinks shared by the mean and variance stages.

    Returns:
        Two-step predictions and comparison metrics.
    """
    _, mean_model, variance_model = train_xsin_two_step(
        data, config, event_sinks=event_sinks
    )
    return summarize_xsin(
        mean_model(data.evaluation_inputs),
        variance_model(data.evaluation_inputs).mean(),
        data,
        config,
    )


def train_xsin_two_step(
    data: XSinData,
    config: XSinConfig,
    *,
    event_sinks: Sequence[EventSink],
) -> tuple[TrainingState, XSinMeanModel, XSinGammaModel]:
    """Train the mean and Gamma variance stages on shared state.

    Args:
        data: Shared XSin data.
        config: Benchmark configuration.
        event_sinks: Sinks shared by the mean and variance stages.

    Returns:
        The trained state, mean model and variance model.
    """
    mean_key, variance_key, train_key = jax.random.split(
        jax.random.key(config.seed + 2),
        3,
    )
    loader = make_xsin_loader(data, config)
    state = TrainingState(rng_state=train_key)

    mean_model = XSinMeanModel(
        config.hidden_features,
        train_domain=(config.train_min, config.train_max),
        rngs=nnx.Rngs(mean_key),
    )
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(
            mean_model,
            optax.adam(config.learning_rate),
        ),
        train_loader=loader,
        options=SupervisedStageOptions(
            epochs=config.mean_epochs,
            event_sinks=event_sinks,
        ),
    )
    mean_stage.prepare(state)
    mean_stage.train(state)

    variance_model = XSinGammaModel(
        config.hidden_features,
        train_domain=(config.train_min, config.train_max),
        rngs=nnx.Rngs(variance_key),
    )
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(
            variance_model,
            optax.adam(config.learning_rate),
        ),
        source_loader=loader,
        options=SupervisedStageOptions(
            epochs=config.variance_epochs,
            event_sinks=event_sinks,
        ),
        splits=("train",),
    )
    variance_stage.prepare(state)
    variance_stage.train(state)
    return state, mean_model, variance_model


def print_xsin_metrics(method: str, result: XSinResult) -> None:
    """Print the common XSin comparison metrics.

    Args:
        method: Human-readable method label.
        result: Benchmark result to report.
    """
    print(f"method={method}")
    print(f"mean_rmse={result.mean_rmse:.6f}")
    print(f"aleatoric_variance_rmse={result.variance_rmse:.6f}")
    print(f"interpolation_mean_rmse={result.interpolation_mean_rmse:.6f}")
    print(
        "interpolation_aleatoric_variance_rmse="
        f"{result.interpolation_variance_rmse:.6f}"
    )
    print(f"extrapolation_mean_rmse={result.extrapolation_mean_rmse:.6f}")
    print(
        "extrapolation_aleatoric_variance_rmse="
        f"{result.extrapolation_variance_rmse:.6f}"
    )


def plot_xsin_result(
    method: str,
    data: XSinData,
    result: XSinResult,
    config: XSinConfig,
) -> None:
    """Show a common interactive mean/aleatoric comparison plot.

    The mean panel shades the true and predicted 95% intervals: each mean plus
    or minus ``INTERVAL_MULTIPLIER`` standard deviations of the aleatoric noise,
    which for a Gaussian is ``loc ± 1.96 * scale``.

    Args:
        method: Human-readable method label used in the figure title.
        data: Shared observations and exact benchmark functions.
        result: Predicted mean and aleatoric variance.
        config: Benchmark domains used to mark extrapolation regions.
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
    true_half_width = INTERVAL_MULTIPLIER * jnp.sqrt(true_variance)
    predicted_half_width = INTERVAL_MULTIPLIER * jnp.sqrt(predicted_variance)
    mean_axis.fill_between(
        inputs,
        true_mean - true_half_width,
        true_mean + true_half_width,
        color="black",
        alpha=0.1,
        label="true 95% interval",
    )
    mean_axis.fill_between(
        inputs,
        predicted_mean - predicted_half_width,
        predicted_mean + predicted_half_width,
        color="C1",
        alpha=0.2,
        label="95% predictive interval",
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

    for axis in (mean_axis, variance_axis):
        axis.axvline(config.train_min, color="C3", linestyle="--")
        axis.axvline(config.train_max, color="C3", linestyle="--")
        axis.axvspan(
            config.evaluation_min,
            config.train_min,
            color="C3",
            alpha=0.05,
        )
        axis.axvspan(
            config.train_max,
            config.evaluation_max,
            color="C3",
            alpha=0.05,
        )

    figure.suptitle(
        f"{method}: mean RMSE={result.mean_rmse:.4f}, "
        f"variance RMSE={result.variance_rmse:.4f}"
    )
    figure.tight_layout()
    plt.show()


def summarize_xsin(
    mean: jax.Array,
    variance: jax.Array,
    data: XSinData,
    config: XSinConfig,
) -> XSinResult:
    """Build common RMSE summaries for mean and aleatoric variance.

    Args:
        mean: Predicted mean on the evaluation grid.
        variance: Predicted aleatoric variance on the evaluation grid.
        data: Shared XSin data.
        config: Benchmark configuration.

    Returns:
        The predictions with their RMSEs over all, interpolation and
        extrapolation inputs.
    """
    mean_errors = jnp.square(mean - data.true_mean)
    variance_errors = jnp.square(variance - data.true_variance)
    interpolation = interpolation_mask(data, config)
    extrapolation = ~interpolation
    return XSinResult(
        mean=mean,
        variance=variance,
        mean_rmse=_masked_rmse(mean_errors, jnp.ones_like(interpolation, dtype=bool)),
        variance_rmse=_masked_rmse(
            variance_errors,
            jnp.ones_like(interpolation, dtype=bool),
        ),
        interpolation_mean_rmse=_masked_rmse(mean_errors, interpolation),
        interpolation_variance_rmse=_masked_rmse(variance_errors, interpolation),
        extrapolation_mean_rmse=_masked_rmse(mean_errors, extrapolation),
        extrapolation_variance_rmse=_masked_rmse(variance_errors, extrapolation),
    )


def _masked_rmse(errors: jax.Array, mask: jax.Array) -> float:
    """Return root mean squared error over a one-dimensional boolean mask."""
    flattened_errors = errors.reshape(-1)
    return float(jnp.sqrt(jnp.mean(flattened_errors[mask])))


def interpolation_mask(data: XSinData, config: XSinConfig) -> jax.Array:
    """Return which evaluation inputs lie strictly inside the training domain.

    Args:
        data: Shared XSin data.
        config: Benchmark configuration.

    Returns:
        A flat boolean mask over the evaluation inputs.
    """
    return (
        (data.evaluation_inputs > config.train_min)
        & (data.evaluation_inputs < config.train_max)
    ).reshape(-1)
