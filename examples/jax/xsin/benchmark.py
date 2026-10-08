"""Shared XSin-inspired benchmark utilities for probabilistic regression examples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from probreg.core import (
    EvaluationGrid,
    NegativeLogLikelihood,
    PointContinuousRankedProbabilityScore,
)
from probreg.core.losses import NegativeLogLikelihoodLoss
from probreg.core.protocols import LoaderFactory
from probreg.core.tracking import EventSink, TrainingEvent
from probreg.core.types import Batch, TrainingState
from probreg.jax import (
    BayesByBackprop,
    Gamma,
    GammaHead,
    GammaVarianceStage,
    Gaussian,
    GaussianHead,
    GaussianPredictor,
    InferenceMethod,
    MeanStage,
    MetricSuite,
    Posterior,
    PosteriorPredictive,
    PosteriorPredictivePredictor,
    PosteriorStage,
    PosteriorStageOptions,
    PreconditionedSGLD,
    SupervisedStageOptions,
    create_optimizer,
    evaluate_loader,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

INTERVAL_MULTIPLIER = 1.96
"""Half-width of the 95% predictive interval, in predictive scales."""

CRPS_GRID = EvaluationGrid(np.linspace(-40.0, 40.0, 801))
"""Target grid the point CRPS is integrated over; it covers every XSin target."""


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
            after burn-in. ``psgld_burn_in + psgld_thinning`` must not exceed
            the batches of one epoch, so the posterior stage's first
            validation finds a retained sample.
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

    @property
    def batches_per_epoch(self) -> int:
        """The number of training batches in one epoch."""
        return math.ceil(self.train_size / self.batch_size)


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

    def __call__(self, inputs: jax.Array) -> Gaussian:
        return self.head(self.backbone(inputs))


class XSinGammaModel(nnx.Module):
    """Gamma residual regressor for the two-step comparison."""

    def __init__(self, hidden_features: int, *, rngs: nnx.Rngs) -> None:
        self.backbone = XSinBackbone(hidden_features, rngs=rngs)
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
    return _summarize(
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
    _, mean_model, variance_model = _train_xsin_two_step(
        data, config, event_sinks=event_sinks
    )
    return _summarize(
        mean_model(data.evaluation_inputs),
        variance_model(data.evaluation_inputs).mean(),
        data,
        config,
    )


@dataclass(frozen=True)
class XSinScores:
    """NLL and point CRPS of a predictive distribution on the evaluation targets.

    Attributes:
        nll: Mean negative log-likelihood over the whole evaluation grid.
        crps: Mean point CRPS over the whole evaluation grid.
        interpolation_nll: NLL inside the training domain.
        interpolation_crps: CRPS inside the training domain.
        extrapolation_nll: NLL outside the training domain.
        extrapolation_crps: CRPS outside the training domain.
    """

    nll: float
    crps: float
    interpolation_nll: float
    interpolation_crps: float
    extrapolation_nll: float
    extrapolation_crps: float


@dataclass(frozen=True)
class XSinPosteriorResult:
    """The posterior predictive of a mean, variance and posterior run.

    Attributes:
        variance_stage: The two-step prediction the posterior stage starts
            from: the trained mean and the aleatoric variance.
        mean: Posterior predictive mean on the evaluation grid, the average
            of the draws' means.
        aleatoric_variance: Aleatoric variance on the evaluation grid.
        epistemic_variance: Variance of the draws' means on the evaluation
            grid.
        variance_stage_scores: Scores of the variance stage's Gaussian.
        posterior_scores: Scores of the exact posterior predictive.
    """

    variance_stage: XSinResult
    mean: jax.Array
    aleatoric_variance: jax.Array
    epistemic_variance: jax.Array
    variance_stage_scores: XSinScores
    posterior_scores: XSinScores


class XSinTwoStepGaussian(nnx.Module):
    """The variance stage's Gaussian: the trained mean and aleatoric scale."""

    def __init__(self, mean_model: nnx.Module, variance_model: nnx.Module) -> None:
        self.mean_model = mean_model
        self.variance_model = variance_model

    def __call__(self, inputs: jax.Array) -> Gaussian:
        return Gaussian(
            loc=self.mean_model(inputs),
            scale=jnp.sqrt(self.variance_model(inputs).mean()),
        )


class XSinPosteriorPredictive(nnx.Module):
    """The posterior predictive at inputs, for a fixed draw key.

    Every call takes the same draws, so draw ``s`` is the same mean function
    at every input. A finite posterior uses all its draws; an unlimited one
    takes ``num_draws``.
    """

    def __init__(
        self,
        posterior: Posterior,
        variance_model: nnx.Module,
        *,
        draw_key: jax.Array,
        num_draws: int,
    ) -> None:
        self.posterior = nnx.static(posterior)
        self.variance_model = variance_model
        self.draw_key = draw_key
        self.num_draws = None if posterior.num_draws is not None else num_draws

    def __call__(self, inputs: jax.Array) -> PosteriorPredictive:
        return PosteriorPredictive(
            draws=self.posterior.sample_means(inputs, self.draw_key, self.num_draws),
            aleatoric_variance=self.variance_model(inputs).mean(),
        )


def xsin_bayes_by_backprop(config: XSinConfig) -> BayesByBackprop:
    """Build the Bayes by Backprop inference method of the XSin example.

    Args:
        config: Benchmark configuration.

    Returns:
        A fresh, uninitialized inference method.
    """
    return BayesByBackprop(
        optimizer=optax.adam(config.bbb_learning_rate),
        initial_std=config.bbb_initial_std,
    )


def xsin_psgld(config: XSinConfig) -> PreconditionedSGLD:
    """Build the pSGLD inference method of the XSin example.

    Args:
        config: Benchmark configuration.

    Returns:
        A fresh, uninitialized inference method.

    Raises:
        ValueError: If ``psgld_burn_in + psgld_thinning`` exceeds the batches
            of one epoch. The posterior stage validates after every epoch, and
            pSGLD has no posterior until its first sample is retained.
    """
    if config.psgld_burn_in + config.psgld_thinning > config.batches_per_epoch:
        raise ValueError(
            "psgld_burn_in + psgld_thinning must fit in the first epoch's "
            f"{config.batches_per_epoch} batches."
        )
    return PreconditionedSGLD(
        step_size=config.psgld_step_size,
        burn_in=config.psgld_burn_in,
        thinning=config.psgld_thinning,
    )


def run_xsin_posterior(
    data: XSinData,
    config: XSinConfig,
    inference_method: InferenceMethod,
    *,
    event_sinks: Sequence[EventSink] = (),
) -> XSinPosteriorResult:
    """Train mean, variance and posterior stages, then score both predictives.

    The posterior stage validates every epoch on the training data with its
    default metrics, the NLL and the CRPS, reported as
    ``posterior/validation/nll`` and ``posterior/validation/crps``.

    Args:
        data: Shared XSin data.
        config: Benchmark configuration.
        inference_method: A fresh inference method, e.g. from
            ``xsin_bayes_by_backprop`` or ``xsin_psgld``.
        event_sinks: Sinks shared by all three stages.

    Returns:
        The posterior predictive on the evaluation grid and the scores of the
        variance stage's Gaussian and of the posterior predictive.
    """
    state, mean_model, variance_model = _train_xsin_two_step(
        data, config, event_sinks=event_sinks
    )
    loader = make_xsin_loader(data, config)
    posterior_stage = PosteriorStage(
        inference_method=inference_method,
        train_loader=loader,
        dataset_size=config.train_size,
        options=PosteriorStageOptions(
            epochs=config.posterior_epochs,
            num_draws=config.posterior_num_draws,
            validation_loader=loader,
            event_sinks=event_sinks,
        ),
    )
    posterior_stage.prepare(state)
    posterior_stage.train(state)

    predictive = XSinPosteriorPredictive(
        state.model_components[posterior_stage.model_name],
        variance_model,
        draw_key=jax.random.key(config.seed + 3),
        num_draws=config.posterior_num_draws,
    )
    moments = predictive(data.evaluation_inputs).moment_matched()
    score_key = jax.random.key(config.seed + 4)
    return XSinPosteriorResult(
        variance_stage=_summarize(
            mean_model(data.evaluation_inputs),
            variance_model(data.evaluation_inputs).mean(),
            data,
            config,
        ),
        mean=moments.loc,
        aleatoric_variance=moments.aleatoric_variance,
        epistemic_variance=moments.epistemic_variance,
        variance_stage_scores=_score(
            XSinTwoStepGaussian(mean_model, variance_model),
            GaussianPredictor(),
            data,
            config,
            key=score_key,
        ),
        posterior_scores=_score(
            predictive,
            PosteriorPredictivePredictor(),
            data,
            config,
            key=score_key,
        ),
    )


def _train_xsin_two_step(
    data: XSinData,
    config: XSinConfig,
    *,
    event_sinks: Sequence[EventSink],
) -> tuple[TrainingState, XSinMeanModel, XSinGammaModel]:
    """Train the mean and Gamma variance stages on shared state."""
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
        options=SupervisedStageOptions(
            epochs=config.mean_epochs,
            event_sinks=event_sinks,
        ),
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
        options=SupervisedStageOptions(
            epochs=config.variance_epochs,
            event_sinks=event_sinks,
        ),
        splits=("train",),
    )
    variance_stage.prepare(state)
    variance_stage.train(state)
    return state, mean_model, variance_model


def _score_suite(
    predictor: GaussianPredictor | PosteriorPredictivePredictor,
) -> MetricSuite:
    """Register the NLL and the point CRPS, reported as ``nll`` and ``crps``."""
    return MetricSuite(
        epoch=(
            NegativeLogLikelihood(),
            PointContinuousRankedProbabilityScore(name="crps"),
        ),
        predictor=predictor,
        predictive_sample_count=128,
        evaluation_grid=CRPS_GRID,
    )


def _score(
    model: nnx.Module,
    predictor: GaussianPredictor | PosteriorPredictivePredictor,
    data: XSinData,
    config: XSinConfig,
    *,
    key: jax.Array,
) -> XSinScores:
    """Score a predictive model on all, interpolation and extrapolation targets."""
    interpolation = _interpolation_mask(data, config)
    regions = {
        "": jnp.ones_like(interpolation),
        "interpolation_": interpolation,
        "extrapolation_": ~interpolation,
    }
    values: dict[str, float] = {}
    for prefix, mask in regions.items():
        batch = Batch(
            inputs=data.evaluation_inputs[mask],
            targets=data.evaluation_targets[mask],
        )
        metrics, _ = evaluate_loader(
            model, [batch], key=key, metrics=_score_suite(predictor)
        )
        values[f"{prefix}nll"] = metrics["nll"]
        values[f"{prefix}crps"] = metrics["crps"]
    return XSinScores(**values)


def print_xsin_scores(method: str, result: XSinPosteriorResult) -> None:
    """Print the variance stage's and the posterior predictive's scores.

    Args:
        method: Human-readable inference-method label.
        result: Posterior run to report.
    """
    print(f"method={method}")
    for label, scores in (
        ("variance_stage", result.variance_stage_scores),
        ("posterior", result.posterior_scores),
    ):
        print(
            f"{label}: nll={scores.nll:.4f} crps={scores.crps:.4f} "
            f"interpolation_nll={scores.interpolation_nll:.4f} "
            f"interpolation_crps={scores.interpolation_crps:.4f} "
            f"extrapolation_nll={scores.extrapolation_nll:.4f} "
            f"extrapolation_crps={scores.extrapolation_crps:.4f}"
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


def plot_xsin_posterior(
    method: str,
    data: XSinData,
    result: XSinPosteriorResult,
    config: XSinConfig,
) -> None:
    """Show the posterior predictive band against the variance stage's band.

    The variance stage's band is its mean plus or minus ``INTERVAL_MULTIPLIER``
    aleatoric standard deviations. The posterior band is the moment-matched
    predictive's, whose variance adds the epistemic variance, so it widens
    where the draws disagree. The lower panel shows both variance parts.

    Args:
        method: Human-readable inference-method label used in the title.
        data: Shared observations and exact benchmark functions.
        result: Posterior run to plot.
        config: Benchmark domains used to mark extrapolation regions.
    """
    import matplotlib.pyplot as plt

    inputs = jax.device_get(data.evaluation_inputs).reshape(-1)
    true_mean = jax.device_get(data.true_mean).reshape(-1)
    stage_mean = jax.device_get(result.variance_stage.mean).reshape(-1)
    stage_half_width = INTERVAL_MULTIPLIER * jnp.sqrt(
        result.variance_stage.variance.reshape(-1)
    )
    posterior_mean = jax.device_get(result.mean).reshape(-1)
    aleatoric = jax.device_get(result.aleatoric_variance).reshape(-1)
    epistemic = jax.device_get(result.epistemic_variance).reshape(-1)
    posterior_half_width = INTERVAL_MULTIPLIER * jnp.sqrt(aleatoric + epistemic)

    figure, (mean_axis, variance_axis) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    mean_axis.scatter(
        jax.device_get(data.train_inputs).reshape(-1),
        jax.device_get(data.train_targets).reshape(-1),
        s=8,
        alpha=0.2,
        label="training observations",
    )
    mean_axis.plot(inputs, true_mean, color="black", label="true mean")
    mean_axis.fill_between(
        inputs,
        stage_mean - stage_half_width,
        stage_mean + stage_half_width,
        color="C1",
        alpha=0.25,
        label="variance stage 95% interval",
    )
    mean_axis.fill_between(
        inputs,
        posterior_mean - posterior_half_width,
        posterior_mean + posterior_half_width,
        color="C0",
        alpha=0.25,
        label="moment-matched predictive 95% interval",
    )
    mean_axis.plot(inputs, posterior_mean, color="C0", label="posterior mean")
    mean_axis.set_ylabel("target")
    mean_axis.legend(ncol=2, fontsize="small")

    variance_axis.plot(inputs, aleatoric, color="C1", label="aleatoric variance")
    variance_axis.plot(inputs, epistemic, color="C0", label="epistemic variance")
    variance_axis.set_yscale("log")
    variance_axis.set_xlabel("x")
    variance_axis.set_ylabel("variance")
    variance_axis.legend()

    for axis in (mean_axis, variance_axis):
        axis.axvspan(config.evaluation_min, config.train_min, color="C3", alpha=0.05)
        axis.axvspan(config.train_max, config.evaluation_max, color="C3", alpha=0.05)

    scores = result.posterior_scores
    stage_scores = result.variance_stage_scores
    figure.suptitle(
        f"{method}: NLL {stage_scores.nll:.3f} -> {scores.nll:.3f}, "
        f"CRPS {stage_scores.crps:.3f} -> {scores.crps:.3f}"
    )
    figure.tight_layout()
    plt.show()


def _summarize(
    mean: jax.Array,
    variance: jax.Array,
    data: XSinData,
    config: XSinConfig,
) -> XSinResult:
    """Build common RMSE summaries for mean and aleatoric variance."""
    mean_errors = jnp.square(mean - data.true_mean)
    variance_errors = jnp.square(variance - data.true_variance)
    interpolation = _interpolation_mask(data, config)
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


def _interpolation_mask(data: XSinData, config: XSinConfig) -> jax.Array:
    """Return which evaluation inputs lie strictly inside the training domain."""
    return (
        (data.evaluation_inputs > config.train_min)
        & (data.evaluation_inputs < config.train_max)
    ).reshape(-1)
