"""The posterior stage on the XSin-inspired benchmark.

Bayes by Backprop and pSGLD inference methods, a mean, variance and posterior
run, the NLL and point CRPS of its predictives, and a plot of the posterior
predictive band against the variance stage's band. The data, models and the
two stages it starts from live in ``benchmark.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from benchmark import (
    INTERVAL_MULTIPLIER,
    XSinConfig,
    XSinData,
    XSinResult,
    interpolation_mask,
    make_xsin_loader,
    summarize_xsin,
    train_xsin_two_step,
)
from flax import nnx

from probreg.core import (
    EvaluationGrid,
    NegativeLogLikelihood,
    PointContinuousRankedProbabilityScore,
)
from probreg.core.tracking import EventSink
from probreg.core.types import Batch
from probreg.jax import (
    BayesByBackprop,
    Gaussian,
    GaussianPredictor,
    InferenceMethod,
    MetricSuite,
    Posterior,
    PosteriorPredictive,
    PosteriorPredictivePredictor,
    PosteriorStage,
    PosteriorStageOptions,
    PreconditionedSGLD,
    evaluate_loader,
)

CRPS_GRID = EvaluationGrid(np.linspace(-40.0, 40.0, 801))
"""Target grid the point CRPS is integrated over; it covers every XSin target."""


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
    """
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
    state, mean_model, variance_model = train_xsin_two_step(
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
        variance_stage=summarize_xsin(
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
    interpolation = interpolation_mask(data, config)
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
