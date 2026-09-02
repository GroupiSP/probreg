from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probreg.core.metric_registry import (
    ContinuousRankedProbabilityScore,
    EpochPredictionData,
    EvaluationGrid,
    IntervalCoverage,
    MetricRequirements,
    PointContinuousRankedProbabilityScore,
    PredictionInterval,
    RootMeanSquaredError,
    WeightedSpread,
)
from probreg.core.types import Batch
from probreg.jax.distributions import Gaussian
from probreg.jax.metrics import (
    GaussianPredictor,
    MetricSuite,
    PredictionRequirements,
    merge_epoch_prediction_data,
    metric_key,
    resolve_epoch_prediction_data,
)


class GaussianModel(nnx.Module):
    def __call__(self, inputs: jax.Array) -> Gaussian:
        return Gaussian(loc=2.0 * inputs, scale=jnp.ones_like(inputs) * 0.5)


class BroadcastScaleGaussianModel(nnx.Module):
    def __call__(self, inputs: jax.Array) -> Gaussian:
        return Gaussian(loc=2.0 * inputs, scale=jnp.array(0.5))


class VectorEventGaussian(Gaussian):
    @property
    def event_shape(self) -> tuple[int, ...]:
        return (1,)


class VectorEventModel(nnx.Module):
    def __call__(self, inputs: jax.Array) -> Gaussian:
        return VectorEventGaussian(loc=inputs, scale=jnp.ones_like(inputs))


class StatefulGaussianModel(nnx.Module):
    def __init__(self) -> None:
        self.call_count = nnx.Variable(jnp.array(0))

    def __call__(self, inputs: jax.Array) -> Gaussian:
        self.call_count[...] += 1
        return Gaussian(loc=inputs, scale=jnp.ones_like(inputs))


def coordinate_from_batch(batch: Batch) -> object:
    return batch.metadata["coordinate"]


def reference_from_batch(batch: Batch, key: jax.Array) -> object:
    del key
    return batch.metadata["reference_samples"]


def stateful_predictor(
    model: nnx.Module,
    batch: Batch,
    requirements: PredictionRequirements,
    key: jax.Array,
) -> EpochPredictionData:
    del requirements, key
    prediction = model(batch.inputs)
    assert isinstance(prediction, Gaussian)
    assert batch.targets is not None
    return EpochPredictionData(
        targets=np.asarray(batch.targets).reshape(-1),
        mean=np.asarray(prediction.mean()).reshape(-1),
    )


def test_suite_combines_requirements_and_requires_explicit_sampling_config() -> None:
    predictor = GaussianPredictor(coordinate_extractor=coordinate_from_batch)
    grid = EvaluationGrid(np.linspace(-5.0, 5.0, 51))
    suite = MetricSuite(
        epoch=(
            RootMeanSquaredError(),
            IntervalCoverage(level=0.9),
            WeightedSpread(level=0.8),
            PointContinuousRankedProbabilityScore(),
        ),
        predictor=predictor,
        predictive_sample_count=16,
        evaluation_grid=grid,
    )

    requirements = suite.prediction_requirements
    assert requirements.fields.predictive_samples
    assert requirements.fields.coordinate
    assert requirements.fields.interval_levels == frozenset({0.8, 0.9})
    assert requirements.predictive_sample_count == 16
    assert requirements.evaluation_grid is grid

    with pytest.raises(ValueError, match="predictive_sample_count"):
        MetricSuite(
            epoch=(PointContinuousRankedProbabilityScore(),),
            predictor=GaussianPredictor(),
            evaluation_grid=grid,
        )
    with pytest.raises(ValueError, match="evaluation_grid"):
        MetricSuite(
            epoch=(PointContinuousRankedProbabilityScore(),),
            predictor=GaussianPredictor(),
            predictive_sample_count=4,
        )
    with pytest.raises(ValueError, match="explicit suite predictor"):
        MetricSuite(epoch=(RootMeanSquaredError(),))


def test_gaussian_predictor_rejects_non_scalar_events() -> None:
    batch = Batch(inputs=jnp.ones((2, 1)), targets=jnp.ones((2, 1)))
    suite = MetricSuite(epoch=(RootMeanSquaredError(),), predictor=GaussianPredictor())

    with pytest.raises(ValueError, match="scalar distribution events only"):
        resolve_epoch_prediction_data(
            suite=suite,
            model=VectorEventModel(),
            batch=batch,
            key=jax.random.key(0),
        )


def test_gaussian_predictor_materializes_only_requested_fields() -> None:
    batch = Batch(
        inputs=jnp.array([[0.0], [1.0]]),
        targets=jnp.array([[0.25], [1.75]]),
    )
    suite = MetricSuite(epoch=(RootMeanSquaredError(),), predictor=GaussianPredictor())

    data = resolve_epoch_prediction_data(
        suite=suite,
        model=GaussianModel(),
        batch=batch,
        key=jax.random.key(0),
    )

    assert np.array_equal(data.targets, np.array([0.25, 1.75]))
    assert np.array_equal(data.mean, np.array([0.0, 2.0]))
    assert data.variance is None
    assert data.predictive_samples is None
    assert data.intervals == ()


def test_gaussian_predictor_broadcasts_variance_to_prediction_shape() -> None:
    batch = Batch(
        inputs=jnp.array([[0.0], [1.0]]),
        targets=jnp.array([[0.25], [1.75]]),
    )
    data = GaussianPredictor()(
        BroadcastScaleGaussianModel(),
        batch,
        PredictionRequirements(fields=MetricRequirements(variance=True)),
        jax.random.key(0),
    )

    assert data.mean.shape == (2,)
    assert data.variance is not None
    assert data.variance.shape == (2,)
    assert np.array_equal(data.variance, np.full(2, 0.25))


def test_epoch_prediction_resolver_does_not_mutate_the_live_model() -> None:
    model = StatefulGaussianModel()
    batch = Batch(inputs=jnp.ones((2, 1)), targets=jnp.ones((2, 1)))
    suite = MetricSuite(epoch=(RootMeanSquaredError(),), predictor=stateful_predictor)

    resolve_epoch_prediction_data(
        suite=suite,
        model=model,
        batch=batch,
        key=jax.random.key(0),
    )

    assert int(model.call_count[...]) == 0


def test_gaussian_predictor_materializes_samples_intervals_and_coordinate() -> None:
    grid = EvaluationGrid(np.linspace(-5.0, 5.0, 51))
    batch = Batch(
        inputs=jnp.array([[0.0], [1.0], [2.0]]),
        targets=jnp.array([[0.0], [2.0], [4.0]]),
        metadata={"coordinate": jnp.array([0.0, 1.0, 2.0])},
    )
    suite = MetricSuite(
        epoch=(
            PointContinuousRankedProbabilityScore(),
            IntervalCoverage(level=0.8),
            WeightedSpread(level=0.8),
        ),
        predictor=GaussianPredictor(coordinate_extractor=coordinate_from_batch),
        predictive_sample_count=7,
        evaluation_grid=grid,
    )

    data = resolve_epoch_prediction_data(
        suite=suite,
        model=GaussianModel(),
        batch=batch,
        key=jax.random.key(4),
    )

    assert data.predictive_samples is not None
    assert data.predictive_samples.shape == (3, 7)
    assert data.coordinate is not None
    assert np.array_equal(data.coordinate, np.array([0.0, 1.0, 2.0]))
    interval = data.interval(0.8)
    assert np.all(interval.lower < data.mean)
    assert np.all(interval.upper > data.mean)
    assert data.evaluation_grid is grid


def test_gaussian_predictor_is_deterministic_for_metric_key() -> None:
    batch = Batch(inputs=jnp.ones((2, 1)), targets=jnp.ones((2, 1)))
    suite = MetricSuite(
        epoch=(PointContinuousRankedProbabilityScore(),),
        predictor=GaussianPredictor(),
        predictive_sample_count=8,
        evaluation_grid=EvaluationGrid(np.linspace(-3.0, 3.0, 31)),
    )

    first = resolve_epoch_prediction_data(
        suite=suite,
        model=GaussianModel(),
        batch=batch,
        key=metric_key(jax.random.key(1)),
    )
    second = resolve_epoch_prediction_data(
        suite=suite,
        model=GaussianModel(),
        batch=batch,
        key=metric_key(jax.random.key(1)),
    )

    assert np.array_equal(first.predictive_samples, second.predictive_samples)


def test_conditional_crps_reference_samples_use_explicit_extractor() -> None:
    reference = jnp.array([[0.0, 0.1], [2.0, 2.1]])
    batch = Batch(
        inputs=jnp.array([[0.0], [1.0]]),
        targets=jnp.array([[0.0], [2.0]]),
        metadata={"reference_samples": reference},
    )
    suite = MetricSuite(
        epoch=(ContinuousRankedProbabilityScore(),),
        predictor=GaussianPredictor(reference_samples_extractor=reference_from_batch),
        predictive_sample_count=3,
        evaluation_grid=EvaluationGrid(np.linspace(-3.0, 3.0, 31)),
    )

    data = resolve_epoch_prediction_data(
        suite=suite, model=GaussianModel(), batch=batch, key=jax.random.key(0)
    )

    assert np.array_equal(data.reference_samples, np.asarray(reference))


def _part(
    targets: list[float],
    *,
    draws: int = 3,
    grid: EvaluationGrid | None = None,
) -> EpochPredictionData:
    n_units = len(targets)
    return EpochPredictionData(
        targets=np.asarray(targets),
        mean=np.asarray(targets),
        variance=np.ones(n_units),
        predictive_samples=np.ones((n_units, draws)),
        intervals=(PredictionInterval(0.9, np.zeros(n_units), np.full(n_units, 2.0)),),
        coordinate=np.arange(n_units, dtype=float),
        evaluation_grid=grid,
    )


def test_merge_handles_unequal_batch_sizes_by_scoring_axis() -> None:
    grid = EvaluationGrid(np.linspace(-2.0, 2.0, 9))

    merged = merge_epoch_prediction_data(
        [_part([0.0, 1.0], grid=grid), _part([2.0], grid=grid)]
    )

    assert merged.targets.shape == (3,)
    assert merged.predictive_samples is not None
    assert merged.predictive_samples.shape == (3, 3)
    assert merged.interval(0.9).lower.shape == (3,)
    assert merged.evaluation_grid is grid


def test_merge_rejects_conflicting_draw_counts_and_grids() -> None:
    first_grid = EvaluationGrid(np.array([0.0, 1.0, 2.0]))
    second_grid = EvaluationGrid(np.array([0.0, 1.5, 2.0]))

    with pytest.raises(ValueError, match="draw counts"):
        merge_epoch_prediction_data([_part([0.0], draws=2), _part([1.0], draws=3)])
    with pytest.raises(ValueError, match="shared"):
        merge_epoch_prediction_data(
            [_part([0.0], grid=first_grid), _part([1.0], grid=second_grid)]
        )


def test_weighted_spread_is_independent_of_loader_coordinate_order() -> None:
    first = EpochPredictionData(
        targets=np.zeros(2),
        mean=np.zeros(2),
        intervals=(
            PredictionInterval(0.9, np.array([1.0, 1.0]), np.array([3.0, 5.0])),
        ),
        coordinate=np.array([2.0, 3.0]),
    )
    second = EpochPredictionData(
        targets=np.zeros(2),
        mean=np.zeros(2),
        intervals=(
            PredictionInterval(0.9, np.array([0.0, 1.0]), np.array([2.0, 3.0])),
        ),
        coordinate=np.array([0.0, 1.0]),
    )

    merged = merge_epoch_prediction_data([first, second])

    assert WeightedSpread(level=0.9)(merged) == pytest.approx(
        WeightedSpread(level=0.9)(
            EpochPredictionData(
                targets=np.zeros(4),
                mean=np.zeros(4),
                intervals=(
                    PredictionInterval(
                        0.9,
                        np.array([0.0, 1.0, 1.0, 1.0]),
                        np.array([2.0, 3.0, 3.0, 5.0]),
                    ),
                ),
                coordinate=np.array([0.0, 1.0, 2.0, 3.0]),
            )
        )
    )
