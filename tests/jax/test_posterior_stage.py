from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st

from probreg.core.checkpoints import Checkpoint, InMemoryCheckpointStore
from probreg.core.early_stopping import EarlyStopper
from probreg.core.metric_registry import (
    EvaluationGrid,
    NegativeLogLikelihood,
    PointContinuousRankedProbabilityScore,
)
from probreg.core.tracking import EventSink, TrackerEventSink
from probreg.core.types import (
    Batch,
    ParameterRole,
    PyTree,
    StageState,
    TrainingState,
)
from probreg.jax import HeldOutValidation, MetricSuite, PosteriorPredictivePredictor
from probreg.jax.distributions import GammaHead
from probreg.jax.posterior import PosteriorProblem
from probreg.jax.posterior_stage import PosteriorStage, PosteriorStageOptions
from probreg.jax.state import create_optimizer
from probreg.jax.supervised_staged import (
    GammaVarianceStage,
    MeanStage,
    SupervisedStageOptions,
)

Tracker = Callable[[], Any]

_DATASET_SIZE = 16
_BATCH_SIZE = 8


def _regression_data() -> tuple[jax.Array, jax.Array]:
    """Sixteen noisy points on the line ``y = 2x``."""
    data_key = jax.random.key(3)
    inputs = jnp.linspace(-1.0, 1.0, _DATASET_SIZE)[:, None]
    targets = 2.0 * inputs + 0.3 * jax.random.normal(data_key, inputs.shape)
    return inputs, targets


def regression_loader(*, split: str, epoch: int) -> list[Batch]:
    """Two batches of eight; the validation split is shifted by 0.25."""
    del epoch
    inputs, targets = _regression_data()
    offset = 0.25 if split == "validation" else 0.0
    return [
        Batch(
            inputs=inputs[start : start + _BATCH_SIZE],
            targets=targets[start : start + _BATCH_SIZE] + offset,
        )
        for start in range(0, _DATASET_SIZE, _BATCH_SIZE)
    ]


@dataclass
class VarianceReadyRun:
    """A training state after real mean and variance stages."""

    state: TrainingState
    mean_model: nnx.Module
    variance_model: nnx.Module


MakeVarianceReadyRun = Callable[..., VarianceReadyRun]


@pytest.fixture(scope="session")
def variance_ready_run(
    linear_model: type[Any], squared_error: Any
) -> MakeVarianceReadyRun:
    """Return a driver of a real mean-then-variance run on `regression_loader`."""

    def run(*sinks: EventSink, mean_only: bool = False) -> VarianceReadyRun:
        """Train a validated mean stage and, unless ``mean_only``, a variance stage.

        Args:
            *sinks: The event sinks attached to both stages.
            mean_only: Stop after the mean stage.

        Returns:
            The shared state and the two trained models.
        """
        mean_model = linear_model(rngs=nnx.Rngs(0))
        state = TrainingState(rng_state=jax.random.key(1))
        mean_stage = MeanStage(
            model=mean_model,
            optimizer=create_optimizer(mean_model, optax.adam(0.1)),
            train_loader=regression_loader,
            options=SupervisedStageOptions(
                epochs=20,
                validation=HeldOutValidation(
                    model=mean_model, loader=regression_loader, loss=squared_error
                ),
                event_sinks=sinks,
            ),
        )
        mean_stage.prepare(state)
        mean_stage.train(state)
        variance_model = GammaHead(1, 1, rngs=nnx.Rngs(2))
        if mean_only:
            return VarianceReadyRun(state, mean_model, variance_model)
        variance_stage = GammaVarianceStage(
            model=variance_model,
            optimizer=create_optimizer(variance_model, optax.adam(0.05)),
            source_loader=regression_loader,
            options=SupervisedStageOptions(epochs=5, event_sinks=sinks),
            validation_factory=lambda residuals: HeldOutValidation(
                model=variance_model, loader=residuals, loss=variance_stage.loss
            ),
        )
        variance_stage.prepare(state)
        variance_stage.train(state)
        return VarianceReadyRun(state, mean_model, variance_model)

    return run


@pytest.fixture(scope="session")
def shared_variance_ready_run(
    variance_ready_run: MakeVarianceReadyRun,
) -> VarianceReadyRun:
    """One variance-ready run shared by tests that prepare but never train it."""
    return variance_ready_run()


@dataclass(frozen=True)
class OffsetDrawsPosterior:
    """Draws that shift the posterior network's mean by fixed offsets.

    Finite when ``offsets`` is given: then every call returns one draw per
    offset and refuses an explicit draw count. Unlimited otherwise: each draw
    shifts the mean by a standard normal offset drawn from the key.
    """

    mean_function: Callable[[PyTree, PyTree], jax.Array]
    parameters: PyTree
    offsets: tuple[float, ...] | None = None
    calls: list[tuple[jax.Array, int | None]] = field(default_factory=list)

    @property
    def num_draws(self) -> int | None:
        return None if self.offsets is None else len(self.offsets)

    def sample_means(
        self, inputs: PyTree, key: jax.Array, num_samples: int | None = None
    ) -> jax.Array:
        self.calls.append((key, num_samples))
        if self.offsets is not None:
            if num_samples is not None:
                raise ValueError("a finite posterior uses all its draws.")
            offsets = jnp.asarray(self.offsets)
        else:
            if num_samples is None:
                raise ValueError("an unlimited posterior needs num_samples.")
            offsets = jax.random.normal(key, (num_samples,))
        means = self.mean_function(self.parameters, inputs)
        return means[None] + offsets.reshape((-1,) + (1,) * means.ndim)


@dataclass
class GradientAscentMethod:
    """A deterministic inference method: gradient ascent on the log posterior."""

    learning_rate: float = 0.01
    offsets: tuple[float, ...] | None = (-0.5, 0.5)
    early_stopping: bool = True
    problem: PosteriorProblem | None = None
    parameters: PyTree = None
    steps: int = 0
    posteriors: list[OffsetDrawsPosterior] = field(default_factory=list)

    @property
    def supports_early_stopping(self) -> bool:
        return self.early_stopping

    def init(self, problem: PosteriorProblem) -> None:
        self.problem = problem
        self.parameters = problem.initial_parameters
        self.steps = 0

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        del key
        assert self.problem is not None
        problem = self.problem

        def negative_log_posterior(parameters: PyTree) -> jax.Array:
            return -(
                problem.log_likelihood(parameters, batch)
                + problem.prior.log_prob(parameters)
            )

        loss, grads = jax.value_and_grad(negative_log_posterior)(self.parameters)
        self.parameters = jax.tree.map(
            lambda value, grad: value - self.learning_rate * grad,
            self.parameters,
            grads,
        )
        self.steps += 1
        return {"loss": loss}

    def posterior(self) -> OffsetDrawsPosterior:
        assert self.problem is not None
        posterior = OffsetDrawsPosterior(
            self.problem.mean_function, self.parameters, self.offsets
        )
        self.posteriors.append(posterior)
        return posterior

    def state(self) -> PyTree:
        return {"parameters": self.parameters, "steps": jnp.asarray(self.steps)}

    def load_state(self, state: PyTree) -> None:
        self.parameters = state["parameters"]
        self.steps = int(state["steps"])

    def posterior_state(self) -> PyTree:
        return self.parameters

    def load_posterior(self, state: PyTree) -> None:
        self.parameters = state


class RecordingStore(InMemoryCheckpointStore):
    """A store that also records every checkpoint saved, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.saves: list[tuple[str, Checkpoint]] = []

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        self.saves.append((key, checkpoint))
        super().save(key, checkpoint)


def _posterior_stage(method: GradientAscentMethod, **options: Any) -> PosteriorStage:
    return PosteriorStage(
        inference_method=method,
        train_loader=regression_loader,
        dataset_size=_DATASET_SIZE,
        options=PosteriorStageOptions(**{"epochs": 3, **options}),
    )


def _leaves_equal(left: PyTree, right: PyTree) -> bool:
    return all(
        jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _observable(state: TrainingState) -> tuple[Any, ...]:
    """Everything a refused prepare must leave unchanged, copied."""
    return (
        dict(state.model_components),
        dict(state.parameter_roles),
        state.frozen_components,
        dict(state.optimizer_states),
        state.posterior_state,
        state.rng_state,
        state.lifecycle_state,
        state.stage,
        {tag: list(values) for tag, values in state.metric_history.items()},
    )


def _copy(module: nnx.Module) -> PyTree:
    return jax.tree.map(jnp.copy, nnx.state(module))


@pytest.mark.parametrize("mean_only", [False, True])
def test_posterior_stage_refuses_to_run_without_ready_mean_and_variance(
    variance_ready_run: MakeVarianceReadyRun, mean_only: bool
) -> None:
    run = variance_ready_run(mean_only=mean_only)
    if not mean_only:
        run.state.lifecycle_state = StageState.MEAN_READY
    stage = _posterior_stage(GradientAscentMethod())
    before = _observable(run.state)

    with pytest.raises(ValueError, match="VARIANCE_READY"):
        stage.prepare(run.state)

    assert _observable(run.state) == before


def test_posterior_stage_refuses_a_fresh_state() -> None:
    state = TrainingState(rng_state=jax.random.key(0))

    with pytest.raises(ValueError, match="VARIANCE_READY"):
        _posterior_stage(GradientAscentMethod()).prepare(state)

    assert _observable(state) == _observable(TrainingState(rng_state=state.rng_state))


@pytest.mark.parametrize("network_seed", [None, 42])
def test_the_warm_start_equals_the_mean_weights(
    shared_variance_ready_run: VarianceReadyRun,
    linear_model: type[Any],
    network_seed: int | None,
) -> None:
    run = shared_variance_ready_run
    method = GradientAscentMethod()
    network = (
        None if network_seed is None else linear_model(rngs=nnx.Rngs(network_seed))
    )
    stage = dataclasses.replace(_posterior_stage(method), model=network)

    stage.prepare(run.state)

    assert method.problem is not None
    assert _leaves_equal(
        method.problem.initial_parameters, nnx.state(run.mean_model, nnx.Param)
    )
    inputs, _ = _regression_data()
    assert jnp.allclose(
        method.problem.mean_function(method.problem.initial_parameters, inputs),
        run.mean_model(inputs),
    )


def test_training_leaves_the_mean_and_variance_weights_unchanged(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    mean_before = _copy(run.mean_model)
    variance_before = _copy(run.variance_model)
    method = GradientAscentMethod()
    stage = _posterior_stage(method)

    stage.prepare(run.state)
    initial = method.parameters
    stage.train(run.state)

    assert not _leaves_equal(method.parameters, initial)
    assert _leaves_equal(nnx.state(run.mean_model), mean_before)
    assert _leaves_equal(nnx.state(run.variance_model), variance_before)


def _independent_log_likelihood(
    run: VarianceReadyRun, batch: Batch, dataset_size: int
) -> float:
    """``N`` times the mean per-row Gaussian log-density under the trained models."""
    assert batch.targets is not None
    variance = run.variance_model(batch.inputs).mean()
    rows = jax.scipy.stats.norm.logpdf(
        batch.targets, run.mean_model(batch.inputs), jnp.sqrt(variance)
    )
    return dataset_size * float(jnp.mean(rows))


@settings(deadline=None, max_examples=10)
@given(
    repeats=st.integers(1, 4),
    dataset_size=st.integers(1, 1000),
    rows=st.integers(1, _DATASET_SIZE),
)
def test_the_likelihood_is_scaled_by_dataset_size_over_batch_size(
    shared_variance_ready_run: VarianceReadyRun,
    repeats: int,
    dataset_size: int,
    rows: int,
) -> None:
    run = shared_variance_ready_run
    method = GradientAscentMethod()
    stage = dataclasses.replace(_posterior_stage(method), dataset_size=dataset_size)
    stage.prepare(run.state)
    assert method.problem is not None
    inputs, targets = _regression_data()
    batch = Batch(inputs=inputs[:rows], targets=targets[:rows])
    repeated = Batch(
        inputs=jnp.tile(batch.inputs, (repeats, 1)),
        targets=jnp.tile(batch.targets, (repeats, 1)),
    )

    log_likelihood = method.problem.log_likelihood(
        method.problem.initial_parameters, repeated
    )

    assert float(log_likelihood) == pytest.approx(
        _independent_log_likelihood(run, batch, dataset_size), rel=1e-4
    )


def _validation_suite() -> MetricSuite:
    return MetricSuite(
        epoch=(
            NegativeLogLikelihood(),
            PointContinuousRankedProbabilityScore(name="crps"),
        ),
        predictor=PosteriorPredictivePredictor(),
        predictive_sample_count=64,
        evaluation_grid=EvaluationGrid(np.linspace(-6.0, 6.0, 121)),
    )


def test_the_three_stages_of_a_posterior_run_never_share_a_tag(
    in_memory_tracker: Tracker,
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    recorder = in_memory_tracker()
    tracker = in_memory_tracker()
    sinks = (recorder, TrackerEventSink(tracker))
    run = variance_ready_run(*sinks)
    stage = _posterior_stage(
        GradientAscentMethod(),
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
        event_sinks=sinks,
    )

    stage.prepare(run.state)
    stage.train(run.state)

    expected = {
        "mean/train/loss",
        "mean/validation/loss",
        "variance/train/loss",
        "variance/validation/loss",
        "posterior/train/loss",
        "posterior/validation/nll",
        "posterior/validation/crps",
    }
    assert {
        f"{event.stage}/{event.split}/{metric}"
        for event in recorder.events
        for metric in event.metrics
    } == expected
    assert tracker.tags == expected
    assert set(run.state.metric_history) == expected


def test_validation_scores_the_current_posterior_predictive(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    method = GradientAscentMethod()
    stage = _posterior_stage(
        method,
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
    )

    stage.prepare(run.state)
    stage.train(run.state)

    history = run.state.metric_history
    assert len(history["posterior/validation/nll"]) == 3
    assert all(math.isfinite(value) for value in history["posterior/validation/crps"])
    final = method.posteriors[-1]
    targets = jnp.concatenate(
        [b.targets for b in regression_loader(split="validation", epoch=2)]
    )
    inputs = jnp.concatenate(
        [b.inputs for b in regression_loader(split="validation", epoch=2)]
    )
    draws = (
        final.mean_function(final.parameters, inputs)[None]
        + jnp.array([-0.5, 0.5])[:, None, None]
    )
    scale = jnp.sqrt(run.variance_model(inputs).mean())
    mixture = jsp.special.logsumexp(
        jax.scipy.stats.norm.logpdf(targets, draws, scale), axis=0
    ) - math.log(2.0)
    assert history["posterior/validation/nll"][-1] == pytest.approx(
        -float(jnp.mean(mixture)), rel=1e-4
    )


def test_an_unlimited_posterior_draws_the_configured_count_with_one_key_per_epoch(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    method = GradientAscentMethod(offsets=None)
    stage = _posterior_stage(
        method,
        num_draws=7,
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
    )

    stage.prepare(run.state)
    stage.train(run.state)

    validated = [posterior for posterior in method.posteriors if posterior.calls]
    assert len(validated) == 3
    for posterior in validated:
        keys = [jax.random.key_data(key) for key, _ in posterior.calls]
        assert len(keys) == 2
        assert all(jnp.array_equal(key, keys[0]) for key in keys)
        assert {count for _, count in posterior.calls} == {7}
    first_keys = [jax.random.key_data(p.calls[0][0]) for p in validated]
    assert not jnp.array_equal(first_keys[0], first_keys[1])


def test_an_early_stopper_is_refused_for_a_method_without_early_stopping(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    stage = _posterior_stage(
        GradientAscentMethod(early_stopping=False),
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
        early_stopper=EarlyStopper(metric="nll", mode="min", patience=0),
    )
    before = _observable(run.state)

    with pytest.raises(ValueError, match="early stopping"):
        stage.prepare(run.state)

    assert _observable(run.state) == before


def test_an_early_stopper_stops_a_method_that_supports_it(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    stage = _posterior_stage(
        GradientAscentMethod(learning_rate=0.0),
        epochs=10,
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
        early_stopper=EarlyStopper(metric="nll", mode="min", patience=0),
    )

    stage.prepare(run.state)
    stage.train(run.state)

    assert len(run.state.metric_history["posterior/train/loss"]) < 10


def test_a_trained_posterior_stage_passes_validate(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    method = GradientAscentMethod()
    stage = _posterior_stage(method)

    stage.prepare(run.state)
    assert not stage.validate(run.state).passed
    result = stage.train(run.state)

    assert result.loss is not None and math.isfinite(result.loss)
    assert stage.validate(run.state).passed
    assert run.state.lifecycle_state is StageState.POSTERIOR_READY
    assert run.state.model_components["posterior"] is method.posteriors[-1]
    assert run.state.parameter_roles["posterior"] is ParameterRole.POSTERIOR
    assert {"mean_model", "variance_model"} <= run.state.frozen_components


def test_a_posterior_stage_must_be_prepared_before_training(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()

    with pytest.raises(ValueError, match="prepared"):
        _posterior_stage(GradientAscentMethod()).train(run.state)


def test_a_posterior_network_with_another_parameter_tree_is_refused(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    stage = dataclasses.replace(
        _posterior_stage(GradientAscentMethod()),
        model=nnx.Linear(1, 2, rngs=nnx.Rngs(0)),
    )

    with pytest.raises(ValueError, match="parameter tree"):
        stage.prepare(run.state)


def test_posterior_stage_declares_its_place_in_the_lifecycle() -> None:
    stage = _posterior_stage(GradientAscentMethod())

    assert stage.name == "posterior"
    assert stage.requires == frozenset({"mean", "variance"})
    assert stage.produces == frozenset({"posterior"})


def test_a_method_without_early_stopping_writes_only_a_finalized_checkpoint(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    store = RecordingStore()
    method = GradientAscentMethod(early_stopping=False)
    stage = _posterior_stage(
        method,
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
        checkpoint_store=store,
    )

    stage.prepare(run.state)
    stage.train(run.state)

    assert [key for key, _ in store.saves] == ["posterior/best"]
    finalized = store.load("posterior/best")
    assert finalized.metadata == {"stage": "posterior", "stage_complete": True}
    assert finalized.state.lifecycle_state is StageState.POSTERIOR_READY
    assert finalized.epoch == 2
    assert _leaves_equal(finalized.state.posterior_state, method.parameters)
    assert finalized.parameters is None
    assert finalized.optimizer_state is None
    assert finalized.state.model_components == {}


def test_the_best_checkpoint_holds_the_method_state_and_is_finalized_as_its_posterior(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    store = RecordingStore()
    method = GradientAscentMethod()
    stage = _posterior_stage(
        method,
        early_stopper=EarlyStopper(
            metric="loss", mode="max", patience=10, source="train"
        ),
        checkpoint_store=store,
    )

    stage.prepare(run.state)
    result = stage.train(run.state)

    *bests, (final_key, finalized) = store.saves
    assert final_key == "posterior/best"
    assert bests and {key for key, _ in bests} == {"posterior/best"}
    _, best = bests[-1]
    assert best.epoch < 2
    assert best.metadata == {}
    assert best.state.lifecycle_state is StageState.VARIANCE_READY
    assert int(best.parameters["steps"]) == 2 * (best.epoch + 1)
    assert finalized.metadata == {"stage": "posterior", "stage_complete": True}
    assert finalized.state.lifecycle_state is StageState.POSTERIOR_READY
    assert finalized.epoch == best.epoch
    assert finalized.early_stopping_state == best.early_stopping_state
    assert finalized.parameters is None
    assert _leaves_equal(finalized.state.posterior_state, best.parameters["parameters"])
    assert _leaves_equal(
        method.posteriors[-1].parameters, best.parameters["parameters"]
    )
    assert result.loss == best.state.metric_history["posterior/train/loss"][-1]
    assert stage.validate(run.state).passed
