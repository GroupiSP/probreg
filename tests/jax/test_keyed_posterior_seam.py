"""The keyed posterior seam: the key reaches the posterior network's RNG streams.

A [`PosteriorStage`][stagereg.jax.PosteriorStage] hands its inference method a
mean function and a log-likelihood that take a PRNG key. A deterministic
network ignores it; a network with ``nnx.Dropout`` draws its masks from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st
from posterior_runs import (
    DATASET_SIZE,
    MakeVarianceReadyRun,
    VarianceReadyRun,
    regression_data,
    regression_loader,
)

from stagereg.core.types import Batch, PyTree
from stagereg.jax import (
    BayesByBackprop,
    InferenceMethod,
    MeanFieldGaussianPosterior,
    PosteriorProblem,
    PosteriorStage,
    PosteriorStageOptions,
    RetainedSamplesPosterior,
)


class DropoutLinearModel(nnx.Module):
    """The mean network's linear layer behind an always-active input dropout."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(1, 1, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(self.dropout(inputs))


@dataclass
class KeyedProbe:
    """Delegates to an inference method, and records each batch's mean function.

    In every ``update`` it evaluates the problem's mean function at the warm
    start on ``probe_inputs`` under the batch's key, so a test can see which
    masks the stage's keys select.
    """

    inner: InferenceMethod
    probe_inputs: jax.Array
    problem: PosteriorProblem | None = None
    probes: list[jax.Array] = field(default_factory=list)

    @property
    def supports_early_stopping(self) -> bool:
        return self.inner.supports_early_stopping

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return self.inner.diagnostics

    @property
    def has_posterior(self) -> bool:
        return self.inner.has_posterior

    def init(self, problem: PosteriorProblem) -> None:
        self.problem = problem
        self.inner.init(problem)

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        assert self.problem is not None
        self.probes.append(
            self.problem.mean_function(
                self.problem.initial_parameters, self.probe_inputs, key
            )
        )
        return self.inner.update(batch, key)

    def posterior(self) -> Any:
        return self.inner.posterior()

    def state(self) -> PyTree:
        return self.inner.state()

    def load_state(self, state: PyTree) -> None:
        self.inner.load_state(state)

    def posterior_state(self) -> PyTree:
        return self.inner.posterior_state()

    def load_posterior(self, state: PyTree) -> None:
        self.inner.load_posterior(state)


def _probe() -> KeyedProbe:
    return KeyedProbe(
        inner=BayesByBackprop(optimizer=optax.adam(0.01), initial_std=0.05),
        probe_inputs=jnp.ones((64, 1)),
    )


def _stage(method: InferenceMethod, model: nnx.Module | None = None) -> PosteriorStage:
    return PosteriorStage(
        inference_method=method,
        train_loader=regression_loader,
        dataset_size=DATASET_SIZE,
        options=PosteriorStageOptions(epochs=3, num_draws=4),
        model=model,
    )


@pytest.fixture(scope="module")
def deterministic_problem(
    variance_ready_run: MakeVarianceReadyRun,
) -> PosteriorProblem:
    """The posterior problem a stage builds for the deterministic mean network."""
    run = variance_ready_run()
    probe = _probe()
    _stage(probe).prepare(run.state)
    assert probe.problem is not None
    return probe.problem


@pytest.fixture(scope="module")
def dropout_run(variance_ready_run: MakeVarianceReadyRun) -> tuple[Any, KeyedProbe]:
    """A dropout posterior network trained by Bayes by Backprop; never mutate it."""
    run: VarianceReadyRun = variance_ready_run()
    probe = _probe()
    stage = _stage(probe, model=DropoutLinearModel(rngs=nnx.Rngs(0)))
    stage.prepare(run.state)
    result = stage.train(run.state)
    return result, probe


@settings(deadline=None, max_examples=20)
@given(
    first_seed=st.integers(0, 2**31 - 1),
    second_seed=st.integers(0, 2**31 - 1),
    rows=st.integers(1, DATASET_SIZE),
)
def test_a_deterministic_network_gives_identical_results_for_any_key(
    deterministic_problem: PosteriorProblem,
    first_seed: int,
    second_seed: int,
    rows: int,
) -> None:
    problem = deterministic_problem
    inputs, targets = regression_data()
    batch = Batch(inputs=inputs[:rows], targets=targets[:rows])
    first, second = jax.random.key(first_seed), jax.random.key(second_seed)
    parameters = problem.initial_parameters

    assert jnp.array_equal(
        problem.mean_function(parameters, batch.inputs, first),
        problem.mean_function(parameters, batch.inputs, second),
    )
    assert jnp.array_equal(
        problem.log_likelihood(parameters, batch, first),
        problem.log_likelihood(parameters, batch, second),
    )


def test_a_dropout_network_trains_under_the_posterior_stage(
    dropout_run: tuple[Any, KeyedProbe],
) -> None:
    result, _ = dropout_run

    assert result.loss is not None
    assert jnp.isfinite(result.loss)


def test_a_dropout_networks_masks_vary_across_batches(
    dropout_run: tuple[Any, KeyedProbe],
) -> None:
    _, probe = dropout_run

    assert len(probe.probes) == 6
    assert all(
        not jnp.array_equal(earlier, later)
        for index, earlier in enumerate(probe.probes)
        for later in probe.probes[index + 1 :]
    )


@settings(deadline=None, max_examples=10)
@given(seed=st.integers(0, 2**31 - 1))
def test_a_dropout_networks_masks_come_from_the_key(
    dropout_run: tuple[Any, KeyedProbe], seed: int
) -> None:
    _, probe = dropout_run
    problem = probe.problem
    assert problem is not None
    key = jax.random.key(seed)
    parameters = problem.initial_parameters

    first = problem.mean_function(parameters, probe.probe_inputs, key)
    again = jax.jit(problem.mean_function)(parameters, probe.probe_inputs, key)

    assert jnp.allclose(first, again)


def _keyed_line(parameters: PyTree, inputs: jax.Array, key: jax.Array) -> jax.Array:
    """A line shifted by one standard normal offset drawn from the key."""
    return inputs * parameters["w"] + parameters["b"] + jax.random.normal(key)


_LINE = {"w": jnp.asarray(1.5), "b": jnp.asarray(-0.5)}

_KEYED_POSTERIORS: dict[str, Any] = {
    "mean-field gaussian": lambda: MeanFieldGaussianPosterior(
        mean_function=_keyed_line,
        means=_LINE,
        stds=jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.1), _LINE),
    ),
    "retained samples": lambda: RetainedSamplesPosterior(
        mean_function=_keyed_line,
        samples=jax.tree.map(lambda leaf: jnp.stack([leaf] * 4), _LINE),
    ),
}


def _draws(posterior: Any, inputs: jax.Array, key: jax.Array) -> jax.Array:
    num_samples = None if posterior.num_draws is not None else 4
    return posterior.sample_means(inputs, key, num_samples)


@settings(deadline=None, max_examples=20)
@given(
    posterior_name=st.sampled_from(sorted(_KEYED_POSTERIORS)),
    seed=st.integers(0, 2**31 - 1),
    rows=st.lists(st.integers(0, 7), min_size=1, max_size=8),
)
def test_a_keyed_draw_is_the_same_function_at_every_input(
    posterior_name: str, seed: int, rows: list[int]
) -> None:
    posterior = _KEYED_POSTERIORS[posterior_name]()
    inputs = jnp.linspace(-1.0, 1.0, 8)[:, None]
    key = jax.random.key(seed)
    subset = jnp.asarray(rows)

    full = _draws(posterior, inputs, key)
    partial = _draws(posterior, inputs[subset], key)

    assert jnp.allclose(partial, full[:, subset], atol=1e-6)


@pytest.mark.parametrize("posterior_name", sorted(_KEYED_POSTERIORS))
def test_keyed_draws_differ_from_each_other(posterior_name: str) -> None:
    posterior = _KEYED_POSTERIORS[posterior_name]()

    draws = _draws(posterior, jnp.zeros((1, 1)), jax.random.key(0))

    assert len(set(draws.ravel().tolist())) == draws.shape[0]
