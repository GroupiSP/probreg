"""MC-Dropout inside a real mean, variance and posterior run.

The mean network is a small MLP; the posterior network is the same MLP with a hidden
dropout layer whose mask is broadcast over the batch axis, so both have the same
parameter tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    RecordingStore,
    build_stages,
    leaves_equal,
    regression_data,
    regression_loader,
    restore_mean_and_variance,
    run_to_variance_ready,
)

from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.types import StageState, TrainingState
from stagereg.jax import MCDropout, PosteriorStage, PosteriorStageOptions

_HIDDEN = 8
_NUM_DRAWS = 8


class MLP(nnx.Module):
    """The mean network: one hidden layer of eight units."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.hidden = nnx.Linear(1, _HIDDEN, rngs=rngs)
        self.output = nnx.Linear(_HIDDEN, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.output(nnx.relu(self.hidden(inputs)))


class DropoutMLP(MLP):
    """The mean network with one hidden dropout mask shared by the whole batch."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__(rngs=rngs)
        self.dropout = nnx.Dropout(0.5, broadcast_dims=(0,), rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.output(self.dropout(nnx.relu(self.hidden(inputs))))


def _stage(store: RecordingStore | None) -> PosteriorStage:
    return PosteriorStage(
        inference_method=MCDropout(optimizer=optax.adam(0.01)),
        train_loader=regression_loader,
        dataset_size=DATASET_SIZE,
        options=PosteriorStageOptions(
            epochs=8,
            num_draws=_NUM_DRAWS,
            validation_loader=regression_loader,
            early_stopper=EarlyStopper(metric="nll", mode="min", patience=2),
            checkpoint_store=store,
        ),
        model=DropoutMLP(rngs=nnx.Rngs(0)),
    )


@dataclass
class MCDropoutRun:
    """A trained MC-Dropout posterior stage and where it checkpointed."""

    state: TrainingState
    store: RecordingStore
    result: Any

    @property
    def posterior(self) -> Any:
        return self.state.model_components["posterior"]


@pytest.fixture(scope="module")
def mc_dropout_run(squared_error: Any) -> MCDropoutRun:
    """A real three-stage run ending in MC-Dropout; never mutate it."""
    store = RecordingStore()
    run = run_to_variance_ready(
        build_stages(MLP, squared_error, checkpoint_store=store)
    )
    stage = _stage(store)
    stage.prepare(run.state)
    result = stage.train(run.state)
    return MCDropoutRun(run.state, store, result)


def test_mc_dropout_trains_with_early_stopping_and_checkpoints(
    mc_dropout_run: MCDropoutRun,
) -> None:
    run = mc_dropout_run

    assert run.result.loss is not None and math.isfinite(run.result.loss)
    *bests, finalized = [c for key, c in run.store.saves if key == "posterior/best"]
    assert bests
    assert finalized.metadata == {"stage": "posterior", "stage_complete": True}
    assert finalized.state.lifecycle_state is StageState.POSTERIOR_READY
    assert finalized.epoch == bests[-1].epoch
    assert leaves_equal(
        finalized.state.posterior_state, bests[-1].parameters["parameters"]
    )


def test_mc_dropout_draws_differ_from_each_other(
    mc_dropout_run: MCDropoutRun,
) -> None:
    inputs, _ = regression_data()

    draws = mc_dropout_run.posterior.sample_means(inputs, jax.random.key(0), _NUM_DRAWS)

    assert all(
        not jnp.allclose(draws[earlier], draws[later])
        for earlier in range(_NUM_DRAWS)
        for later in range(earlier + 1, _NUM_DRAWS)
    )


@settings(deadline=None, max_examples=20)
@given(
    seed=st.integers(0, 2**31 - 1),
    rows=st.lists(st.integers(0, DATASET_SIZE - 1), min_size=1, max_size=8),
)
def test_an_mc_dropout_draw_is_the_same_function_on_any_input_subset(
    mc_dropout_run: MCDropoutRun, seed: int, rows: list[int]
) -> None:
    inputs, _ = regression_data()
    key = jax.random.key(seed)
    subset = jnp.asarray(rows)

    full = mc_dropout_run.posterior.sample_means(inputs, key, _NUM_DRAWS)
    partial = mc_dropout_run.posterior.sample_means(inputs[subset], key, _NUM_DRAWS)

    assert jnp.allclose(partial, full[:, subset], atol=1e-6)


@settings(deadline=None, max_examples=10)
@given(seed=st.integers(0, 2**31 - 1))
def test_the_same_key_gives_the_same_mc_dropout_draws(
    mc_dropout_run: MCDropoutRun, seed: int
) -> None:
    inputs, _ = regression_data()
    key = jax.random.key(seed)
    posterior = mc_dropout_run.posterior

    assert jnp.array_equal(
        posterior.sample_means(inputs, key, _NUM_DRAWS),
        posterior.sample_means(inputs, key, _NUM_DRAWS),
    )


def test_restoring_an_mc_dropout_checkpoint_gives_identical_draws(
    mc_dropout_run: MCDropoutRun, squared_error: Any
) -> None:
    run = mc_dropout_run
    state = restore_mean_and_variance(
        build_stages(MLP, squared_error, seed=7), run.store
    )
    restored_stage = _stage(None)

    restored_stage.restore(state, run.store.load("posterior/best"))

    assert restored_stage.validate(state).passed
    inputs, _ = regression_data()
    key = jax.random.key(5)
    assert jnp.array_equal(
        state.model_components["posterior"].sample_means(inputs, key, _NUM_DRAWS),
        run.posterior.sample_means(inputs, key, _NUM_DRAWS),
    )
