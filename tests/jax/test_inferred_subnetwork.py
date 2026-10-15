"""Posterior stages that infer only an inferred subnetwork of the mean network.

The mean network of `posterior_runs` is one linear layer. Inferring its kernel
alone leaves the bias as the frozen remainder, which every draw then shares:
at input zero each draw is exactly the bias.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st
from posterior_runs import (
    DATASET_SIZE,
    MakeStages,
    MakeVarianceReadyRun,
    RecordingStore,
    VarianceReadyRun,
    regression_loader,
    restore_mean_and_variance,
)

from stagereg.core.types import PyTree
from stagereg.jax import (
    BayesByBackprop,
    InferenceMethod,
    PosteriorStage,
    PosteriorStageOptions,
    PreconditionedSGLD,
)

_KERNEL = nnx.PathContains("kernel")

_METHODS: dict[str, Callable[[], InferenceMethod]] = {
    "bayes by backprop": lambda: BayesByBackprop(
        optimizer=optax.adam(0.01), initial_std=0.05
    ),
    "psgld": lambda: PreconditionedSGLD(step_size=1e-3, burn_in=1, thinning=2),
}


def _stage(
    method: InferenceMethod, inferred_subnetwork: Any = nnx.Param, **options: Any
) -> PosteriorStage:
    return PosteriorStage(
        inference_method=method,
        train_loader=regression_loader,
        dataset_size=DATASET_SIZE,
        options=PosteriorStageOptions(**{"epochs": 3, "num_draws": 4, **options}),
        inferred_subnetwork=inferred_subnetwork,
    )


@dataclass
class TrainedRun:
    """A posterior stage trained on a variance-ready run."""

    run: VarianceReadyRun
    method: InferenceMethod


def _train(
    variance_ready_run: MakeVarianceReadyRun,
    method_name: str,
    inferred_subnetwork: Any = nnx.Param,
) -> TrainedRun:
    run = variance_ready_run()
    method = _METHODS[method_name]()
    stage = _stage(method, inferred_subnetwork)
    stage.prepare(run.state)
    stage.train(run.state)
    return TrainedRun(run, method)


def _parameter_names(tree: PyTree) -> set[str]:
    """The names of the linear layer's parameters in a parameter tree."""
    return set(tree["linear"].keys())


@pytest.fixture(scope="module")
def kernel_runs(variance_ready_run: MakeVarianceReadyRun) -> dict[str, TrainedRun]:
    """Each method trained on the kernel alone; never mutate them."""
    return {name: _train(variance_ready_run, name, _KERNEL) for name in _METHODS}


@pytest.mark.parametrize("method_name", sorted(_METHODS))
def test_a_method_infers_only_the_inferred_subnetwork(
    kernel_runs: dict[str, TrainedRun], method_name: str
) -> None:
    posterior_state = kernel_runs[method_name].method.posterior_state()
    trees = (
        [posterior_state["means"], posterior_state["stds"]]
        if method_name == "bayes by backprop"
        else [posterior_state]
    )

    assert all(_parameter_names(tree) == {"kernel"} for tree in trees)


@settings(deadline=None, max_examples=10)
@given(
    method_name=st.sampled_from(sorted(_METHODS)),
    seed=st.integers(0, 2**31 - 1),
    rows=st.integers(1, 5),
)
def test_the_frozen_remainder_keeps_its_trained_mean_values_bit_for_bit(
    kernel_runs: dict[str, TrainedRun], method_name: str, seed: int, rows: int
) -> None:
    trained = kernel_runs[method_name]
    posterior = trained.run.state.model_components["posterior"]
    num_samples = None if posterior.num_draws is not None else 4

    draws = posterior.sample_means(
        jnp.zeros((rows, 1)), jax.random.key(seed), num_samples
    )

    bias = trained.run.mean_model.linear.bias.get_value()
    assert draws.shape[1:] == (rows, 1)
    np.testing.assert_array_equal(draws, jnp.broadcast_to(bias, draws.shape))


# The training losses of the default filter under stagereg 0.1.0, which
# inferred the whole network.
_LOSSES_OF_0_1_0 = {
    "bayes by backprop": [17.569884300231934, 17.512011528015137, 17.620736122131348],
    "psgld": [14.574500560760498, 14.39689016342163, 14.398993968963623],
}


@pytest.mark.parametrize("method_name", sorted(_METHODS))
def test_the_default_filter_reproduces_stagereg_0_1_0(
    variance_ready_run: MakeVarianceReadyRun, method_name: str
) -> None:
    trained = _train(variance_ready_run, method_name)

    np.testing.assert_allclose(
        trained.run.state.metric_history["posterior/train/loss"],
        _LOSSES_OF_0_1_0[method_name],
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "inferred_subnetwork",
    [nnx.PathContains("conv"), nnx.RngState, nnx.Not(nnx.Param)],
    ids=["unknown path", "rng state", "no param"],
)
def test_a_filter_selecting_no_param_leaf_is_refused_at_prepare(
    variance_ready_run: MakeVarianceReadyRun, inferred_subnetwork: Any
) -> None:
    state = variance_ready_run().state
    lifecycle, frozen, stage_name = (
        state.lifecycle_state,
        state.frozen_components,
        state.active_stage,
    )
    stage = _stage(_METHODS["bayes by backprop"](), inferred_subnetwork)

    with pytest.raises(ValueError, match="selects no Param leaf"):
        stage.prepare(state)

    assert (state.lifecycle_state, state.frozen_components, state.active_stage) == (
        lifecycle,
        frozen,
        stage_name,
    )


@pytest.mark.parametrize("method_name", sorted(_METHODS))
@pytest.mark.parametrize(
    ("trained_with", "restored_with"),
    [(_KERNEL, nnx.Param), (nnx.Param, _KERNEL), (_KERNEL, nnx.PathContains("bias"))],
    ids=["kernel then all", "all then kernel", "kernel then bias"],
)
def test_restoring_under_another_filter_is_refused(
    variance_ready_run: MakeVarianceReadyRun,
    make_stages: MakeStages,
    method_name: str,
    trained_with: Any,
    restored_with: Any,
) -> None:
    store = RecordingStore()
    run = variance_ready_run(checkpoint_store=store)
    trained = _stage(_METHODS[method_name](), trained_with, checkpoint_store=store)
    trained.prepare(run.state)
    trained.train(run.state)
    state = restore_mean_and_variance(make_stages(seed=7), store)
    stage = _stage(_METHODS[method_name](), restored_with)

    with pytest.raises(ValueError, match="inferred_subnetwork"):
        stage.restore(state, store.load("posterior/best"))

    assert stage.inference_method.has_posterior is False
    assert "posterior" not in state.model_components


@pytest.mark.parametrize("method_name", sorted(_METHODS))
def test_restoring_under_the_same_filter_rebuilds_the_posterior(
    variance_ready_run: MakeVarianceReadyRun,
    make_stages: MakeStages,
    method_name: str,
) -> None:
    store = RecordingStore()
    run = variance_ready_run(checkpoint_store=store)
    trained = _stage(_METHODS[method_name](), _KERNEL, checkpoint_store=store)
    trained.prepare(run.state)
    trained.train(run.state)
    state = restore_mean_and_variance(make_stages(seed=7), store)
    stage = _stage(_METHODS[method_name](), _KERNEL)

    stage.restore(state, store.load("posterior/best"))

    inputs = jnp.linspace(-1.0, 1.0, 5)[:, None]
    key = jax.random.key(0)
    restored = state.model_components["posterior"]
    num_samples = None if restored.num_draws is not None else 4
    np.testing.assert_array_equal(
        restored.sample_means(inputs, key, num_samples),
        run.state.model_components["posterior"].sample_means(inputs, key, num_samples),
    )
