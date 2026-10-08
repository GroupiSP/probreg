"""Bayes by Backprop inside a real mean, variance and posterior run.

The mean and variance fixtures are shared with `test_posterior_stage.py`,
imported from it rather than moved into `conftest.py`.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax
from test_posterior_stage import (
    _DATASET_SIZE,
    MakeStages,
    MakeVarianceReadyRun,
    RecordingStore,
    _leaves_equal,
    _regression_data,
    _restore_mean_and_variance,
    make_stages,  # noqa: F401  (fixture)
    regression_loader,
    variance_ready_run,  # noqa: F401  (fixture)
)

from probreg.core.early_stopping import EarlyStopper
from probreg.core.types import StageState
from probreg.jax import BayesByBackprop, PosteriorStage, PosteriorStageOptions


def _stage(method: BayesByBackprop, store: RecordingStore | None) -> PosteriorStage:
    return PosteriorStage(
        inference_method=method,
        train_loader=regression_loader,
        dataset_size=_DATASET_SIZE,
        options=PosteriorStageOptions(
            epochs=8,
            num_draws=16,
            validation_loader=regression_loader,
            early_stopper=EarlyStopper(metric="nll", mode="min", patience=2),
            checkpoint_store=store,
        ),
    )


def test_bayes_by_backprop_checkpoints_and_restores_through_the_posterior_stage(
    variance_ready_run: MakeVarianceReadyRun,  # noqa: F811
    make_stages: MakeStages,  # noqa: F811
) -> None:
    store = RecordingStore()
    run = variance_ready_run(checkpoint_store=store)
    method = BayesByBackprop(optimizer=optax.adam(0.01), initial_std=0.05)
    stage = _stage(method, store)

    stage.prepare(run.state)
    result = stage.train(run.state)

    assert result.loss is not None and math.isfinite(result.loss)
    *bests, finalized = [c for key, c in store.saves if key == "posterior/best"]
    assert bests and all(best.metadata == {} for best in bests)
    best = bests[-1]
    assert finalized.metadata == {"stage": "posterior", "stage_complete": True}
    assert finalized.state.lifecycle_state is StageState.POSTERIOR_READY
    assert finalized.epoch == best.epoch
    saved_posterior = finalized.state.posterior_state
    assert _leaves_equal(
        saved_posterior["means"], best.parameters["variational"]["means"]
    )
    assert all(
        bool(jnp.all(std > 0.0)) for std in jax.tree.leaves(saved_posterior["stds"])
    )

    # The best checkpoint resumes inference where it left off.
    resumed = BayesByBackprop(optimizer=optax.adam(0.01), initial_std=0.05)
    _stage(resumed, None).prepare(variance_ready_run().state)
    resumed.load_state(best.parameters)
    assert _leaves_equal(resumed.posterior_state(), saved_posterior)
    batch = regression_loader(split="train", epoch=0)[0]
    assert math.isfinite(float(resumed.update(batch, jax.random.key(0))["loss"]))

    # The finalized checkpoint rebuilds the same posterior in a fresh state.
    state = _restore_mean_and_variance(make_stages(seed=7), store)
    restored_stage = _stage(BayesByBackprop(), None)
    restored_stage.restore(state, store.load("posterior/best"))
    assert restored_stage.validate(state).passed
    inputs, _ = _regression_data()
    key = jax.random.key(5)
    trained = run.state.model_components["posterior"]
    restored = state.model_components["posterior"]
    assert jnp.allclose(
        restored.sample_means(inputs, key, 16),
        trained.sample_means(inputs, key, 16),
        atol=1e-6,
    )
