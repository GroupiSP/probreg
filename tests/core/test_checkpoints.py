from __future__ import annotations

from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from stagereg.core.checkpoints import (
    STAGE_COMPLETE_METADATA_KEY,
    CachedCheckpointStore,
    Checkpoint,
    CheckpointStore,
    InMemoryCheckpointStore,
    always,
    every_nth_save,
    finalized_only,
)
from stagereg.core.types import TrainingState


def test_checkpoint_store_protocol_preserves_resume_state() -> None:
    store: CheckpointStore = InMemoryCheckpointStore()
    checkpoint = Checkpoint(
        state=TrainingState(rng_state=42, active_stage="mean"),
        epoch=3,
        parameters={"weight": 1.0},
        optimizer_state={"step": 3},
        early_stopping_state={"best_loss": 0.1},
    )

    store.save("mean-best", checkpoint)

    assert store.exists("mean-best")
    assert store.load("mean-best") == checkpoint


def test_in_memory_store_reports_absent_keys_as_missing() -> None:
    store = InMemoryCheckpointStore()

    assert not store.exists("mean-best")


@given(
    keys=st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=6, unique=True),
    epochs=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=6),
)
def test_in_memory_store_loads_back_every_saved_checkpoint(
    keys: list[str], epochs: list[int]
) -> None:
    store = InMemoryCheckpointStore()
    checkpoints = {
        key: Checkpoint(
            state=TrainingState(active_stage="mean"), epoch=epochs[index % len(epochs)]
        )
        for index, key in enumerate(keys)
    }

    for key, checkpoint in checkpoints.items():
        store.save(key, checkpoint)

    assert all(store.exists(key) for key in checkpoints)
    assert {key: store.load(key) for key in checkpoints} == checkpoints


@given(epoch=st.integers(min_value=0, max_value=100))
def test_in_memory_store_save_overwrites_the_previous_checkpoint(epoch: int) -> None:
    store = InMemoryCheckpointStore()
    first = Checkpoint(state=TrainingState(active_stage="mean"), epoch=epoch)
    second = Checkpoint(state=TrainingState(active_stage="mean"), epoch=epoch + 1)

    store.save("best", first)
    store.save("best", second)

    assert store.load("best") == second


def test_a_checkpoint_carries_its_random_key_only_in_its_state() -> None:
    checkpoint = Checkpoint(state=TrainingState(rng_state=42), epoch=0)

    assert checkpoint.state.rng_state == 42
    with pytest.raises(TypeError, match="rng_state"):
        Checkpoint(  # type: ignore[call-arg]
            state=TrainingState(rng_state=42), epoch=0, rng_state=43
        )


checkpoint_keys = st.text(min_size=1, max_size=8)
checkpoints = st.builds(
    Checkpoint,
    state=st.builds(TrainingState, active_stage=st.sampled_from(["mean", "variance"])),
    epoch=st.integers(min_value=0, max_value=100),
    metadata=st.dictionaries(
        st.sampled_from([STAGE_COMPLETE_METADATA_KEY, "stage", "note"]),
        st.booleans(),
        max_size=3,
    ),
)
saves = st.lists(st.tuples(st.sampled_from(["a", "b", "c"]), checkpoints), max_size=12)


@given(key=checkpoint_keys, checkpoint=checkpoints)
def test_cached_store_loads_back_what_it_saved(
    key: str, checkpoint: Checkpoint
) -> None:
    store: CheckpointStore = CachedCheckpointStore(InMemoryCheckpointStore())

    store.save(key, checkpoint)

    assert store.exists(key)
    assert store.load(key) == checkpoint


@given(saves=saves)
def test_cached_store_persists_exactly_the_saves_the_predicate_selects(
    saves: list[tuple[str, Checkpoint]], recording_store: Callable
) -> None:
    def every_odd_epoch(key: str, checkpoint: Checkpoint) -> bool:
        del key
        return checkpoint.epoch % 2 == 1

    expectations = {
        always: saves,
        finalized_only: [
            (key, checkpoint)
            for key, checkpoint in saves
            if checkpoint.metadata.get(STAGE_COMPLETE_METADATA_KEY) is True
        ],
        every_odd_epoch: [
            (key, checkpoint) for key, checkpoint in saves if checkpoint.epoch % 2
        ],
    }
    for predicate, expected in expectations.items():
        persistent = recording_store()
        store = CachedCheckpointStore(persistent, persist_when=predicate)
        for key, checkpoint in saves:
            store.save(key, checkpoint)
        assert persistent.saves == expected


@given(saves=saves, nth=st.integers(min_value=1, max_value=4))
def test_every_nth_save_persists_each_keys_nth_save(
    saves: list[tuple[str, Checkpoint]], nth: int, recording_store: Callable
) -> None:
    persistent = recording_store()
    store = CachedCheckpointStore(persistent, persist_when=every_nth_save(nth))

    for key, checkpoint in saves:
        store.save(key, checkpoint)

    expected = [
        (key, checkpoint)
        for index, (key, checkpoint) in enumerate(saves)
        if [saved_key for saved_key, _ in saves[: index + 1]].count(key) % nth == 0
    ]
    assert persistent.saves == expected


def test_cached_store_defaults_to_persisting_finalized_checkpoints_only(
    recording_store: Callable,
) -> None:
    persistent = recording_store()
    store = CachedCheckpointStore(persistent)
    best = Checkpoint(state=TrainingState(), epoch=1)
    finalized = Checkpoint(
        state=TrainingState(), epoch=1, metadata={STAGE_COMPLETE_METADATA_KEY: True}
    )

    store.save("mean/best", best)
    store.save("mean/best", finalized)

    assert persistent.saves == [("mean/best", finalized)]


@pytest.mark.parametrize("nth", [0, -1])
def test_every_nth_save_rejects_a_non_positive_period(nth: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        every_nth_save(nth)


@given(key=checkpoint_keys, checkpoint=checkpoints)
def test_a_fresh_cache_falls_back_to_a_populated_persistent_store(
    key: str, checkpoint: Checkpoint
) -> None:
    persistent = InMemoryCheckpointStore()
    persistent.save(key, checkpoint)

    store = CachedCheckpointStore(persistent)

    assert store.exists(key)
    assert store.load(key) == checkpoint


def test_cached_store_raises_key_error_on_a_key_neither_holds() -> None:
    store = CachedCheckpointStore(InMemoryCheckpointStore())
    store.save("mean/best", Checkpoint(state=TrainingState(), epoch=0))

    assert not store.exists("variance/best")
    with pytest.raises(KeyError):
        store.load("variance/best")


def test_cached_store_reads_the_cache_before_the_persistent_store() -> None:
    persistent = InMemoryCheckpointStore()
    persistent.save("mean/best", Checkpoint(state=TrainingState(), epoch=0))
    store = CachedCheckpointStore(persistent)
    newer = Checkpoint(state=TrainingState(), epoch=5)

    store.save("mean/best", newer)

    assert store.load("mean/best") == newer
