from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from probreg.core.checkpoints import (
    Checkpoint,
    CheckpointStore,
    InMemoryCheckpointStore,
)
from probreg.core.types import TrainingState


def test_checkpoint_store_protocol_preserves_resume_state() -> None:
    store: CheckpointStore = InMemoryCheckpointStore()
    checkpoint = Checkpoint(
        state=TrainingState(stage="mean"),
        epoch=3,
        parameters={"weight": 1.0},
        optimizer_state={"step": 3},
        rng_state=42,
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
            state=TrainingState(stage="mean"), epoch=epochs[index % len(epochs)]
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
    first = Checkpoint(state=TrainingState(stage="mean"), epoch=epoch)
    second = Checkpoint(state=TrainingState(stage="mean"), epoch=epoch + 1)

    store.save("best", first)
    store.save("best", second)

    assert store.load("best") == second
