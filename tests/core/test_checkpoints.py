from __future__ import annotations

from probreg.core.checkpoints import Checkpoint, CheckpointStore
from probreg.core.types import TrainingState


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self.checkpoints: dict[str, Checkpoint] = {}

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        self.checkpoints[key] = checkpoint

    def load(self, key: str) -> Checkpoint:
        return self.checkpoints[key]

    def exists(self, key: str) -> bool:
        return key in self.checkpoints


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
