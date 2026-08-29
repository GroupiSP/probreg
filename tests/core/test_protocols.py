from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from probreg.core.protocols import Dataset, LoaderFactory, Optimizer, Step
from probreg.core.types import Batch, StageResult, TrainingState


class ExampleDataset:
    def __len__(self) -> int:
        return 1

    def get(self, index: int) -> Mapping[str, Any]:
        return {"index": index}


def example_loader(*, split: str, epoch: int) -> Iterable[Batch]:
    del split, epoch
    return [Batch(inputs=[1], targets=[2])]


class ExampleOptimizer:
    def init(self, parameters: Any) -> dict[str, int]:
        del parameters
        return {"updates": 0}

    def update(
        self, gradients: Any, state: dict[str, int], parameters: Any
    ) -> tuple[Any, dict[str, int]]:
        del gradients
        return parameters, {"updates": state["updates"] + 1}


def example_step(
    batch: Batch, state: TrainingState, *, key: Any, training: bool
) -> StageResult:
    del batch, key, training
    return StageResult(state=state, loss=0.0)


def test_protocol_implementations_are_usable() -> None:
    dataset: Dataset = ExampleDataset()
    loader: LoaderFactory = example_loader
    optimizer: Optimizer = ExampleOptimizer()
    step: Step = example_step

    assert dataset.get(0) == {"index": 0}
    assert next(iter(loader(split="train", epoch=0))).targets == [2]
    assert optimizer.update({}, optimizer.init({}), {"weight": 1}) == (
        {"weight": 1},
        {"updates": 1},
    )
    assert step(Batch(inputs=[]), TrainingState(), key=None, training=True).loss == 0.0
