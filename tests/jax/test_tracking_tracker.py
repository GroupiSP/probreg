"""Tests for the tracking example's TensorBoard experiment tracker.

The tracker is exercised at its constructor seam: a fake writer records
the calls it receives, so nothing here imports `tensorboardX` or JAX.
That is what makes the tracker's backend neutrality checkable rather
than merely intended.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from probreg.core.tracking import ExperimentTracker


class FakeWriter:
    """A summary writer recording every call it receives."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int | None]] = []
        self.figures: list[tuple[str, Any]] = []
        self.texts: list[tuple[str, str]] = []
        self.hparams: list[dict[str, Any]] = []
        self.flushes = 0
        self.closes = 0

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: int | None = None
    ) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def add_figure(self, tag: str, figure: Any, global_step: int | None = None) -> None:
        self.figures.append((tag, figure))

    def add_text(
        self, tag: str, text_string: str, global_step: int | None = None
    ) -> None:
        self.texts.append((tag, text_string))

    def add_hparams(
        self, hparam_dict: dict[str, Any], metric_dict: dict[str, float]
    ) -> None:
        self.hparams.append(hparam_dict)

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closes += 1


def count_leaves(values: Mapping[str, Any]) -> int:
    """Count the non-mapping leaves of a nested mapping."""
    return sum(
        count_leaves(value) if isinstance(value, Mapping) else 1
        for value in values.values()
    )


keys = st.text(alphabet="abcde", min_size=1, max_size=3)
leaves = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False), st.text(), st.booleans()
)
nested_params = st.recursive(
    st.dictionaries(keys, leaves, min_size=1, max_size=3),
    lambda children: st.dictionaries(
        keys, st.one_of(leaves, children), min_size=1, max_size=3
    ),
    max_leaves=6,
)


@pytest.fixture
def writer() -> FakeWriter:
    return FakeWriter()


def test_the_tracker_satisfies_the_experiment_tracker_protocol(
    tracking_tracker: ModuleType, writer: FakeWriter
) -> None:
    tracker: ExperimentTracker = tracking_tracker.TensorBoardTracker(
        "unused", writer=writer
    )

    assert tracker is not None


@given(
    values=st.dictionaries(
        keys, st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=4
    ),
    step=st.integers(min_value=0, max_value=1000),
)
def test_log_metrics_writes_one_scalar_per_entry_at_the_given_step(
    tracking_tracker: ModuleType, values: dict[str, float], step: int
) -> None:
    writer = FakeWriter()
    tracker = tracking_tracker.TensorBoardTracker("unused", writer=writer)

    tracker.log_metrics(values, step=step)

    assert len(writer.scalars) == len(values)
    assert {tag for tag, _, _ in writer.scalars} == set(values)
    assert all(recorded == step for _, _, recorded in writer.scalars)
    assert {tag: value for tag, value, _ in writer.scalars} == values


@given(values=nested_params)
def test_log_params_flattening_is_injective_in_the_nested_key_path(
    tracking_tracker: ModuleType, values: dict[str, Any]
) -> None:
    writer = FakeWriter()
    tracker = tracking_tracker.TensorBoardTracker("unused", writer=writer)

    tracker.log_params(values)

    (recorded,) = writer.hparams
    assert len(recorded) == count_leaves(values)


@given(values=nested_params)
def test_log_params_records_only_flat_scalars_or_strings(
    tracking_tracker: ModuleType, values: dict[str, Any]
) -> None:
    writer = FakeWriter()
    tracker = tracking_tracker.TensorBoardTracker("unused", writer=writer)

    tracker.log_params(values)

    (recorded,) = writer.hparams
    assert all(
        isinstance(value, bool | int | float | str) for value in recorded.values()
    )


def test_log_params_stringifies_values_the_hparams_plugin_cannot_carry(
    tracking_tracker: ModuleType, writer: FakeWriter
) -> None:
    tracker = tracking_tracker.TensorBoardTracker("unused", writer=writer)

    tracker.log_params({"metrics": ("rmse", "point_crps"), "loss": None})

    (recorded,) = writer.hparams
    assert recorded == {"metrics": "('rmse', 'point_crps')", "loss": "None"}


def test_log_params_rejects_a_key_that_would_collide_with_a_nested_path(
    tracking_tracker: ModuleType, writer: FakeWriter
) -> None:
    tracker = tracking_tracker.TensorBoardTracker("unused", writer=writer)

    with pytest.raises(ValueError, match="/"):
        tracker.log_params({"optimizer/learning_rate": 0.05})


def test_log_artifact_routes_a_figure_to_the_image_summary(
    tracking_tracker: ModuleType, writer: FakeWriter
) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    figure, _ = plt.subplots()
    tracker = tracking_tracker.TensorBoardTracker("unused", writer=writer)

    tracker.log_artifact("predictions", figure)

    assert writer.figures == [("predictions", figure)]
    assert writer.texts == []
    plt.close(figure)


def test_log_artifact_routes_a_string_to_the_text_summary(
    tracking_tracker: ModuleType, writer: FakeWriter
) -> None:
    tracker = tracking_tracker.TensorBoardTracker("unused", writer=writer)

    tracker.log_artifact("summary", "best epoch: 42")

    assert writer.texts == [("summary", "best epoch: 42")]
    assert writer.figures == []


def test_log_artifact_rejects_an_unsupported_value(
    tracking_tracker: ModuleType, writer: FakeWriter
) -> None:
    tracker = tracking_tracker.TensorBoardTracker("unused", writer=writer)

    with pytest.raises(TypeError, match="list"):
        tracker.log_artifact("predictions", [1.0, 2.0])


def test_flush_and_close_reach_the_injected_writer(
    tracking_tracker: ModuleType, writer: FakeWriter
) -> None:
    tracker = tracking_tracker.TensorBoardTracker("unused", writer=writer)

    tracker.flush()
    tracker.close()

    assert (writer.flushes, writer.closes) == (1, 1)
