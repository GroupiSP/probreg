"""Tests for the metric-tag and parameter-path naming scheme.

The naming module is backend-neutral, so every test here runs under a bare `pytest
tests/core`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from probreg.core import (
    SEPARATOR,
    MetricTag,
    Split,
    flatten_parameters,
    metric_tag,
    parse_metric_tag,
)

segments = st.text(min_size=1).filter(lambda text: SEPARATOR not in text)
splits = st.sampled_from(Split)
invalid_segments = st.one_of(
    st.just(""),
    st.tuples(st.text(), st.text()).map(lambda parts: SEPARATOR.join(parts)),
)
leaves = st.one_of(st.integers(), st.floats(allow_nan=False), st.text(), st.booleans())
nested_parameters = st.recursive(
    st.dictionaries(segments, leaves, max_size=4),
    lambda children: st.dictionaries(segments, st.one_of(leaves, children), max_size=4),
    max_leaves=12,
)


def leaf_paths(mapping: Mapping[str, Any]) -> list[tuple[tuple[str, ...], Any]]:
    """Return every leaf of a nested mapping with its key path."""
    found: list[tuple[tuple[str, ...], Any]] = []
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            found.extend(((key, *path), leaf) for path, leaf in leaf_paths(value))
        else:
            found.append(((key,), value))
    return found


def test_split_vocabulary_is_train_and_validation() -> None:
    assert {split.value for split in Split} == {"train", "validation"}


@given(stage=segments, split=splits, metric=segments)
def test_building_then_parsing_round_trips(
    stage: str, split: Split, metric: str
) -> None:
    tag = metric_tag(stage, split, metric)

    assert parse_metric_tag(tag) == MetricTag(stage, split, metric)
    assert metric_tag(*parse_metric_tag(tag)) == tag


@given(stage=segments, split=splits, metric=segments)
def test_parsed_split_is_the_enumeration(stage: str, split: Split, metric: str) -> None:
    assert parse_metric_tag(metric_tag(stage, split, metric)).split is split


@given(
    valid=st.tuples(segments, segments),
    invalid=invalid_segments,
    position=st.sampled_from(["stage", "metric"]),
    split=splits,
)
def test_building_rejects_empty_or_separator_segments(
    valid: tuple[str, str], invalid: str, position: str, split: Split
) -> None:
    stage, metric = valid
    if position == "stage":
        stage = invalid
    else:
        metric = invalid

    with pytest.raises(ValueError):
        metric_tag(stage, split, metric)


@given(parts=st.lists(segments, max_size=6))
def test_parsing_rejects_segment_counts_other_than_three(parts: list[str]) -> None:
    assume(len(parts) != 3)

    with pytest.raises(ValueError):
        parse_metric_tag(SEPARATOR.join(parts))


@given(stage=segments, split=segments, metric=segments)
def test_parsing_rejects_unknown_splits(stage: str, split: str, metric: str) -> None:
    assume(split not in {member.value for member in Split})

    with pytest.raises(ValueError):
        parse_metric_tag(SEPARATOR.join([stage, split, metric]))


@given(
    valid=st.tuples(segments, splits, segments),
    position=st.sampled_from([0, 2]),
)
def test_parsing_rejects_empty_segments(
    valid: tuple[str, Split, str], position: int
) -> None:
    parts = [valid[0], valid[1].value, valid[2]]
    parts[position] = ""

    with pytest.raises(ValueError):
        parse_metric_tag(SEPARATOR.join(parts))


@pytest.mark.parametrize("stage", ["Mean", "camelCase", "with space", "stage-1"])
def test_non_snake_case_segments_are_accepted(stage: str) -> None:
    tag = metric_tag(stage, Split.TRAIN, "Loss")

    assert tag == f"{stage}/train/Loss"
    assert parse_metric_tag(tag) == MetricTag(stage, Split.TRAIN, "Loss")


@given(parameters=nested_parameters)
def test_flattening_keeps_every_leaf_on_a_distinct_path(
    parameters: dict[str, Any],
) -> None:
    flat = flatten_parameters(parameters)
    expected = {SEPARATOR.join(path): leaf for path, leaf in leaf_paths(parameters)}

    assert len(flat) == len(leaf_paths(parameters))
    assert flat == expected


@given(parameters=nested_parameters, invalid=invalid_segments, data=st.data())
def test_flattening_rejects_empty_or_separator_keys(
    parameters: dict[str, Any], invalid: str, data: st.DataObject
) -> None:
    mappings = [parameters] + [
        value for _, value in _nested_mappings(parameters) if isinstance(value, dict)
    ]
    target = data.draw(st.sampled_from(mappings))
    target[invalid] = data.draw(leaves)

    with pytest.raises(ValueError):
        flatten_parameters(parameters)


def _nested_mappings(mapping: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return every mapping nested anywhere below `mapping`."""
    found: list[tuple[str, Any]] = []
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            found.append((key, value))
            found.extend(_nested_mappings(value))
    return found


def test_split_is_a_path_segment_of_its_own() -> None:
    parameters = {"data": {Split.TRAIN.value: {"samples": 128, "batch_size": 32}}}

    assert flatten_parameters(parameters) == {
        "data/train/samples": 128,
        "data/train/batch_size": 32,
    }
