"""Tests for the metric-tag and parameter-path naming scheme.

The naming module is backend-neutral, so every test here runs under a bare `pytest
tests/core`. The separator is private to the module, so the tests spell the contract's
`/` literally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from probreg.core import (
    MetricTag,
    Split,
    flatten_parameters,
    metric_tag,
    parse_metric_tag,
)

SEPARATOR = "/"

segments = st.text(min_size=1).filter(lambda text: SEPARATOR not in text)
splits = st.sampled_from(Split)
invalid_segments = st.one_of(
    st.just(""),
    st.tuples(st.text(), st.text()).map(lambda parts: SEPARATOR.join(parts)),
)
leaves = st.one_of(
    st.integers(),
    st.floats(allow_nan=False),
    st.text(),
    st.booleans(),
    st.just({}),
)
key_paths = st.lists(segments, min_size=1, max_size=4).map(tuple)


def _is_prefix_free(paths: list[tuple[str, ...]]) -> bool:
    """Return whether no path is a proper prefix of another."""
    return not any(
        len(short) < len(long) and long[: len(short)] == short
        for short in paths
        for long in paths
    )


path_values = st.dictionaries(key_paths, leaves, max_size=8).filter(
    lambda mapping: _is_prefix_free(list(mapping))
)


def _nest(path_values: Mapping[tuple[str, ...], Any]) -> dict[str, Any]:
    """Build the nested mapping that places each value at its key path."""
    nested: dict[str, Any] = {}
    for path, value in path_values.items():
        node = nested
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return nested


def _lookup(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Return the value found by following `path` through `mapping`."""
    value: Any = mapping
    for key in path:
        value = value[key]
    return value


def test_split_vocabulary_is_train_and_validation() -> None:
    assert {split.value for split in Split} == {"train", "validation"}


@given(stage=segments, split=splits, metric=segments)
def test_building_then_parsing_round_trips(
    stage: str, split: Split, metric: str
) -> None:
    tag = metric_tag(stage, split, metric)

    assert tag == f"{stage}/{split.value}/{metric}"
    assert parse_metric_tag(tag) == MetricTag(stage, split, metric)
    assert parse_metric_tag(tag).split is split


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


def test_non_snake_case_stage_is_accepted() -> None:
    tag = metric_tag("Mean", Split.TRAIN, "loss")

    assert tag == "Mean/train/loss"
    assert parse_metric_tag(tag) == MetricTag("Mean", Split.TRAIN, "loss")


@given(path_values=path_values)
def test_flattening_places_each_value_at_its_joined_key_path(
    path_values: dict[tuple[str, ...], Any],
) -> None:
    assert flatten_parameters(_nest(path_values)) == {
        SEPARATOR.join(path): value for path, value in path_values.items()
    }


@given(path_values=path_values)
def test_flattening_yields_only_leaves_of_the_input(
    path_values: dict[tuple[str, ...], Any],
) -> None:
    parameters = _nest(path_values)

    for parameter_path, value in flatten_parameters(parameters).items():
        found = _lookup(parameters, tuple(parameter_path.split(SEPARATOR)))
        assert found == value
        assert not isinstance(found, Mapping) or not found


@given(path_values=path_values, invalid=invalid_segments, data=st.data())
def test_flattening_rejects_empty_or_separator_keys(
    path_values: dict[tuple[str, ...], Any], invalid: str, data: st.DataObject
) -> None:
    parameters = _nest(path_values)
    target = parameters
    if path_values:
        path = data.draw(st.sampled_from(list(path_values)))
        depth = data.draw(st.integers(min_value=0, max_value=len(path) - 1))
        target = _lookup(parameters, path[:depth])
    target[invalid] = data.draw(leaves)

    with pytest.raises(ValueError):
        flatten_parameters(parameters)


def test_empty_nested_mapping_is_kept_as_a_leaf() -> None:
    assert flatten_parameters({"optimizer": {}, "seed": 0}) == {
        "optimizer": {},
        "seed": 0,
    }


def test_split_is_a_path_segment_of_its_own() -> None:
    parameters = {"data": {Split.TRAIN.value: {"samples": 128, "batch_size": 32}}}

    assert flatten_parameters(parameters) == {
        "data/train/samples": 128,
        "data/train/batch_size": 32,
    }
