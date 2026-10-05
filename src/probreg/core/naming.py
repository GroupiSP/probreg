"""The metric-tag and parameter-path naming scheme.

This module is the only code that joins or splits a metric tag
(``stage/split/metric``) or a parameter path (a ``/``-joined key path through
a nested parameter mapping). No segment may be empty or contain the
separator, so every tag and path splits back into exactly the segments it was
built from. ``snake_case`` is a convention for segments, not enforced here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from enum import StrEnum
from typing import Any, NamedTuple

SEPARATOR = "/"


class Split(StrEnum):
    """The partition of data a metric was measured on."""

    TRAIN = "train"
    VALIDATION = "validation"


class MetricTag(NamedTuple):
    """The parts of a metric tag.

    Attributes:
        stage: Name of the stage that produced the metric.
        split: Split the metric was measured on.
        metric: Bare name of the metric.
    """

    stage: str
    split: Split
    metric: str


def _check_segment(segment: str, role: str) -> None:
    """Reject a segment that would make a joined name ambiguous.

    Args:
        segment: The segment to check.
        role: What the segment names, used in the error message.

    Raises:
        ValueError: If `segment` is empty or contains the separator.
    """
    if not segment:
        raise ValueError(f"{role} must be a non-empty string.")
    if SEPARATOR in segment:
        raise ValueError(f"{role} {segment!r} may not contain {SEPARATOR!r}.")


def metric_tag(stage: str, split: Split, metric: str) -> str:
    """Build the metric tag for a metric measured in a stage on a split.

    Args:
        stage: Name of the stage that produced the metric.
        split: Split the metric was measured on.
        metric: Bare name of the metric.

    Returns:
        The tag ``stage/split/metric``.

    Raises:
        ValueError: If `stage` or `metric` is empty or contains the
            separator, or if `split` is not a known split.
    """
    _check_segment(stage, "stage")
    _check_segment(metric, "metric")
    return SEPARATOR.join((stage, Split(split).value, metric))


def parse_metric_tag(tag: str) -> MetricTag:
    """Split a metric tag back into its stage, split and metric name.

    This is the exact inverse of :func:`metric_tag`.

    Args:
        tag: A tag of the form ``stage/split/metric``.

    Returns:
        The stage, split and metric name the tag was built from.

    Raises:
        ValueError: If `tag` does not have exactly three segments, has an
            empty segment, or names an unknown split.
    """
    segments = tag.split(SEPARATOR)
    if len(segments) != 3:
        raise ValueError(
            f"metric tag {tag!r} must have exactly three {SEPARATOR!r}-separated "
            f"segments, not {len(segments)}."
        )
    stage, split, metric = segments
    _check_segment(stage, "stage")
    _check_segment(metric, "metric")
    try:
        parsed_split = Split(split)
    except ValueError:
        known = ", ".join(repr(member.value) for member in Split)
        raise ValueError(
            f"metric tag {tag!r} names unknown split {split!r}; expected one of "
            f"{known}."
        ) from None
    return MetricTag(stage, parsed_split, metric)


def flatten_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a nested parameter mapping into parameter paths.

    Every non-mapping value is a leaf and is kept unchanged under its key
    path joined with the separator. Because no key may be empty or contain
    the separator, two distinct key paths never map onto one parameter path.

    Args:
        parameters: The parameters to flatten, possibly nested.

    Returns:
        One entry per leaf, keyed by its parameter path.

    Raises:
        ValueError: If any key, at any depth, is empty or contains the
            separator.
    """

    def walk(mapping: Mapping[str, Any], prefix: str) -> Iterator[tuple[str, Any]]:
        for key, value in mapping.items():
            _check_segment(key, "parameter key")
            path = f"{prefix}{key}"
            if isinstance(value, Mapping):
                yield from walk(value, f"{path}{SEPARATOR}")
            else:
                yield path, value

    return dict(walk(parameters, ""))
