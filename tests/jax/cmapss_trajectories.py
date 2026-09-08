"""Synthetic CMAPSS-shaped trajectory frames shared by the example's tests.

Every CMAPSS test module needs the same DataFrame shape — a `unit_id`
column, an ascending `time_cycles` column per unit, and a set of sensor
columns — so the builders live here once rather than being re-derived in
each module. They are plain functions rather than fixtures because most
callers are Hypothesis `@given` bodies, which cannot take a
function-scoped fixture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

#: The sensor columns `build_trajectories` emits.
FEATURE_COLUMNS = ["sensor_a", "sensor_b"]


def build_trajectories(lifetimes: Mapping[int, int]) -> pd.DataFrame:
    """Build deterministic multi-unit trajectories with the given lifetimes.

    Sensor values are arbitrary but unique per unit and cycle, so a test
    that mixes up two units or two cycles fails rather than coincidentally
    passing. They are not standardized: no test asserting on windowing,
    selection, or plotting depends on their scale.

    Args:
        lifetimes: Mapping of unit ID to that unit's number of cycles. The
            unit's `time_cycles` run from 1 to that count inclusive, so the
            count is also its lifetime.

    Returns:
        A DataFrame with `unit_id`, `time_cycles`, and `FEATURE_COLUMNS`,
        one row per unit and cycle, in the order the units were given.
    """
    rows = []
    for unit_id, n_cycles in lifetimes.items():
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "unit_id": unit_id,
                    "time_cycles": cycle,
                    "sensor_a": unit_id * 1000.0 + cycle,
                    "sensor_b": unit_id * 10.0 - 0.25 * cycle,
                }
            )
    return pd.DataFrame(rows)


def build_random_trajectories(
    rng: np.random.Generator,
    *,
    unit_ids: Sequence[int] | range,
    n_cycles: int,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Build equal-length trajectories of random readings over given columns.

    Intended for the end-to-end tests, which stand in for the real FD001
    splits and therefore need the example's full sensor column set and
    readings with actual variance to standardize.

    Args:
        rng: Random generator drawing the sensor readings.
        unit_ids: Unit IDs to generate one trajectory each for.
        n_cycles: Number of cycles per trajectory.
        feature_columns: Sensor column names to emit.

    Returns:
        A DataFrame with `unit_id`, `time_cycles`, and `feature_columns`,
        one row per unit and cycle.
    """
    frames = []
    for unit_id in unit_ids:
        frame = pd.DataFrame(
            rng.normal(loc=float(unit_id), size=(n_cycles, len(feature_columns))),
            columns=list(feature_columns),
        )
        frame.insert(0, "unit_id", unit_id)
        frame.insert(1, "time_cycles", np.arange(1, n_cycles + 1))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
