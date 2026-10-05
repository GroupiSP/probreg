"""End-to-end tests for the tracking example's runner.

One test drives the runner's `main` with a temporary log directory and
asserts that a real TensorBoard event file was written, so that a
breaking change in the writer's API is caught here rather than
discovered by a user. Another drives the training through a recording
tracker and asserts the parameter paths it logs. Both are skipped when
the example's dependency group is absent, which keeps the default suite
runnable everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from probreg.core.naming import Split, flatten_parameters


def test_main_writes_one_timestamped_run_of_event_files(
    tracking_run: ModuleType, tmp_path: Path
) -> None:
    logdir = tmp_path / "runs"

    tracking_run.main(["--logdir", str(logdir), "--epochs", "2"])

    run_directories = sorted(path for path in logdir.iterdir() if path.is_dir())
    assert len(run_directories) == 1
    event_files = list(run_directories[0].rglob("events.out.tfevents.*"))
    assert event_files
    assert all(path.stat().st_size > 0 for path in event_files)


class RecordingTracker:
    """An experiment tracker recording the parameters it is given."""

    def __init__(self) -> None:
        self.params: list[Mapping[str, Any]] = []

    def log_params(self, values: Mapping[str, Any]) -> None:
        self.params.append(values)

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        pass

    def log_artifact(self, name: str, value: Any) -> None:
        pass


def test_the_run_logs_grouped_parameter_paths_with_splits_as_segments(
    tracking_run: ModuleType,
) -> None:
    tracker = RecordingTracker()

    tracking_run.run_tracked_training(tracker, epochs=1)

    (params,) = tracker.params
    paths = flatten_parameters(params)
    assert {
        "data/train/samples",
        "data/validation/samples",
        "data/train/batch_size",
        "data/validation/batch_size",
    } <= paths.keys()
    # Fully grouped: no parameter sits flat at the top level.
    assert all("/" in path for path in paths)
    # No leaf key encodes a split; a split is only ever a segment of its own.
    assert not any(
        split.value in path.rsplit("/", 1)[-1] for path in paths for split in Split
    )
