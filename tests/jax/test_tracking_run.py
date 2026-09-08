"""End-to-end test for the tracking example's runner.

One test drives the runner's `main` with a temporary log directory and
asserts that a real TensorBoard event file was written, so that a
breaking change in the writer's API is caught here rather than
discovered by a user. It is skipped when the example's dependency group
is absent, which keeps the default suite runnable everywhere.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType


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
