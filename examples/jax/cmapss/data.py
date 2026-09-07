"""Load and visualize selected NASA CMAPSS FD001 sensor trajectories.

The CMAPSS archive is fetched at most once: subsequent loader calls reuse a
persistent on-disk cache instead of re-downloading it. If NASA's servers
can't be reached at all, a manually obtained copy of the archive can be
supplied via `PROBREG_CMAPSS_ARCHIVE` (or the `archive_override` parameter)
to bypass the network entirely.

Run this module directly to pre-fetch the archive into the cache without
loading or plotting anything:

    uv run --group example-cmapss python examples/jax/cmapss/data.py --fetch
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
_TRAIN_MEMBER = "train_FD001.txt"
_TEST_MEMBER = "test_FD001.txt"
_RUL_MEMBER = "RUL_FD001.txt"
_COLUMN_NAMES = [
    "unit_id",
    "time_cycles",
    *(f"operational_setting_{index}" for index in range(1, 4)),
    *(f"sensor_{index}" for index in range(1, 22)),
]
_SENSOR_NAMES = [f"sensor_{i}" for i in (11, 12, 4, 7, 15, 20, 21, 2, 17)]
_SELECTED_COLUMNS = ["unit_id", "time_cycles", *_SENSOR_NAMES]

_CACHE_DIR_ENV_VAR = "PROBREG_CMAPSS_CACHE_DIR"
_ARCHIVE_OVERRIDE_ENV_VAR = "PROBREG_CMAPSS_ARCHIVE"
_CACHED_ARCHIVE_NAME = "CMAPSSData.zip"
_DOWNLOAD_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0


def _default_cache_dir() -> Path:
    """Resolve the per-user cache directory used when no override is set.

    Returns:
        `$XDG_CACHE_HOME/probreg-cmapss` if `XDG_CACHE_HOME` is set,
        otherwise `~/.cache/probreg-cmapss`.
    """
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
    return base / "probreg-cmapss"


def _cache_dir() -> Path:
    """Resolve the archive cache directory, honoring `PROBREG_CMAPSS_CACHE_DIR`."""
    override = os.environ.get(_CACHE_DIR_ENV_VAR)
    return Path(override) if override else _default_cache_dir()


def _archive_override(archive_override: Path | None) -> Path | None:
    """Resolve a manually supplied archive path, if any.

    Args:
        archive_override: An explicit override path, or `None` to fall back
            to the `PROBREG_CMAPSS_ARCHIVE` environment variable.

    Returns:
        The resolved override path, or `None` if neither was supplied.
    """
    if archive_override is not None:
        return archive_override
    env_override = os.environ.get(_ARCHIVE_OVERRIDE_ENV_VAR)
    return Path(env_override) if env_override else None


def _is_valid_archive(archive_path: Path) -> bool:
    """Check that `archive_path` is a well-formed, complete zip archive."""
    if not archive_path.is_file():
        return False
    if not zipfile.is_zipfile(archive_path):
        return False
    try:
        with zipfile.ZipFile(archive_path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def _download_archive(url: str, destination: Path) -> None:
    """Download the CMAPSS archive to `destination`, retrying transient failures.

    Each attempt is validated as a well-formed zip before being accepted;
    a corrupt or incomplete download is treated as a failed attempt.

    Args:
        url: URL of the CMAPSS ZIP archive.
        destination: Local path where the archive will be written.

    Raises:
        RuntimeError: All download attempts failed or produced a corrupt
            archive, and no manual override was available to fall back on.
    """
    # Downloaded to a sibling temp file and atomically renamed into place so
    # a concurrent reader never observes a partially written or corrupt
    # `destination`.
    temp_destination = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        if attempt > 0:
            time.sleep(_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
        try:
            urllib.request.urlretrieve(url, temp_destination)
        except OSError as error:
            last_error = error
            continue
        if _is_valid_archive(temp_destination):
            os.replace(temp_destination, destination)
            return
        last_error = RuntimeError("downloaded archive is corrupt or incomplete.")
    temp_destination.unlink(missing_ok=True)
    raise RuntimeError(
        f"failed to download the CMAPSS archive from {url} after "
        f"{_DOWNLOAD_ATTEMPTS} attempts: {last_error}. If the archive has "
        "been downloaded manually, point the "
        f"{_ARCHIVE_OVERRIDE_ENV_VAR} environment variable at the local "
        f"{_CACHED_ARCHIVE_NAME} file instead."
    )


def _ensure_cached_archive(archive_override: Path | None = None) -> Path:
    """Ensure a valid CMAPSS archive is available in the cache, and return its path.

    A manual override (explicit argument or `PROBREG_CMAPSS_ARCHIVE`) is
    used to seed the cache if present. Otherwise, a cached archive is
    reused if it's still a valid zip; a missing or corrupt cache entry is
    (re-)downloaded.

    Args:
        archive_override: An explicit path to a pre-downloaded archive,
            bypassing the network. Falls back to the
            `PROBREG_CMAPSS_ARCHIVE` environment variable when omitted.

    Returns:
        Path to a valid, cached CMAPSS archive.
    """
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_archive = cache_dir / _CACHED_ARCHIVE_NAME

    override = _archive_override(archive_override)
    if override is not None:
        if not _is_valid_archive(override):
            raise RuntimeError(f"archive override {override} is not a valid zip file.")
        temp_archive = cached_archive.with_name(cached_archive.name + ".part")
        temp_archive.write_bytes(override.read_bytes())
        os.replace(temp_archive, cached_archive)
        return cached_archive

    if not _is_valid_archive(cached_archive):
        _download_archive(_CMAPSS_URL, cached_archive)
    return cached_archive


def _load_member(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """Read one CMAPSS trajectories member into the selected-column shape.

    Args:
        archive: Open CMAPSS ZIP archive.
        member: Name of the whitespace-separated trajectories file to read
            from the archive.

    Returns:
        The trajectories with unit ID, time cycle, and the selected sensor
        columns.
    """
    with archive.open(member) as member_data:
        return pd.read_csv(
            member_data,
            sep=r"\s+",
            header=None,
            names=_COLUMN_NAMES,
            usecols=_SELECTED_COLUMNS,
        )[_SELECTED_COLUMNS]


def load_fd001_data(archive_override: Path | None = None) -> pd.DataFrame:
    """Load selected sensor columns from the CMAPSS FD001 training split.

    The CMAPSS archive is fetched into a persistent cache on first use and
    reused on subsequent calls; see the module docstring for cache and
    override configuration.

    Args:
        archive_override: An explicit path to a pre-downloaded archive,
            bypassing the network. Falls back to the
            `PROBREG_CMAPSS_ARCHIVE` environment variable when omitted.

    Returns:
        The FD001 training trajectories with unit ID, time cycle, and the
        selected sensor columns.
    """
    archive_path = _ensure_cached_archive(archive_override)
    with zipfile.ZipFile(archive_path) as archive:
        return _load_member(archive, _TRAIN_MEMBER)


def load_fd001_test_data(archive_override: Path | None = None) -> pd.DataFrame:
    """Load selected sensor columns from the CMAPSS FD001 test split.

    The CMAPSS archive is fetched into a persistent cache on first use and
    reused on subsequent calls; see the module docstring for cache and
    override configuration.

    Args:
        archive_override: An explicit path to a pre-downloaded archive,
            bypassing the network. Falls back to the
            `PROBREG_CMAPSS_ARCHIVE` environment variable when omitted.

    Returns:
        The FD001 truncated test trajectories with unit ID, time cycle, and
        the selected sensor columns.
    """
    archive_path = _ensure_cached_archive(archive_override)
    with zipfile.ZipFile(archive_path) as archive:
        return _load_member(archive, _TEST_MEMBER)


def load_fd001_test_rul(archive_override: Path | None = None) -> np.ndarray:
    """Load the ground-truth remaining RUL for each FD001 test unit.

    The CMAPSS archive is fetched into a persistent cache on first use and
    reused on subsequent calls; see the module docstring for cache and
    override configuration. Values are returned in the same order as the
    RUL file's lines, which matches the order in which unit IDs first
    appear in the FD001 test trajectories.

    Args:
        archive_override: An explicit path to a pre-downloaded archive,
            bypassing the network. Falls back to the
            `PROBREG_CMAPSS_ARCHIVE` environment variable when omitted.

    Returns:
        A 1D array of ground-truth remaining useful life values, one per
        test unit.
    """
    archive_path = _ensure_cached_archive(archive_override)
    with (
        zipfile.ZipFile(archive_path) as archive,
        archive.open(_RUL_MEMBER) as rul_data,
    ):
        return pd.read_csv(rul_data, sep=r"\s+", header=None, names=["rul"])[
            "rul"
        ].to_numpy(dtype=float)


def plot_sensor_data(data: pd.DataFrame) -> None:
    """Plot every FD001 unit trajectory in a separate facet per sensor.

    Args:
        data: Wide CMAPSS DataFrame containing unit ID, time cycle, and each
            selected sensor column.
    """
    plot_data = data.melt(
        id_vars=["unit_id", "time_cycles"],
        value_vars=_SENSOR_NAMES,
        var_name="sensor",
        value_name="reading",
    )
    grid = sns.relplot(
        data=plot_data,
        kind="line",
        x="time_cycles",
        y="reading",
        units="unit_id",
        estimator=None,
        col="sensor",
        col_order=_SENSOR_NAMES,
        col_wrap=3,
        height=2.5,
        aspect=1.3,
        alpha=0.35,
        linewidth=0.8,
        facet_kws={"sharey": False},
    )
    grid.set_axis_labels("Time cycles", "Sensor reading")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("CMAPSS FD001 training sensor trajectories", y=1.02)
    grid.tight_layout()
    plt.show()


def main() -> None:
    """Load the selected FD001 sensor data and display its trajectories.

    Pass `--fetch` to only pre-populate the archive cache, without loading or plotting
    anything.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Only fetch the CMAPSS archive into the cache, then exit.",
    )
    args = parser.parse_args()

    if args.fetch:
        archive_path = _ensure_cached_archive()
        print(f"CMAPSS archive cached at {archive_path}")
        return

    data = load_fd001_data()
    plot_sensor_data(data)


if __name__ == "__main__":
    main()
