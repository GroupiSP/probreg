"""Load and visualize selected NASA CMAPSS FD001 sensor trajectories."""

from __future__ import annotations

import tempfile
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
_TRAIN_MEMBER = "train_FD001.txt"
_COLUMN_NAMES = [
    "unit_id",
    "time_cycles",
    *(f"operational_setting_{index}" for index in range(1, 4)),
    *(f"sensor_{index}" for index in range(1, 22)),
]
_SENSOR_NAMES = [f"sensor_{i}" for i in (11, 12, 4, 7, 15, 20, 21, 2, 17)]
_SELECTED_COLUMNS = ["unit_id", "time_cycles", *_SENSOR_NAMES]


def _download_archive(url: str, destination: Path) -> None:
    """Download the CMAPSS archive to a temporary destination.

    Args:
        url: URL of the CMAPSS ZIP archive.
        destination: Local path where the archive will be written.
    """
    urllib.request.urlretrieve(url, destination)


def load_fd001_data() -> pd.DataFrame:
    """Load selected sensor columns from the CMAPSS FD001 training split.

    The downloaded archive is stored only for the duration of this function.
    The returned DataFrame remains available in memory after the temporary
    directory and archive have been removed.

    Returns:
        The FD001 training trajectories with unit ID, time cycle, and the
        selected sensor columns.
    """
    with tempfile.TemporaryDirectory(prefix="probreg-cmapss-") as temporary_dir:
        archive_path = Path(temporary_dir) / "CMAPSSData.zip"
        _download_archive(_CMAPSS_URL, archive_path)

        with (
            zipfile.ZipFile(archive_path) as archive,
            archive.open(_TRAIN_MEMBER) as training_data,
        ):
            return pd.read_csv(
                training_data,
                sep=r"\s+",
                header=None,
                names=_COLUMN_NAMES,
                usecols=_SELECTED_COLUMNS,
            )[_SELECTED_COLUMNS]


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
    """Load the selected FD001 sensor data and display its trajectories."""
    data = load_fd001_data()
    plot_sensor_data(data)


if __name__ == "__main__":
    main()
