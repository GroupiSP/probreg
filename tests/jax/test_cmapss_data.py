"""Tests for the CMAPSS FD001 data exploration example."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("seaborn")

_DATA_PATH = Path(__file__).parents[2] / "examples" / "jax" / "cmapss" / "data.py"
_SPEC = importlib.util.spec_from_file_location("cmapss_data", _DATA_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the CMAPSS data module.")
_DATA = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DATA
_SPEC.loader.exec_module(_DATA)


def _cmapss_row(unit_id: int, time_cycles: int, sensor_offset: int) -> str:
    values = [
        unit_id,
        time_cycles,
        0.1,
        0.2,
        100.0,
        *(sensor_offset + sensor for sensor in range(1, 22)),
    ]
    return " ".join(str(value) for value in values)


def _fake_download_writing_members(
    monkeypatch: pytest.MonkeyPatch, members: dict[str, str]
) -> list[Path | None]:
    """Patch `_DATA._download_archive` to write an in-memory zip.

    Args:
        monkeypatch: The active monkeypatch fixture.
        members: Mapping of archive member name to its file contents.

    Returns:
        A single-element list that receives the downloaded archive's path
        once the patched function has been called, so callers can assert on
        cleanup after the archive-consuming call returns.
    """
    downloaded_archive: list[Path | None] = [None]

    def fake_download(url: str, destination: Path) -> None:
        assert url == _DATA._CMAPSS_URL
        downloaded_archive[0] = destination
        with zipfile.ZipFile(destination, mode="w") as archive:
            for member, contents in members.items():
                archive.writestr(member, contents)

    monkeypatch.setattr(_DATA, "_download_archive", fake_download)
    return downloaded_archive


def _assert_archive_cleaned_up(downloaded_archive: list[Path | None]) -> None:
    archive_path = downloaded_archive[0]
    assert archive_path is not None
    assert not archive_path.exists()
    assert not archive_path.parent.exists()


def test_load_fd001_data_selects_columns_and_removes_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded_archive = _fake_download_writing_members(
        monkeypatch,
        {
            _DATA._TRAIN_MEMBER: "\n".join(
                [
                    _cmapss_row(unit_id=1, time_cycles=1, sensor_offset=100),
                    _cmapss_row(unit_id=2, time_cycles=3, sensor_offset=200),
                ]
            )
        },
    )

    data = _DATA.load_fd001_data()

    assert data.columns.tolist() == _DATA._SELECTED_COLUMNS
    assert data["unit_id"].tolist() == [1, 2]
    assert data["time_cycles"].tolist() == [1, 3]
    assert data["sensor_11"].tolist() == [111, 211]
    assert data["sensor_17"].tolist() == [117, 217]
    _assert_archive_cleaned_up(downloaded_archive)


def test_load_fd001_test_data_selects_columns_and_removes_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded_archive = _fake_download_writing_members(
        monkeypatch,
        {
            _DATA._TEST_MEMBER: "\n".join(
                [
                    _cmapss_row(unit_id=1, time_cycles=1, sensor_offset=100),
                    _cmapss_row(unit_id=2, time_cycles=3, sensor_offset=200),
                ]
            )
        },
    )

    data = _DATA.load_fd001_test_data()

    assert data.columns.tolist() == _DATA._SELECTED_COLUMNS
    assert data["unit_id"].tolist() == [1, 2]
    assert data["time_cycles"].tolist() == [1, 3]
    assert data["sensor_11"].tolist() == [111, 211]
    assert data["sensor_17"].tolist() == [117, 217]
    _assert_archive_cleaned_up(downloaded_archive)


def test_load_fd001_test_rul_returns_values_in_unit_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded_archive = _fake_download_writing_members(
        monkeypatch, {_DATA._RUL_MEMBER: "112\n98\n45\n"}
    )

    rul = _DATA.load_fd001_test_rul()

    assert isinstance(rul, _DATA.np.ndarray)
    assert rul.tolist() == [112.0, 98.0, 45.0]
    _assert_archive_cleaned_up(downloaded_archive)


def test_plot_sensor_data_facets_raw_trajectories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "time_cycles": [1, 2, 1, 2],
            **{
                sensor: [index, index + 1, index + 2, index + 3]
                for index, sensor in enumerate(_DATA._SENSOR_NAMES)
            },
        }
    )
    calls: dict[str, Any] = {}

    class FakeGrid:
        def __init__(self) -> None:
            self.figure = SimpleNamespace(
                suptitle=lambda *args, **kwargs: calls.update(suptitle=(args, kwargs))
            )

        def set_axis_labels(self, *args: str) -> None:
            calls["axis_labels"] = args

        def set_titles(self, template: str) -> None:
            calls["titles"] = template

        def tight_layout(self) -> None:
            calls["tight_layout"] = True

    def fake_relplot(**kwargs: Any) -> FakeGrid:
        calls["relplot"] = kwargs
        return FakeGrid()

    monkeypatch.setattr(_DATA.sns, "relplot", fake_relplot)
    monkeypatch.setattr(_DATA.plt, "show", lambda: calls.update(show=True))

    _DATA.plot_sensor_data(data)

    plot_call = calls["relplot"]
    assert plot_call["kind"] == "line"
    assert plot_call["units"] == "unit_id"
    assert plot_call["estimator"] is None
    assert plot_call["col"] == "sensor"
    assert plot_call["col_order"] == _DATA._SENSOR_NAMES
    assert plot_call["facet_kws"] == {"sharey": False}
    assert plot_call["data"]["sensor"].drop_duplicates().tolist() == (
        _DATA._SENSOR_NAMES
    )
    assert len(plot_call["data"]) == len(data) * len(_DATA._SENSOR_NAMES)
    assert calls["show"] is True
