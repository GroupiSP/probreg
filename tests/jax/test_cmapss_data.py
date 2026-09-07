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


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the archive cache at a per-test directory, never the real one."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(_DATA._CACHE_DIR_ENV_VAR, str(cache_dir))
    monkeypatch.delenv(_DATA._ARCHIVE_OVERRIDE_ENV_VAR, raising=False)
    return cache_dir


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


def _write_archive(destination: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(destination, mode="w") as archive:
        for member, contents in members.items():
            archive.writestr(member, contents)


def _fake_urlretrieve_writing_members(
    monkeypatch: pytest.MonkeyPatch, members: dict[str, str]
) -> list[int]:
    """Patch `urllib.request.urlretrieve` to write an in-memory zip.

    Args:
        monkeypatch: The active monkeypatch fixture.
        members: Mapping of archive member name to its file contents.

    Returns:
        A single-element list tracking how many times the fake download
        was invoked, so callers can assert on caching/retry behavior.
    """
    call_count = [0]

    def fake_urlretrieve(url: str, destination: str) -> None:
        assert url == _DATA._CMAPSS_URL
        call_count[0] += 1
        _write_archive(Path(destination), members)

    monkeypatch.setattr(_DATA.urllib.request, "urlretrieve", fake_urlretrieve)
    return call_count


def test_load_fd001_data_selects_columns_and_caches_archive(
    monkeypatch: pytest.MonkeyPatch, isolated_cache: Path
) -> None:
    _fake_urlretrieve_writing_members(
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
    assert (isolated_cache / _DATA._CACHED_ARCHIVE_NAME).is_file()


def test_load_fd001_test_data_selects_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_urlretrieve_writing_members(
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


def test_load_fd001_test_rul_returns_values_in_unit_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_urlretrieve_writing_members(monkeypatch, {_DATA._RUL_MEMBER: "112\n98\n45\n"})

    rul = _DATA.load_fd001_test_rul()

    assert isinstance(rul, _DATA.np.ndarray)
    assert rul.tolist() == [112.0, 98.0, 45.0]


def test_second_load_reuses_cache_without_redownloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = _fake_urlretrieve_writing_members(
        monkeypatch, {_DATA._TRAIN_MEMBER: _cmapss_row(1, 1, 100)}
    )

    _DATA.load_fd001_data()
    _DATA.load_fd001_data()

    assert call_count[0] == 1


def test_download_retries_transient_failures_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_DATA.time, "sleep", lambda _seconds: None)
    attempts = [0]

    def flaky_urlretrieve(url: str, destination: str) -> None:
        attempts[0] += 1
        if attempts[0] < _DATA._DOWNLOAD_ATTEMPTS:
            raise OSError("transient failure")
        _write_archive(Path(destination), {_DATA._TRAIN_MEMBER: _cmapss_row(1, 1, 100)})

    monkeypatch.setattr(_DATA.urllib.request, "urlretrieve", flaky_urlretrieve)

    data = _DATA.load_fd001_data()

    assert attempts[0] == _DATA._DOWNLOAD_ATTEMPTS
    assert data["unit_id"].tolist() == [1]


def test_download_raises_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_DATA.time, "sleep", lambda _seconds: None)

    def always_fails(url: str, destination: str) -> None:
        raise OSError("permanently unreachable")

    monkeypatch.setattr(_DATA.urllib.request, "urlretrieve", always_fails)

    with pytest.raises(RuntimeError) as excinfo:
        _DATA.load_fd001_data()

    assert _DATA._ARCHIVE_OVERRIDE_ENV_VAR in str(excinfo.value)


def test_corrupt_download_is_retried_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_DATA.time, "sleep", lambda _seconds: None)

    def writes_garbage(url: str, destination: str) -> None:
        Path(destination).write_bytes(b"not a zip file")

    monkeypatch.setattr(_DATA.urllib.request, "urlretrieve", writes_garbage)

    with pytest.raises(RuntimeError, match="corrupt"):
        _DATA.load_fd001_data()


def test_corrupt_cached_archive_is_redownloaded(
    monkeypatch: pytest.MonkeyPatch, isolated_cache: Path
) -> None:
    isolated_cache.mkdir(parents=True)
    (isolated_cache / _DATA._CACHED_ARCHIVE_NAME).write_bytes(b"not a zip file")
    call_count = _fake_urlretrieve_writing_members(
        monkeypatch, {_DATA._TRAIN_MEMBER: _cmapss_row(1, 1, 100)}
    )

    data = _DATA.load_fd001_data()

    assert call_count[0] == 1
    assert data["unit_id"].tolist() == [1]


def test_archive_override_env_var_bypasses_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override_path = tmp_path / "manual" / "CMAPSSData.zip"
    override_path.parent.mkdir()
    _write_archive(override_path, {_DATA._TRAIN_MEMBER: _cmapss_row(1, 1, 100)})
    monkeypatch.setenv(_DATA._ARCHIVE_OVERRIDE_ENV_VAR, str(override_path))

    def fails_if_called(url: str, destination: str) -> None:
        raise AssertionError("network download should not be attempted")

    monkeypatch.setattr(_DATA.urllib.request, "urlretrieve", fails_if_called)

    data = _DATA.load_fd001_data()

    assert data["unit_id"].tolist() == [1]


def test_archive_override_parameter_bypasses_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override_path = tmp_path / "manual" / "CMAPSSData.zip"
    override_path.parent.mkdir()
    _write_archive(override_path, {_DATA._TRAIN_MEMBER: _cmapss_row(1, 1, 100)})

    def fails_if_called(url: str, destination: str) -> None:
        raise AssertionError("network download should not be attempted")

    monkeypatch.setattr(_DATA.urllib.request, "urlretrieve", fails_if_called)

    data = _DATA.load_fd001_data(archive_override=override_path)

    assert data["unit_id"].tolist() == [1]


def test_invalid_archive_override_raises(tmp_path: Path) -> None:
    override_path = tmp_path / "not_a_zip.zip"
    override_path.write_bytes(b"not a zip file")

    with pytest.raises(RuntimeError, match="not a valid zip"):
        _DATA.load_fd001_data(archive_override=override_path)


def test_default_cache_dir_honors_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert _DATA._default_cache_dir() == tmp_path / "probreg-cmapss"


def test_default_cache_dir_falls_back_to_home_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(_DATA.Path, "home", classmethod(lambda cls: tmp_path))

    assert _DATA._default_cache_dir() == tmp_path / ".cache" / "probreg-cmapss"


def test_main_fetch_only_populates_cache_without_plotting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_urlretrieve_writing_members(
        monkeypatch, {_DATA._TRAIN_MEMBER: _cmapss_row(1, 1, 100)}
    )
    monkeypatch.setattr(sys, "argv", ["data.py", "--fetch"])

    def fails_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("plotting should not happen with --fetch")

    monkeypatch.setattr(_DATA, "plot_sensor_data", fails_if_called)

    _DATA.main()


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
