"""Tests of the distribution a user downloads from PyPI.

The module builds the wheel and the sdist once with `uv build`, then checks what a user
sees: the metadata on the package page, the files each archive holds, and that a clean
install of the wheel imports. It never inspects how the build backend is configured.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.message import Message
from email.parser import HeaderParser
from pathlib import Path, PurePosixPath

import pytest
import yaml


REPOSITORY = Path(__file__).parent.parent
PROJECT = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())["project"]
DECLARED_VERSION = PROJECT["version"]
GROUP = "Intelligent System Prognostics for Operations & Maintenance Research Group"
EXPECTED_CLASSIFIERS = {
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
}
REPO_ONLY_TOP_LEVEL = {"examples", "docs", "docs_site", "research", "site", ".github"}


def _uv() -> str:
    """Return the path of the `uv` executable, skipping when it is not installed."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed.")
    return uv


@pytest.fixture(scope="module")
def dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the wheel and sdist from the working tree into a fresh directory."""
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [_uv(), "build", "--quiet", "--out-dir", str(out), str(REPOSITORY)],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def wheel(dist: Path) -> Path:
    """Return the path of the single wheel the build produced."""
    (path,) = dist.glob("*.whl")
    return path


@pytest.fixture(scope="module")
def sdist(dist: Path) -> Path:
    """Return the path of the single sdist the build produced."""
    (path,) = dist.glob("*.tar.gz")
    return path


@pytest.fixture(scope="module")
def metadata(wheel: Path) -> Message:
    """Return the core metadata a package index shows for the wheel."""
    with zipfile.ZipFile(wheel) as archive:
        (name,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        return HeaderParser().parsestr(archive.read(name).decode())


@pytest.fixture(scope="module")
def sdist_members(sdist: Path) -> set[PurePosixPath]:
    """Return the sdist's file paths, relative to its single top-level directory."""
    with tarfile.open(sdist) as archive:
        return {
            PurePosixPath(*PurePosixPath(m.name).parts[1:])
            for m in archive.getmembers()
            if m.isfile()
        }


def test_metadata_names_the_package_and_its_declared_version(metadata: Message) -> None:
    assert metadata["Name"] == "stagereg"
    assert metadata["Version"] == DECLARED_VERSION
    assert metadata["Requires-Python"] == ">=3.12"


def test_metadata_credits_the_author_and_the_research_group(metadata: Message) -> None:
    authors = metadata["Author"] or ""
    emails = metadata["Author-email"] or ""
    credited = authors + ", " + emails
    assert "Giorgio Tosti Balducci <g.b.l.tostibalducci@tudelft.nl>" in emails
    assert GROUP in credited


def test_metadata_declares_bsd_3_clause_with_the_licence_file(
    metadata: Message, wheel: Path
) -> None:
    assert metadata["License-Expression"] == "BSD-3-Clause"
    assert metadata.get_all("License-File") == ["LICENSE"]
    with zipfile.ZipFile(wheel) as archive:
        licences = [n for n in archive.namelist() if n.endswith("licenses/LICENSE")]
        (licence,) = licences
        assert "Redistribution and use in source and binary forms" in (
            archive.read(licence).decode()
        )


def test_metadata_lists_the_agreed_classifiers(metadata: Message) -> None:
    assert set(metadata.get_all("Classifier") or []) == EXPECTED_CLASSIFIERS


def test_metadata_links_the_repository_docs_and_issues(metadata: Message) -> None:
    urls = set(metadata.get_all("Project-URL") or [])
    assert urls == {
        "Repository, https://github.com/GroupiSP/stagereg",
        "Documentation, https://groupisp.github.io/stagereg/",
        "Issues, https://github.com/GroupiSP/stagereg/issues",
    }


def test_metadata_offers_the_jax_and_plot_extras(metadata: Message) -> None:
    assert set(metadata.get_all("Provides-Extra") or []) == {"jax", "plot"}


def test_wheel_holds_only_the_stagereg_package(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = [PurePosixPath(n) for n in archive.namelist()]
    top_level = {n.parts[0] for n in names}
    assert top_level == {"stagereg", f"stagereg-{DECLARED_VERSION}.dist-info"}
    assert PurePosixPath("stagereg/py.typed") in names
    assert PurePosixPath("stagereg/jax/__init__.py") in names


def test_sdist_ships_the_source_metadata_and_tests(
    sdist_members: set[PurePosixPath],
) -> None:
    for required in ("PKG-INFO", "pyproject.toml", "README.md", "LICENSE"):
        assert PurePosixPath(required) in sdist_members
    assert PurePosixPath("src/stagereg/py.typed") in sdist_members
    assert PurePosixPath("tests/conftest.py") in sdist_members
    assert PurePosixPath("tests/core/test_stages.py") in sdist_members


def test_sdist_leaves_out_repo_only_material(
    sdist_members: set[PurePosixPath],
) -> None:
    top_level = {m.parts[0] for m in sdist_members}
    assert not top_level & REPO_ONLY_TOP_LEVEL


@pytest.mark.parametrize(
    ("requirement", "module"),
    [("stagereg", "stagereg"), ("stagereg[jax]", "stagereg.jax")],
)
def test_clean_install_of_the_wheel_imports(
    wheel: Path, tmp_path: Path, requirement: str, module: str
) -> None:
    uv = _uv()
    venv = tmp_path / "venv"
    subprocess.run(
        [uv, "venv", "--quiet", "--python", sys.executable, str(venv)],
        check=True,
        capture_output=True,
    )
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--quiet",
            "--python",
            str(python),
            f"{requirement} @ {wheel.as_uri()}",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(python), "-I", "-c", f"import {module}"],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )


def test_citation_matches_the_declared_version_and_authors() -> None:
    citation = yaml.safe_load((REPOSITORY / "CITATION.cff").read_text())
    assert citation["version"] == DECLARED_VERSION
    assert citation["license"] == PROJECT["license"]
    assert citation["repository-code"] == PROJECT["urls"]["Repository"]
    assert citation["url"] == PROJECT["urls"]["Documentation"]
    cited = [
        author.get("name") or f"{author['given-names']} {author['family-names']}"
        for author in citation["authors"]
    ]
    declared = [author["name"] for author in PROJECT["authors"]]
    assert cited == declared == ["Giorgio Tosti Balducci", GROUP]
