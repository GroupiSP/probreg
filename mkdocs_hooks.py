"""MkDocs hooks for the probreg docs site.

mkdocstrings reads the packages by static analysis, so a `probreg.jax` that cannot
import (JAX missing, a broken module) would still render its reference page from the
source alone. Importing both packages before the build turns that into a build failure.
"""

from __future__ import annotations

import importlib

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError

DOCUMENTED_PACKAGES = ("probreg.core", "probreg.jax")


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    """Fail the build unless every documented package imports.

    Args:
        config: The loaded MkDocs configuration, returned unchanged.

    Returns:
        The configuration, unchanged.

    Raises:
        PluginError: If a documented package fails to import.
    """
    for package in DOCUMENTED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError as error:
            raise PluginError(
                f"{package} failed to import, so its API reference cannot be trusted: "
                f"{error}. Build with `uv run --extra jax --group docs mkdocs build`."
            ) from error
    return config
