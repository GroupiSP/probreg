# The library is named stagereg

The library was developed as `probreg`, but that name is taken on PyPI by an unrelated point-cloud registration library, so it cannot be published under it. We renamed it to `stagereg` before the first release. The name matches **Stage**, the central term in the glossary, and the library's own description as a library for stage-oriented probabilistic regression: a model is trained as a sequence of stages, and the name says so. `stagereg` was free on PyPI, TestPyPI and GitHub when we chose it.

## Considered Options

- **Keep `probreg`** and publish under a different distribution name. Rejected: users would `pip install` one name and `import` another, and the PyPI name would always point readers to the wrong project.
- **`musigma`**, after the mean and standard deviation the library predicts. Dropped: it describes the output of any probabilistic regressor, not what sets this library apart, and it has no tie to the glossary.

## Consequences

The rename covers everything: the distribution name, the import package, the GitHub repository (`GroupiSP/stagereg`), the docs site URL, the CMAPSS environment variables (`STAGEREG_CMAPSS_*`), the per-user CMAPSS cache directory, and all prose. There is no compatibility shim: nothing re-exports under the old import name and the old environment variable names are not read, because there were no released users to protect. This ADR is the only place in the repository where the old name stays, as history; a check fails if it comes back anywhere else. Once `stagereg` is on PyPI, changing the name again is expensive, since every pinned install, citation and link would break, so do not rename it back.
