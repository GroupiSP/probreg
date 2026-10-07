# User docs are built with MkDocs and mkdocstrings; docstring cross-references use autorefs syntax

The user documentation site is MkDocs Material with `mkdocstrings[python]`, built from a top-level `docs_site/` that is kept apart from `docs/`, and every cross-reference inside a docstring is written in mkdocstrings' autorefs form (`` [`EarlyStopper`][probreg.core.EarlyStopper] ``) instead of a Sphinx role (`` :class:`EarlyStopper` ``). The repo's prose is already plain Markdown and its docstrings are already Google-style, so MkDocs adds a theme and a plugin where Sphinx would add a `conf.py`, MyST directive syntax and a separate doctest builder. Writing references in autorefs form gets back most of what Sphinx's `--nitpicky` would have given: `mkdocs build --strict` fails on a reference that does not resolve, so a renamed symbol breaks the docs build rather than quietly leaving dead text in the reference.

`docs_site/` is separate so that publishing never becomes a reason to reshape `docs/`, which holds the internal design records (ADRs and agent notes) that are written for contributors, not users. Only `GLOSSARY.md` and the README's minimal example reach the site, and both are pulled in by include rather than copied, so each stays the single source.

## Considered Options

- **Sphinx + MyST + autodoc.** It has the strongest reference checking and intersphinx links into NumPy and JAX. Rejected while the reference is two modules: that is setup cost for a guarantee the strict build mostly covers. Revisit if the reference grows large or links into NumPy/JAX become valuable.
- **GitHub Pages serving `docs/` through built-in Jekyll.** Rejected: it cannot generate an API reference from docstrings.
- **Leaving the Sphinx roles in place**, or rewriting them as plain inline code. Rejected: mkdocstrings renders a role as a literal `:class:` in front of the code, and plain code drops the check that a strict build performs on a real cross-reference.

## Consequences

The toolchain choice is now built into the source tree as well as the build config. Moving to Sphinx later means rewriting every docstring cross-reference, not just replacing `mkdocs.yml`. Because docstrings now carry references the strict build checks, `mkdocs build --strict` is part of the contributor verification gate next to pre-commit and pytest: a symbol rename can break the docs without failing a single test.
