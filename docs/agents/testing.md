# Testing

Tests use `pytest` and live in `tests/`, in modules mirroring the structure of `src/probreg/`
(so `tests/jax/` mirrors the JAX backend, `tests/core/` the core). A test not tied to one
`src/probreg` module, such as a check of the docs site or the API reference against the source,
lives at the top of `tests/` (for example `tests/test_reference_completeness.py`).

## Fixtures

Reuse fixtures aggressively; introduce or extend a `conftest.py` rather than repeating setup
across modules.

## Property-based over example-based

`hypothesis` is a dev dependency and is the preferred tool. Before writing a table of example
cases, ask which mathematical property the code must satisfy — invariance, symmetry,
monotonicity, idempotence, a conservation law — and test that instead. Example-based tests are
for the specific regressions and boundary cases a property cannot express.

Design for this: when writing implementation code, decide up front which properties of it are
testable.

## Markdown snippets

`pytest-markdown-docs` collects every fenced `python` block in `docs_site/` and `README.md` as a
test; `uv run pytest tests/core` (or any narrower path) skips them. Mark a fragment that cannot
run standalone as ```` ```{.python notest} ````; see [`CONTRIBUTING.md`](../../CONTRIBUTING.md)
for why the brace form is required.
