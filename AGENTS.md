# AGENTS.md

`probreg` is a Python library for stage-oriented probabilistic regression, with a JAX/Flax
backend and runnable examples under `examples/jax/`.

## Environment

The package manager is **`uv`**, not pip. Every Python command runs through it: `uv run pytest`,
`uv run pre-commit run --all-files`. Dev setup is `uv sync --group dev` then
`uv run pre-commit install`. Examples have their own dependency groups (`example-cmapss`,
`example-tracking`); install one with `uv sync --group <name>`.

## Verification gate

Before presenting changes for review, all three must pass:

- `uv run pre-commit run --all-files` (ruff check, ruff format, docformatter)
- `uv run pytest`
- A smoke test of the entry points under `examples/jax/` that your change could affect.

## Reference

| Topic | Document |
| --- | --- |
| Issues and specs live as GitHub issues in `GroupiSP/probreg`, driven by the `gh` CLI | [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md) |
| Glossary and architecture decisions: read before exploring an unfamiliar area | [`CONTEXT.md`](CONTEXT.md), [`docs/adr/`](docs/adr/), [`docs/agents/domain.md`](docs/agents/domain.md) |
| Planning flow, branches, draft PRs, commit messages | [`docs/agents/workflow.md`](docs/agents/workflow.md) |
| Code structure, typing, interfaces, docstrings | [`docs/agents/coding-style.md`](docs/agents/coding-style.md) |
| Test layout, fixtures, property-based testing | [`docs/agents/testing.md`](docs/agents/testing.md) |
| Human-facing contribution rules | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## Replies

Answer concisely. Reach for a table, diagram, code snippet or worked example when prose alone
would not carry the point — not by default.
