# Coding style

## Structure

- Favour free functions and data-container classes over wrapping behaviour in one large class.
- Keep responsibilities narrow. When reviewing code, flag any function or class that has taken on
  too many.
- Favour composition over inheritance; at most one level of inheritance.
- Abstract behaviour behind `typing.Protocol` rather than abstract base classes.
- When reviewing, name a design pattern that would measurably improve the code, and say what it
  buys.

## Typing

Annotate every function signature, class attribute and dataclass field. `pyright` runs in `basic`
mode (see `pyproject.toml`), so unannotated code passes the checker while still being wrong here.

## Docstrings

- Google style, enforced in format by `docformatter` via pre-commit. Include `Args`, `Returns`
  and `Raises` sections wherever they apply.
- For dataclasses, document the attributes, except those declared `init=False`.
  `tests/test_dataclass_attributes.py` enforces this for exported dataclasses and NamedTuples.
- Cross-references use mkdocstrings autorefs syntax to the public path,
  `` [`EarlyStopper`][probreg.core.EarlyStopper] ``, never a Sphinx role (`:class:`,
  `:func:`, …), which renders as literal text and escapes the strict docs build
  ([ADR 0007](../adr/0007-user-docs-are-mkdocs-and-docstrings-use-autorefs-cross-references.md)).
  Link a module by its reference-page section anchor, `` [`probreg.jax.state`][probregjaxstate] ``.
  Double-backtick literals stay for values and non-symbol code.
