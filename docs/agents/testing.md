# Testing

Tests use `pytest` and live in `tests/`, in modules mirroring the structure of `src/probreg/`
(so `tests/jax/` mirrors the JAX backend, `tests/core/` the core).

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
