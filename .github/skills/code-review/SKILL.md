---
name: code-review
description: Review a pull request against Probreg's architecture and coding guidelines (staged training, type hints, Protocols, docstrings, testing).
---

# Code Review Skill

Apply this checklist when reviewing a PR diff to this repository. Focus on
high-confidence, actionable issues — do not comment on style already enforced
by `ruff`/`docformatter` (run via `pre-commit`).

## Architecture Checklist

- [ ] Changes respect the staged-training design (e.g. mean-predictor stage
      trained/frozen before the variance stage is introduced)
- [ ] Favours standalone functions and data container classes over a single
      monolithic class
- [ ] No god functions/classes — flag entities with too many responsibilities
- [ ] Composition over inheritance; inheritance depth is at most one level
- [ ] Cross-cutting behaviour is abstracted via `typing.Protocol`, not an
      abstract base class
- [ ] Note any spot where introducing a design pattern would meaningfully
      improve clarity or extensibility

## Code Quality Checklist

- [ ] All new/changed functions and classes have type hints
- [ ] No bare `except` clauses
- [ ] No mutable default arguments
- [ ] Context managers used for file/resource I/O
- [ ] Variable and function names follow PEP 8 (snake_case)

## Documentation Checklist

- [ ] Public functions/classes have Google-style docstrings with `Args`,
      `Returns`, and `Raises` sections
- [ ] Docstrings describe intent/behavior, not just restate the signature

## Testing Checklist

- [ ] New/changed code has corresponding `pytest` tests
- [ ] Tests live in a module mirroring the source structure under `tests/`
- [ ] Reusable fixtures are used (or added to `conftest.py`) instead of
      duplicated setup
- [ ] Mathematical properties (invariance, symmetry, monotonicity, etc.) are
      tested via property-based tests where applicable, rather than only
      example-based tests
- [ ] Edge cases are covered (empty inputs, `None`, out-of-range values)

## Validation Checklist

- [ ] `pre-commit run --all-files` passes
- [ ] `pytest` passes
- [ ] Any touched scripts under `scripts/` were smoke-tested (unless excluded
      by repo instructions)

## Output Format

Present findings as:

```
## Code Review: [PR title / diff scope]

### Architecture
- [PASS/FAIL] Description of finding (file:line)

### Code Quality
- [PASS/FAIL] Description of finding (file:line)

### Documentation
- [PASS/FAIL] Description of finding (file:line)

### Testing
- [PASS/FAIL] Description of finding (file:line)

### Validation
- [PASS/FAIL] pre-commit / pytest results

### Summary
[X] items need attention before merge, ranked by severity.
```
