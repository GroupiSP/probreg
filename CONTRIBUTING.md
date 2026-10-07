# Contributing to probreg

Thanks for your interest in contributing to `probreg`.

## Opening issues

- Open issues using one of the templates in GitHub:
  - **Bug report**
  - **Feature request**
- Choose the template that best matches your case and fill in all relevant fields.
- If you plan to implement the change, tick **"I would like to work on it"** in the issue.

## Contributing to the codebase

1. Open an issue first, and explicitly state that you would like to work on it.
2. Set up your development environment with `uv`, then install pre-commit hooks:
   - `uv sync --group dev`
   - `uv run pre-commit install`
   - `uv sync --extra jax --group dev --group docs`, to build the documentation site
3. Open a **draft pull request** targeting the `main` branch as soon as you start the work, and
   link it to the issue you opened.
4. Implement your changes on that branch, pushing commits to the draft PR as you go.
5. Run the required checks locally:
   - `uv run pre-commit run --all-files`
   - `uv run pytest`
   - `uv run --extra jax --group docs mkdocs build --strict`
6. When the checks pass, mark the PR as ready and request review from one of the maintainers.

## Documentation

The user documentation site is built with MkDocs Material from `docs_site/`; `docs/` holds
design records and is not published. Preview it with
`uv run --extra jax --group docs mkdocs serve`.

Every fenced `python` block in `docs_site/` and `README.md` runs as a test under `uv run pytest`,
so write snippets that stand alone and finish in about a second. For a fragment that cannot run on
its own (a dataset download, a full training run), mark the fence as non-executable with the
superfences brace form, and link to the runnable example:

````markdown
```{.python notest}
result = run_cmapss_example()
```
````

The form is `{.python notest}`, not `python notest`: Material's superfences does not recognise a
bare word after the language and renders the block as plain text, while pytest is configured with
`--markdown-docs-syntax=superfences` and skips brace-form fences carrying `notest`.

The API reference in `docs_site/reference/` has one page per package and one `## module`
section per source module. Exporting a symbol from `probreg.core` or `probreg.jax` means adding
`::: probreg.core.Name` (its public path) under the section of the module that defines it;
`tests/test_reference_completeness.py` fails until you do, and fails on an entry that names
something not exported.
