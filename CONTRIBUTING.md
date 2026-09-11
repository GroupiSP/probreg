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
3. Open a **draft pull request** targeting the `main` branch as soon as you start the work, and
   link it to the issue you opened.
4. Implement your changes on that branch, pushing commits to the draft PR as you go.
5. Run the required checks locally:
   - `uv run pre-commit run --all-files`
   - `uv run pytest`
6. When the checks pass, mark the PR as ready and request review from one of the maintainers.
