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
3. Implement your changes and run relevant checks locally.
4. Open a **draft pull request** targeting the `main` branch.
5. Link the pull request to the issue you opened and push your changes to that PR.
6. When the draft PR is ready for review, mark it as ready and request review from one of the maintainers.
