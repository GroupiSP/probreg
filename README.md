# probreg

`probreg` is a stage-oriented library for probabilistic regression.

The initial release establishes backend-neutral contracts for batches, datasets,
optimizers, staged training, predictive distributions, likelihoods, loss
functions, checkpoints, tracking, and numerical metrics. JAX/NNX training
adapters and probabilistic algorithms are intentionally separate future layers.

## Development

Install the development dependencies and run the core tests:

```bash
uv sync --group dev
uv run pytest tests/core -v
uv run ruff check src tests
```