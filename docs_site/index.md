# probreg

`probreg` is a library for stage-oriented probabilistic regression: training
models that predict a distribution over the target, not just a point estimate.
Its core is a set of backend-neutral contracts for batches, predictive
distributions, losses, metrics, checkpoints, early stopping and tracking. An
optional JAX backend built on Flax NNX and Optax implements them.

## Why train in stages

Splitting training into separate stages fixes two problems that fitting
everything at once leaves open:

- **It avoids the pathologies of joint training.** Mean-variance estimation
  ([Nix & Weigend, 1994](https://doi.org/10.1109/ICNN.1994.374138)) fits a mean
  and a variance together under one Gaussian negative log-likelihood. That
  couples the two fits: the variance can grow to excuse a poor mean fit, and the
  mean stalls where the variance is large
  ([Detlefsen et al., 2019](https://doi.org/10.48550/arXiv.1906.03260)).
  Fitting the mean before the variance, as warm-up schemes also do
  ([Sluijterman et al., 2024](https://doi.org/10.1016/j.neucom.2024.127929)),
  removes the coupling.
- **It disentangles aleatoric and epistemic uncertainty.** Once its own stage
  has fixed the aleatoric variance, a later Bayesian stage can attribute the
  remaining spread of mean functions to epistemic variance, so neither absorbs
  the other ([Yi & Bessa, 2025](https://doi.org/10.48550/arXiv.2505.02743)).

## How probreg trains in stages

A run is a sequence of named stages: a mean stage, a variance stage, and an
optional posterior stage. Each stage checks that the lifecycle of the shared
training state allows it to run, trains and validates, and hands on a finalized
checkpoint the next stage resumes from. Every metric a run reports is tagged
with the stage that produced it. See
[two-step training](guides/two-step-training.md) for the mean and variance
stages and [the posterior stage](guides/posterior-stage.md) for the third.

## Install

The base package has only NumPy as a dependency and holds the backend-neutral
contracts:

```bash
uv add "probreg @ git+https://github.com/GroupiSP/probreg"
```

The `jax` extra adds the JAX backend (JAX, Flax and Optax):

```bash
uv add "probreg[jax] @ git+https://github.com/GroupiSP/probreg"
```

With pip, use `pip install "probreg[jax] @ git+https://github.com/GroupiSP/probreg"`.

## Minimal example

--8<-- "README.md:minimal-example"

## Contributing

Development setup, the checks a change must pass and how to build this site
are in
[`CONTRIBUTING.md`](https://github.com/GroupiSP/probreg/blob/main/CONTRIBUTING.md).
