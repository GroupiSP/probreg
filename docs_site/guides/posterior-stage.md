# Posterior stage

The [`PosteriorStage`][probreg.jax.PosteriorStage] is an optional third stage
after [two-step mean/variance training](two-step-training.md). It replaces the
single trained mean function with a posterior over mean functions, starting
from the trained mean and holding the aleatoric variance fixed, so predictions
also carry the epistemic variance: the uncertainty about the mean that more
data would reduce.

## When to add it

When a run that stops after the variance stage is complete, its predictive is a Gaussian with the trained mean and the aleatoric variance, which is all you need when the inputs you predict at look like the training inputs.

The posterior stage allows to quantify the epistemic uncertainty of the model predictions. This is useful in at least two situations:

1. When you want to know the uncertainty of the model given the data, which is important for decision making and risk assessment.
2. When you seek the uncertainty in regions where the data is scarse or absent, such as extrapolation or rare operating conditions. In these cases, the trained mean is only one of many mean functions that fit the data equally well, and the aleatoric variance alone understates the uncertainty.

## Terms

The stage and its inference methods share a small vocabulary, defined in the
[glossary](../glossary.md):

| Term | Meaning |
| --- | --- |
| Inference method | The interchangeable strategy that turns the stage's posterior problem into a posterior, such as [`BayesByBackprop`][probreg.jax.BayesByBackprop] or [`PreconditionedSGLD`][probreg.jax.PreconditionedSGLD]. The stage decides what is inferred; the method decides how. |
| Posterior | What the inference method produces: an approximate posterior over mean functions, from which draws can be taken. |
| Draw | One mean function taken from the posterior, the same function at every input. A finite posterior, such as pSGLD's retained samples, always uses all its draws. |
| Posterior predictive | The [`PosteriorPredictive`][probreg.jax.PosteriorPredictive]: the equally weighted mixture, over the draws, of Gaussians centred on each draw's mean with the aleatoric variance. |
| Moment-matched predictive | The [`MomentMatchedPredictive`][probreg.jax.MomentMatchedPredictive]: the single Gaussian with the posterior predictive's mean and variance, the aleatoric plus the epistemic variance. A summary, not a substitute. |

A draw is a mean function, not a target value; a value drawn from a predictive
distribution, as the CRPS uses, is a predictive sample.

## A three-stage run

The run below trains a linear mean model and a
[`GammaHead`][probreg.jax.GammaHead] variance model as in
[the two-step guide](two-step-training.md), then a posterior stage with Bayes by
Backprop. The stage needs the number of training examples, `dataset_size`, to
scale each batch's log-likelihood to the whole data set. It validates every
epoch on the current posterior predictive and reports the NLL as
`posterior/validation/nll`.

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from probreg.core import Batch, ParameterRole, StageState, TrainingState
from probreg.jax import (
    BayesByBackprop,
    GammaHead,
    GammaVarianceStage,
    MeanStage,
    PosteriorPredictive,
    PosteriorStage,
    PosteriorStageOptions,
    SupervisedStageOptions,
    create_optimizer,
)

inputs = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
targets = 2.0 * inputs + 0.1 * jax.random.normal(jax.random.key(0), inputs.shape)


def loader(*, split: str, epoch: int) -> list[Batch]:
    return [Batch(inputs=inputs, targets=targets)]


state = TrainingState(rng_state=jax.random.key(0))

mean_model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
mean_stage = MeanStage(
    model=mean_model,
    optimizer=create_optimizer(mean_model, optax.adam(0.1)),
    train_loader=loader,
    options=SupervisedStageOptions(epochs=3),
)
mean_stage.prepare(state)
mean_stage.train(state)

variance_model = GammaHead(1, 1, rngs=nnx.Rngs(1))
variance_stage = GammaVarianceStage(
    model=variance_model,
    optimizer=create_optimizer(variance_model, optax.adam(0.1)),
    source_loader=loader,
    options=SupervisedStageOptions(epochs=3),
)
variance_stage.prepare(state)
variance_stage.train(state)

# Step 3: a posterior over mean functions, warm-started from the trained mean.
posterior_stage = PosteriorStage(
    inference_method=BayesByBackprop(optimizer=optax.adam(0.01)),
    train_loader=loader,
    dataset_size=8,
    options=PosteriorStageOptions(epochs=3, num_draws=16, validation_loader=loader),
)
posterior_stage.prepare(state)
posterior_stage.train(state)

assert state.lifecycle_state is StageState.POSTERIOR_READY
assert state.parameter_roles["posterior"] is ParameterRole.POSTERIOR
assert state.frozen_components == {"mean_model", "variance_model"}
assert len(state.metric_history["posterior/validation/nll"]) == 3
assert len(state.metric_history["posterior/validation/crps"]) == 3
```

The mean network is copied for the posterior network, so `mean_model` itself is
untouched. To use a different module with the same parameter tree, pass it as
`model`; a copy of it is warm-started, so it is untouched as well. The prior defaults to an
[`IsotropicGaussianPrior`][probreg.jax.IsotropicGaussianPrior] with precision 1;
pass `prior` to change it.

Validation reports two metrics by default: `nll`, the exact negative
log-likelihood of the posterior predictive, and `crps`, the CRPS of 100
predictive samples per target, scored by
[`SampleContinuousRankedProbabilityScore`][probreg.core.SampleContinuousRankedProbabilityScore]
without an evaluation grid. To score other metrics, give the options a
`validation_metrics` suite with a
[`PosteriorPredictivePredictor`][probreg.jax.PosteriorPredictivePredictor];
[Epoch metrics](epoch-metrics.md) explains how.

## Predicting with the posterior

The trained posterior is registered as `state.model_components["posterior"]`.
It is consumed only through draws: `sample_means` returns each draw's mean at
the inputs, shaped `[S, *batch]`. Combined with the frozen variance model's
aleatoric variance, they form the posterior predictive, continuing the run
above:

```{.python continuation}
posterior = state.model_components["posterior"]
draws = posterior.sample_means(inputs, jax.random.key(1), 64)
assert draws.shape == (64, 8, 1)

predictive = PosteriorPredictive(
    draws=draws, aleatoric_variance=variance_model(inputs).mean()
)
nll = -predictive.log_prob(targets).mean()
summary = predictive.moment_matched()
assert bool(jnp.all(summary.epistemic_variance >= 0.0))
```

An unlimited posterior, such as Bayes by Backprop's, needs the number of draws;
a finite one, such as pSGLD's, refuses it and always uses all its draws, so call
`sample_means(inputs, key)` without it. Score with the posterior predictive
itself: it is exact for the draws taken and may be skewed or multimodal. The
moment-matched predictive is for reporting the variance split or drawing a
band.

## Restoring three stages

The posterior stage's finalized checkpoint holds the posterior alone, not the
mean and variance weights, so a later process restores the three stages in
order: mean, then variance, then posterior. Each stage's `restore` refuses,
before changing anything, while the stages it depends on are missing.
[Checkpoints](checkpoints.md#the-posterior-stage) shows the calls and what each
checkpoint holds.

## Bayes by Backprop or pSGLD

| | [`BayesByBackprop`][probreg.jax.BayesByBackprop] | [`PreconditionedSGLD`][probreg.jax.PreconditionedSGLD] |
| --- | --- | --- |
| Kind | Mean-field Gaussian variational inference | SG-MCMC: a preconditioned Langevin chain |
| Posterior | Unlimited: takes `num_draws` draws per prediction | Finite: the retained samples, all used |
| Early stopping | Supported, with a best checkpoint | Refused; only a finalized checkpoint |
| Cost of a prediction | `num_draws` forward passes, chosen freely | One forward pass per retained sample |
| Checkpoint size | Twice the network's parameters | The network's parameters times the retained samples |
| Approximation | Independent Gaussians ignore correlations between parameters, so the spread of the draws can have the wrong shape | The retained samples approach the posterior as the chain runs, if the step size is small enough |
| Tuning | Learning rate and initial standard deviation | Step size, burn-in and thinning; too large a step inflates the epistemic variance everywhere |

Start with Bayes by Backprop when you want early stopping, small checkpoints
and cheap predictions. Choose pSGLD when the shape of the epistemic variance
matters, as it does for extrapolation, and you can afford tuning the step size
and storing samples. In the XSin example below, pSGLD's epistemic variance
grows away from the training data, while Bayes by Backprop's does not.

pSGLD counts `burn_in` and `thinning` in update steps, that is training
batches. It has no posterior before its first retained sample, so the stage
skips validation for the epochs that end before then: they record no
validation metrics. The chain starts from the trained mean, which already fits
the data, so a short burn-in is usually enough:

```python
from probreg.jax import PreconditionedSGLD

# Eight batches per epoch: retain the position at the end of every epoch.
method = PreconditionedSGLD(step_size=3e-3, burn_in=0, thinning=8)
```

## The XSin example

`examples/jax/xsin/posterior.py` runs mean, variance and posterior stages on the
XSin benchmark with each method, prints the NLL and CRPS of the variance stage
and of the posterior predictive, and plots the moment-matched predictive band
against the variance stage's band. [Examples](examples.md#xsin-benchmark) lists
the run command.
