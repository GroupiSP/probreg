# Posterior stage drives inference methods that yield mean-function draws

The posterior stage must host inference methods as different as variational inference, SG-MCMC, ensembles and MC-Dropout. We fixed the seam between the stage and a method in two places. First, a posterior is consumed only through draws of the mean function at given inputs, never through its parameters, so a variational distribution, a stack of retained samples, a set of ensemble members and a dropout network all look the same to prediction, and prediction (the posterior predictive mixture over draws under the fixed aleatoric variance) lives outside every method. Second, the stage owns the epoch loop and an inference method only steps (`init`, `update` on one batch, `posterior`), so training events, validation on the current posterior predictive, early stopping and checkpoints are written once in the stage rather than once per method.

## Considered Options

- **A posterior exposes parameter samples** and prediction applies the network to each. Rejected: MC-Dropout has no parameter samples, and it ties prediction to one network layout per posterior.
- **A posterior returns a predictive distribution directly.** Rejected: every method would reimplement the mixture, and a draw could not be shared across inputs, which a RUL curve needs.
- **Each method runs its own training loop** and the stage only calls it. Rejected: events, validation, early stopping and checkpointing would be duplicated, and drift, in every method.

## Consequences

A method that needs its own control flow (full-batch HMC) fits by taking a one-batch loader, with one epoch being one or more transitions. A method declares whether it supports early stopping; the stage refuses an early stopper for one that does not, and SG-MCMC writes only a finalized checkpoint.
