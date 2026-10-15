# Changelog

All notable changes to `stagereg` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Breaking changes

### Added

- `PosteriorStage.inferred_subnetwork`, an NNX filter selecting the inferred
  subnetwork; the frozen remainder keeps its trained-mean values. The default,
  `nnx.Param`, infers the whole network as in 0.1.0 (#105).
- `stagereg.core.expected_sample_crps(predictive_samples, reference_samples)`: the
  exact, grid-free expected CRPS of each scoring unit's empirical predictive under
  its reference samples, batched over units.

### Changed

- `PosteriorProblem.initial_parameters`, `mean_function` and `prior` cover the
  inferred subnetwork only. `BayesByBackprop.load_posterior` and
  `PreconditionedSGLD.load_posterior` refuse a posterior whose tree differs
  from it, and `PreconditionedSGLD.load_posterior` now requires `init` (#105).

## [0.1.0]

First release under the `stagereg` name.
