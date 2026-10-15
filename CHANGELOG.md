# Changelog

All notable changes to `stagereg` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Breaking changes

- `TrainingState.active_stage` is now a dataclass field. The `TrainingState.stage`
  field, which the `active_stage` property used to wrap, is removed; pass and read
  `active_stage` instead.
- `Checkpoint.rng_state` is removed. A checkpoint's random key lives only in
  `Checkpoint.state.rng_state`, and restores read it from there.
- `TrainingState.outer_iteration` and `TrainingState.data_fingerprint` are removed.
  Nothing in the library populated them.
- `TrainingEvent.iteration` is removed. It always reported `0`, as no run has outer
  iterations yet.
- `InferenceMethod` requires a `diagnostics` property; a custom method
  declares `()` to keep recording only its loss (#106).

### Added

- `PosteriorProblem.aleatoric_variance`: the fixed, stop-gradient aleatoric
  variance a `PosteriorStage` uses for its likelihood, so an implicit method
  can turn draws into predictive samples. Optional for hand-built problems (#106).
- `InferenceMethod.diagnostics`: the keys of `update`'s output that
  `PosteriorStage` records as `posterior/train/<key>` epoch means. Undeclared
  keys are ignored; a key colliding with a recorded tag is refused at `prepare`.
  `BayesByBackprop` and `PreconditionedSGLD` declare `()` (#106).
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
