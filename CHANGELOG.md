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
- `stagereg.core.expected_sample_crps(predictive_samples, reference_samples)`: the
  exact, grid-free expected CRPS of each scoring unit's empirical predictive under
  its reference samples, batched over units.

### Changed

- `MeanStage` and `GammaVarianceStage` with a checkpoint store but no early stopper
  now save their last epoch as their finalized checkpoint under the stage's
  checkpoint key, so the stage can be restored in a later process. Previously they
  saved nothing. With an early stopper the best epoch is still finalized
  (ADR 0011).

## [0.1.0]

First release under the `stagereg` name.
