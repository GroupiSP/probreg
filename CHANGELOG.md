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

### Added

- `stagereg.core.expected_sample_crps(predictive_samples, reference_samples)`: the
  exact, grid-free expected CRPS of each scoring unit's empirical predictive under
  its reference samples, batched over units.
- `stagereg.core.CachedCheckpointStore(persistent, persist_when=finalized_only)`: a
  checkpoint store that caches every save in memory and forwards to `persistent`
  only the saves its predicate selects, reading the cache first and falling back to
  `persistent`. Ready-made predicates `finalized_only`, `every_nth_save(n)` and
  `always`, typed as `PersistencePredicate`.
- `stagereg.core.STAGE_COMPLETE_METADATA_KEY` and `stagereg.core.is_finalized`: the
  checkpoint-metadata marker of a finalized checkpoint, now public. The stages write
  and check it as before.

### Changed

## [0.1.0]

First release under the `stagereg` name.
