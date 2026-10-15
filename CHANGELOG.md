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

### Added

### Changed

## [0.1.0]

First release under the `stagereg` name.
