# Changelog

All notable changes to `stagereg` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Breaking changes

### Added

- `PosteriorProblem.aleatoric_variance`: the fixed, stop-gradient aleatoric
  variance a `PosteriorStage` uses for its likelihood, so an implicit method
  can turn draws into predictive samples. Optional for hand-built problems (#106).

### Changed

## [0.1.0]

First release under the `stagereg` name.
