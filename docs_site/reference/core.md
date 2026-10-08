# `probreg.core`

Every symbol exported from `probreg.core`, the backend-neutral contracts and utilities,
grouped by the module that defines it.

## `probreg.core.checkpoints`

::: probreg.core.Checkpoint

::: probreg.core.CheckpointStore

::: probreg.core.InMemoryCheckpointStore

## `probreg.core.distributions`

::: probreg.core.PredictiveDistribution

::: probreg.core.DistributionHead

::: probreg.core.Likelihood

::: probreg.core.DistributionLoss

::: probreg.core.PredictionLoss

::: probreg.core.Loss

## `probreg.core.early_stopping`

::: probreg.core.OptimizationMode

::: probreg.core.EarlyStoppingState

::: probreg.core.EarlyStoppingDecision

::: probreg.core.EarlyStopper

## `probreg.core.losses`

::: probreg.core.NegativeLogLikelihoodLoss

::: probreg.core.SquaredErrorLoss

::: probreg.core.add_epsilon

::: probreg.core.GaussianNLLLoss

::: probreg.core.BetaNLLLoss

## `probreg.core.metric_registry`

::: probreg.core.EvaluationGrid

::: probreg.core.PredictionInterval

::: probreg.core.EpochPredictionData

::: probreg.core.MetricRequirements

::: probreg.core.EpochMetric

::: probreg.core.RootMeanSquaredError

::: probreg.core.NegativeLogLikelihood

::: probreg.core.IntervalCoverage

::: probreg.core.WeightedSpread

::: probreg.core.PointContinuousRankedProbabilityScore

::: probreg.core.ContinuousRankedProbabilityScore

## `probreg.core.metrics`

::: probreg.core.cdf

::: probreg.core.crps

::: probreg.core.point_crps

::: probreg.core.rmse

::: probreg.core.coverage

::: probreg.core.wsu

## `probreg.core.naming`

::: probreg.core.Split

::: probreg.core.MetricTag

::: probreg.core.metric_tag

::: probreg.core.parse_metric_tag

::: probreg.core.flatten_parameters

## `probreg.core.protocols`

::: probreg.core.Dataset

::: probreg.core.LoaderFactory

::: probreg.core.Optimizer

::: probreg.core.Step

::: probreg.core.ValidationStrategy

## `probreg.core.stages`

::: probreg.core.validate_transition

::: probreg.core.TrainingStage

## `probreg.core.tracking`

::: probreg.core.Decision

::: probreg.core.TrainingEvent

::: probreg.core.EventSink

::: probreg.core.ExperimentTracker

::: probreg.core.TrackerEventSink

## `probreg.core.types`

::: probreg.core.Array

::: probreg.core.PyTree

::: probreg.core.ParameterRole

::: probreg.core.StageState

::: probreg.core.Batch

::: probreg.core.CheckpointRef

::: probreg.core.TrainingState

::: probreg.core.StageResult

::: probreg.core.ValidationResult
