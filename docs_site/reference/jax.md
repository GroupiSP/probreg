# `probreg.jax`

Every symbol exported from `probreg.jax`, the JAX/Flax NNX training backend, grouped by the
module that defines it. Using it needs the `jax` extra.

## `probreg.jax.distributions`

::: probreg.jax.Gaussian

::: probreg.jax.Gamma

::: probreg.jax.PosteriorPredictive

::: probreg.jax.MomentMatchedPredictive

::: probreg.jax.GaussianHead

::: probreg.jax.GammaHead

## `probreg.jax.evaluation`

::: probreg.jax.SupervisedLoss

::: probreg.jax.make_evaluation_step

::: probreg.jax.evaluate_loader

## `probreg.jax.losses`

::: probreg.jax.make_supervised_loss

## `probreg.jax.metrics`

::: probreg.jax.BatchMetric

::: probreg.jax.BatchMetricSpec

::: probreg.jax.PredictionRequirements

::: probreg.jax.CoordinateExtractor

::: probreg.jax.ReferenceSamplesExtractor

::: probreg.jax.Predictor

::: probreg.jax.GaussianPredictor

::: probreg.jax.PosteriorPredictivePredictor

::: probreg.jax.MetricSuite

::: probreg.jax.merge_epoch_prediction_data

## `probreg.jax.rng`

::: probreg.jax.split_key

## `probreg.jax.state`

::: probreg.jax.NnxSnapshot

::: probreg.jax.create_optimizer

::: probreg.jax.initialize_training_state

::: probreg.jax.freeze_training_state

::: probreg.jax.snapshot

::: probreg.jax.restore_checkpoint

## `probreg.jax.supervised`

::: probreg.jax.make_train_step

::: probreg.jax.run_supervised

## `probreg.jax.supervised_staged`

::: probreg.jax.materialize_residual_loader

::: probreg.jax.SupervisedStageOptions

::: probreg.jax.MeanStage

::: probreg.jax.GammaVarianceStage

## `probreg.jax.validation`

::: probreg.jax.HeldOutValidation
