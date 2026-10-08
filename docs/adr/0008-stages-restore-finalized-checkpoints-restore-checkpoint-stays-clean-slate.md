# Stages restore finalized checkpoints; `restore_checkpoint` stays clean-slate

Restoring a checkpoint into a staged workflow is the job of the stage that wrote it: `MeanStage` and `GammaVarianceStage` each expose `restore(state, checkpoint)`, which accepts only the stage's finalized checkpoint. At the end of `train`, each stage restores its own best checkpoint, which is not finalized yet, through the same restore path minus the finalized-checkpoint check, and then saves it again under the same key as a finalized checkpoint. `restore_checkpoint` keeps its clean-slate semantics: it empties `model_components` and `optimizer_states` and registers only the model and optimizer it is given.

The problem this solves is the variance stage's checkpoint. It snapshots only the variance model and optimizer, because the mean model is frozen and its weights already live in the mean stage's finalized checkpoint. So restoring a variance checkpoint, whether at the end of training or in a later process after the mean checkpoint, must keep the mean model registered, and a clean-slate restore drops it. The variance stage knows `mean_model_name` and `mean_optimizer_name`, so it carries those registrations over the restore. `restore_checkpoint` would have to guess which surviving component an optimizer belongs to, and `TrainingState` does not record that.

A stage's `restore` accepts only a finalized checkpoint: the stage's ready lifecycle state, and metadata naming that stage and marking it complete. The variance stage also refuses when the mean model is not registered yet. Both checks run before anything is mutated, so a successful `restore` always leaves a state for which the stage's `validate` passes, and a best checkpoint left behind by a crashed run is rejected instead of being restored halfway.

## Considered Options

- **Make `restore_checkpoint` keep live registrations the checkpoint does not cover** (those named in the restored `parameter_roles` or `frozen_components`). Rejected: it changes a documented, general primitive for the sake of one workflow, and it needs a component-to-optimizer pairing that `TrainingState` does not have.
- **Document the manual recovery** (re-register the mean model and optimizer, then set `lifecycle_state` by hand). Rejected: every caller would have to rediscover it, and leaving out a step silently produces a state that cannot predict.
- **Restore leniently**: accept any checkpoint and force the ready lifecycle state. Rejected: it would treat a mid-training snapshot as a completed stage.

## Consequences

Resuming across stages reads `mean_stage.restore(...)` followed by `variance_stage.restore(...)`, with no registry names passed by hand, and the order is enforced. Code that restores a staged checkpoint with `restore_checkpoint` directly still gets the clean-slate behaviour, and the guides point it to the stage methods instead.
