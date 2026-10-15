# Stages without an early stopper finalize their last epoch

`MeanStage` and `GammaVarianceStage` with a checkpoint store but no early stopper now save their last epoch as their finalized checkpoint, under the stage's checkpoint key, with the stage's ready lifecycle state and the finalized metadata. With an early stopper nothing changes: the stage restores its best checkpoint and saves it again, finalized.

The problem this solves is restoring such a stage in a later process. Under [ADR 0008](0008-stages-restore-finalized-checkpoints-restore-checkpoint-stays-clean-slate.md) a stage restores only its finalized checkpoint, and a finalized checkpoint was always a best checkpoint saved again. Without an early stopper there is no best checkpoint, so the stage saved nothing even when it was given a store, and a later process could resume it only by configuring early stopping it did not want. A stage without an early stopper trains for all its epochs, so the last epoch is the model the stage leaves live and hands on; finalizing it keeps the checkpoint and the live state in agreement.

## What this amends in ADR 0008

ADR 0008 stands: a stage's `restore` accepts only its finalized checkpoint, and `restore_checkpoint` stays clean-slate. Only the source of the finalized checkpoint changes: it is the best checkpoint when an early stopper is configured, otherwise the last epoch's state.

## Considered Options

- **Keep saving nothing without an early stopper.** Rejected: a user who wants every epoch run, such as a fixed training budget, could not restore the stage later, and a persistent store would have nothing to persist.
- **Let `restore` accept a non-finalized checkpoint without an early stopper.** Rejected: there is still no checkpoint to accept, and it would weaken the finalized-checkpoint check that rejects snapshots of crashed runs.
- **Save a checkpoint on every epoch without an early stopper.** Rejected: it writes a checkpoint per epoch that nothing restores from, which a persistent store would pay for in I/O.

## Consequences

A store passed to a stage always ends up holding that stage's finalized checkpoint, with or without an early stopper, so the resume chain `mean_stage.restore(...)`, `variance_stage.restore(...)` works for both. The finalized checkpoint of a stage without an early stopper carries no early-stopping state. The posterior stage already finalized its last state without an early stopper and is unchanged.
