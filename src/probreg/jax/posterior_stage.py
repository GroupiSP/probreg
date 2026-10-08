"""The posterior stage: VeBNN's Step 3 after the mean and variance stages."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from flax import nnx

from probreg.core.checkpoints import Checkpoint, CheckpointStore
from probreg.core.early_stopping import EarlyStopper
from probreg.core.metric_registry import EpochPredictionData, NegativeLogLikelihood
from probreg.core.naming import Split, metric_tag
from probreg.core.protocols import LoaderFactory, ValidationStrategy
from probreg.core.tracking import EventSink
from probreg.core.types import (
    Batch,
    CheckpointRef,
    ParameterRole,
    PyTree,
    StageResult,
    StageState,
    TrainingState,
    ValidationResult,
)
from probreg.jax.distributions import PosteriorPredictive
from probreg.jax.epoch_loop import (
    StateSnapshot,
    check_epoch_loop_arguments,
    resolve_checkpoint_key,
    run_epoch_loop,
)
from probreg.jax.metrics import (
    MetricSuite,
    PosteriorPredictivePredictor,
    metric_key,
    reduce_metric_suite,
)
from probreg.jax.posterior import (
    InferenceMethod,
    IsotropicGaussianPrior,
    Posterior,
    PosteriorProblem,
    Prior,
)
from probreg.jax.rng import split_key
from probreg.jax.state import _restore_training_state, freeze_training_state
from probreg.jax.supervised_staged import (
    _STAGE_COMPLETE_METADATA_KEY,
    _STAGE_METADATA_KEY,
    _latest_training_metrics,
    _require_component_names,
    _require_finalized,
)


def _default_validation_metrics() -> MetricSuite:
    """Score the exact posterior predictive's NLL."""
    return MetricSuite(
        epoch=(NegativeLogLikelihood(),),
        predictor=PosteriorPredictivePredictor(),
    )


@dataclass(frozen=True)
class PosteriorStageOptions:
    """Epoch-loop and prediction options of a posterior stage.

    Attributes:
        epochs: Maximum number of training epochs.
        num_draws: The number of draws ``S`` taken from an unlimited posterior
            (e.g. a variational distribution) to form the posterior predictive.
            A finite posterior always uses all its draws and ignores it.
            Defaults to ``32``.
        validation_loader: Factory producing the ``"validation"`` batches the
            posterior predictive is scored on after each epoch. Defaults to
            ``None``, which disables validation.
        validation_metrics: The epoch metrics the posterior predictive is
            scored with, reported under their usual names (``nll``, ``crps``,
            ...). Its predictor receives a model mapping inputs to a
            [`PosteriorPredictive`][probreg.jax.PosteriorPredictive], so it is
            normally a
            [`PosteriorPredictivePredictor`][probreg.jax.PosteriorPredictivePredictor].
            Defaults to the NLL alone; add, for example,
            ``PointContinuousRankedProbabilityScore(name="crps")`` with a
            predictive sample count and an evaluation grid to report the CRPS.
        early_stopper: Optional early-stopping policy. Refused for an inference
            method that does not support early stopping.
        event_sinks: Event consumers notified of ``posterior/...`` events.
        checkpoint_store: Optional store for the best checkpoint, which holds
            the inference method's full state, and for the finalized
            checkpoint that replaces it at the end of training and holds the
            posterior alone.
        checkpoint_key: The key of the best and finalized checkpoints.
            Defaults to ``None``, which resolves to ``posterior/best``.
    """

    epochs: int
    num_draws: int = 32
    validation_loader: LoaderFactory | None = None
    validation_metrics: MetricSuite = field(default_factory=_default_validation_metrics)
    early_stopper: EarlyStopper | None = None
    event_sinks: Sequence[EventSink] = ()
    checkpoint_store: CheckpointStore | None = None
    checkpoint_key: str | None = None

    def __post_init__(self) -> None:
        """Validate the draw count and the validation metrics.

        Raises:
            ValueError: If ``num_draws`` is not positive, or if
                ``validation_metrics`` registers batch metrics, which need a
                trained NNX model rather than a posterior.
        """
        if self.num_draws <= 0:
            raise ValueError("num_draws must be positive.")
        if self.validation_metrics.batch:
            raise ValueError(
                "posterior validation scores epoch metrics only; "
                "batch metrics are not supported."
            )


@dataclass
class PosteriorStage:
    """Optional Step 3 stage inferring a posterior over mean functions.

    It follows the variance stage. The posterior network has the mean
    network's parameter tree and is warm-started from a copy of the trained
    mean weights; the mean and variance models stay frozen, and the aleatoric
    variance is computed on the fly as the mean of the variance model's
    prediction. The stage hands its inference method a
    [`PosteriorProblem`][probreg.jax.PosteriorProblem], drives the epoch loop
    with the method's ``update``, and validates every epoch on the current
    posterior predictive.

    Attributes:
        inference_method: The [`InferenceMethod`][probreg.jax.InferenceMethod]
            deciding how the posterior is inferred.
        train_loader: Factory producing the original regression batches.
        dataset_size: The number of training examples ``N``, which scales each
            batch's log-likelihood by ``N / B``.
        options: Epoch-loop and prediction options.
        prior: The prior over the posterior network's parameters. Defaults to
            [`IsotropicGaussianPrior`][probreg.jax.IsotropicGaussianPrior] with
            precision 1.
        model: The posterior network, whose parameters are overwritten by the
            warm start. Its parameter tree must equal the mean network's, but
            its module may differ (e.g. add dropout). Defaults to ``None``,
            which uses a copy of the mean network.
        mean_model_name: Registry name of the trained mean model.
        variance_model_name: Registry name of the trained variance model.
        model_name: Registry name the posterior is registered under.
    """

    inference_method: InferenceMethod
    train_loader: LoaderFactory
    dataset_size: int
    options: PosteriorStageOptions
    prior: Prior = field(default_factory=IsotropicGaussianPrior)
    model: nnx.Module | None = None
    mean_model_name: str = "mean_model"
    variance_model_name: str = "variance_model"
    model_name: str = "posterior"
    name: str = field(default="posterior", init=False)
    requires: frozenset[str] = field(
        default_factory=lambda: frozenset({"mean", "variance"}),
        init=False,
    )
    produces: frozenset[str] = field(
        default_factory=lambda: frozenset({"posterior"}),
        init=False,
    )
    _validation: ValidationStrategy | None = field(default=None, init=False, repr=False)
    _prepared: bool = field(default=False, init=False, repr=False)
    _posterior: Posterior | None = field(default=None, init=False, repr=False)

    def prepare(self, state: TrainingState) -> None:
        """Warm-start the posterior network and initialize the inference method.

        Every check runs before ``state`` changes. Afterwards the mean and
        variance models are frozen components.

        Args:
            state: Shared state whose mean and variance stages are ready.

        Raises:
            ValueError: If the lifecycle state is not ``VARIANCE_READY``, if
                the mean or variance model is not registered with its role, if
                ``model_name`` is taken by another component, if the posterior
                network's parameter tree differs from the mean network's, if
                ``dataset_size`` is not positive, if an early stopper is
                configured for a method that does not support early stopping,
                or if the epoch-loop configuration is refused.
            TypeError: If the mean or variance model is not an NNX module, or
                ``state.rng_state`` is not a JAX random key.
        """
        if state.lifecycle_state is not StageState.VARIANCE_READY:
            raise ValueError("posterior stage requires VARIANCE_READY lifecycle state.")
        mean_model = _registered_module(
            state, self.mean_model_name, ParameterRole.MEAN, kind="mean"
        )
        variance_model = _registered_module(
            state, self.variance_model_name, ParameterRole.VARIANCE, kind="variance"
        )
        if self.model_name in state.model_components:
            raise ValueError(
                f"model component {self.model_name!r} is already registered."
            )
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        options = self.options
        if (
            options.early_stopper is not None
            and not self.inference_method.supports_early_stopping
        ):
            raise ValueError(
                "the inference method does not support early stopping; "
                "remove the early stopper."
            )
        validation = (
            None
            if options.validation_loader is None
            else _PosteriorPredictiveValidation(
                inference_method=self.inference_method,
                aleatoric_variance=_aleatoric_variance(variance_model),
                loader=options.validation_loader,
                metrics=options.validation_metrics,
                num_draws=options.num_draws,
            )
        )
        check_epoch_loop_arguments(
            state=state,
            epochs=options.epochs,
            stage=self.name,
            validation=validation,
            early_stopper=options.early_stopper,
        )
        self.inference_method.init(self._problem(mean_model, variance_model))
        state.frozen_components = state.frozen_components | {
            self.mean_model_name,
            self.variance_model_name,
        }
        state.active_stage = self.name
        self._validation = validation
        self._prepared = True
        self._posterior = None

    def train(self, state: TrainingState) -> StageResult:
        """Infer the posterior, register it and transition to ``POSTERIOR_READY``.

        With an early stopper and a checkpoint store, every improvement saves a
        best checkpoint whose ``parameters`` hold the inference method's full
        [`state`][probreg.jax.InferenceMethod.state]. After the last epoch the
        method resumes from the best checkpoint, if one was saved, and
        ``state`` returns to the best epoch. The method's
        [`posterior_state`][probreg.jax.InferenceMethod.posterior_state] is
        then stored as ``state.posterior_state``, and, with a checkpoint store,
        saved as the finalized checkpoint under the checkpoint key: lifecycle
        state ``POSTERIOR_READY``, metadata ``{"stage": "posterior",
        "stage_complete": True}``, and no parameters, so neither the mean and
        variance weights nor the method's optimizer state are duplicated.

        Args:
            state: State the stage has been prepared on.

        Returns:
            The epoch loop's result: the training metrics of the best epoch
            when a best checkpoint was restored, otherwise of the last epoch.

        Raises:
            ValueError: If the stage was not prepared or training produced a
                non-finite final loss.
        """
        if state.lifecycle_state is not StageState.VARIANCE_READY or not self._prepared:
            raise ValueError("posterior stage must be prepared before training.")
        method = self.inference_method
        options = self.options
        result = run_epoch_loop(
            step=method.update,
            snapshot_state=lambda: StateSnapshot(parameters=method.state()),
            train_loader=self.train_loader,
            state=state,
            epochs=options.epochs,
            stage=self.name,
            checkpoint_key=options.checkpoint_key,
            validation=self._validation,
            early_stopper=options.early_stopper,
            event_sinks=options.event_sinks,
            checkpoint_store=options.checkpoint_store,
        )
        if result.loss is None or not math.isfinite(result.loss):
            raise ValueError(f"{self.name} stage produced a non-finite final loss.")
        store = options.checkpoint_store
        key = resolve_checkpoint_key(options.checkpoint_key, self.name)
        epoch = (
            len(state.metric_history[metric_tag(self.name, Split.TRAIN, "loss")]) - 1
        )
        early_stopping_state = None
        if (
            options.early_stopper is not None
            and store is not None
            and store.exists(key)
        ):
            best = store.load(key)
            method.load_state(best.parameters)
            _restore_keeping_components(state, best, drop=())
            epoch = best.epoch
            early_stopping_state = best.early_stopping_state
            metrics = _latest_training_metrics(state, self.name)
            result = StageResult(state=state, metrics=metrics, loss=metrics["loss"])
        self._register_posterior(state, method.posterior_state())
        if store is not None:
            store.save(
                key,
                Checkpoint(
                    state=freeze_training_state(state),
                    epoch=epoch,
                    rng_state=state.rng_state,
                    early_stopping_state=early_stopping_state,
                    metadata={
                        _STAGE_METADATA_KEY: self.name,
                        _STAGE_COMPLETE_METADATA_KEY: True,
                    },
                ),
            )
        self._prepared = False
        return result

    def restore(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore the posterior stage's finalized checkpoint into ``state``.

        Call it after the mean stage's
        [`restore`][probreg.jax.MeanStage.restore] and the variance stage's
        [`restore`][probreg.jax.GammaVarianceStage.restore]. The inference
        method is initialized on the posterior problem of the registered mean
        and variance models, then rebuilds the saved posterior with
        [`load_posterior`][probreg.jax.InferenceMethod.load_posterior]; the
        posterior is registered under ``model_name``. Every model component
        and optimizer registered before, such as the mean and variance models,
        stays registered. Afterwards the state passes
        [`validate`][probreg.jax.PosteriorStage.validate].

        Args:
            state: Training state the mean and variance stages have been
                restored into.
            checkpoint: The posterior stage's finalized checkpoint.

        Raises:
            ValueError: If ``checkpoint`` is not finalized by this stage, that
                is its lifecycle state is not ``POSTERIOR_READY``, its metadata
                lacks ``"stage": "posterior"`` and ``"stage_complete": True``,
                or it holds no posterior state; if it was saved with another
                ``model_name``, ``mean_model_name`` or ``variance_model_name``;
                if the mean or variance model is not registered with its role;
                if ``dataset_size`` is not positive; or if the posterior
                network's parameter tree differs from the mean network's. All
                are checked before ``state`` changes.
            TypeError: If the mean or variance model is not an NNX module, or
                the checkpoint holds no JAX random key.
        """
        _require_finalized(
            checkpoint, stage=self.name, ready=StageState.POSTERIOR_READY
        )
        for frozen in (self.mean_model_name, self.variance_model_name):
            _require_component_names(
                checkpoint,
                model_name=self.model_name,
                role=ParameterRole.POSTERIOR,
                frozen=frozen,
            )
        posterior_state = checkpoint.state.posterior_state
        if posterior_state is None:
            raise ValueError("checkpoint holds no posterior state.")
        if not isinstance(checkpoint.rng_state, jax.Array):
            raise TypeError("checkpoint.rng_state must be a JAX random key.")
        mean_model = _registered_module(
            state, self.mean_model_name, ParameterRole.MEAN, kind="mean"
        )
        variance_model = _registered_module(
            state, self.variance_model_name, ParameterRole.VARIANCE, kind="variance"
        )
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        problem = self._problem(mean_model, variance_model)

        self.inference_method.init(problem)
        self.inference_method.load_posterior(posterior_state)
        _restore_keeping_components(state, checkpoint, drop={self.model_name})
        self._register_posterior(state, posterior_state)
        self._validation = None
        self._prepared = False

    def _problem(
        self, mean_model: nnx.Module, variance_model: nnx.Module
    ) -> PosteriorProblem:
        """Warm-start the posterior network and build the posterior problem.

        Raises:
            ValueError: If the posterior network's parameter tree differs from
                the mean network's, before the network changes.
        """
        network = self.model if self.model is not None else nnx.clone(mean_model)
        mean_parameters = nnx.state(mean_model, nnx.Param)
        _require_same_parameter_tree(nnx.state(network, nnx.Param), mean_parameters)

        nnx.update(network, jax.tree.map(jnp.copy, mean_parameters))
        mean_function = _mean_function(network)
        return PosteriorProblem(
            initial_parameters=jax.tree.map(jnp.copy, nnx.state(network, nnx.Param)),
            mean_function=mean_function,
            log_likelihood=_scaled_log_likelihood(
                mean_function,
                _aleatoric_variance(variance_model),
                dataset_size=self.dataset_size,
            ),
            prior=self.prior,
            train_loader=self.train_loader,
            dataset_size=self.dataset_size,
        )

    def _register_posterior(
        self, state: TrainingState, posterior_state: PyTree
    ) -> None:
        """Register the method's posterior and mark the state ``POSTERIOR_READY``."""
        posterior = self.inference_method.posterior()
        state.register_component(self.model_name, posterior)
        state.parameter_roles[self.model_name] = ParameterRole.POSTERIOR
        state.posterior_state = posterior_state
        state.lifecycle_state = StageState.POSTERIOR_READY
        state.active_stage = self.name
        self._posterior = posterior

    def validate(self, state: TrainingState) -> ValidationResult:
        """Validate posterior-stage lifecycle, ownership and freezing.

        Args:
            state: Shared staged training state.

        Returns:
            A validation result describing whether Step 3 is ready.
        """
        passed = (
            state.lifecycle_state is StageState.POSTERIOR_READY
            and self._posterior is not None
            and state.model_components.get(self.model_name) is self._posterior
            and state.parameter_roles.get(self.model_name) is ParameterRole.POSTERIOR
            and state.parameter_roles.get(self.mean_model_name) is ParameterRole.MEAN
            and state.parameter_roles.get(self.variance_model_name)
            is ParameterRole.VARIANCE
            and self.mean_model_name in state.model_components
            and self.variance_model_name in state.model_components
            and {self.mean_model_name, self.variance_model_name}
            <= state.frozen_components
        )
        return ValidationResult(
            passed=passed,
            message=None if passed else "posterior stage invariants are not satisfied.",
        )

    def select_checkpoint(self, state: TrainingState) -> CheckpointRef:
        """Return a reference to the posterior stage's best checkpoint.

        Args:
            state: Shared staged training state.

        Returns:
            Reference to the configured ``checkpoint_key``, or
            ``posterior/best`` when none was configured.

        Raises:
            ValueError: If no checkpoint exists under that key.
        """
        del state
        store = self.options.checkpoint_store
        key = resolve_checkpoint_key(self.options.checkpoint_key, self.name)
        if store is None or not store.exists(key):
            raise ValueError(f"checkpoint {key!r} is not available.")
        return CheckpointRef(key=key, metadata={"stage": self.name})


class _PosteriorPredictiveModel(nnx.Module):
    """The posterior predictive at inputs, for a fixed draw key.

    Every call with the same instance takes the same draws, so draw ``s`` is
    the same mean function in every batch of an epoch.
    """

    def __init__(
        self,
        posterior: Posterior,
        aleatoric_variance: Callable[[PyTree], jax.Array],
        draw_key: jax.Array,
        num_samples: int | None,
    ) -> None:
        self.posterior = nnx.static(posterior)
        self.aleatoric_variance = nnx.static(aleatoric_variance)
        self.draw_key = draw_key
        self.num_samples = num_samples

    def __call__(self, inputs: PyTree) -> PosteriorPredictive:
        return PosteriorPredictive(
            draws=self.posterior.sample_means(inputs, self.draw_key, self.num_samples),
            aleatoric_variance=self.aleatoric_variance(inputs),
        )


@dataclass(frozen=True)
class _PosteriorPredictiveValidation:
    """Score the inference method's current posterior predictive.

    Each epoch draws one key for the posterior's draws, then one key per
    validation batch for the metrics, both from ``state.rng_state``.
    """

    inference_method: InferenceMethod
    aleatoric_variance: Callable[[PyTree], jax.Array]
    loader: LoaderFactory
    metrics: MetricSuite
    num_draws: int

    def __call__(self, state: TrainingState, *, epoch: int) -> ValidationResult:
        posterior = self.inference_method.posterior()
        state.rng_state, draw_key = split_key(state.rng_state)
        model = _PosteriorPredictiveModel(
            posterior,
            self.aleatoric_variance,
            draw_key,
            None if posterior.num_draws is not None else self.num_draws,
        )
        predictor = self.metrics.predictor
        parts: list[EpochPredictionData] = []
        for batch in self.loader(split="validation", epoch=epoch):
            state.rng_state, batch_key = split_key(state.rng_state)
            if predictor is not None:
                parts.append(
                    predictor(
                        model,
                        batch,
                        self.metrics.prediction_requirements,
                        metric_key(batch_key),
                    )
                )
        metrics = reduce_metric_suite(
            suite=self.metrics,
            losses=None,
            batch_metric_values={},
            epoch_metric_parts=parts if self.metrics.epoch else None,
        )
        return ValidationResult(passed=True, metrics=metrics, message=None)


def _restore_keeping_components(
    state: TrainingState, checkpoint: Checkpoint, *, drop: Collection[str]
) -> None:
    """Restore a checkpoint's saved state fields, keeping the live registrations.

    A posterior-stage checkpoint holds no mean or variance weights, so every
    registered model component and optimizer except those in ``drop`` is
    carried across the restore.
    """
    components = {
        name: component
        for name, component in state.model_components.items()
        if name not in drop
    }
    optimizers = dict(state.optimizer_states)
    _restore_training_state(state, checkpoint.state)
    state.rng_state = checkpoint.rng_state
    for name, component in components.items():
        state.register_component(name, component)
    for name, optimizer in optimizers.items():
        state.register_optimizer(name, optimizer)


def _registered_module(
    state: TrainingState, name: str, role: ParameterRole, *, kind: str
) -> nnx.Module:
    """Return a registered NNX component after checking its role."""
    if name not in state.model_components:
        raise ValueError(f"{kind} model component {name!r} is not registered.")
    module = state.model_components[name]
    if not isinstance(module, nnx.Module):
        raise TypeError(f"registered {kind} model must be an NNX module.")
    if state.parameter_roles.get(name) is not role:
        raise ValueError(
            f"registered {kind} model must have the {role.name} parameter role."
        )
    return module


def _require_same_parameter_tree(candidate: PyTree, reference: PyTree) -> None:
    """Refuse a posterior network whose parameters differ from the mean's."""
    same = jax.tree.structure(candidate) == jax.tree.structure(reference) and all(
        jnp.shape(left) == jnp.shape(right)
        for left, right in zip(
            jax.tree.leaves(candidate), jax.tree.leaves(reference), strict=True
        )
    )
    if not same:
        raise ValueError(
            "the posterior network must have the mean network's parameter tree."
        )


def _mean_function(network: nnx.Module) -> Callable[[PyTree, PyTree], jax.Array]:
    """Return the posterior network as a pure function of its parameters."""
    graph, _, rest = nnx.split(network, nnx.Param, ...)

    def mean_function(parameters: PyTree, inputs: PyTree) -> jax.Array:
        return nnx.merge(graph, parameters, rest)(inputs)

    return mean_function


def _aleatoric_variance(
    variance_model: nnx.Module,
) -> Callable[[PyTree], jax.Array]:
    """Return the frozen aleatoric variance, the variance prediction's mean.

    The variance model is copied in inference mode, so neither training nor later
    changes to the live model affect it, and it carries no gradient.
    """
    frozen = nnx.clone(variance_model)
    frozen.eval()
    graph, frozen_state = nnx.split(frozen)

    def aleatoric_variance(inputs: PyTree) -> jax.Array:
        return jax.lax.stop_gradient(nnx.merge(graph, frozen_state)(inputs).mean())

    return aleatoric_variance


def _scaled_log_likelihood(
    mean_function: Callable[[PyTree, PyTree], jax.Array],
    aleatoric_variance: Callable[[PyTree], jax.Array],
    *,
    dataset_size: int,
) -> Callable[[PyTree, Batch], jax.Array]:
    """Return the batch Gaussian log-likelihood scaled by ``N / B``."""

    def log_likelihood(parameters: PyTree, batch: Batch) -> jax.Array:
        if batch.targets is None:
            raise ValueError("training batches must provide targets.")
        targets = jnp.asarray(batch.targets)
        means = mean_function(parameters, batch.inputs)
        scale = jnp.sqrt(aleatoric_variance(batch.inputs))
        log_density = jstats.norm.logpdf(targets, loc=means, scale=scale)
        return dataset_size / targets.shape[0] * jnp.sum(log_density)

    return log_likelihood
