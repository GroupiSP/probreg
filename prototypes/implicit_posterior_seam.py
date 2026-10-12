"""PROTOTYPE (throwaway) for stagereg#106 -- not production code.

Question: can an implicit, generator-based posterior (QAVI-shaped) fit the
``InferenceMethod`` seam? A classical stand-in replaces the PQC: a scalar
latent ``z ~ U(0, 2pi)`` goes through ``cos(a * z + b)`` "qubit expectations"
and a linear postprocessor that emits the inferred subnetwork's weights. The
generator is trained adversarially against an NNX discriminator that judges
``(input, predictive sample)`` pairs, plus the problem's log-likelihood.

Run:  uv run python prototypes/implicit_posterior_seam.py
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from stagereg.core.checkpoints import InMemoryCheckpointStore
from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.types import Batch, PyTree, TrainingState
from stagereg.jax import (
    GammaHead,
    GammaVarianceStage,
    HeldOutValidation,
    MeanStage,
    PosteriorProblem,
    PosteriorStage,
    PosteriorStageOptions,
    SupervisedStageOptions,
    create_optimizer,
)

N, B = 64, 16


# --------------------------------------------------------------------------- data
def data() -> tuple[jax.Array, jax.Array]:
    x = jnp.linspace(-2.0, 2.0, N)[:, None]
    y = jnp.sin(2.0 * x) + 0.2 * jax.random.normal(jax.random.key(0), x.shape)
    return x, y


def loader(*, split: str, epoch: int) -> list[Batch]:
    del epoch
    x, y = data()
    shift = 0.05 if split == "validation" else 0.0
    return [
        Batch(inputs=x[i : i + B], targets=y[i : i + B] + shift) for i in range(0, N, B)
    ]


class TwoLayer(nnx.Module):
    """Stand-in for the battery CNN; ``hidden.kernel`` stands in for the conv kernel."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.hidden = nnx.Linear(1, 16, rngs=rngs)
        self.out = nnx.Linear(16, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.out(jnp.tanh(self.hidden(x)))


# ------------------------------------------------- inferred subnetwork (cf. #105)
def is_inferred(path: tuple[Any, ...]) -> bool:
    keys = [str(getattr(p, "key", p)) for p in path]
    return "hidden" in keys and "kernel" in keys


def inferred_leaf(tree: PyTree) -> jax.Array:
    (leaf,) = [
        v for p, v in jax.tree_util.tree_leaves_with_path(tree) if is_inferred(p)
    ]
    return leaf


def assemble(frozen: PyTree, inferred: jax.Array) -> PyTree:
    """Network tree = frozen remainder + one generated inferred leaf."""
    return jax.tree_util.tree_map_with_path(
        lambda p, leaf: inferred if is_inferred(p) else leaf, frozen
    )


# --------------------------------------------------------- classical generator
N_QUBITS = 4


def init_generator(key: jax.Array, size: int) -> PyTree:
    k1, k2 = jax.random.split(key)
    return {
        "circuit": jax.random.uniform(k1, (N_QUBITS, 2), maxval=2 * jnp.pi),
        "post_w": 1e-2 * jax.random.normal(k2, (N_QUBITS, size)),
        "post_b": jnp.zeros((size,)),
    }


def generate(gen: PyTree, anchor: jax.Array, key: jax.Array) -> jax.Array:
    """One draw of the inferred weights; ``anchor`` = trained mean weights (warm
    start)."""
    z = jax.random.uniform(key, (), maxval=2 * jnp.pi)
    feats = jnp.cos(gen["circuit"][:, 0] * z + gen["circuit"][:, 1])
    return anchor + (feats @ gen["post_w"] + gen["post_b"]).reshape(anchor.shape)


# ------------------------------------------------------------- discriminator
class Discriminator(nnx.Module):
    def __init__(self, in_dim: int, *, rngs: nnx.Rngs) -> None:
        self.l1 = nnx.Linear(in_dim + 1, 32, rngs=rngs)
        self.l2 = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        h = jnp.concatenate([x.reshape(x.shape[0], -1), y.reshape(y.shape[0], 1)], 1)
        return self.l2(nnx.leaky_relu(self.l1(h), 0.2))[:, 0]  # logits


# ---------------------------------------------------------------- posterior
@dataclass(frozen=True)
class GeneratorPosterior:
    """Implicit posterior: draw s = mean function with weights G(z_s)."""

    mean_function: Callable[[PyTree, PyTree], jax.Array]
    frozen: PyTree
    generator: PyTree

    @property
    def num_draws(self) -> int | None:
        return None

    def sample_means(
        self, inputs: PyTree, key: jax.Array, num_samples: int | None = None
    ) -> jax.Array:
        if num_samples is None:
            raise ValueError("an unlimited posterior needs num_samples.")
        anchor = inferred_leaf(self.frozen)

        def draw(s: jax.Array) -> jax.Array:
            w = generate(self.generator, anchor, jax.random.fold_in(key, s))
            return self.mean_function(assemble(self.frozen, w), inputs)

        return jax.vmap(draw)(jnp.arange(num_samples))


# ---------------------------------------------------------- inference method
@dataclass
class AdversarialVI:
    """QAVI-shaped InferenceMethod with a classical generator."""

    adversarial_weight: float = 0.1
    disc_steps: int = 1
    generator_optimizer: optax.GradientTransformation = field(
        default_factory=lambda: optax.adam(3e-3)
    )
    discriminator_optimizer: optax.GradientTransformation = field(
        default_factory=lambda: optax.adam(3e-3)
    )
    seed: int = 7
    _problem: PosteriorProblem | None = field(default=None, init=False, repr=False)
    _gen: PyTree = field(default=None, init=False, repr=False)
    _disc: PyTree = field(default=None, init=False, repr=False)  # nnx.State
    _disc_graph: Any = field(default=None, init=False, repr=False)
    _gen_opt: PyTree = field(default=None, init=False, repr=False)
    _disc_opt: PyTree = field(default=None, init=False, repr=False)
    _step: Callable[..., Any] | None = field(default=None, init=False, repr=False)

    supports_early_stopping = True

    @property
    def has_posterior(self) -> bool:
        return self._problem is not None

    def init(self, problem: PosteriorProblem) -> None:
        if problem.aleatoric_variance is None:  # the seam gap this prototype probes
            raise ValueError(
                "QAVI needs problem.aleatoric_variance for predictive samples."
            )
        self._problem = problem
        g_key, d_key = jax.random.split(jax.random.key(self.seed))
        size = inferred_leaf(problem.initial_parameters).size
        self._gen = init_generator(g_key, size)
        x0 = jnp.asarray(
            next(iter(problem.train_loader(split="train", epoch=0))).inputs
        )
        disc = Discriminator(int(x0[0].size), rngs=nnx.Rngs(d_key))
        self._disc_graph, self._disc = nnx.split(disc)
        self._gen_opt = self.generator_optimizer.init(self._gen)
        self._disc_opt = self.discriminator_optimizer.init(self._disc)
        self._step = self._make_step(problem)

    def _make_step(self, problem: PosteriorProblem) -> Callable[..., Any]:
        frozen = problem.initial_parameters
        anchor = inferred_leaf(frozen)
        graph, lam, n = self._disc_graph, self.adversarial_weight, problem.dataset_size
        g_tx, d_tx = self.generator_optimizer, self.discriminator_optimizer
        aleatoric = problem.aleatoric_variance
        assert aleatoric is not None

        def fake_samples(
            gen: PyTree, x: jax.Array, key: jax.Array
        ) -> tuple[jax.Array, PyTree]:
            """One weight draw per example, as qmodem does; y ~ N(mu_s(x),
            sigma^2(x))."""
            k_w, k_eps = jax.random.split(key)
            ws = jax.vmap(lambda k: generate(gen, anchor, k))(
                jax.random.split(k_w, x.shape[0])
            )
            mu = jax.vmap(
                lambda w, xi: problem.mean_function(assemble(frozen, w), xi[None])[0]
            )(ws, x)
            eps = jax.random.normal(k_eps, mu.shape)
            return mu + jnp.sqrt(aleatoric(x)).reshape(mu.shape) * eps, ws

        def disc_loss(disc: PyTree, gen: PyTree, x, y, key) -> jax.Array:
            fake, _ = fake_samples(jax.lax.stop_gradient(gen), x, key)
            d = nnx.merge(graph, disc)
            return jnp.mean(
                optax.sigmoid_binary_cross_entropy(d(x, y), 1.0)
                + optax.sigmoid_binary_cross_entropy(d(x, fake), 0.0)
            )

        def gen_loss(gen: PyTree, disc: PyTree, x, y, key) -> jax.Array:
            fake, ws = fake_samples(gen, x, key)
            adv = jnp.mean(
                optax.sigmoid_binary_cross_entropy(nnx.merge(graph, disc)(x, fake), 1.0)
            )
            # per-example log-likelihood under that example's draw (problem's N/B scaling undone)
            ll = jax.vmap(
                lambda w, xi, yi: problem.log_likelihood(
                    assemble(frozen, w), Batch(inputs=xi[None], targets=yi[None])
                )
                / n
            )(ws, x, y)
            return -jnp.mean(ll) + lam * adv

        @jax.jit
        def step(gen, disc, g_opt, d_opt, x, y, key):
            d_keys = jax.random.split(key, self.disc_steps + 1)
            for k in d_keys[:-1]:
                d_l, d_g = jax.value_and_grad(disc_loss)(disc, gen, x, y, k)
                upd, d_opt = d_tx.update(d_g, d_opt, disc)
                disc = optax.apply_updates(disc, upd)
            g_l, g_g = jax.value_and_grad(gen_loss)(gen, disc, x, y, d_keys[-1])
            upd, g_opt = g_tx.update(g_g, g_opt, gen)
            return optax.apply_updates(gen, upd), disc, g_opt, d_opt, g_l, d_l

        return step

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        assert self._step is not None
        self._gen, self._disc, self._gen_opt, self._disc_opt, g_l, d_l = self._step(
            self._gen,
            self._disc,
            self._gen_opt,
            self._disc_opt,
            jnp.asarray(batch.inputs),
            jnp.asarray(batch.targets),
            key,
        )
        return {"loss": g_l, "discriminator_loss": d_l}

    def posterior(self) -> GeneratorPosterior:
        assert self._problem is not None
        return GeneratorPosterior(
            self._problem.mean_function, self._problem.initial_parameters, self._gen
        )

    def state(self) -> PyTree:
        return {
            "generator": self._gen,
            "discriminator": self._disc,
            "generator_opt": self._gen_opt,
            "discriminator_opt": self._disc_opt,
        }

    def load_state(self, state: PyTree) -> None:
        self._gen = state["generator"]
        self._disc = state["discriminator"]
        self._gen_opt = state["generator_opt"]
        self._disc_opt = state["discriminator_opt"]

    def posterior_state(self) -> PyTree:
        return {"generator": self._gen}  # discriminator is training-only

    def load_posterior(self, state: PyTree) -> None:
        self._gen = state["generator"]


# --------------------------------------------------------------------- driver
def mse(model, inputs, targets, sample_weight, key, training):
    del sample_weight, key, training
    return jnp.mean(jnp.square(model(inputs) - targets))


def build(
    store: InMemoryCheckpointStore,
) -> tuple[MeanStage, GammaVarianceStage, PosteriorStage]:
    mean_model = TwoLayer(rngs=nnx.Rngs(0))
    mean = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(mean_model, optax.adam(0.03)),
        train_loader=loader,
        options=SupervisedStageOptions(
            epochs=150,
            checkpoint_store=store,
            validation=HeldOutValidation(model=mean_model, loader=loader, loss=mse),
            early_stopper=EarlyStopper(metric="loss", mode="min", patience=1000),
        ),
    )
    var_model = GammaHead(1, 1, rngs=nnx.Rngs(2))
    var = GammaVarianceStage(
        model=var_model,
        optimizer=create_optimizer(var_model, optax.adam(0.05)),
        source_loader=loader,
        options=SupervisedStageOptions(
            epochs=60,
            checkpoint_store=store,
            early_stopper=EarlyStopper(metric="loss", mode="min", patience=1000),
        ),
        validation_factory=lambda residuals: HeldOutValidation(
            model=var_model, loader=residuals, loss=var.loss
        ),
    )
    post = PosteriorStage(
        inference_method=AdversarialVI(),
        train_loader=loader,
        dataset_size=N,
        options=PosteriorStageOptions(
            epochs=40,
            validation_loader=loader,
            early_stopper=EarlyStopper(metric="nll", mode="min", patience=10),
            checkpoint_store=store,
        ),
    )
    return mean, var, post


def show(title: str, **kv: Any) -> None:
    print(f"\n== {title}")
    for k, v in kv.items():
        print(f"   {k:<38} {v}")


def main() -> None:
    store = InMemoryCheckpointStore()
    mean, var, post = build(store)
    state = TrainingState(rng_state=jax.random.key(1))
    for stage in (mean, var):
        stage.prepare(state)
        stage.train(state)
    post.prepare(state)

    method: AdversarialVI = post.inference_method  # type: ignore[assignment]
    problem = method._problem
    assert problem is not None
    net_count = sum(v.size for v in jax.tree.leaves(problem.initial_parameters))
    gen_count = sum(v.size for v in jax.tree.leaves(method._gen))
    paths = [
        jax.tree_util.keystr(p)
        for p, _ in jax.tree_util.tree_leaves_with_path(problem.initial_parameters)
        if is_inferred(p)
    ]
    x, _ = data()
    trained = problem.mean_function(problem.initial_parameters, x)
    draws0 = method.posterior().sample_means(x, jax.random.key(9), 16)
    show(
        "1. inferred tree != network tree",
        network_params=net_count,
        generator_params=gen_count,
        inferred_leaf=paths,
        warm_start_max_abs_dev=float(jnp.max(jnp.abs(draws0 - trained[None]))),
    )

    post.train(state)
    hist = {
        k: [round(float(v), 3) for v in vs]
        for k, vs in state.metric_history.items()
        if k.startswith("posterior")
    }
    show(
        "2. epoch loop driven by stage (update returns loss + discriminator_loss)",
        **{k: (v[:3] + ["..."] + v[-2:]) for k, v in hist.items()},
        discriminator_loss_recorded=any("discriminator" in k for k in hist),
    )

    p = method.posterior()
    k = jax.random.key(42)
    a = p.sample_means(x, k, 8)
    b = p.sample_means(x[:10], k, 8)
    show(
        "3. draws are whole mean functions",
        same_key_same_draws=bool(jnp.allclose(a, p.sample_means(x, k, 8))),
        draw_consistent_across_inputs=bool(jnp.allclose(a[:, :10], b)),
        epistemic_std_mean=float(jnp.mean(jnp.std(a, 0))),
    )

    best = store.load("posterior/best")
    show(
        "4. checkpoints",
        finalized_posterior_state_keys=list(state.posterior_state),
        finalized_has_discriminator="discriminator" in state.posterior_state,
        finalized_checkpoint_epoch=best.epoch,
    )

    mean2, var2, post2 = build(store)
    fresh = TrainingState()
    mean2.restore(fresh, store.load("mean/best"))
    var2.restore(fresh, store.load("variance/best"))
    post2.restore(fresh, store.load("posterior/best"))
    c = fresh.model_components["posterior"].sample_means(x, k, 8)
    show(
        "5. restore in a fresh process-like setup",
        validate=post2.validate(fresh).passed,
        restored_draws_equal=bool(jnp.allclose(a, c)),
    )


if __name__ == "__main__":
    main()
