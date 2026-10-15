"""Run mean, variance and posterior stages on the XSin-inspired benchmark.

The VeBNN-style workflow: the mean and Gamma variance stages of
``two_steps.py``, then a posterior stage that replaces the trained mean with a
posterior over mean functions, inferred in turn by Bayes by Backprop, pSGLD and
MC-Dropout, the last on the mean network with batch-shared dropout. Each run
prints the NLL and CRPS of the variance stage's Gaussian and of the posterior
predictive on noisy evaluation targets, and plots the posterior predictive band
against the variance stage's band.

Run with:

    uv run --extra jax --extra plot python examples/jax/xsin/posterior.py
"""

from benchmark import StagePrintingEventSink, XSinConfig, make_xsin_data
from posterior_benchmark import (
    plot_xsin_posterior,
    print_xsin_scores,
    run_xsin_posterior,
    xsin_bayes_by_backprop,
    xsin_mc_dropout,
    xsin_mc_dropout_model,
    xsin_psgld,
)


def main() -> None:
    """Run the three stages with each inference method, print and plot them."""
    config = XSinConfig()
    data = make_xsin_data(config)
    for label, make_method, make_model in (
        ("Bayes by Backprop", xsin_bayes_by_backprop, None),
        ("pSGLD", xsin_psgld, None),
        ("MC-Dropout", xsin_mc_dropout, xsin_mc_dropout_model),
    ):
        result = run_xsin_posterior(
            data,
            config,
            make_method(config),
            model=None if make_model is None else make_model(config),
            event_sinks=(StagePrintingEventSink(),),
        )
        print_xsin_scores(label, result)
        plot_xsin_posterior(label, data, result, config)


if __name__ == "__main__":
    main()
