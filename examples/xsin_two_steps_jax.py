"""Run mean-then-Gamma training on the XSin-inspired benchmark.

This practical benchmark follows the qualitative XSin comparison described by
Yi and Bessa (2025), using explicit mean and variance stages to avoid the
gradient coupling present in joint MVE training.

Run with:

    uv run --extra jax --extra plot python examples/xsin_two_steps_jax.py
"""

from xsin_benchmark import (
    StagePrintingEventSink,
    XSinConfig,
    make_xsin_data,
    plot_xsin_result,
    print_xsin_metrics,
    run_xsin_two_step,
)


def main() -> None:
    """Train both stages, print errors, and show their predictions."""
    config = XSinConfig()
    data = make_xsin_data(config)
    result = run_xsin_two_step(
        data,
        config,
        event_sinks=(StagePrintingEventSink(),),
    )
    print_xsin_metrics("mean plus Gamma variance", result)
    plot_xsin_result("Mean plus Gamma variance", data, result, config)


if __name__ == "__main__":
    main()
