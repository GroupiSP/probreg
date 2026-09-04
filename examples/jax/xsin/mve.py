"""Run joint Gaussian MVE on the XSin-inspired benchmark.

This practical benchmark follows the qualitative XSin comparison described by
Yi and Bessa (2025), but uses compact example-oriented settings rather than
claiming exact reproduction of the paper's reported numerical results.

Run with:

    uv run --extra jax --extra plot python examples/jax/xsin/mve.py
"""

from benchmark import (
    XSinConfig,
    make_xsin_data,
    plot_xsin_result,
    print_xsin_metrics,
    run_xsin_mve,
)


def main() -> None:
    """Train the MVE baseline, print errors, and show its predictions."""
    config = XSinConfig()
    data = make_xsin_data(config)
    result = run_xsin_mve(data, config)
    print_xsin_metrics("joint Gaussian MVE", result)
    plot_xsin_result("Joint Gaussian MVE", data, result, config)


if __name__ == "__main__":
    main()
