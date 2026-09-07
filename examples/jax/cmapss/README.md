# NASA CMAPSS Dataset

The example in this directory focuses predicting of remaining useful life (RUL) for the simulated jet engine units that compose the NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset, available from the [NASA CMAPSS dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/5224bcd1-ad61-490b-93b9-2817288accb8).

In particular, the scope is restricted to the FD001 dataset, which consists of 100 train and 100 test time trajectories. The train trajectories are run-to-failure, while the test trajectories are partial histories for which the RUL labels are provided in a separate file.

## Description of the files

### `data.py`

Downloads the CMAPSS archive into temporary storage and loads the FD001
training split into a pandas DataFrame. The DataFrame retains only the unit ID,
time cycles, and sensors 11, 12, 4, 7, 15, 20, 21, 2, and 17; the operational
settings and other sensors are discarded. The downloaded archive is removed
automatically after the DataFrame has been created.

Run the module with:

```shell
uv run --group example-cmapss python examples/jax/cmapss/data.py
```

The `main` function displays all engine trajectories in seaborn line plots,
with one facet per selected sensor and independent sensor scales.

### `model.py`

Contains the classes defining the two-stage predictive model: a Stage-1
deterministic mean model (`Cnn1DMeanModel`), a Stage-2 Gamma residual model
(`Cnn1DGammaModel`), and a `CompositeGaussianModel` combining the frozen
Stage-1 point prediction with the Stage-2 Gamma mean into a single Gaussian
predictive RUL distribution.

### `run.py`

Contains the script to train and evaluate the two-stage predictive model on
the CMAPSS dataset end to end: loading the FD001 train/test splits,
standardizing and windowing them, training the Stage-1 mean model, training
the Stage-2 Gamma model on the frozen Stage-1 model's squared residuals, and
scoring the composite Gaussian model against the official FD001 test split
(`RUL_FD001.txt`), one window per test unit ending at its last observed
cycle.

Run it with:

```shell
uv run --group example-cmapss python examples/jax/cmapss/run.py
```

The script prints per-epoch training metrics for both stages, followed by
the final test-set metrics: RMSE, 95% predictive-interval coverage, and
point-CRPS.
