# Copilot instructions for Probreg

## Project overview

Probreg is a Python library for probabilistic regression. It is structured in stages, in order to allow training of the probabilistic models in steps. For example, a first stage might involve fitting a mean predicting model, and a second stage would instead learn the data variance, using the already trained mean predictor.

The library has been inspired by the work of [Yi and Bessa, 2025](https://arxiv.org/abs/2505.02743), which proposes the step-wise training logic for avoiding gradient pathologies of mean-variance estimators (MVEs) and iterative training of variance predictor and Bayesian Neural Networks (BNNs) to disentangle aleatoric and epistemic uncertainty.
