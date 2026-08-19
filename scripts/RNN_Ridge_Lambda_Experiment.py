"""Regularization comparison for the RNN (random features) basis, mirroring
Laguerre_Ridge_Lambda_Experiment.py: plain least squares vs. a ridge_lambda
grid, at state=S, for dimensions d in {1, 10, 25}, M_t=M_e=10000 (d=1, d=10
use 10 repetitions each; d=25 uses 5), and a matrix of hidden-unit counts
(K = n_hidden) in N_HIDDEN_VALUES. Same fixed-eval/varying-train
methodology: ONE evaluation sample per dimension, fixed across every
repetition and every n_hidden; each repetition draws a new training-only
sample and fits a separate policy per fit configuration (plain, plus each
lambda), all on that SAME training draw.

RNN weights are redrawn per repetition (tied to the training seed, like
State_Representation_Experiment.py) -- CLAUDE.md treats the random feature
draw as part of "how training seed r produces policy pi_r", so it varies
with rep, never resampled mid-run for a single fit. The weight seed formula
depends only on (n_dims, rep), not on n_hidden, so adding a new n_hidden
value never changes the draws already used for an existing one.

LAMBDA_VALUES, DIMS_AND_REPS and N_HIDDEN_VALUES are simple module constants,
deliberately easy to extend -- new values just append new rows to the same
CSV. To backfill only new n_hidden values without re-running ones already in
the CSV, write a small script that imports this module and reuses its
constants/functions for just the new n_hidden (see the Laguerre backfill
scripts under scratchpad/ for the pattern).

Run: python -m scripts.RNN_Ridge_Lambda_Experiment
"""
import functools
import os
import time

import numpy as np

from core.Basis_Functions import make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares, ridge_regression
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_ridge_lambda_experiment.csv")

MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=40.0,
    lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4,
)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

N_HIDDEN_VALUES = [10, 20, 30, 40, 50]   # 60 left out for now, add later if wanted

LAMBDA_VALUES = [0.001, 0.01, 0.1, 1.0, 5.0]
DIMS_AND_REPS = {1: 10, 10: 10, 25: 5}

FIT_CONFIGS = [("plain", least_squares, None)] + [
    ("ridge", functools.partial(ridge_regression, ridge_lambda=lam), lam) for lam in LAMBDA_VALUES
]

EVAL_SEED = 500_000   # fixed per dimension: EVAL_SEED + n_dims
WEIGHT_SEED_BASE = 100_000   # RNN weight seed: WEIGHT_SEED_BASE + n_dims*1000 + rep (independent of n_hidden)


def _base_row(n_dims: int, n_hidden: int) -> dict:
    return {
        "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
        "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
        "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
        "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp, "f_period": MARKET_PARAMS.f_period,
        "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS,
        "n_paths_train": N_SAMPLES, "n_paths_eval": N_SAMPLES, "n_dims": n_dims,
        "K": CONTRACT.K, "q_min": CONTRACT.q_min, "q_max": CONTRACT.q_max,
        "q_tilde": CONTRACT.q_tilde, "L": CONTRACT.L,
        "regression_mode": "per-level",
        "state_input": "S",
        "basis_type": "rnn", "basis_n_hidden": n_hidden, "basis_activation": "tanh",
    }


if __name__ == "__main__":
    print(f"RNN n_hidden values: {N_HIDDEN_VALUES}, state=S, M_t=M_e={N_SAMPLES}")
    print(f"fit configs: {[(f, l) for f, _, l in FIT_CONFIGS]}")
    print(f"dims/reps: {DIMS_AND_REPS}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    for n_hidden in N_HIDDEN_VALUES:
        print(f"n_hidden={n_hidden}")
        for n_dims, n_reps in DIMS_AND_REPS.items():
            base_row = _base_row(n_dims, n_hidden)

            # ONE evaluation sample per dimension, fixed across every
            # repetition, every fit config, and every n_hidden.
            eval_rng = np.random.default_rng(EVAL_SEED + n_dims)
            eval_paths = simulate_hhk(
                eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
            )

            t0 = time.time()
            for rep in range(n_reps):
                train_rng = np.random.default_rng(1000 * n_dims + rep)
                train_paths = simulate_hhk(
                    train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
                )

                # RNN weights fixed for this rep's policy (fit + eval both use
                # it), but redrawn next rep -- part of "how seed r produces pi_r".
                weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + n_dims * 1000 + rep)
                basis = make_random_features_basis(weight_rng, n_dims=n_dims, n_hidden=n_hidden)

                for fit_type, fit, ridge_lambda in FIT_CONFIGS:
                    t_run = time.time()
                    policy = fit_policy(
                        S_train=train_paths.S, regression_state_train=train_paths.S, contract=CONTRACT,
                        aggregate=max_aggregation, basis=basis, fit=fit, alpha=ALPHA, train_itm_only=False,
                    )
                    result = evaluate_policy(
                        policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
                        contract=CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
                    )
                    duration_sec = time.time() - t_run

                    append_result_row(CSV_PATH, {
                        **base_row,
                        "rep": rep, "fit_type": fit_type, "ridge_lambda": ridge_lambda,
                        "price": result["v0"], "duration_sec": duration_sec,
                    })

                print(f"  d={n_dims:2d} rep {rep + 1}/{n_reps} done ({time.time() - t0:.1f}s elapsed)")
            print()
