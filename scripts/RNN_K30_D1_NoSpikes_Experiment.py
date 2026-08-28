"""Spikes-off arm at d=1, completing the dimension sweep alongside the d=10
and d=25 spikes-off experiments (see conversation) -- same lambda grid, K-1,
and seeds as those two, run directly at 40 reps (matching their post-backfill
depth) rather than starting at 10 and backfilling separately.

Run: python -m scripts.RNN_K30_D1_NoSpikes_Experiment
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

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_k30_d1_nospikes_experiment.csv")

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=0.0, mu_up=0.6, lam_down=0.0, mu_down=0.4)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

N_HIDDEN = 30
N_DIMS = 1
LAMBDA_VALUES = [0.001, 0.01, 0.1, 0.3, 0.7, 1.0, 5.0]
N_REPS = 40

EVAL_SEED = 500_000
WEIGHT_SEED_BASE = 100_000


def _base_row() -> dict:
    return {
        "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
        "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
        "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
        "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp, "f_period": MARKET_PARAMS.f_period,
        "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS,
        "n_paths_train": N_SAMPLES, "n_paths_eval": N_SAMPLES, "n_dims": N_DIMS,
        "K": CONTRACT.K, "q_min": CONTRACT.q_min, "q_max": CONTRACT.q_max,
        "q_tilde": CONTRACT.q_tilde, "L": CONTRACT.L,
        "regression_mode": "per-level",
        "state_input": "S",
        "basis_type": "rnn", "basis_n_hidden": N_HIDDEN, "basis_activation": "tanh",
    }


if __name__ == "__main__":
    print(f"RNN K-1={N_HIDDEN}, d={N_DIMS}, SPIKES OFF (lam_up=lam_down=0), lambda grid={LAMBDA_VALUES}, {N_REPS} reps")
    base_row = _base_row()
    fit_configs = [("plain", least_squares, None)] + [
        ("ridge", functools.partial(ridge_regression, ridge_lambda=lam), lam) for lam in LAMBDA_VALUES
    ]

    eval_rng = np.random.default_rng(EVAL_SEED + N_DIMS)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

    t0 = time.time()
    for rep in range(N_REPS):
        train_rng = np.random.default_rng(1000 * N_DIMS + rep)
        train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

        weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + N_DIMS * 1000 + rep)
        basis = make_random_features_basis(weight_rng, n_dims=N_DIMS, n_hidden=N_HIDDEN)

        for fit_type, fit, ridge_lambda in fit_configs:
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

        print(f"  rep {rep + 1}/{N_REPS} done ({time.time() - t0:.1f}s elapsed)")

    print("done")
