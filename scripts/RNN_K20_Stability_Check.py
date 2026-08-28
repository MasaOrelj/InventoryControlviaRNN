"""Targeted diagnostic: is K-1=20's occasional bad-draw behavior at d=10 (reps
0, 8 showing large price drops at lambda in {0.1, 1.0}, see conversation) a
real elevated instability rate, or just 2 unlucky seeds out of the original
10? Adds 30 MORE reps (10-39, fresh seeds, same formula as elsewhere in this
project) at d=10 only, for K-1=20 (the value in question) and K-1=30 (a
comparison baseline that showed no such instability in the original 10 reps),
across the full lambda grid.

A mechanistic check (Gram-matrix condition number at a representative step)
did NOT show reps 0/8 as unusually ill-conditioned, so this direct
statistical route is the more trustworthy way to settle the question: does
K-1=20 show a higher rate of "bad" (outlier) reps than K-1=30 once we have
enough draws to tell?

Run: python -m scripts.RNN_K20_Stability_Check
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

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_k20_stability_check_experiment.csv")

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

N_DIMS = 10
K_VALUES = [20, 30]
LAMBDA_VALUES = [0.001, 0.01, 0.1, 1.0, 5.0]
NEW_REPS = range(10, 40)   # 30 fresh reps, continuing the same seed formula

FIT_CONFIGS = [("plain", least_squares, None)] + [
    ("ridge", functools.partial(ridge_regression, ridge_lambda=lam), lam) for lam in LAMBDA_VALUES
]

EVAL_SEED = 500_000
WEIGHT_SEED_BASE = 100_000


def _base_row(n_hidden: int) -> dict:
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
        "basis_type": "rnn", "basis_n_hidden": n_hidden, "basis_activation": "tanh",
    }


if __name__ == "__main__":
    print(f"K-1=20 stability check: d={N_DIMS}, K in {K_VALUES}, reps {min(NEW_REPS)}-{max(NEW_REPS)}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    eval_rng = np.random.default_rng(EVAL_SEED + N_DIMS)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

    for n_hidden in K_VALUES:
        base_row = _base_row(n_hidden)
        print(f"K-1={n_hidden}")
        t0 = time.time()
        for rep in NEW_REPS:
            train_rng = np.random.default_rng(1000 * N_DIMS + rep)
            train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

            weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + N_DIMS * 1000 + rep)
            basis = make_random_features_basis(weight_rng, n_dims=N_DIMS, n_hidden=n_hidden)

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

            print(f"  K-1={n_hidden} rep {rep} done ({time.time() - t0:.1f}s elapsed)")
        print()

    print("done")
