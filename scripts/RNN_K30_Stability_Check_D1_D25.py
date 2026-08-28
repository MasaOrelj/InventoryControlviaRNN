"""K-1=30 stability check, extended to d=1 and d=25 (d=10 already has 40 reps
from RNN_K20_Stability_Check.py). Adds 30 more reps (10-39, same seed formula)
for K-1=30 only, full lambda grid, at d=1 and d=25 -- so all three dimensions
have a 40-rep sample for K-1=30 to check whether the same kind of occasional
bad-draw instability found at d=10 (see conversation) shows up elsewhere too.

Run: python -m scripts.RNN_K30_Stability_Check_D1_D25
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

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_k30_stability_check_d1_d25_experiment.csv")

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

N_HIDDEN = 30
DIMS = [1, 25]
LAMBDA_VALUES = [0.001, 0.01, 0.1, 1.0, 5.0]
NEW_REPS = range(10, 40)

FIT_CONFIGS = [("plain", least_squares, None)] + [
    ("ridge", functools.partial(ridge_regression, ridge_lambda=lam), lam) for lam in LAMBDA_VALUES
]

EVAL_SEED = 500_000
WEIGHT_SEED_BASE = 100_000


def _base_row(n_dims: int) -> dict:
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
        "basis_type": "rnn", "basis_n_hidden": N_HIDDEN, "basis_activation": "tanh",
    }


if __name__ == "__main__":
    print(f"K-1=30 stability check: d in {DIMS}, reps {min(NEW_REPS)}-{max(NEW_REPS)}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    for n_dims in DIMS:
        base_row = _base_row(n_dims)
        eval_rng = np.random.default_rng(EVAL_SEED + n_dims)
        eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)

        print(f"d={n_dims}")
        t0 = time.time()
        for rep in NEW_REPS:
            train_rng = np.random.default_rng(1000 * n_dims + rep)
            train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)

            weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + n_dims * 1000 + rep)
            basis = make_random_features_basis(weight_rng, n_dims=n_dims, n_hidden=N_HIDDEN)

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

            print(f"  d={n_dims:2d} rep {rep} done ({time.time() - t0:.1f}s elapsed)")
        print()

    print("done")
