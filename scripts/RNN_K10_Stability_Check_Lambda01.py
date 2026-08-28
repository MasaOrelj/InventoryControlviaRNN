"""Backfill reps 10-39 for K-1=10, lambda=0.01, at d=10 and d=25 -- the two
additional df(lambda) in [3.5,10] cells found when extending the table (see
conversation). Separate output file from RNN_K10_Stability_Check.py's to
avoid concurrent writes to the same CSV.

Run: python -m scripts.RNN_K10_Stability_Check_Lambda01
"""
import functools
import os
import time

import numpy as np

from core.Basis_Functions import make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, ridge_regression
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_k10_stability_check_lambda01_experiment.csv")

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

N_HIDDEN = 10
LAMBDA = 0.01
DIMS = [10, 25]
NEW_REPS = range(10, 40)

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
    print(f"K-1=10, lambda={LAMBDA} backfill: dims={DIMS}, reps {min(NEW_REPS)}-{max(NEW_REPS)}")
    fit = functools.partial(ridge_regression, ridge_lambda=LAMBDA)

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
                "rep": rep, "fit_type": "ridge", "ridge_lambda": LAMBDA,
                "price": result["v0"], "duration_sec": duration_sec,
            })

            print(f"  d={n_dims:2d} rep {rep} done ({time.time() - t0:.1f}s elapsed)")
        print()

    print("done")
