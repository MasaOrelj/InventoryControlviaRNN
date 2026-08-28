"""Backfill reps 10-19 for Laguerre, d=1, contract L=1, plain and
lambda=0.001 only (see conversation) -- the plain-vs-0.001 comparison was the
one close call in the significance check (non-significant at L=1, p=0.51),
so more reps sharpen that specific comparison rather than the whole L=1
table. Same seeds/conventions as Laguerre_Ridge_Lambda_Experiment_OneL.py.

Run: python -m scripts.Laguerre_D1_L1_Plain_Lambda001_Backfill
"""
import os
import time

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares, ridge_regression
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row
import functools

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "laguerre_ridge_lambda_L1_experiment.csv")

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=1)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

LAGUERRE_K = 1.0
LAGUERRE_DEGREE = 2
N_DIMS = 1
NEW_REPS = range(10, 20)

EVAL_SEED = 500_000

FIT_CONFIGS = [("plain", least_squares, None), ("ridge", functools.partial(ridge_regression, ridge_lambda=0.001), 0.001)]


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
        "basis_type": "laguerre", "basis_degree": LAGUERRE_DEGREE, "basis_K": LAGUERRE_K,
    }


if __name__ == "__main__":
    print(f"Laguerre backfill: d={N_DIMS}, L={CONTRACT.L}, plain+lambda=0.001, reps {min(NEW_REPS)}-{max(NEW_REPS)}")
    base_row = _base_row()

    eval_rng = np.random.default_rng(EVAL_SEED + N_DIMS)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

    basis = WeightedLaguerreBasis(n_dims=N_DIMS, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)

    t0 = time.time()
    for rep in NEW_REPS:
        train_rng = np.random.default_rng(1000 * N_DIMS + rep)
        train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

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

        print(f"  rep {rep} done ({time.time() - t0:.1f}s elapsed)")

    print("done")
