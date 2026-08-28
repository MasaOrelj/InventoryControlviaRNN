"""Backfill reps 10-39 for the Laguerre ridge-lambda experiment, d=10,
lambda in {0.05, 0.1} (see conversation) -- brings these two cells to 40
reps, matching the depth used elsewhere for stability checks (K-1=30, K-1=10
RNN backfills). Same market/contract/basis/seeds as
Laguerre_Ridge_Lambda_Experiment.py, just narrowed to this one dimension and
these two lambdas, reps 10-39 only.

Run: python -m scripts.Laguerre_D10_Lambda005_01_Backfill
"""
import functools
import os
import time

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, ridge_regression
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "laguerre_ridge_lambda_experiment.csv")

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

LAGUERRE_K = 1.0
LAGUERRE_DEGREE = 2
N_DIMS = 10
LAMBDA_VALUES = [0.05, 0.1]
NEW_REPS = range(10, 40)

EVAL_SEED = 500_000


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
    print(f"Laguerre backfill: d={N_DIMS}, lambda={LAMBDA_VALUES}, reps {min(NEW_REPS)}-{max(NEW_REPS)}")
    base_row = _base_row()
    fit_configs = [functools.partial(ridge_regression, ridge_lambda=lam) for lam in LAMBDA_VALUES]

    eval_rng = np.random.default_rng(EVAL_SEED + N_DIMS)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

    basis = WeightedLaguerreBasis(n_dims=N_DIMS, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)

    t0 = time.time()
    for rep in NEW_REPS:
        train_rng = np.random.default_rng(1000 * N_DIMS + rep)
        train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

        for lam, fit in zip(LAMBDA_VALUES, fit_configs):
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
                "rep": rep, "fit_type": "ridge", "ridge_lambda": lam,
                "price": result["v0"], "duration_sec": duration_sec,
            })

        print(f"  rep {rep} done ({time.time() - t0:.1f}s elapsed)")

    print("done")
