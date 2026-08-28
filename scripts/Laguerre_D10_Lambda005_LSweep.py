"""Fill the lambda=0.05 gap between 0.01 and 0.1 for Laguerre d=10, across
all four L values in the sweep (see conversation) -- 0.01 was beating 0.1 on
price at every L, so this checks whether the price peak actually sits
between them. Same seeds/conventions as Laguerre_Ridge_Lambda_Experiment_OneL.py,
10 reps (matching d=10's rep count there), appended to each L's own CSV.

Run: python -m scripts.Laguerre_D10_Lambda005_LSweep
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

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

LAGUERRE_K = 1.0
LAGUERRE_DEGREE = 2
N_DIMS = 10
N_REPS = 10
LAMBDA = 0.05
L_VALUES = [1, 5, 25, 40]

EVAL_SEED = 500_000


def _base_row(contract: SwingContract) -> dict:
    return {
        "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
        "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
        "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
        "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp, "f_period": MARKET_PARAMS.f_period,
        "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS,
        "n_paths_train": N_SAMPLES, "n_paths_eval": N_SAMPLES, "n_dims": N_DIMS,
        "K": contract.K, "q_min": contract.q_min, "q_max": contract.q_max,
        "q_tilde": contract.q_tilde, "L": contract.L,
        "regression_mode": "per-level",
        "state_input": "S",
        "basis_type": "laguerre", "basis_degree": LAGUERRE_DEGREE, "basis_K": LAGUERRE_K,
    }


if __name__ == "__main__":
    print(f"Laguerre d={N_DIMS}, lambda={LAMBDA}, L in {L_VALUES}, {N_REPS} reps each")
    basis = WeightedLaguerreBasis(n_dims=N_DIMS, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)
    fit = functools.partial(ridge_regression, ridge_lambda=LAMBDA)

    eval_rng = np.random.default_rng(EVAL_SEED + N_DIMS)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

    for L in L_VALUES:
        contract = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=L)
        csv_path = os.path.join(os.path.dirname(__file__), "..", "results", f"laguerre_ridge_lambda_L{L}_experiment.csv")
        base_row = _base_row(contract)

        t0 = time.time()
        for rep in range(N_REPS):
            train_rng = np.random.default_rng(1000 * N_DIMS + rep)
            train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=N_DIMS)

            t_run = time.time()
            policy = fit_policy(
                S_train=train_paths.S, regression_state_train=train_paths.S, contract=contract,
                aggregate=max_aggregation, basis=basis, fit=fit, alpha=ALPHA, train_itm_only=False,
            )
            result = evaluate_policy(
                policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
                contract=contract, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
            )
            duration_sec = time.time() - t_run

            append_result_row(csv_path, {
                **base_row,
                "rep": rep, "fit_type": "ridge", "ridge_lambda": LAMBDA,
                "price": result["v0"], "duration_sec": duration_sec,
            })

        print(f"  L={L:2d} done ({time.time() - t0:.1f}s elapsed)")

    print("done")
