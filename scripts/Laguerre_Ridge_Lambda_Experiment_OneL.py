"""Same as Laguerre_Ridge_Lambda_Experiment_LSweep.py, but for ONE L value
passed on the command line, writing to its own CSV -- lets several L values
run as SEPARATE OS processes in parallel (one per available core budget)
instead of one process working through all of them sequentially. Each L gets
its own output file specifically so concurrent processes never write to the
same CSV (Experiment_Log.append_result_row's header-write isn't safe for
concurrent writers on a shared file).

Run: python -m scripts.Laguerre_Ridge_Lambda_Experiment_OneL <L>
"""
import functools
import os
import sys
import time

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares, ridge_regression
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row

L = int(sys.argv[1])
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", f"laguerre_ridge_lambda_L{L}_experiment.csv")

MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=40.0,
    lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4,
)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

LAGUERRE_K = 1.0
LAGUERRE_DEGREE = 2

LAMBDA_VALUES = [0.001, 0.01, 0.1, 1.0]
EXTRA_LAMBDA_BY_DIM = {25: [10.0]}

DIMS_AND_REPS = {1: 10, 10: 10, 25: 5}

EVAL_SEED = 500_000


def _fit_configs_for(n_dims: int) -> list:
    lambdas = LAMBDA_VALUES + EXTRA_LAMBDA_BY_DIM.get(n_dims, [])
    return [("plain", least_squares, None)] + [
        ("ridge", functools.partial(ridge_regression, ridge_lambda=lam), lam) for lam in lambdas
    ]


def _base_row(n_dims: int, contract: SwingContract) -> dict:
    return {
        "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
        "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
        "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
        "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp, "f_period": MARKET_PARAMS.f_period,
        "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS,
        "n_paths_train": N_SAMPLES, "n_paths_eval": N_SAMPLES, "n_dims": n_dims,
        "K": contract.K, "q_min": contract.q_min, "q_max": contract.q_max,
        "q_tilde": contract.q_tilde, "L": contract.L,
        "regression_mode": "per-level",
        "state_input": "S",
        "basis_type": "laguerre", "basis_degree": LAGUERRE_DEGREE, "basis_K": LAGUERRE_K,
    }


if __name__ == "__main__":
    contract = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=L)
    print(f"Laguerre L={L}: deg={LAGUERRE_DEGREE}, state=S, M_t=M_e={N_SAMPLES}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    for n_dims, n_reps in DIMS_AND_REPS.items():
        base_row = _base_row(n_dims, contract)
        fit_configs = _fit_configs_for(n_dims)

        eval_rng = np.random.default_rng(EVAL_SEED + n_dims)
        eval_paths = simulate_hhk(
            eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
        )

        basis = WeightedLaguerreBasis(n_dims=n_dims, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)

        t0 = time.time()
        for rep in range(n_reps):
            train_rng = np.random.default_rng(1000 * n_dims + rep)
            train_paths = simulate_hhk(
                train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
            )

            for fit_type, fit, ridge_lambda in fit_configs:
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

                append_result_row(CSV_PATH, {
                    **base_row,
                    "rep": rep, "fit_type": fit_type, "ridge_lambda": ridge_lambda,
                    "price": result["v0"], "duration_sec": duration_sec,
                })

            print(f"  L={L:2d} d={n_dims:2d} rep {rep + 1}/{n_reps} done ({time.time() - t0:.1f}s elapsed)")
        print()

    print("done")
