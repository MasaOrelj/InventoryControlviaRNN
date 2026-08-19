"""Regularization comparison for the Laguerre (degree 2) basis: plain least
squares vs. a ridge_lambda grid, at state=S, for dimensions d in {1, 10, 25},
M_t=M_e=10000. Same fixed-eval/varying-train methodology as
State_Representation_Experiment.py / Basis_Function_Experiment.py: ONE
evaluation sample per dimension, fixed across every repetition; each
repetition draws a new training-only sample and fits a separate policy per
fit configuration (plain, plus each lambda), all on that SAME training draw
-- so plain and every ridge_lambda are directly comparable, not just
independently noisy.

d=1 and d=10 use 10 repetitions each; d=25 uses 5 -- cheap dimensions get
more reps, matching the established convention elsewhere in this project.

LAMBDA_VALUES and DIMS_AND_REPS are simple module constants, deliberately easy
to extend -- this experiment is meant to grow incrementally as more lambda
values / dimensions get added later; each run just appends new rows to the
same CSV without disturbing what's already there.

Run: python -m scripts.Laguerre_Ridge_Lambda_Experiment
"""
import functools
import os
import time

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares, ridge_regression
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "laguerre_ridge_lambda_experiment.csv")

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

LAGUERRE_K = 1.0
LAGUERRE_DEGREE = 2

# Easy to extend: add more values, rerun -- new combinations append to the
# same CSV, existing rows untouched. Log-spaced (each 10x the last) so the
# grid spans "negligible" through "destroys the fit" in effective-df terms
# (see CLAUDE.md / mentor comment on the ridge regularization table): 1.0 is
# kept specifically because it's the mentor's own worked example (df=0.6 at
# d=1 out of 3 features vs. df=3.1 at d=25 out of 351 -- same raw lambda,
# very different actual regularization).
LAMBDA_VALUES = [0.001, 0.01, 0.1, 1.0]

# d=25 additionally gets lambda=10.0: verified (see conversation) to bring
# ITS effective df down to ~0.57 -- almost exactly matching d=1's df=0.58 at
# lambda=1.0. Same "destroyed fit" level, reached with a 10x larger raw
# lambda because d=25 has 10x-ish more nominal capacity (351 vs 3 features)
# to shrink away first.
EXTRA_LAMBDA_BY_DIM = {25: [10.0]}

DIMS_AND_REPS = {1: 10, 10: 10, 25: 5}


def _fit_configs_for(n_dims: int) -> list:
    """(fit_type, fit_function, ridge_lambda_or_None) -- "plain" first, then
    the ridge grid (base + any dimension-specific extras), all fit on the
    SAME training draw per repetition."""
    lambdas = LAMBDA_VALUES + EXTRA_LAMBDA_BY_DIM.get(n_dims, [])
    return [("plain", least_squares, None)] + [
        ("ridge", functools.partial(ridge_regression, ridge_lambda=lam), lam) for lam in lambdas
    ]

EVAL_SEED = 500_000   # fixed per dimension: EVAL_SEED + n_dims


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
        "basis_type": "laguerre", "basis_degree": LAGUERRE_DEGREE, "basis_K": LAGUERRE_K,
    }


if __name__ == "__main__":
    print(f"Laguerre deg={LAGUERRE_DEGREE}, state=S, M_t=M_e={N_SAMPLES}")
    print(f"base lambda values: {LAMBDA_VALUES}, extra by dim: {EXTRA_LAMBDA_BY_DIM}")
    print(f"dims/reps: {DIMS_AND_REPS}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    for n_dims, n_reps in DIMS_AND_REPS.items():
        base_row = _base_row(n_dims)
        fit_configs = _fit_configs_for(n_dims)

        # ONE evaluation sample per dimension, fixed across every repetition
        # and every fit config.
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
