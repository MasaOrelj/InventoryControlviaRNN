"""E1 -- Laguerre, JOINT (B) mode only. Separate (A) mode results already
exist for Laguerre in training_sample_sensitivity_experiment.csv (same
seeds, per-level/"separate" regression_mode, all three dimensions, 10 reps
each) -- this script fills in only the missing joint-mode side, so it's
compared against that existing data rather than re-running the (already
expensive) separate mode again.

Same nested-M_t-prefix design and SAME SEEDS as
Training_Sample_Sensitivity_Experiment.py / E1_Inventory_Mode_Experiment_RNN.py.
d=1, d=10: 10 reps (matching existing convention). d=25: 5 reps only
(explicit request, given joint mode's cost there).

Run: python -m scripts.E1_Inventory_Mode_Experiment_Laguerre
"""
import os
import time

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "e1_inventory_mode_laguerre_experiment.csv")

MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=40.0,
    lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4,
)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
ALPHA = np.exp(-R * MATURITY / N_STEPS)

LAGUERRE_DEGREE = 2
LAGUERRE_K = 1.0
M_T_VALUES = [250, 1_000, 5_000, 10_000]
M_E = 10_000
DIMS_AND_REPS = {1: 10, 10: 10, 25: 5}

# SAME seeds as Training_Sample_Sensitivity_Experiment.py / RNN E1 script.
EVAL_SEED_BASE = 700_000
TRAIN_SEED_BASE = 300_000


if __name__ == "__main__":
    print(f"E1 (Laguerre, joint only): M_t in {M_T_VALUES}, M_e={M_E}, dims/reps={DIMS_AND_REPS}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    max_m_t = max(M_T_VALUES)

    for n_dims, n_reps in DIMS_AND_REPS.items():
        eval_rng = np.random.default_rng(EVAL_SEED_BASE + n_dims)
        eval_paths = simulate_hhk(
            eval_rng, MARKET_PARAMS, n_paths=M_E, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
        )

        basis = WeightedLaguerreBasis(n_dims=n_dims + 1, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)

        t0 = time.time()
        for rep in range(n_reps):
            seed_key = n_dims * 1_000_000 + rep

            train_rng = np.random.default_rng(TRAIN_SEED_BASE + seed_key)
            train_paths_full = simulate_hhk(
                train_rng, MARKET_PARAMS, n_paths=max_m_t, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
            )

            for m_t in M_T_VALUES:
                S_train = train_paths_full.S[:, :m_t, :]

                t_run = time.time()
                policy = fit_policy(
                    S_train=S_train, regression_state_train=S_train, contract=CONTRACT,
                    aggregate=max_aggregation, basis=basis, fit=least_squares, alpha=ALPHA,
                    train_itm_only=False, inventory_mode="joint",
                )
                result = evaluate_policy(
                    policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
                    contract=CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
                )
                duration_sec = time.time() - t_run

                append_result_row(CSV_PATH, {
                    "rep": rep,
                    "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
                    "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
                    "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
                    "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp,
                    "f_period": MARKET_PARAMS.f_period,
                    "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS,
                    "n_paths_train": m_t, "n_paths_eval": M_E, "n_dims": n_dims,
                    "K": CONTRACT.K, "q_min": CONTRACT.q_min, "q_max": CONTRACT.q_max,
                    "q_tilde": CONTRACT.q_tilde, "L": CONTRACT.L,
                    "regression_mode": "joint", "state_input": "S",
                    "basis_type": "laguerre", "basis_degree": LAGUERRE_DEGREE,
                    "basis_n_hidden": "", "basis_activation": "", "basis_K": LAGUERRE_K,
                    "fit_type": "plain", "ridge_lambda": "",
                    "price": result["v0"], "duration_sec": duration_sec,
                })
                print(f"    m_t={m_t:5d} done ({time.time() - t0:.1f}s elapsed)")

            print(f"  d={n_dims:2d} rep {rep + 1}/{n_reps} done ({time.time() - t0:.1f}s elapsed)")
        print()

    print("done")
