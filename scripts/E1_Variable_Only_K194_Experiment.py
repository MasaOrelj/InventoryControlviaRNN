"""E1 follow-up at K-1=194 (see conversation): does Variable/joint inventory
mode's own convergence keep improving as K-1 grows past 100? Joint mode ONLY
(not separate/Fixed) -- the time estimate given to the user (~5.3h, range
4.7-5.9h) was extrapolated from real joint-mode-only duration data at
K-1=20/50/100 and specifically excludes separate mode, which would roughly
double the cost. Mirrors E1_Inventory_Mode_Experiment_RNN_OneK.py exactly
(same dims, M_t grid, reps, seeds, contract, market) but with
INVENTORY_MODES restricted to ["joint"].

Run: python -m scripts.E1_Variable_Only_K194_Experiment
"""
import os
import time

import numpy as np

from core.Basis_Functions import make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row

N_HIDDEN = 194
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", f"e1_inventory_mode_rnn_K{N_HIDDEN}_experiment.csv")

MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=40.0,
    lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4,
)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
ALPHA = np.exp(-R * MATURITY / N_STEPS)

DIMS = [1, 10, 25]
M_T_VALUES = [250, 1_000, 5_000, 10_000]
N_REPS = 10
M_E = 10_000
INVENTORY_MODES = ["joint"]
REGRESSION_MODE_LABEL = {"separate": "per-level", "joint": "joint"}

EVAL_SEED_BASE = 700_000
TRAIN_SEED_BASE = 300_000
WEIGHT_SEED_BASE = 400_000


if __name__ == "__main__":
    print(f"E1 (RNN, K-1={N_HIDDEN}, VARIABLE/joint ONLY): M_t in {M_T_VALUES}, M_e={M_E}, "
          f"{N_REPS} reps, dims={DIMS}, inventory_mode in {INVENTORY_MODES}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    max_m_t = max(M_T_VALUES)

    for n_dims in DIMS:
        eval_rng = np.random.default_rng(EVAL_SEED_BASE + n_dims)
        eval_paths = simulate_hhk(
            eval_rng, MARKET_PARAMS, n_paths=M_E, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
        )

        t0 = time.time()
        for rep in range(N_REPS):
            seed_key = n_dims * 1_000_000 + rep

            train_rng = np.random.default_rng(TRAIN_SEED_BASE + seed_key)
            train_paths_full = simulate_hhk(
                train_rng, MARKET_PARAMS, n_paths=max_m_t, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
            )

            for inventory_mode in INVENTORY_MODES:
                weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + seed_key)
                basis_dims = n_dims if inventory_mode == "separate" else n_dims + 1
                basis = make_random_features_basis(weight_rng, n_dims=basis_dims, n_hidden=N_HIDDEN)

                for m_t in M_T_VALUES:
                    S_train = train_paths_full.S[:, :m_t, :]

                    t_run = time.time()
                    policy = fit_policy(
                        S_train=S_train, regression_state_train=S_train, contract=CONTRACT,
                        aggregate=max_aggregation, basis=basis, fit=least_squares, alpha=ALPHA,
                        train_itm_only=False, inventory_mode=inventory_mode,
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
                        "regression_mode": REGRESSION_MODE_LABEL[inventory_mode], "state_input": "S",
                        "basis_type": "rnn", "basis_degree": "",
                        "basis_n_hidden": N_HIDDEN,
                        "basis_activation": "tanh",
                        "basis_K": "",
                        "fit_type": "plain", "ridge_lambda": "",
                        "price": result["v0"], "duration_sec": duration_sec,
                    })

            print(f"  K-1={N_HIDDEN} d={n_dims:2d} rep {rep + 1}/{N_REPS} done ({time.time() - t0:.1f}s elapsed)")
        print()

    print("done")
