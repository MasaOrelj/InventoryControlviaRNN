"""Beta spike-sensitivity experiment, d=25 follow-up (see conversation):
persists the ad-hoc d=25, beta=40, RNN spot-check from earlier in the
session (mean relative spike contribution 183.68%, sd 2.47%, 10 reps --
printed to console only, never saved to a CSV) as a proper, reproducible
full beta sweep, matching Beta_Spike_Sensitivity_Experiment_D10.py's design
exactly (same seed reused across the whole beta sweep within a repetition;
no-jump baseline built by zeroing Y out of an already-simulated Z path,
never a separate lambda=0 simulation) and the SAME beta grid as d=10, for
direct cross-dimension comparability.

5 reps (down from d=10's 10), matching this project's established
"cheap dimensions get more reps" convention (DIMS_AND_REPS={1:10,10:10,25:5}
elsewhere) -- Laguerre degree-2 at d=25 has 351 features and costs roughly
112s per single fit even after this session's separate-mode optimization
(measured directly from the joint L-sweep run), ~40x RNN's cost at the same
dimension, so 5 reps already means ~1.25h dominated by Laguerre alone.

Distinct seed ranges from both the d=1 and d=10 scripts.

Run: python -m scripts.Beta_Spike_Sensitivity_Experiment_D25
"""
import csv
import os
import time

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis, make_random_features_basis
from core.Electricity_Market_Model import HHKParams, seasonality, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares
from core.Swing import SwingContract

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "beta_spike_sensitivity_experiment_d25.csv")

KAPPA, SIGMA = 7.0, 1.4
LAM_UP, MU_UP, LAM_DOWN, MU_DOWN = 5.0, 0.6, 3.0, 0.4
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
DELTA = MATURITY / N_STEPS
ALPHA = np.exp(-R * MATURITY / N_STEPS)
D = 25
M = 10_000
N_REPS = 5
BETAS = [30, 35, 40, 45, 50, 75, 100]
BASIS_TYPES = ["laguerre", "rnn"]
LAGUERRE_DEGREE = 2
LAGUERRE_K = 1.0
RNN_N_HIDDEN = 20

# Distinct seed ranges from both the d=1 and d=10 scripts.
TRAIN_SEED_BASE = 980_000
EVAL_SEED_BASE = 990_000
WEIGHT_SEED_BASE = 1_000_000

FIELDNAMES = [
    "rep", "n_dims", "beta", "half_life_over_delta", "basis_type",
    "price_with_spikes", "price_no_spikes", "diff", "rel_pct", "duration_sec",
]


def append_row(row: dict) -> None:
    is_new_file = not os.path.exists(CSV_PATH)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def make_basis(basis_type: str, weight_rng: np.random.Generator):
    if basis_type == "laguerre":
        return WeightedLaguerreBasis(n_dims=D, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)
    return make_random_features_basis(weight_rng, n_dims=D, n_hidden=RNN_N_HIDDEN)


def hhk_params(beta: float) -> HHKParams:
    return HHKParams(kappa=KAPPA, sigma=SIGMA, beta=beta, lam_up=LAM_UP, mu_up=MU_UP,
                      lam_down=LAM_DOWN, mu_down=MU_DOWN)


if __name__ == "__main__":
    print(f"betas={BETAS}, {N_REPS} reps, d={D}, bases={BASIS_TYPES}, Delta={DELTA:.4f} yr")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    for rep in range(N_REPS):
        t0 = time.time()

        bases = {
            bt: make_basis(bt, np.random.default_rng(WEIGHT_SEED_BASE + rep))
            for bt in BASIS_TYPES
        }

        ref_params = hhk_params(BETAS[0])
        train_paths_ref = simulate_hhk(
            np.random.default_rng(TRAIN_SEED_BASE + rep), ref_params,
            n_paths=M, n_steps=N_STEPS, maturity=MATURITY, n_dims=D,
        )
        eval_paths_ref = simulate_hhk(
            np.random.default_rng(EVAL_SEED_BASE + rep), ref_params,
            n_paths=M, n_steps=N_STEPS, maturity=MATURITY, n_dims=D,
        )
        f = seasonality(train_paths_ref.t, ref_params)
        S_train_nojump = np.exp(f[:, None, None] + train_paths_ref.Z)
        S_eval_nojump = np.exp(f[:, None, None] + eval_paths_ref.Z)

        price_nojump = {}
        for basis_type in BASIS_TYPES:
            basis = bases[basis_type]
            policy = fit_policy(
                S_train=S_train_nojump, regression_state_train=S_train_nojump, contract=CONTRACT,
                aggregate=max_aggregation, basis=basis, fit=least_squares, alpha=ALPHA, train_itm_only=False,
            )
            result = evaluate_policy(
                policy, S_eval=S_eval_nojump, regression_state_eval=S_eval_nojump,
                contract=CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
            )
            price_nojump[basis_type] = result["v0"]
            print(f"  rep {rep}: {basis_type} no-jump baseline done ({time.time() - t0:.1f}s elapsed)")

        for beta in BETAS:
            params = hhk_params(beta)
            train_paths = simulate_hhk(
                np.random.default_rng(TRAIN_SEED_BASE + rep), params,
                n_paths=M, n_steps=N_STEPS, maturity=MATURITY, n_dims=D,
            )
            eval_paths = simulate_hhk(
                np.random.default_rng(EVAL_SEED_BASE + rep), params,
                n_paths=M, n_steps=N_STEPS, maturity=MATURITY, n_dims=D,
            )
            half_life_over_delta = (np.log(2.0) / beta) / DELTA

            for basis_type in BASIS_TYPES:
                basis = bases[basis_type]
                t_run = time.time()
                policy = fit_policy(
                    S_train=train_paths.S, regression_state_train=train_paths.S, contract=CONTRACT,
                    aggregate=max_aggregation, basis=basis, fit=least_squares, alpha=ALPHA, train_itm_only=False,
                )
                result = evaluate_policy(
                    policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
                    contract=CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
                )
                duration_sec = time.time() - t_run
                price_with = result["v0"]
                diff = price_with - price_nojump[basis_type]
                rel_pct = 100.0 * diff / price_nojump[basis_type]

                append_row({
                    "rep": rep, "n_dims": D, "beta": beta, "half_life_over_delta": half_life_over_delta,
                    "basis_type": basis_type,
                    "price_with_spikes": price_with, "price_no_spikes": price_nojump[basis_type],
                    "diff": diff, "rel_pct": rel_pct,
                    "duration_sec": duration_sec,
                })
                print(f"  rep {rep}: beta={beta} {basis_type} rel={rel_pct:.2f}% done ({time.time() - t0:.1f}s elapsed)")

        print(f"rep {rep + 1:2d}/{N_REPS} done ({time.time() - t0:.1f}s elapsed)\n")

    print("done")
