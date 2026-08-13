"""Consistency check for the CLT shortcut: fix ONE policy per (dimension,
basis) cell (same M_t=10000, same seeds as Evaluation_Sample_Size_Experiment.py
-- literally the same policies), then evaluate each against 100 DIFFERENT
evaluation samples of fixed size M_e=10000. Compares:
  - empirical mean/SD of v0 across the 100 draws (a direct, assumption-free
    measurement of how much the reported price actually varies with the
    evaluation sample)
  - the CLT-predicted SE from a single draw's path_level_sd/sqrt(M_e) (the
    cheap shortcut normally used for one real run -- see
    Evaluation_Sample_Size_Experiment.py)
If they agree, the CLT shortcut is trustworthy at this M_e; if they diverge,
it isn't (see conversation: this was exactly the case at M_e=5000, and much
less so by M_e=10000-15000).

Run: python -m scripts.Evaluation_Consistency_Check
"""
import csv
import os
import time

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis, make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares
from core.Swing import SwingContract

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "evaluation_consistency_check.csv")

MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=200.0,
    lam_up=0.5, mu_up=0.6, lam_down=0.3, mu_down=0.4,
)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
DIMS = [1, 10]
BASIS_TYPES = ["rnn", "laguerre"]
LAGUERRE_K = 1.0
M_T = 10_000
M_E = 10_000
N_DRAWS = 100
ALPHA = np.exp(-R * MATURITY / N_STEPS)

EVAL_SEED_BASE = 800_000   # distinct from Evaluation_Sample_Size_Experiment.py's 600_000 range

FIELDNAMES = [
    "n_dims", "state_input", "basis_type", "basis_degree", "n_paths_train", "n_paths_eval",
    "draw", "v0", "path_level_sd", "duration_sec",
]


def append_row(row: dict) -> None:
    is_new_file = not os.path.exists(CSV_PATH)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def make_basis(basis_type: str, n_dims: int):
    if basis_type == "laguerre":
        return WeightedLaguerreBasis(n_dims=n_dims, degree=2, K=LAGUERRE_K), 2
    weight_rng = np.random.default_rng(42 + n_dims)
    return make_random_features_basis(weight_rng, n_dims=n_dims, n_hidden=20), None


if __name__ == "__main__":
    print(f"M_t={M_T}, M_e={M_E}, {N_DRAWS} draws, dims={DIMS}, basis={BASIS_TYPES}, state=S, fit=plain")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    for n_dims in DIMS:
        train_rng = np.random.default_rng(1000 * n_dims)   # same seed as Evaluation_Sample_Size_Experiment.py
        train_paths = simulate_hhk(
            train_rng, MARKET_PARAMS, n_paths=M_T, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
        )

        for basis_type in BASIS_TYPES:
            basis, basis_degree = make_basis(basis_type, n_dims)   # same weight seed too -> identical policy
            policy = fit_policy(
                S_train=train_paths.S, regression_state_train=train_paths.S, contract=CONTRACT,
                aggregate=max_aggregation, basis=basis, fit=least_squares, alpha=ALPHA, train_itm_only=False,
            )

            t0 = time.time()
            v0s = []
            for draw in range(N_DRAWS):
                eval_rng = np.random.default_rng(EVAL_SEED_BASE + n_dims * 1_000_000 + draw)
                eval_paths = simulate_hhk(
                    eval_rng, MARKET_PARAMS, n_paths=M_E, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
                )
                t_run = time.time()
                result = evaluate_policy(
                    policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
                    contract=CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
                )
                duration_sec = time.time() - t_run
                path_level_sd = result["cashflows"].std(ddof=1)
                v0s.append(result["v0"])

                append_row({
                    "n_dims": n_dims, "state_input": "S", "basis_type": basis_type, "basis_degree": basis_degree,
                    "n_paths_train": M_T, "n_paths_eval": M_E,
                    "draw": draw, "v0": result["v0"], "path_level_sd": path_level_sd,
                    "duration_sec": duration_sec,
                })
                if (draw + 1) % 25 == 0:
                    print(f"  d={n_dims} {basis_type}: {draw + 1}/{N_DRAWS} draws done "
                          f"({time.time() - t0:.1f}s elapsed)")

            v0s = np.array(v0s)
            print(f"d={n_dims:2d} {basis_type:9s}: empirical mean={v0s.mean():.4f}  "
                  f"empirical SD={v0s.std(ddof=1):.4f}\n")
