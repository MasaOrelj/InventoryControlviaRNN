"""Component-B demonstration: fix ONE policy per (dimension, basis) cell (fit
once, M_t=10000), then evaluate that SAME fixed policy against evaluation
samples of growing size M_e in {5000, 10000, 15000, 20000}, to show how the
CLT confidence interval (built from evaluate_policy's cashflows) tightens as
M_e grows -- SE scales as path-level SD / sqrt(M_e), so quadrupling M_e
should roughly halve SE.

Cells: (d, basis) in {1, 10} x {rnn, laguerre deg2}, state fixed to S, fit
fixed to plain least squares. This is purely about evaluation Monte Carlo noise
(component B) for a SINGLE fixed policy per cell -- no repeated training
seeds here, that's the separate, already-covered component-A question (see
State_Representation_Experiment.py / Basis_Function_Experiment.py).
Confidence intervals themselves are computed downstream (in R) from this
script's logged v0 / path-level SD / M_e columns, not here -- see CLAUDE.md's
clt_confidence_interval for the formula (Ṽ_0 ± z * path_level_sd/sqrt(M_e)).

Run: python -m scripts.Evaluation_Sample_Size_Experiment
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

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "evaluation_sample_size_experiment.csv")

MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=200.0,
    lam_up=0.5, mu_up=0.6, lam_down=0.3, mu_down=0.4,
)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
DIMS = [1, 10, 25]
BASIS_TYPES = ["rnn", "laguerre"]
LAGUERRE_K = 1.0
M_T = 10_000
M_E_VALUES = [5_000, 10_000, 15_000, 20_000]
ALPHA = np.exp(-R * MATURITY / N_STEPS)

EVAL_SEED_BASE = 600_000   # eval seed = EVAL_SEED_BASE + n_dims*100_000 + m_e -- one fresh sample per (d, M_e)

FIELDNAMES = [
    "n_dims", "state_input", "basis_type", "basis_degree", "n_paths_train", "n_paths_eval",
    "v0", "path_level_sd", "mc_se", "duration_sec",
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
    return make_random_features_basis(weight_rng, n_dims=n_dims, n_hidden=20), None   # default activation: tanh


if __name__ == "__main__":
    print(f"M_t={M_T}, M_e in {M_E_VALUES}, dims={DIMS}, basis={BASIS_TYPES}, state=S, fit=plain")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    for n_dims in DIMS:
        train_rng = np.random.default_rng(1000 * n_dims)
        train_paths = simulate_hhk(
            train_rng, MARKET_PARAMS, n_paths=M_T, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
        )

        for basis_type in BASIS_TYPES:
            basis, basis_degree = make_basis(basis_type, n_dims)

            t0 = time.time()
            policy = fit_policy(
                S_train=train_paths.S, regression_state_train=train_paths.S, contract=CONTRACT,
                aggregate=max_aggregation, basis=basis, fit=least_squares, alpha=ALPHA, train_itm_only=False,
            )
            print(f"d={n_dims} basis={basis_type}: policy fit in {time.time() - t0:.1f}s")

            for m_e in M_E_VALUES:
                t_run = time.time()
                eval_rng = np.random.default_rng(EVAL_SEED_BASE + n_dims * 100_000 + m_e)
                eval_paths = simulate_hhk(
                    eval_rng, MARKET_PARAMS, n_paths=m_e, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
                )
                result = evaluate_policy(
                    policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
                    contract=CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
                )
                cashflows = result["cashflows"]
                path_level_sd = cashflows.std(ddof=1)
                mc_se = path_level_sd / np.sqrt(m_e)
                duration_sec = time.time() - t_run

                append_row({
                    "n_dims": n_dims, "state_input": "S", "basis_type": basis_type, "basis_degree": basis_degree,
                    "n_paths_train": M_T, "n_paths_eval": m_e,
                    "v0": result["v0"], "path_level_sd": path_level_sd, "mc_se": mc_se,
                    "duration_sec": duration_sec,
                })
                print(f"  M_e={m_e:6d}: v0={result['v0']:.4f}  path_level_sd={path_level_sd:.4f}  "
                      f"mc_se={mc_se:.4f}  ({duration_sec:.1f}s)")
            print()
