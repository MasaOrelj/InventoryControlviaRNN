"""Joint Laguerre-vs-RNN experiment (see conversation): for each basis
(Laguerre degree 2, RNN K-1=30), each dimension d in {1,10,25}, each swing-
rights count L in {1,5,10,25,40}, run 40 repetitions using the SINGLE
regularization value settled on for that (basis, d, L) cell (the table
pasted into the conversation) -- no more full lambda grids, regularization
choice is now fixed.

Rerun from scratch (not reusing any earlier duration_sec data) because
core/Regression.py's separate-mode backward induction was optimized this
session (duplicate beta1/beta0 fits eliminated -- see conversation, proven
bit-for-bit identical prices, only faster) AFTER all the earlier L-sweep
experiments were run. Their price/SD numbers are still valid, but their
duration_sec numbers reflect the OLD, slower code and would misrepresent
current runtime, which is itself one of this experiment's deliverables.

Seeds: eval_rng depends only on n_dims (ONE fixed evaluation sample per
dimension, shared across every L and both bases). train_rng depends only on
(n_dims, rep) -- NOT on L or basis -- so the SAME 40 training draws are
reused across every L and both bases at a given dimension, per the
conversation's explicit methodology. weight_rng (RNN only) depends on
(n_dims, rep) too, matching every prior RNN script's convention.

Execution order is cheapest-to-most-expensive (RNN entirely, then Laguerre
d=1, d=10, d=25 last) -- Laguerre d=25 alone (351 features per fit at
degree-2/d=25) accounts for ~9.6 of this run's ~11.3 total estimated hours,
so front-loading everything else means a useful, nearly-complete dataset
exists early even if the long tail needs to be checked on later.

Run: python -m scripts.Joint_LSweep_40Rep_Experiment
"""
import functools
import os
import time

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis, make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares, ridge_regression
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "joint_lsweep_40rep_experiment.csv")

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

DIMS = [1, 10, 25]
L_VALUES = [1, 5, 10, 25, 40]
N_REPS = 40
N_HIDDEN = 30
LAGUERRE_DEGREE = 2
LAGUERRE_K = 1.0

EVAL_SEED = 500_000
WEIGHT_SEED_BASE = 100_000

LAGUERRE_LAMBDA = {
    (1, 1): "plain", (1, 5): "plain", (1, 10): "plain", (1, 25): "plain", (1, 40): "plain",
    (10, 1): 0.05, (10, 5): 0.05, (10, 10): 0.05, (10, 25): 0.01, (10, 40): 0.01,
    (25, 1): 1.0, (25, 5): 1.0, (25, 10): 1.0, (25, 25): 0.1, (25, 40): 0.01,
}
RNN_LAMBDA = {
    (1, 1): 0.01, (1, 5): 0.01, (1, 10): 0.01, (1, 25): 0.01, (1, 40): 0.01,
    (10, 1): 1.0, (10, 5): 1.0, (10, 10): 0.1, (10, 25): 0.01, (10, 40): 0.01,
    (25, 1): 1.0, (25, 5): 1.0, (25, 10): 1.0, (25, 25): 1.0, (25, 40): 0.1,
}


def _base_row(basis_type: str, n_dims: int, contract: SwingContract) -> dict:
    row = {
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
        "basis_type": basis_type,
    }
    if basis_type == "laguerre":
        row.update({"basis_degree": LAGUERRE_DEGREE, "basis_K": LAGUERRE_K})
    else:
        row.update({"basis_n_hidden": N_HIDDEN, "basis_activation": "tanh"})
    return row


def _fit_config(lam):
    if lam == "plain":
        return "plain", least_squares, None
    return "ridge", functools.partial(ridge_regression, ridge_lambda=lam), lam


def run_cell(basis_type: str, n_dims: int, L: int, lam, n_reps: int = N_REPS, rep_start: int = 0) -> None:
    contract = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=L)
    fit_type, fit, ridge_lambda = _fit_config(lam)
    base_row = _base_row(basis_type, n_dims, contract)

    eval_rng = np.random.default_rng(EVAL_SEED + n_dims)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)

    t0 = time.time()
    for rep in range(rep_start, n_reps):
        train_rng = np.random.default_rng(1000 * n_dims + rep)
        train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)

        if basis_type == "laguerre":
            basis = WeightedLaguerreBasis(n_dims=n_dims, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)
        else:
            weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + n_dims * 1000 + rep)
            basis = make_random_features_basis(weight_rng, n_dims=n_dims, n_hidden=N_HIDDEN)

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

    print(f"  {basis_type:8s} d={n_dims:2d} L={L:2d} lambda={lam!s:>6}: reps {rep_start}-{n_reps-1} done ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    print(f"Joint L-sweep, 40 reps: dims={DIMS}, L={L_VALUES}, bases=[rnn(K-1={N_HIDDEN}), laguerre(deg={LAGUERRE_DEGREE})]")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    t_start = time.time()

    print("=== RNN ===")
    for d in DIMS:
        for L in L_VALUES:
            run_cell("rnn", d, L, RNN_LAMBDA[(d, L)])

    print("=== Laguerre d=1, d=10 ===")
    for d in [1, 10]:
        for L in L_VALUES:
            run_cell("laguerre", d, L, LAGUERRE_LAMBDA[(d, L)])

    print("=== Laguerre d=25 (20 reps for now -- more to be added later) ===")
    for L in L_VALUES:
        run_cell("laguerre", 25, L, LAGUERRE_LAMBDA[(25, L)], n_reps=20)

    print(f"\ndone, total {time.time()-t_start:.1f}s")
