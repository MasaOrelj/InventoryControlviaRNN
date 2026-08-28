"""Quick regularization tuning pass for the "rare spikes" diagnostic market
(see conversation): lam_up=1.5, lam_down=0.5 (2 jumps/year per dimension,
fixed and identical across d -- unlike the earlier rejected "scale lambda
down with d" idea, this keeps each dimension's own spike behavior physically
consistent regardless of portfolio size), mu_up/mu_down unchanged from the
standard calibration (only jump FREQUENCY is being thinned, not jump SIZE).

Small lambda grid (plain, 0.01, 0.1, 1.0), 5 reps, at L=10 (default
contract), RNN (K-1=30) and Laguerre (degree 2) at d=10 and d=25; RNN also
at d=50 (Laguerre skipped there -- infeasible feature count regardless of
spike rate, see conversation). Purpose: since the spikes-off ablation
earlier this session showed optimal df(lambda) can shift substantially when
the jump process itself changes, this checks whether the standard
(spikes-on-calibrated) lambda choices still hold under the rarer-spike
market before running the real 10-rep price/SD comparison across d.

Run: python -m scripts.RareSpikes_Lambda_Sweep_Experiment
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

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rare_spikes_lambda_sweep_experiment.csv")

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=1.5, mu_up=0.6, lam_down=0.5, mu_down=0.4)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

N_HIDDEN = 30
LAGUERRE_DEGREE = 2
LAGUERRE_K = 1.0
LAMBDA_VALUES = [0.01, 0.1, 1.0]
N_REPS = 5

RNN_DIMS = [10, 25, 50]
LAGUERRE_DIMS = [10, 25]

EVAL_SEED = 500_000
WEIGHT_SEED_BASE = 100_000


def _base_row(basis_type: str, n_dims: int) -> dict:
    row = {
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
        "basis_type": basis_type,
    }
    if basis_type == "laguerre":
        row.update({"basis_degree": LAGUERRE_DEGREE, "basis_K": LAGUERRE_K})
    else:
        row.update({"basis_n_hidden": N_HIDDEN, "basis_activation": "tanh"})
    return row


def run_dim(basis_type: str, n_dims: int) -> None:
    base_row = _base_row(basis_type, n_dims)
    fit_configs = [("plain", least_squares, None)] + [
        ("ridge", functools.partial(ridge_regression, ridge_lambda=lam), lam) for lam in LAMBDA_VALUES
    ]

    eval_rng = np.random.default_rng(EVAL_SEED + n_dims)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)

    t0 = time.time()
    for rep in range(N_REPS):
        train_rng = np.random.default_rng(1000 * n_dims + rep)
        train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)

        if basis_type == "laguerre":
            basis = WeightedLaguerreBasis(n_dims=n_dims, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)
        else:
            weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + n_dims * 1000 + rep)
            basis = make_random_features_basis(weight_rng, n_dims=n_dims, n_hidden=N_HIDDEN)

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

        print(f"  {basis_type:8s} d={n_dims:2d} rep {rep + 1}/{N_REPS} done ({time.time() - t0:.1f}s elapsed)")


if __name__ == "__main__":
    print(f"Rare-spikes lambda sweep: lam_up={MARKET_PARAMS.lam_up}, lam_down={MARKET_PARAMS.lam_down}, "
          f"lambda grid=[plain]+{LAMBDA_VALUES}, {N_REPS} reps")
    print(f"RNN dims={RNN_DIMS}, Laguerre dims={LAGUERRE_DIMS}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    for d in RNN_DIMS:
        run_dim("rnn", d)
    for d in LAGUERRE_DIMS:
        run_dim("laguerre", d)

    print("\ndone")
