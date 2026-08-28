"""Component-B CLT confidence intervals for the joint L-sweep table (see
conversation): for ONE fixed, already-fitted policy per (basis, d, L) cell
(rep=0, same seeds/lambda as Joint_LSweep_40Rep_Experiment.py), computes
core.Regression.clt_confidence_interval on evaluate_policy's own "cashflows"
array -- CLAUDE.md's Component B (evaluation Monte Carlo noise for a FIXED
policy), not the between-rep spread (Component A, already reported as the
table's SD column). Joint_LSweep_40Rep_Experiment.py only ever logged the
aggregate price per rep, discarding cashflows, so this can't be recovered
from the existing CSV -- has to be refit.

Run: python -m scripts.Joint_LSweep_CLT_CI
"""
import os
import time

import numpy as np
import pandas as pd

from core.Basis_Functions import WeightedLaguerreBasis, make_random_features_basis
from core.Electricity_Market_Model import simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import clt_confidence_interval, evaluate_policy, fit_policy
from core.Swing import SwingContract
from scripts.Joint_LSweep_40Rep_Experiment import (
    ALPHA, EVAL_SEED, LAGUERRE_DEGREE, LAGUERRE_K, LAGUERRE_LAMBDA, L_VALUES,
    MARKET_PARAMS, MATURITY, N_HIDDEN, N_SAMPLES, N_STEPS, RNN_LAMBDA,
    WEIGHT_SEED_BASE, _fit_config,
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "joint_lsweep_clt_ci.csv")
DIMS = [1, 10, 25]
REP = 0


def run_cell(basis_type: str, n_dims: int, L: int, lam) -> dict:
    contract = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=L)
    fit_type, fit, ridge_lambda = _fit_config(lam)

    eval_rng = np.random.default_rng(EVAL_SEED + n_dims)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)
    train_rng = np.random.default_rng(1000 * n_dims + REP)
    train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)

    if basis_type == "laguerre":
        basis = WeightedLaguerreBasis(n_dims=n_dims, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)
    else:
        weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + n_dims * 1000 + REP)
        basis = make_random_features_basis(weight_rng, n_dims=n_dims, n_hidden=N_HIDDEN)

    policy = fit_policy(
        S_train=train_paths.S, regression_state_train=train_paths.S, contract=contract,
        aggregate=max_aggregation, basis=basis, fit=fit, alpha=ALPHA, train_itm_only=False,
    )
    result = evaluate_policy(
        policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
        contract=contract, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
    )
    lo, hi = clt_confidence_interval(result["cashflows"], confidence=0.95)
    return {"n_dims": n_dims, "L": L, "basis_type": basis_type, "v0": result["v0"], "ci_lower": lo, "ci_upper": hi}


if __name__ == "__main__":
    rows = []
    t0 = time.time()
    for d in DIMS:
        for L in L_VALUES:
            rows.append(run_cell("laguerre", d, L, LAGUERRE_LAMBDA[(d, L)]))
            rows.append(run_cell("rnn", d, L, RNN_LAMBDA[(d, L)]))
        print(f"d={d} done ({time.time()-t0:.1f}s elapsed)")

    out = pd.DataFrame(rows)
    out.to_csv(CSV_PATH, index=False)
    print(f"wrote {os.path.abspath(CSV_PATH)}")
    print(out.to_string(index=False))
