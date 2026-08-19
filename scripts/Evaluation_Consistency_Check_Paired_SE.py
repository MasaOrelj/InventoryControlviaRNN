"""Paired-difference companion to Evaluation_Consistency_Check.py: for the
SAME fixed RNN/Laguerre policies (identical seeds -- same M_t=10000 training
draw), evaluated on the SAME single shared M_e=10000 evaluation sample
(draw=0, matching Evaluation_Consistency_Check.py's own "first draw" CLT-SE
convention), compute the per-path cashflow DIFFERENCE (RNN - Laguerre) and
its CLT standard error.

Mentor comment (Table 2.6): sharing the evaluation sample doesn't lower
either price's own marginal SE, but it makes the two errors correlated, so
the SE of the DIFFERENCE is far smaller than either marginal SE (or a naive
independent-errors combination) would suggest. This script produces the
"single fixed run" companion to the empirical (100 shared draws) paired SD
already computable directly from evaluation_consistency_check.csv -- the two
should roughly agree, the same way each basis's own Empirical SD already
agrees with its single-draw CLT SE in that table.

Run: python -m scripts.Evaluation_Consistency_Check_Paired_SE
"""
import os

import numpy as np

import scripts.Evaluation_Consistency_Check as ECC
from core.Electricity_Market_Model import simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "evaluation_consistency_check_paired_se.csv",
)

DRAW = 0   # matches evaluation_consistency_check_table.R's "first_draw_se" convention

if __name__ == "__main__":
    rows = []
    for n_dims in ECC.DIMS:
        train_rng = np.random.default_rng(1000 * n_dims)   # same seed as Evaluation_Consistency_Check.py
        train_paths = simulate_hhk(
            train_rng, ECC.MARKET_PARAMS, n_paths=ECC.M_T, n_steps=ECC.N_STEPS,
            maturity=ECC.MATURITY, n_dims=n_dims,
        )

        cashflows = {}
        for basis_type in ECC.BASIS_TYPES:
            basis, _ = ECC.make_basis(basis_type, n_dims)   # same weight seed -> identical policy
            policy = fit_policy(
                S_train=train_paths.S, regression_state_train=train_paths.S, contract=ECC.CONTRACT,
                aggregate=max_aggregation, basis=basis, fit=least_squares, alpha=ECC.ALPHA,
                train_itm_only=False,
            )
            eval_rng = np.random.default_rng(ECC.EVAL_SEED_BASE + n_dims * 1_000_000 + DRAW)
            eval_paths = simulate_hhk(
                eval_rng, ECC.MARKET_PARAMS, n_paths=ECC.M_E, n_steps=ECC.N_STEPS,
                maturity=ECC.MATURITY, n_dims=n_dims,
            )
            result = evaluate_policy(
                policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
                contract=ECC.CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ECC.ALPHA,
            )
            cashflows[basis_type] = result["cashflows"]

        diff = cashflows["rnn"] - cashflows["laguerre"]
        paired_se = diff.std(ddof=1) / np.sqrt(ECC.M_E)
        rows.append({"n_dims": n_dims, "draw": DRAW, "paired_first_draw_se": paired_se})
        print(f"d={n_dims}: first-draw paired SE (RNN - Laguerre) = {paired_se:.4f}")

    import csv
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_dims", "draw", "paired_first_draw_se"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {os.path.abspath(CSV_PATH)}")
