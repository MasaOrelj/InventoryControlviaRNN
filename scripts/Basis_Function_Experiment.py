"""Experiment: basis function selection -- polynomial (degree 2, 3) vs weighted
Laguerre (degree 2, 3) -- for the electricity swing option, across market
dimensions d in {1, 10, 25}. Same market/contract as State_Representation_Experiment.py:
double-jump HHK, symmetric contract centered near the median price (see
conversation for why symmetry still makes sense for a basis-function
comparison). State input fixed to (S_n, I_n); regression fixed to plain least
squares (no ridge), per the experiment's own scope.

poly deg3 at d=25 (3276 features, vs M_t=5000 training paths) is skipped:
computationally infeasible (multi-hour+ per rep) and statistically fragile
even if computed (near-interpolating, unregularized) -- see conversation.

Every individual run is appended as its own row to
results/basis_function_experiment.csv (same schema as
state_representation_experiment.csv), for analysis in R.

Run: python -m scripts.Basis_Function_Experiment
"""

import os
import time
from collections import defaultdict

import numpy as np

from Basis_Functions import PolynomialBasis, WeightedLaguerreBasis
from Electricity_Market_Model import HHKParams, simulate_hhk
from Experiment_Log import append_result_row
from Payoff_Aggregation import sum_aggregation
from Regression import least_squares, price_swing
from Swing import SwingContract

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "basis_function_experiment.csv")

# Same double-jump market and symmetric contract as State_Representation_Experiment.py.
MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=200.0,
    lam_up=5.0, mu_up=0.4, lam_down=5.0, mu_down=0.6,
)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)

R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_PATHS = 10_000
N_REPS = 5

DIMS = [1, 10, 25]
BASIS_CONFIGS = [("poly", 2), ("poly", 3), ("laguerre", 2), ("laguerre", 3)]
SKIP = {("poly", 3, 25)}   # (basis_type, degree, n_dims): infeasible, see module docstring

ALPHA = np.exp(-R * MATURITY / N_STEPS)

BASE_ROW = {
    "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
    "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
    "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
    "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp, "f_period": MARKET_PARAMS.f_period,
    "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS, "n_paths": N_PATHS,
    "K": CONTRACT.K, "q_min": CONTRACT.q_min, "q_max": CONTRACT.q_max,
    "q_tilde": CONTRACT.q_tilde, "L": CONTRACT.L,
    "regression_mode": "per-level",
    "state_input": "S",
    "fit_type": "plain",
}


def make_basis(basis_type: str, degree: int, n_dims: int):
    if basis_type == "poly":
        return PolynomialBasis(n_dims=n_dims, degree=degree)
    if basis_type == "laguerre":
        return WeightedLaguerreBasis(n_dims=n_dims, degree=degree)
    raise ValueError(basis_type)


if __name__ == "__main__":
    print(f"K={CONTRACT.K}, L={CONTRACT.L}, q=({CONTRACT.q_min},{CONTRACT.q_tilde},{CONTRACT.q_max}), "
          f"r={R}, T={MATURITY}, N={N_STEPS}, M={N_PATHS} ({N_PATHS//2} train/{N_PATHS//2} eval), {N_REPS} reps")
    print(f"dims={DIMS}, basis_configs={BASIS_CONFIGS}, skipping={SKIP}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    results = defaultdict(list)
    t0 = time.time()

    for n_dims in DIMS:
        for rep in range(N_REPS):
            path_rng = np.random.default_rng(1000 * n_dims + rep)
            paths = simulate_hhk(
                path_rng, MARKET_PARAMS, n_paths=N_PATHS, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
            )

            for basis_type, degree in BASIS_CONFIGS:
                if (basis_type, degree, n_dims) in SKIP:
                    continue

                basis = make_basis(basis_type, degree, n_dims)

                t_run = time.time()
                result = price_swing(
                    S=paths.S, regression_state=paths.S, contract=CONTRACT,
                    aggregate=sum_aggregation, basis=basis, fit=least_squares, alpha=ALPHA,
                    train_itm_only=False,
                )
                duration_sec = time.time() - t_run
                results[(n_dims, basis_type, degree)].append(result["v0"])

                append_result_row(CSV_PATH, {
                    **BASE_ROW,
                    "n_dims": n_dims, "rep": rep,
                    "basis_type": basis_type, "basis_degree": degree,
                    "price": result["v0"], "duration_sec": duration_sec,
                })

            print(f"  d={n_dims:>2} rep {rep + 1}/{N_REPS} done ({time.time() - t0:.1f}s elapsed)")

    print()
    for n_dims in DIMS:
        for basis_type, degree in BASIS_CONFIGS:
            if (basis_type, degree, n_dims) in SKIP:
                continue
            prices = np.array(results[(n_dims, basis_type, degree)])
            print(f"d={n_dims:>2}  basis={basis_type:>9}  degree={degree}: "
                  f"mean={prices.mean():.4f}  std={prices.std(ddof=1):.4f}")
