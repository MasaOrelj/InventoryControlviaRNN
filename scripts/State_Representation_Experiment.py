"""Experiment: state representation (S,I) vs (Z,Y,I) for the electricity swing
option, crossed with basis (degree-2 polynomial vs RNN, K-1=20) and regression
(plain least squares vs ridge). d=1 pass done (see conversation); this is the
d=10 follow-up, at N_REPS=5 for a faster first look given poly+(Z,Y)'s much
larger feature count (C(22,2)=231) at this dimension.

Every repetition simulates ONE set of market paths, reused across all 8
(state, basis, fit) combinations for that repetition -- CLAUDE.md's "compare
on identical paths" rule, so differences reflect the method, not the RNG.

poly+(Z,Y) at d=10 (231 features) is far slower than the other 6 cells (fixed
RNN feature count; S alone is only 66 features), so BATCH selects a subset.
Args: batch [fast|slow|all], dimension d, number of repetitions -- e.g.:
    python -m scripts.State_Representation_Experiment all 1 10    # d=1, 10 reps, everything
    python -m scripts.State_Representation_Experiment fast 10 5   # d=10, 5 reps, fast cells
    python -m scripts.State_Representation_Experiment slow 10 5   # d=10, 5 reps, ZY+poly only

Every individual run (one repetition x one state/basis/fit combination) is
appended as its own row to results/state_representation_experiment.csv, for
further analysis outside Python (e.g. in R).
"""

import functools
import os
import sys
import time
from collections import defaultdict

import numpy as np

from Basis_Functions import PolynomialBasis, make_random_features_basis
from Electricity_Market_Model import HHKParams, simulate_hhk
from Experiment_Log import append_result_row
from Payoff_Aggregation import sum_aggregation
from Regression import least_squares, price_swing, ridge_regression
from Swing import SwingContract

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "state_representation_experiment.csv")

# Double-jump market (same params as scripts/plot_market_sanity_checks.py's
# double_jump variant) -- kept asymmetric deliberately: the market's up/down
# jump asymmetry is a real feature of electricity prices, not a nuisance
# parameter, unlike the swing contract's quantities (see conversation).
MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=200.0,
    lam_up=5.0, mu_up=0.4, lam_down=5.0, mu_down=0.6,
)

# Symmetric contract (q_max-q_tilde = q_tilde-q_min = 25) + strike centered on
# the empirical median of S at this horizon -- deliberately symmetric so as
# not to confound the state-representation/algorithm comparison with
# contract-driven asymmetry (see conversation).
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)

R = 0.02             # discount rate; not fixed anywhere for the swing case --
                      # using the legacy code's rate=0.02 (distinct from the
                      # max-call validation's paper-mandated r=0).
MATURITY = 1.0
N_STEPS = 50
N_PATHS = 10_000
RIDGE_LAMBDA = 1.0
# D and N_REPS are set from the command line (see __main__) -- d=1 uses all 10
# reps in one go; d=10's poly+(Z,Y) cell is slow enough to warrant fewer reps
# and the fast/slow BATCH split below.
DEFAULT_D = 10
DEFAULT_N_REPS = 5

# sum_aggregation (not max_aggregation, which was specifically a max-call
# validation artifact): for a portfolio of d delivery periods, one shared
# swing right adjusts all of them at once, so total profit is additive.

ALPHA = np.exp(-R * MATURITY / N_STEPS)

FITS = ["plain", "ridge"]
BATCHES = {
    "fast": [("S", "poly"), ("S", "rnn"), ("ZY", "rnn")],
    "slow": [("ZY", "poly")],
    "all": [("S", "poly"), ("S", "rnn"), ("ZY", "poly"), ("ZY", "rnn")],
}


def _base_row(d: int) -> dict:
    """Fields shared by every row this script writes -- everything that's
    fixed across the whole run, independent of which (state, basis, fit) cell."""
    return {
        "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
        "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
        "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
        "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp, "f_period": MARKET_PARAMS.f_period,
        "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS, "n_paths": N_PATHS, "n_dims": d,
        "K": CONTRACT.K, "q_min": CONTRACT.q_min, "q_max": CONTRACT.q_max,
        "q_tilde": CONTRACT.q_tilde, "L": CONTRACT.L,
        "regression_mode": "per-level",   # joint mode not implemented yet (see CLAUDE.md)
    }


def leaky_relu(x: np.ndarray, negative_slope: float = 0.5) -> np.ndarray:
    return np.where(x > 0.0, x, negative_slope * x)


def _regression_state(Z: np.ndarray, Y: np.ndarray, S: np.ndarray, state_input: str) -> np.ndarray:
    if state_input == "S":
        return S
    if state_input == "ZY":
        return np.concatenate([Z, Y], axis=-1)
    raise ValueError(state_input)


# Fixed (not enumeration-order-dependent) so RNN weight seeds stay identical
# regardless of which BATCH subset is running -- fast/slow/all must agree.
STATE_INDEX = {"S": 0, "ZY": 1}

if __name__ == "__main__":
    batch = sys.argv[1] if len(sys.argv) > 1 else "all"
    D = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_D
    N_REPS = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N_REPS
    state_basis_pairs = BATCHES[batch]
    base_row = _base_row(D)

    print(f"batch={batch}: {state_basis_pairs}")
    print(f"d={D}, K={CONTRACT.K}, L={CONTRACT.L}, q=({CONTRACT.q_min},{CONTRACT.q_tilde},{CONTRACT.q_max}), "
          f"r={R}, T={MATURITY}, N={N_STEPS}, M={N_PATHS} ({N_PATHS//2} train/{N_PATHS//2} eval), {N_REPS} reps")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    results = defaultdict(list)
    t0 = time.time()

    for rep in range(N_REPS):
        path_rng = np.random.default_rng(rep)   # same paths across batches for a given rep
        paths = simulate_hhk(path_rng, MARKET_PARAMS, n_paths=N_PATHS, n_steps=N_STEPS, maturity=MATURITY, n_dims=D)

        for state_input, basis_name in state_basis_pairs:
            regression_state = _regression_state(paths.Z, paths.Y, paths.S, state_input)
            k = regression_state.shape[-1]

            if basis_name == "poly":
                basis = PolynomialBasis(n_dims=k, degree=2)
                basis_row = {"basis_type": "poly", "basis_degree": 2}
            else:
                weight_rng = np.random.default_rng(100_000 + rep * 10 + STATE_INDEX[state_input])
                basis = make_random_features_basis(weight_rng, n_dims=k, n_hidden=20, activation=leaky_relu)
                basis_row = {"basis_type": "rnn", "basis_n_hidden": 20, "basis_activation": "leaky_relu_0.5"}

            for fit_name in FITS:
                if fit_name == "plain":
                    fit = least_squares
                    fit_row = {"fit_type": "plain"}
                else:
                    fit = functools.partial(ridge_regression, ridge_lambda=RIDGE_LAMBDA)
                    fit_row = {"fit_type": "ridge", "ridge_lambda": RIDGE_LAMBDA}

                t_run = time.time()
                result = price_swing(
                    S=paths.S, regression_state=regression_state, contract=CONTRACT,
                    aggregate=sum_aggregation, basis=basis, fit=fit, alpha=ALPHA, train_itm_only=False,
                )
                duration_sec = time.time() - t_run
                results[(state_input, basis_name, fit_name)].append(result["v0"])

                append_result_row(CSV_PATH, {
                    **base_row, **basis_row, **fit_row,
                    "rep": rep, "state_input": state_input,
                    "price": result["v0"], "duration_sec": duration_sec,
                })

        print(f"  rep {rep + 1}/{N_REPS} done ({time.time() - t0:.1f}s elapsed)")

    print()
    for state_input, basis_name in state_basis_pairs:
        for fit_name in FITS:
            prices = np.array(results[(state_input, basis_name, fit_name)])
            print(f"state={state_input:>2}  basis={basis_name:>4}  fit={fit_name:>5}: "
                  f"mean={prices.mean():.4f}  std={prices.std(ddof=1):.4f}")
