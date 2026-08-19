"""Experiment: basis function selection -- polynomial (degree 2, 3) vs weighted
Laguerre (degree 2, 3) -- for the electricity swing option, across market
dimensions d in {1, 10, 25}. Same market/contract as State_Representation_Experiment.py:
double-jump HHK, symmetric contract centered near the median price (see
conversation for why symmetry still makes sense for a basis-function
comparison). State input fixed to (S_n, I_n); regression fixed to plain least
squares (no ridge), per the experiment's own scope.

CLAUDE.md's "Reported standard deviation" methodology: for each n_dims, ONE
evaluation sample is simulated up front, from a FIXED seed (EVAL_SEED + n_dims),
and reused for every repetition at that dimension. Each repetition draws a
NEW, independent training-only sample (seed = 1000*n_dims + rep) and fits a
policy against it (fit_policy), then evaluates that policy against the SAME
fixed evaluation sample (evaluate_policy). This isolates policy-fit
variability (component A) in the between-rep spread of v0, uncontaminated by
evaluation Monte Carlo noise (component B) -- unlike price_swing's
single-seed-for-everything convention, where both halves are redrawn together
each rep. The CLT confidence interval / multi-eval-sample consistency check
(component B's own diagnostics) are not computed here yet.

poly deg3 at d=25 (3276 features, vs M_t=5000 training paths) is skipped:
computationally infeasible (multi-hour+ per rep) and statistically fragile
even if computed (near-interpolating, unregularized) -- see conversation.
laguerre deg3 at d=25 is skipped for the identical reason: with cross terms
(see Basis_Functions.py), laguerre's feature count now exactly matches
poly's at the same (degree, n_dims), so the same 3276-feature blowup applies.

Every individual run is appended as its own row to
results/basis_function_experiment.csv (same schema as
state_representation_experiment.csv), for analysis in R.

Run: python -m scripts.Basis_Function_Experiment [n_samples]
    python -m scripts.Basis_Function_Experiment          # M_t=M_e=5000 (default)
    python -m scripts.Basis_Function_Experiment 20000     # M_t=M_e=20000, logs to its own suffixed CSV
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np

from core.Basis_Functions import PolynomialBasis, WeightedLaguerreBasis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from scripts.Experiment_Log import append_result_row
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares
from core.Swing import SwingContract


def _csv_path(n_samples: int) -> str:
    """Default sample size (5000) keeps the original, unsuffixed filename for
    backward compatibility; any other sample size gets its own file, so
    different-M_t/M_e runs never mix in one CSV."""
    suffix = "" if n_samples == 5_000 else f"_n{n_samples}"
    return os.path.join(os.path.dirname(__file__), "..", "results", f"basis_function_experiment{suffix}.csv")

# Same double-jump market and symmetric contract as State_Representation_Experiment.py
# -- asymmetric in both lam and mu: down-jumps are both less frequent AND
# smaller on average than up-jumps -- user's explicit choice (see conversation).
MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=40.0,
    lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4,
)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)

R = 0.02
MATURITY = 1.0
N_STEPS = 50
DEFAULT_N_SAMPLES = 5_000   # training AND evaluation sample size, unless overridden on the command line
N_REPS = 5

# Fixed per n_dims, distinct from the training seeds (1000*n_dims + rep, rep
# in 0..4) -- one evaluation sample per dimension, shared by every repetition.
EVAL_SEED = 500_000

# WeightedLaguerreBasis's own input-rescaling K -- unrelated to CONTRACT.K (the
# strike). K=1.0 is the class default (the reference implementation's
# un-rescaled form). An earlier version of this script set K=100 to
# compensate for RAW (unstandardized) price levels ~100 -- K=1 on raw prices
# badly underflowed exp(-S/(2*K)) (93% of observations gave a weight <
# 1e-10). Now that price_swing standardizes the state (scale-only, x/std)
# before build_features, that rescaling job is already done: K=100 stacked on
# TOP of an already-standardized input (~O(1-10)) instead flattens the weight
# to ~1 everywhere (verified empirically: mean weight ~0.98 across the full
# price range, effectively discarding the weighting -- this is what caused
# the "difficulties" noticed when running Laguerre). K=1.0 is correct again
# once standardization exists (verified: weight then spans ~1e-5 to ~0.9,
# still with zero underflow).
LAGUERRE_K = 1.0

DIMS = [1, 10, 25]
BASIS_CONFIGS = [("poly", 2), ("poly", 3), ("laguerre", 2), ("laguerre", 3)]
# (basis_type, degree, n_dims): infeasible, see module docstring. laguerre now
# has cross terms (see Basis_Functions.py), so its feature count exactly
# matches poly's at the same (degree, n_dims) -- laguerre deg3 at d=25 is
# therefore equally infeasible and must be skipped too.
SKIP = {("poly", 3, 25), ("laguerre", 3, 25)}

ALPHA = np.exp(-R * MATURITY / N_STEPS)


def _base_row(n_samples: int) -> dict:
    return {
        "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
        "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
        "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
        "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp, "f_period": MARKET_PARAMS.f_period,
        "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS,
        "n_paths_train": n_samples, "n_paths_eval": n_samples,
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
        return WeightedLaguerreBasis(n_dims=n_dims, degree=degree, K=LAGUERRE_K)
    raise ValueError(basis_type)


if __name__ == "__main__":
    N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_SAMPLES
    CSV_PATH = _csv_path(N_SAMPLES)
    base_row = _base_row(N_SAMPLES)

    print(f"K={CONTRACT.K}, L={CONTRACT.L}, q=({CONTRACT.q_min},{CONTRACT.q_tilde},{CONTRACT.q_max}), "
          f"r={R}, T={MATURITY}, N={N_STEPS}, M_t={N_SAMPLES} (new seed per rep), "
          f"M_e={N_SAMPLES} (fixed across all reps), {N_REPS} reps")
    print(f"dims={DIMS}, basis_configs={BASIS_CONFIGS}, skipping={SKIP}")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    results = defaultdict(list)
    t0 = time.time()

    for n_dims in DIMS:
        # ONE evaluation sample per dimension, fixed across every repetition
        # at that dimension (see module docstring).
        eval_rng = np.random.default_rng(EVAL_SEED + n_dims)
        eval_paths = simulate_hhk(
            eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
        )

        for rep in range(N_REPS):
            train_rng = np.random.default_rng(1000 * n_dims + rep)
            train_paths = simulate_hhk(
                train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
            )

            for basis_type, degree in BASIS_CONFIGS:
                if (basis_type, degree, n_dims) in SKIP:
                    continue

                basis = make_basis(basis_type, degree, n_dims)

                t_run = time.time()
                policy = fit_policy(
                    S_train=train_paths.S, regression_state_train=train_paths.S, contract=CONTRACT,
                    aggregate=max_aggregation, basis=basis, fit=least_squares, alpha=ALPHA,
                    train_itm_only=False,
                )
                result = evaluate_policy(
                    policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
                    contract=CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
                )
                duration_sec = time.time() - t_run
                results[(n_dims, basis_type, degree)].append(result["v0"])

                basis_K = LAGUERRE_K if basis_type == "laguerre" else ""
                append_result_row(CSV_PATH, {
                    **base_row,
                    "n_dims": n_dims, "rep": rep,
                    "basis_type": basis_type, "basis_degree": degree, "basis_K": basis_K,
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
