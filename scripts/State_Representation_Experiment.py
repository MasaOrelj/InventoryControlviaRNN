"""Experiment: state representation (S,I) vs (Z,Y,I) for the electricity swing
option, crossed with basis (degree-2 polynomial vs RNN, K-1=20) and regression
(plain least squares vs ridge). d=1 uses all 10 reps in one go; d=10 and d=25
use N_REPS=5, split into fast/slow BATCHes (see below) given poly+(Z,Y)'s much
larger feature count at those dimensions (C(22,2)=231 at d=10, C(52,2)=1326 at
d=25 -- the d=25 ZY+poly+plain cell alone is ~114 min for 5 reps, measured by
timing a single fit at that feature count and extrapolating by fit count; see
conversation. Consider ridge-only or fewer reps for that cell if it's ever a
bottleneck in practice.).

CLAUDE.md's "Reported standard deviation" methodology: ONE evaluation sample
is simulated up front, from a FIXED seed (EVAL_SEED), and reused for every
repetition. Each repetition draws a NEW, independent training-only sample
(seed = rep) and fits a policy against it (fit_policy), then evaluates that
policy against the SAME fixed evaluation sample (evaluate_policy). This
isolates policy-fit variability (component A) in the between-rep spread of
v0, uncontaminated by evaluation Monte Carlo noise (component B) -- unlike
price_swing's single-seed-for-everything convention, where both halves are
redrawn together each rep. The CLT confidence interval / multi-eval-sample
consistency check (component B's own diagnostics) are not computed here yet.

Every repetition's training draw is reused across all 8 (state, basis, fit)
combinations for that repetition -- CLAUDE.md's "compare on identical paths"
rule, so differences reflect the method, not the RNG. The evaluation sample is
shared across every repetition AND every combination, for the same reason.

poly+(Z,Y) is far slower than the other 6 cells at any dimension (fixed RNN
feature count regardless of d), so BATCH selects a subset -- run "fast" and
"slow" as separate invocations at the same (d, N_REPS) to cover all 8 cells.
Args: batch [fast|slow|all], dimension d, number of repetitions, M_t=M_e
sample size (optional, default 5000 -- a non-default value logs to its own
suffixed CSV, e.g. state_representation_experiment_n20000.csv, so different
sample sizes never mix in one file) -- e.g.:
    python -m scripts.State_Representation_Experiment all 1 10          # d=1, 10 reps, everything, M=5000
    python -m scripts.State_Representation_Experiment fast 10 5         # d=10, 5 reps, fast cells
    python -m scripts.State_Representation_Experiment slow 10 5         # d=10, 5 reps, ZY+poly only
    python -m scripts.State_Representation_Experiment fast 25 5         # d=25, 5 reps, fast cells
    python -m scripts.State_Representation_Experiment slow 25 5         # d=25, 5 reps, ZY+poly only (~2 hours)
    python -m scripts.State_Representation_Experiment all 1 10 20000    # d=1, 10 reps, M=20000

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

from core.Basis_Functions import PolynomialBasis, make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from scripts.Experiment_Log import append_result_row
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares, ridge_regression
from core.Swing import SwingContract

def _csv_path(n_samples: int) -> str:
    """Default sample size (5000) keeps the original, unsuffixed filename for
    backward compatibility; any other sample size gets its own file, so
    different-M_t/M_e runs never mix in one CSV -- a naive group_by in R
    would otherwise silently average across incompatible sample sizes."""
    suffix = "" if n_samples == 5_000 else f"_n{n_samples}"
    return os.path.join(os.path.dirname(__file__), "..", "results", f"state_representation_experiment{suffix}.csv")

# Double-jump market -- kept asymmetric deliberately: the market's up/down
# jump asymmetry is a real feature of electricity prices, not a nuisance
# parameter, unlike the swing contract's quantities (see conversation).
# Both jump intensity (lam) AND mean jump size (mu) are asymmetric: down-jumps
# are both less frequent AND smaller on average than up-jumps -- user's
# explicit choice (see conversation), not the reference codebase's own
# double-jump convention (which shares one lam for both directions).
MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=40.0,
    lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4,
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
DEFAULT_N_SAMPLES = 5_000   # training AND evaluation sample size, unless overridden on the command line
RIDGE_LAMBDA = 1.0
# D and N_REPS are set from the command line (see __main__) -- d=1 uses all 10
# reps in one go; d=10's poly+(Z,Y) cell is slow enough to warrant fewer reps
# and the fast/slow BATCH split below.
DEFAULT_D = 10
DEFAULT_N_REPS = 5

# max_aggregation: the portfolio's shared swing right applies to all d units
# at once, and the immediate reward is the maximal gain across dimensions
# (thesis "Payoff Structure": r_n(x,a) = max_j Q(S_n^(j)) for a=1). NOT
# sum_aggregation -- an earlier, incorrect guess (additive profit across
# dimensions) corrected once the actual thesis text was available.

ALPHA = np.exp(-R * MATURITY / N_STEPS)

# Fixed, distinct from the training seeds (rep=0..9) and RNN weight seeds
# (100_000+) below -- one evaluation sample, shared by every repetition.
EVAL_SEED = 500_000

FITS = ["plain", "ridge"]
BATCHES = {
    "fast": [("S", "poly"), ("S", "rnn"), ("ZY", "rnn")],
    "slow": [("ZY", "poly")],
    "all": [("S", "poly"), ("S", "rnn"), ("ZY", "poly"), ("ZY", "rnn")],
    # RNN's feature count is fixed (n_hidden+1) regardless of dimension, so
    # this stays cheap even at high d, unlike poly -- useful for state-input
    # comparisons at dimensions where poly's feature count would be infeasible.
    "rnn_only": [("S", "rnn"), ("ZY", "rnn")],
}


def _base_row(d: int, n_samples: int) -> dict:
    """Fields shared by every row this script writes -- everything that's
    fixed across the whole run, independent of which (state, basis, fit) cell."""
    return {
        "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
        "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
        "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
        "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp, "f_period": MARKET_PARAMS.f_period,
        "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS,
        "n_paths_train": n_samples, "n_paths_eval": n_samples, "n_dims": d,
        "K": CONTRACT.K, "q_min": CONTRACT.q_min, "q_max": CONTRACT.q_max,
        "q_tilde": CONTRACT.q_tilde, "L": CONTRACT.L,
        "regression_mode": "per-level",   # joint mode not implemented yet (see CLAUDE.md)
    }


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
    N_SAMPLES = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_N_SAMPLES
    state_basis_pairs = BATCHES[batch]
    base_row = _base_row(D, N_SAMPLES)
    CSV_PATH = _csv_path(N_SAMPLES)

    print(f"batch={batch}: {state_basis_pairs}")
    print(f"d={D}, K={CONTRACT.K}, L={CONTRACT.L}, q=({CONTRACT.q_min},{CONTRACT.q_tilde},{CONTRACT.q_max}), "
          f"r={R}, T={MATURITY}, N={N_STEPS}, M_t={N_SAMPLES} (new seed per rep), "
          f"M_e={N_SAMPLES} (fixed across all reps), {N_REPS} reps")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    # ONE evaluation sample, fixed across every repetition (see module
    # docstring) -- both state-input views are precomputed once, since the
    # underlying paths never change across reps or (state, basis, fit) cells.
    eval_rng = np.random.default_rng(EVAL_SEED)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=D)
    eval_regression_state = {
        state_input: _regression_state(eval_paths.Z, eval_paths.Y, eval_paths.S, state_input)
        for state_input in STATE_INDEX
    }

    results = defaultdict(list)
    t0 = time.time()

    for rep in range(N_REPS):
        train_rng = np.random.default_rng(rep)   # same training paths across batches for a given rep
        train_paths = simulate_hhk(
            train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=D,
        )

        for state_input, basis_name in state_basis_pairs:
            train_regression_state = _regression_state(train_paths.Z, train_paths.Y, train_paths.S, state_input)
            k = train_regression_state.shape[-1]

            if basis_name == "poly":
                basis = PolynomialBasis(n_dims=k, degree=2)
                basis_row = {"basis_type": "poly", "basis_degree": 2}
            else:
                weight_rng = np.random.default_rng(100_000 + rep * 10 + STATE_INDEX[state_input])
                basis = make_random_features_basis(weight_rng, n_dims=k, n_hidden=20)   # default activation: tanh
                basis_row = {"basis_type": "rnn", "basis_n_hidden": 20, "basis_activation": "tanh"}

            for fit_name in FITS:
                if fit_name == "plain":
                    fit = least_squares
                    fit_row = {"fit_type": "plain"}
                else:
                    fit = functools.partial(ridge_regression, ridge_lambda=RIDGE_LAMBDA)
                    fit_row = {"fit_type": "ridge", "ridge_lambda": RIDGE_LAMBDA}

                t_run = time.time()
                policy = fit_policy(
                    S_train=train_paths.S, regression_state_train=train_regression_state, contract=CONTRACT,
                    aggregate=max_aggregation, basis=basis, fit=fit, alpha=ALPHA, train_itm_only=False,
                )
                result = evaluate_policy(
                    policy, S_eval=eval_paths.S, regression_state_eval=eval_regression_state[state_input],
                    contract=CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
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
