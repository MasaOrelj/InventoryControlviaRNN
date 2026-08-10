"""Validation benchmark: classic multi-stock American max-call, modeled as a
swing option with L=1 (see conversation derivation: q_min=0, q_tilde=0,
q_max=1 with max_aggregation reduces Swing's payoff exactly to the plain
multi-stock max-call g(x) = (max_j x_j - K)^+). Reproduces the Black-Scholes
Bermudan max-call setup from Herrera, Krach, Ruyssen & Teichmann (2021) /
Becker et al., to check the LSM engine against published numbers before
trusting it on the (unverifiable) HHK swing case.

Run: python -m scripts.Validate_Max_Call_Benchmark
"""

import time

import numpy as np

import Swing
from Basis_Functions import PolynomialBasis, make_random_features_basis
from Black_Scholes_Model import BlackScholesParams, simulate_black_scholes
from Payoff_Aggregation import max_aggregation
from Regression import least_squares, price_swing
from Swing import SwingContract

K = 100.0
SPOT = 100.0
R = 0.0
SIGMA = 0.2
MATURITY = 1.0
N_STEPS = 10
N_PATHS = 20_000
N_REPS = 10
DIMS = [5, 10]

CONTRACT = SwingContract(K=K, q_min=0.0, q_max=1.0, q_tilde=0.0, L=1)
ALPHA = np.exp(-R * MATURITY / N_STEPS)


def leaky_relu(x: np.ndarray, negative_slope: float = 0.5) -> np.ndarray:
    """RLSM's activation in the paper: LeakyReLU with negative_slope=0.5."""
    return np.where(x > 0.0, x, negative_slope * x)


def _regression_state(paths_S: np.ndarray, payoff_as_input: bool) -> np.ndarray:
    """Optionally append the current-date payoff Q_total(S_n) as an extra
    regression input coordinate (the paper's 'payoff as input' convention)."""
    if not payoff_as_input:
        return paths_S
    Q_total = Swing.total_gain(paths_S, CONTRACT, max_aggregation)   # (N+1, M)
    return np.concatenate([paths_S, Q_total[..., None]], axis=-1)    # (N+1, M, d+1)


def run_lsm(rng: np.random.Generator, d: int, payoff_as_input: bool) -> float:
    params = BlackScholesParams(r=R, sigma=SIGMA)
    paths = simulate_black_scholes(
        rng, params, n_paths=N_PATHS, n_steps=N_STEPS, maturity=MATURITY, n_dims=d, spot=SPOT,
    )
    regression_state = _regression_state(paths.S, payoff_as_input)
    basis = PolynomialBasis(n_dims=regression_state.shape[-1], degree=2)
    result = price_swing(
        S=paths.S, regression_state=regression_state, contract=CONTRACT, aggregate=max_aggregation,
        basis=basis, fit=least_squares, alpha=ALPHA, train_itm_only=True,
    )
    return result["v0"]


def run_rlsm(rng: np.random.Generator, d: int, payoff_as_input: bool) -> float:
    params = BlackScholesParams(r=R, sigma=SIGMA)
    paths = simulate_black_scholes(
        rng, params, n_paths=N_PATHS, n_steps=N_STEPS, maturity=MATURITY, n_dims=d, spot=SPOT,
    )
    regression_state = _regression_state(paths.S, payoff_as_input)
    basis = make_random_features_basis(rng, n_dims=regression_state.shape[-1], n_hidden=20, activation=leaky_relu)
    result = price_swing(
        S=paths.S, regression_state=regression_state, contract=CONTRACT, aggregate=max_aggregation,
        basis=basis, fit=least_squares, alpha=ALPHA, train_itm_only=True,
    )
    return result["v0"]


if __name__ == "__main__":
    print(f"K=spot={K}, r={R}, sigma={SIGMA}, T={MATURITY}, N={N_STEPS}, "
          f"M={N_PATHS} ({N_PATHS//2} train / {N_PATHS//2} eval), {N_REPS} reps\n")
    for payoff_as_input in [False, True]:
        print(f"--- payoff_as_input={payoff_as_input} ---")
        for d in DIMS:
            for name, run_fn in [("LSM", run_lsm), ("RLSM", run_rlsm)]:
                t0 = time.time()
                prices = np.array(
                    [run_fn(np.random.default_rng(1000 * d + rep), d, payoff_as_input) for rep in range(N_REPS)]
                )
                elapsed = time.time() - t0
                print(f"d={d:>3}  {name:>4}: mean={prices.mean():.4f}  std={prices.std(ddof=1):.4f}  "
                      f"({elapsed:.1f}s for {N_REPS} reps)")
        print()
