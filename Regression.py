"""LSM regression core: the two regression solvers, plus the backward-induction
engine that drives them using Swing.py's payoff/inventory logic.
"""

from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import Swing


def least_squares(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain (unregularized) least squares: beta = argmin ||Phi @ beta - y||^2.

    Solved via SVD (np.linalg.lstsq), not the normal equations -- robust to
    rank-deficient/collinear Phi (e.g. more features than in-the-money samples),
    which ridge_regression's Cholesky solve cannot handle at ridge_lambda=0."""
    beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return beta


def ridge_regression(Phi: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Solve  (Phi^T Phi + ridge_lambda * n_samples * I_0) beta = Phi^T y  for beta.

    Phi: (n_samples, n_features), constant feature in the LAST column (see
    Basis_Functions.py). I_0 = identity except a 0 on that last (constant) position, so
    the constant is never penalized -- only the non-constant coefficients are
    shrunk. Requires ridge_lambda > 0 (otherwise Phi^T Phi alone must already be
    positive-definite for the Cholesky solve below to succeed) -- use
    least_squares for the unregularized case instead.

    Solved via the symmetric positive-definite normal equations (Cholesky),
    not an explicit matrix inverse.
    """
    n_samples, n_features = Phi.shape
    I_0 = np.eye(n_features)
    I_0[-1, -1] = 0.0

    A = Phi.T @ Phi + ridge_lambda * n_samples * I_0
    b = Phi.T @ y

    beta = cho_solve(cho_factor(A), b)
    return beta


def price_swing(
    S: np.ndarray,
    regression_state: np.ndarray,
    contract: Swing.SwingContract,
    aggregate: Callable[[np.ndarray], np.ndarray],
    basis,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray],
    alpha: float,
    train_itm_only: bool = False,
) -> dict:
    """Backward-induction LSM engine, per-level regression mode (CLAUDE.md's
    Boogert-de Jong knob -- joint mode not implemented yet).

    S:                 (N+1, M, d) simulated underlying prices, used for the payoff.
    regression_state:  (N+1, M, k) state fed to the regression -- may be S itself
                        ((S,I) knob) or e.g. concat(Z,Y) ((Z,Y,I) knob); decoupled
                        from S so both state-input choices are just different
                        arguments here, never a branch inside this function.
    contract:          Swing.SwingContract.
    aggregate:         Payoff_Aggregation function (sum_aggregation, max_aggregation).
    basis:             feature map with .build_features(state) -> Phi, already
                        constructed for regression_state's dimension k (for
                        RandomFeaturesBasis: already carrying its fixed weights --
                        this function never draws or resamples them).
    fit:               (Phi, y) -> beta, e.g. least_squares or a ridge_regression partial.
    alpha:             one-step discount factor (e.g. exp(-r*dt)); caller's choice
                        of formula, this function only ever multiplies by it.
    train_itm_only:    if True, fit continuation regressions only on training paths
                        where the current step's Q_total > 0. Settled OFF for swing
                        (CLAUDE.md: Q_total=0 has ~zero probability there), but this
                        engine exposes it since other payoffs (e.g. a classic
                        American max-call, used to validate this engine) need it.

    Returns {"v0": Ṽ_0, "p": final propagated cash flows, shape (L+1, M)}.
    """
    n_steps_plus_1, M, _ = S.shape
    N = n_steps_plus_1 - 1
    M_t = M // 2
    L = contract.L

    Q_total = Swing.total_gain(S, contract, aggregate)   # (N+1, M)

    p = np.zeros((L + 1, M))
    p[1:, :] = Q_total[N, :]   # terminal: g_N(x) = 1{i>=1} * Q_total(s), same for every i>=1

    for n in range(N - 1, 0, -1):
        grid = Swing.inventory_grid(n, L)
        p_prev = p
        p_new = np.zeros((L + 1, M))

        Phi = basis.build_features(regression_state[n])
        immediate = Q_total[n]

        for i in grid:
            if i == 0:
                continue   # p_new[0,:] stays 0 -- V_n(.,.,0)=0, no regression needed

            c0 = _fit_and_predict(Phi, alpha * p_prev[i, :], fit, train_itm_only, immediate, M_t)
            c1 = _fit_and_predict(Phi, alpha * p_prev[i - 1, :], fit, train_itm_only, immediate, M_t)

            action = Swing.decide(immediate, c0, c1)
            reward = Swing.reward(immediate, action)
            continuation = np.where(action == 1, alpha * p_prev[i - 1, :], alpha * p_prev[i, :])
            p_new[i, :] = reward + continuation

        p = p_new

    # n=0: fixed initial state, evaluation paths only, plain Monte Carlo mean -- no regression.
    c0_eval = alpha * p[L, M_t:].mean()
    c1_eval = alpha * p[L - 1, M_t:].mean()
    v0 = max(c0_eval, Q_total[0, M_t:].mean() + c1_eval)

    return {"v0": v0, "p": p}


def _fit_and_predict(
    Phi: np.ndarray,
    target: np.ndarray,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray],
    train_itm_only: bool,
    immediate: np.ndarray,
    M_t: int,
) -> np.ndarray:
    """Fit on training paths (optionally ITM-filtered) only, predict on all paths."""
    if train_itm_only:
        train_idx = np.where(immediate[:M_t] > 0.0)[0]
    else:
        train_idx = np.arange(M_t)

    if len(train_idx) == 0:
        return np.zeros(Phi.shape[0])

    beta = fit(Phi[train_idx], target[train_idx])
    return Phi @ beta
