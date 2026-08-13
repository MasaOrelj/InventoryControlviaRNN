"""LSM regression core: the two regression solvers, plus the backward-induction
engine that drives them using Swing.py's payoff/inventory logic.

price_swing fits and evaluates in one call, on a single simulated path array
split train/eval in half -- its original interface, unchanged behavior.
fit_policy/evaluate_policy expose the same backward induction as two separable
steps, so ONE fitted policy can be evaluated against several DIFFERENT
evaluation samples: CLAUDE.md's SD/CI methodology needs this to isolate
policy-fit variability (vary the training seed, keep eval fixed -- component A)
from evaluation Monte Carlo noise (a CLT confidence interval for one fixed
policy -- component B). price_swing is now a thin wrapper composing the two.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

from core import Swing


def least_squares(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain (unregularized) least squares: beta = argmin ||Phi @ beta - y||^2.
    Solved via SVD (np.linalg.lstsq)"""
    beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return beta


def ridge_regression(Phi: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Solve  (Phi^T Phi + ridge_lambda * n_samples * I_0) beta = Phi^T y  for beta.

    Phi: (n_samples, n_features), constant feature in the LAST column (see
    Basis_Functions.py). I_0 = identity except a 0 on that last (constant) position, so
    the constant is never penalized -- only the non-constant coefficients are
    shrunk. Requires ridge_lambda > 0, Cholesky solve requires positive-definiteness.
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


@dataclass(frozen=True, eq=False)
class Policy:
    """A fitted swing-exercise policy: backward-induction regression
    coefficients and standardization stats from ONE training sample --
    everything evaluate_policy needs to replay the exercise decisions on a
    DIFFERENT (evaluation) sample without refitting. Built by fit_policy.

    betas:              (n, i) -> (beta0, beta1) for n in 1..N-1, i in
                         inventory_grid(n, L) except i=0 (no decision there;
                         beta0 predicts continuing at i, beta1 predicts
                         swinging down to i-1, matching price_swing's c0/c1).
    standardize_stats:   n -> (mean, std), the training-rows-only stats used
                         to standardize step n's state (same n range as betas).
    """

    N: int
    L: int
    betas: dict
    standardize_stats: dict


def fit_policy(
    S_train: np.ndarray,
    regression_state_train: np.ndarray,
    contract: Swing.SwingContract,
    aggregate: Callable[[np.ndarray], np.ndarray],
    basis,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray],
    alpha: float,
    train_itm_only: bool = False,
) -> Policy:
    """Backward-induction fit phase, restricted to ONE path set (S_train /
    regression_state_train -- every row is training data, no internal
    train/eval split here). Mirrors price_swing's interior loop exactly, but
    returns the fitted per-step coefficients instead of propagated cash flows
    -- see evaluate_policy for applying them out-of-sample.
    """
    policy, _p_train = _fit_backward_induction(
        S_train, regression_state_train, contract, aggregate, basis, fit, alpha, train_itm_only,
    )
    return policy


def evaluate_policy(
    policy: Policy,
    S_eval: np.ndarray,
    regression_state_eval: np.ndarray,
    contract: Swing.SwingContract,
    aggregate: Callable[[np.ndarray], np.ndarray],
    basis,
    alpha: float,
) -> dict:
    """Out-of-sample evaluation: applies policy's FIXED coefficients (fit
    elsewhere, on a separate sample -- never refit here) to a new price/state
    sample, propagating cash flows the same way price_swing's interior loop
    does. Returns {"v0", "p", "cashflows"}: "cashflows" is the (M_e,) array of
    discounted per-path values at n=0 that v0 averages -- its sample std /
    sqrt(M_e) is the CLT standard error for v0 (see clt_confidence_interval).
    """
    n_steps_plus_1, M_e, _ = S_eval.shape
    N, L = policy.N, policy.L

    Q_total = Swing.total_gain(S_eval, contract, aggregate)

    p = np.zeros((L + 1, M_e))
    p[1:, :] = Q_total[N, :]

    for n in range(N - 1, 0, -1):
        grid = Swing.inventory_grid(n, L)
        p_prev = p
        p_new = np.zeros((L + 1, M_e))

        mean, std = policy.standardize_stats[n]
        standardized_state = basis.standardize(regression_state_eval[n], mean, std)
        Phi = basis.build_features(standardized_state)
        immediate = Q_total[n]

        for i in grid:
            if i == 0:
                continue   # p_new[0,:] stays 0 -- V_n(.,.,0)=0, no decision needed

            beta0, beta1 = policy.betas[(n, i)]
            c0 = Phi @ beta0
            c1 = Phi @ beta1

            action = Swing.decide(immediate, c0, c1)
            reward = Swing.reward(immediate, action)
            continuation = np.where(action == 1, alpha * p_prev[i - 1, :], alpha * p_prev[i, :])
            p_new[i, :] = reward + continuation

        p = p_new

    # n=0: the initial state is deterministic (every path starts at the same
    # S_0), so this is ONE shared decision for the whole sample -- compare
    # path-averaged continuation values, not a per-path regression decision
    # (mirrors price_swing's original n=0 step). cashflows picks the per-path
    # terms behind whichever branch wins, so cashflows.mean() reproduces v0
    # exactly while still giving a genuine per-path array to compute a CLT
    # standard error from.
    c0_eval = alpha * p[L, :].mean()
    c1_eval = alpha * p[L - 1, :].mean()
    if c0_eval >= Q_total[0, :].mean() + c1_eval:
        cashflows = alpha * p[L, :]
    else:
        cashflows = Q_total[0, :] + alpha * p[L - 1, :]
    v0 = cashflows.mean()

    return {"v0": v0, "p": p, "cashflows": cashflows}


def clt_confidence_interval(cashflows: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    M_e = cashflows.shape[0]
    v0 = cashflows.mean()
    se = cashflows.std(ddof=1) / np.sqrt(M_e)
    z = norm.ppf(0.5 + confidence / 2.0)
    half_width = z * se
    return v0 - half_width, v0 + half_width


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
    """Backward-induction LSM engine, per-level regression mode. 

    S:                 (N+1, M, d) simulated underlying prices, used for the payoff.
    regression_state:  (N+1, M, k) state fed to the regression -- may be S itself
                        ((S,I) knob) or e.g. concat(Z,Y) ((Z,Y,I) knob); decoupled
                        from S so both state-input choices are just different
                        arguments here, never a branch inside this function.
    contract:          Swing.SwingContract.
    aggregate:         Payoff_Aggregation function (sum_aggregation, max_aggregation).
    basis:             feature map with .build_features(state) 
    fit:               (Phi, y) -> beta, e.g. least_squares or a ridge_regression partial.
    alpha:             one-step discount factor (e.g. exp(-r*dt)); caller's choice
                        of formula, this function only ever multiplies by it.
    train_itm_only:    if True, fit continuation regressions only on training paths
                        where the current step's Q_total > 0. Settled OFF for swing
                        (CLAUDE.md: Q_total=0 has ~zero probability there), but this
                        engine exposes it since other payoffs (e.g. a classic
                        American max-call, used to validate this engine) need it.

    Returns {"v0": Ṽ_0, "p": propagated cash flows shape (L+1, M) (training
    columns first, then evaluation, same order as the input), "cashflows":
    evaluation-only per-path terms at n=0, see evaluate_policy}.
    """
    M = S.shape[1]
    M_t = M // 2

    policy, p_train = _fit_backward_induction(
        S[:, :M_t], regression_state[:, :M_t], contract, aggregate, basis, fit, alpha, train_itm_only,
    )
    eval_result = evaluate_policy(
        policy, S[:, M_t:], regression_state[:, M_t:], contract, aggregate, basis, alpha,
    )
    p = np.concatenate([p_train, eval_result["p"]], axis=1)
    return {"v0": eval_result["v0"], "p": p, "cashflows": eval_result["cashflows"]}


def _fit_backward_induction(
    S_train: np.ndarray,
    regression_state_train: np.ndarray,
    contract: Swing.SwingContract,
    aggregate: Callable[[np.ndarray], np.ndarray],
    basis,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray],
    alpha: float,
    train_itm_only: bool,
) -> tuple[Policy, np.ndarray]:
    """Shared core of fit_policy and price_swing: backward induction over ONE
    path set, every row used as training data. Returns (Policy, p_train) --
    p_train (propagated training-path cash flows) is dead weight for
    fit_policy's own purpose (only the fitted betas matter once you can
    evaluate out-of-sample) but lets price_swing reconstruct its historical
    train+eval-combined `p` return value exactly; fit_policy discards it.
    """
    n_steps_plus_1, M_t, _ = S_train.shape
    N = n_steps_plus_1 - 1
    L = contract.L

    Q_total = Swing.total_gain(S_train, contract, aggregate)

    p = np.zeros((L + 1, M_t))
    p[1:, :] = Q_total[N, :]   # terminal: g_N(x) = 1{i>=1} * Q_total(s), same for every i>=1

    betas: dict = {}
    standardize_stats: dict = {}

    for n in range(N - 1, 0, -1):
        grid = Swing.inventory_grid(n, L)
        p_prev = p
        p_new = np.zeros((L + 1, M_t))

        # Standardize using this step's (training-only, by construction --
        # every row here IS training data) statistics, then build features
        # from the standardized state. Recomputed fresh each step, since the
        # state's own distribution generally shifts with n.
        mean = regression_state_train[n].mean(axis=0)
        std = regression_state_train[n].std(axis=0)
        standardize_stats[n] = (mean, std)
        standardized_state = basis.standardize(regression_state_train[n], mean, std)
        Phi = basis.build_features(standardized_state)
        immediate = Q_total[n]

        for i in grid:
            if i == 0:
                continue   # p_new[0,:] stays 0 -- V_n(.,.,0)=0, no regression needed

            beta0 = _fit_beta(Phi, alpha * p_prev[i, :], fit, train_itm_only, immediate)
            beta1 = _fit_beta(Phi, alpha * p_prev[i - 1, :], fit, train_itm_only, immediate)
            betas[(n, i)] = (beta0, beta1)

            c0 = Phi @ beta0
            c1 = Phi @ beta1

            action = Swing.decide(immediate, c0, c1)
            reward = Swing.reward(immediate, action)
            continuation = np.where(action == 1, alpha * p_prev[i - 1, :], alpha * p_prev[i, :])
            p_new[i, :] = reward + continuation

        p = p_new

    return Policy(N=N, L=L, betas=betas, standardize_stats=standardize_stats), p


def _fit_beta(
    Phi: np.ndarray,
    target: np.ndarray,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray],
    train_itm_only: bool,
    immediate: np.ndarray,
) -> np.ndarray:
    """Fit on training rows (optionally ITM-filtered), returning beta itself
    rather than a prediction -- evaluate_policy applies the same beta to a
    DIFFERENT (evaluation) sample's Phi."""
    if train_itm_only:
        train_idx = np.where(immediate > 0.0)[0]
    else:
        train_idx = np.arange(Phi.shape[0])

    if len(train_idx) == 0:
        return np.zeros(Phi.shape[1])

    return fit(Phi[train_idx], target[train_idx])
