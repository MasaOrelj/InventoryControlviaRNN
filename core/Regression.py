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

    inventory_mode:      "separate" (Boogert-de Jong, one regression per
                         (n, a, i), inventory never an input) or "joint"
                         (one regression per (n, a), stacked over the
                         admissible i grid, inventory an input coordinate).
                         Stored here (not re-passed to evaluate_policy) so
                         fit and eval can never disagree on which mode a
                         given Policy was actually fit under.
    betas:               mode "separate": (n, i) -> (beta0, beta1) for
                         n in 1..N-1, i in inventory_grid(n, L) except i=0.
                         mode "joint": (n, a) -> beta for a in {0, 1} --
                         ONE fitted function per action, evaluated at
                         different i inputs for different grid levels.
    standardize_stats:   mode "separate": n -> (mean, std) over the market
                         state alone. mode "joint": (n, a) -> (mean, std)
                         over the stacked (s, i) rows actually used in that
                         action's fit -- NOT shared between a=0 and a=1,
                         since their admissible row sets differ.
    """

    N: int
    L: int
    betas: dict
    standardize_stats: dict
    inventory_mode: str = "separate"


def fit_policy(
    S_train: np.ndarray,
    regression_state_train: np.ndarray,
    contract: Swing.SwingContract,
    aggregate: Callable[[np.ndarray], np.ndarray],
    basis,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray],
    alpha: float,
    train_itm_only: bool = False,
    inventory_mode: str = "separate",
    include_zero_level: bool = False,
) -> Policy:
    """Backward-induction fit phase, restricted to ONE path set (S_train /
    regression_state_train -- every row is training data, no internal
    train/eval split here). Mirrors price_swing's interior loop exactly, but
    returns the fitted per-step coefficients instead of propagated cash flows
    -- see evaluate_policy for applying them out-of-sample.

    inventory_mode:      "separate" (default, byte-for-byte the original
                         behavior) or "joint" -- see Policy docstring.
                         `basis` must already be constructed with the right
                         input dimension for the chosen mode (d for
                         "separate", d+1 for "joint", inventory appended as
                         the last input coordinate) -- this function does not
                         rebuild the basis itself.
    include_zero_level:  "joint" mode only. Whether i=0 rows (whose target is
                         always exactly 0, since V_n(.,.,0)=0 is handled
                         analytically in both modes) are included as training
                         rows in the a=0 stacked fit. Default False, matching
                         "separate" mode's convention of never fitting
                         anything at i=0. Ignored in "separate" mode.
    """
    policy, _p_train = _fit_backward_induction(
        S_train, regression_state_train, contract, aggregate, basis, fit, alpha, train_itm_only,
        inventory_mode, include_zero_level,
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

    Branches on policy.inventory_mode (read from the Policy itself, never
    re-passed here, so fit and eval can never disagree on mode). `basis` must
    be the SAME basis object (or an equivalent one with the same input
    dimension) used to fit `policy`.
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
        immediate = Q_total[n]

        if policy.inventory_mode == "separate":
            mean, std = policy.standardize_stats[n]
            standardized_state = basis.standardize(regression_state_eval[n], mean, std)
            Phi = basis.build_features(standardized_state)

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

        elif policy.inventory_mode == "joint":
            state_market = regression_state_eval[n]
            beta0, beta1 = policy.betas[(n, 0)], policy.betas[(n, 1)]
            mean0, std0 = policy.standardize_stats[(n, 0)]
            mean1, std1 = policy.standardize_stats[(n, 1)]

            for i in grid:
                if i == 0:
                    continue   # p_new[0,:] stays 0 -- V_n(.,.,0)=0, handled analytically in both modes

                # Same two fitted functions (n,0)/(n,1) for every i -- only
                # the inventory INPUT coordinate changes across grid levels.
                state_i = _append_inventory_column(state_market, i)
                Phi0 = basis.build_features(basis.standardize(state_i, mean0, std0))
                Phi1 = basis.build_features(basis.standardize(state_i, mean1, std1))
                c0 = Phi0 @ beta0
                c1 = Phi1 @ beta1

                action = Swing.decide(immediate, c0, c1)
                reward = Swing.reward(immediate, action)
                continuation = np.where(action == 1, alpha * p_prev[i - 1, :], alpha * p_prev[i, :])
                p_new[i, :] = reward + continuation

        else:
            raise ValueError(f"unknown inventory_mode: {policy.inventory_mode!r}")

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
    inventory_mode: str = "separate",
    include_zero_level: bool = False,
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
        inventory_mode, include_zero_level,
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
    inventory_mode: str = "separate",
    include_zero_level: bool = False,
) -> tuple[Policy, np.ndarray]:
    """Shared core of fit_policy and price_swing: backward induction over ONE
    path set, every row used as training data. Returns (Policy, p_train) --
    p_train (propagated training-path cash flows) is dead weight for
    fit_policy's own purpose (only the fitted betas matter once you can
    evaluate out-of-sample) but lets price_swing reconstruct its historical
    train+eval-combined `p` return value exactly; fit_policy discards it.

    inventory_mode="separate" is byte-for-byte the original code path (kept
    completely intact, not refactored, precisely so this is verifiably
    unchanged). "joint" is a new sibling path -- see Policy docstring and
    fit_policy's inventory_mode/include_zero_level docs.
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
        immediate = Q_total[n]

        if inventory_mode == "separate":
            # Standardize using this step's (training-only, by construction
            # -- every row here IS training data) statistics, then build
            # features from the standardized state. Recomputed fresh each
            # step, since the state's own distribution generally shifts with n.
            mean = regression_state_train[n].mean(axis=0)
            std = regression_state_train[n].std(axis=0)
            standardize_stats[n] = (mean, std)
            standardized_state = basis.standardize(regression_state_train[n], mean, std)
            Phi = basis.build_features(standardized_state)

            # beta1 at level i targets p_prev[i-1], on the same Phi -- exactly
            # what beta0 targeted one grid step earlier, at level i-1 (see
            # conversation). Cache the previous iteration's beta0 and reuse it
            # as beta1 instead of refitting; bit-for-bit identical since
            # _fit_beta is a deterministic function of (Phi, target) and both
            # are literally unchanged, just called once instead of twice.
            prev_beta0 = None
            for i in grid:
                if i == 0:
                    continue   # p_new[0,:] stays 0 -- V_n(.,.,0)=0, no regression needed

                beta0 = _fit_beta(Phi, alpha * p_prev[i, :], fit, train_itm_only, immediate)
                if prev_beta0 is not None:
                    beta1 = prev_beta0
                else:
                    beta1 = _fit_beta(Phi, alpha * p_prev[i - 1, :], fit, train_itm_only, immediate)
                betas[(n, i)] = (beta0, beta1)

                c0 = Phi @ beta0
                c1 = Phi @ beta1

                action = Swing.decide(immediate, c0, c1)
                reward = Swing.reward(immediate, action)
                continuation = np.where(action == 1, alpha * p_prev[i - 1, :], alpha * p_prev[i, :])
                p_new[i, :] = reward + continuation

                prev_beta0 = beta0

        elif inventory_mode == "joint":
            state_market = regression_state_train[n]   # (M_t, d), inventory NOT yet appended

            for a in (0, 1):
                # Admissibility: a=1 only ever valid for i>=1 (Swing.can_swing).
                # a=0's own rows additionally depend on include_zero_level --
                # i=0's target is always exactly 0 (V_n(.,.,0)=0), and
                # dumping that block of exact zeros into the stacked fit is
                # what creates the kink Boogert-de Jong warn about, so it's
                # excluded by default, matching "separate" mode's convention
                # of never fitting anything at i=0 at all.
                if a == 1:
                    h_values = [h for h in grid if h >= 1]
                else:
                    h_values = [h for h in grid if include_zero_level or h >= 1]

                stacked_state = np.concatenate(
                    [_append_inventory_column(state_market, h) for h in h_values], axis=0,
                )
                target_source = p_prev[[h - a for h in h_values], :]   # (len(h_values), M_t)
                stacked_target = alpha * target_source.reshape(-1)
                stacked_immediate = np.tile(immediate, len(h_values)) if train_itm_only else immediate

                # Moments over the STACKED rows actually used in this
                # action's fit -- per (n, a), never reused across n or a
                # (a=0/a=1 have different admissible row sets, so their
                # moments genuinely differ).
                mean = stacked_state.mean(axis=0)
                std = stacked_state.std(axis=0)
                standardize_stats[(n, a)] = (mean, std)
                standardized = basis.standardize(stacked_state, mean, std)
                Phi_stacked = basis.build_features(standardized)

                betas[(n, a)] = _fit_beta(Phi_stacked, stacked_target, fit, train_itm_only, stacked_immediate)

            # Propagate: SAME two fitted functions (n,0)/(n,1) for every i,
            # inventory entering as an input coordinate rather than selecting
            # a different beta.
            beta0, beta1 = betas[(n, 0)], betas[(n, 1)]
            mean0, std0 = standardize_stats[(n, 0)]
            mean1, std1 = standardize_stats[(n, 1)]

            for i in grid:
                if i == 0:
                    continue   # p_new[0,:] stays 0 -- V_n(.,.,0)=0, handled analytically in both modes

                state_i = _append_inventory_column(state_market, i)
                Phi0 = basis.build_features(basis.standardize(state_i, mean0, std0))
                Phi1 = basis.build_features(basis.standardize(state_i, mean1, std1))
                c0 = Phi0 @ beta0
                c1 = Phi1 @ beta1

                action = Swing.decide(immediate, c0, c1)
                reward = Swing.reward(immediate, action)
                continuation = np.where(action == 1, alpha * p_prev[i - 1, :], alpha * p_prev[i, :])
                p_new[i, :] = reward + continuation

        else:
            raise ValueError(f"unknown inventory_mode: {inventory_mode!r}")

        p = p_new

    return Policy(N=N, L=L, betas=betas, standardize_stats=standardize_stats, inventory_mode=inventory_mode), p


def _append_inventory_column(state_market: np.ndarray, i: int) -> np.ndarray:
    """(M, d) market state -> (M, d+1), with inventory level i (constant
    across all M rows) appended as the LAST market-side column -- i.e. right
    before build_features' own constant-feature column. Used both when
    stacking training rows across h (joint-mode fit) and when evaluating a
    joint-mode policy at one fixed grid level i."""
    M = state_market.shape[0]
    return np.column_stack([state_market, np.full(M, i, dtype=float)])


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
