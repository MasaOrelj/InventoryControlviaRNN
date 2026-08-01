"""Swing option: payoff, reward, terminal condition, admissible actions,
inventory grid/transition, and the exercise decision rule. See CLAUDE.md's
"Swing option specification" for the full derivation of every formula here.

Shape convention: price arrays carry the market dimension as their LAST axis,
size d (even when d=1, never squeezed away -- matches Electricity_Market_Model.py
and Payoff_Aggregation.py). Every other array here (Q_total, inventory, action)
has no such axis -- the aggregation step in `total_gain` removes it.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SwingContract:
    """K may be a scalar (shared strike) or a (d,)-shaped array (per-dimension
    strike), broadcasting against price arrays' last axis either way."""

    K: float | np.ndarray
    q_min: float
    q_max: float
    q_tilde: float
    L: int   # number of swing rights


def swing_gain(S: np.ndarray, contract: SwingContract) -> np.ndarray:
    """Q(S) = max{ (S-K)(q_max-q_tilde), (K-S)(q_tilde-q_min), 0 } >= 0, per
    market dimension. S: (..., d) -> Q: (..., d), same shape."""
    up = (S - contract.K) * (contract.q_max - contract.q_tilde)
    down = (contract.K - S) * (contract.q_tilde - contract.q_min)
    return np.maximum(np.maximum(up, down), 0.0)


def total_gain(
    S: np.ndarray, contract: SwingContract, aggregate: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Q_total(S) = aggregate(Q(S^(1)), ..., Q(S^(d))) -- see Payoff_Aggregation.py
    for `aggregate` (sum, max, ...). S: (..., d) -> Q_total: (...), one axis
    shorter (the market-dimension axis is consumed by `aggregate`)."""
    return aggregate(swing_gain(S, contract))


def reward(Q_total: np.ndarray, action: np.ndarray) -> np.ndarray:
    """r_n(x, a) = Q_total(s) if a=1, else 0."""
    return np.where(action == 1, Q_total, 0.0)


def terminal_value(Q_total: np.ndarray, inventory: np.ndarray) -> np.ndarray:
    """g_N(x) = 1{i >= 1} * Q_total(s)."""
    return np.where(inventory >= 1, Q_total, 0.0)


def can_swing(inventory: np.ndarray) -> np.ndarray:
    """Whether action a=1 is admissible: D_n(i) = {0,1} if i>=1, else {0}.
    a=0 is always admissible, so this alone fully characterizes D_n(i)."""
    return inventory >= 1


def next_inventory(inventory: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Psi_n^I(i, a) = i - a. Always lands back on {0,...,L} -- no
    interpolation needed for swing (see CLAUDE.md)."""
    return inventory - action


def inventory_grid(n: int, L: int) -> np.ndarray:
    """𝓘(n) = {max(0, L-n), ..., L}: the inventory levels reachable at step n
    (at most one swing per date, so you can't have used more rights than
    elapsed dates). 𝓘(0) = {L}; reaches the full {0,...,L} once n >= L."""
    lo = max(0, L - n)
    return np.arange(lo, L + 1)


def grid_size(n: int, L: int) -> int:
    """H_n = min(n, L) + 1 = len(inventory_grid(n, L))."""
    return min(n, L) + 1


def decide(Q_total: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Swing exercise decision: a=1 iff Q_total(s) > 0 AND Q_total(s) + c1 > c0
    (swinging beats continuing), else a=0. The Q_total>0 guard protects
    against a noise-driven "free" swing when c1 > c0 purely from regression
    estimation error, even though the true continuation values always satisfy
    c0 >= c1 (extra optionality is never worse). Ties (Q_total+c1 == c0) go to
    a=0, the smaller action index."""
    swing = (Q_total > 0) & (Q_total + c1 > c0)
    return swing.astype(int)
