"""Multi-dimensional swing reward aggregation.

Combines per-dimension one-swing gains Q(S_n^(1)), ..., Q(S_n^(d)) into a
single Q_total for one shared swing right across the whole d-dimensional
portfolio. Injected into Swing.py's reward computation rather than branched on
internally -- add new aggregations here as plain functions with this same
signature, nothing else needs to change.

Convention: Q always has the market dimension as its LAST axis (size d, even
when d=1 -- never squeezed away, matching Electricity_Market_Model.py's
(n_steps+1, n_paths, n_dims) convention). Every aggregation here reduces
exactly that last axis, so d=1 trivially returns Q itself (squeezed).
"""

import numpy as np


def sum_aggregation(Q: np.ndarray) -> np.ndarray:
    """Q_total = sum_j Q(S_n^(j)). One swing right adjusts every dimension
    of the portfolio simultaneously; gains add up."""
    return Q.sum(axis=-1)


def max_aggregation(Q: np.ndarray) -> np.ndarray:
    """Q_total = max_j Q(S_n^(j))."""
    return Q.max(axis=-1)
