"""LSM regression core. For now: the two regression solvers shared by every
feature-map choice from Basis_Functions.py. The backward-induction engine is
added once swing.py's payoff/inventory logic exists to drive it.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


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
