"""Multi-stock Black-Scholes model (independent GBM coordinates).

    dX_t = (r - delta) X_t dt + sigma X_t dW_t

`d` independent stocks (no correlation between coordinates), matching the
classic multi-asset Bermudan max-call benchmark (Becker et al.; Herrera,
Krach, Ruyssen & Teichmann). Used to validate the LSM engine against published
results before trusting it on the HHK swing case, where there's no published
ground truth to check against.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlackScholesParams:
    r: float               # risk-free rate (drift)
    sigma: float            # volatility
    dividend: float = 0.0   # continuous dividend yield


@dataclass(frozen=True)
class BlackScholesPaths:
    """Simulated paths. S has shape (n_steps+1, n_paths, n_dims)."""

    t: np.ndarray
    S: np.ndarray


def simulate_black_scholes(
    rng: np.random.Generator,
    params: BlackScholesParams,
    n_paths: int,
    n_steps: int,
    maturity: float,
    n_dims: int,
    spot: float | np.ndarray,
) -> BlackScholesPaths:
    """Exact lognormal GBM solution, independent coordinates (no discretization
    bias -- the log-transform of GBM is an exact affine SDE):
        X_{n+1} = X_n * exp( (r-delta-sigma^2/2)*dt + sigma*sqrt(dt)*eps ),  eps ~ N(0,1) i.i.d.
    """
    dt = maturity / n_steps
    t_grid = np.linspace(0.0, maturity, n_steps + 1)

    eps = rng.standard_normal(size=(n_steps, n_paths, n_dims))
    drift = params.r - params.dividend - 0.5 * params.sigma**2
    log_increments = drift * dt + params.sigma * np.sqrt(dt) * eps
    log_paths = np.concatenate(
        [np.zeros((1, n_paths, n_dims)), np.cumsum(log_increments, axis=0)], axis=0,
    )
    S = spot * np.exp(log_paths)

    return BlackScholesPaths(t=t_grid, S=S)
