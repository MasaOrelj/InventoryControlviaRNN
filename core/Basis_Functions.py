"""Feature maps for the LSM regression: build_features(state) -> Phi.

Every family follows the same convention: `state` has shape (n_samples, n_dims)
(n_dims = regression state dimension, e.g. 2 for (Z, I) or 1 for (S,)), and
`build_features` returns Phi of shape (n_samples, n_features) with the constant
feature always in the LAST column. Regression.py's ridge solver relies on this to know
which column to exempt from the ridge penalty.

Standardization: state values are large (spot price ~10^2). Two distinct
formulas, not one:
  - PolynomialBasis, RandomFeaturesBasis: full z-score, (x-mean)/std. 
  - WeightedLaguerreBasis: scale-only, x/std (no centering). Its weight
    exp(-x/(2K)) is unbounded for negative x, so centering (which can flip a
    positive price negative) would blow it up; scaling alone preserves sign.
"""

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import eval_laguerre


def _monomial_exponents(n_dims: int, degree: int) -> list[tuple[int, ...]]:
    """Exponent tuples alpha in N^n_dims with 1 <= sum(alpha) <= degree, i.e.
    every monomial of total degree 1..degree (the constant, degree 0, is
    handled separately by each basis as its own last column)."""
    exponents = []
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_dims), d):
            alpha = [0] * n_dims
            for i in combo:
                alpha[i] += 1
            exponents.append(tuple(alpha))
    return exponents


@dataclass(frozen=True)
class PolynomialBasis:
    n_dims: int
    degree: int

    def __post_init__(self):
        object.__setattr__(self, "exponents", _monomial_exponents(self.n_dims, self.degree))

    @property
    def n_features(self) -> int:
        return len(self.exponents) + 1

    def build_features(self, state: np.ndarray) -> np.ndarray:
        Phi = np.empty((state.shape[0], self.n_features))
        for col, alpha in enumerate(self.exponents):
            Phi[:, col] = np.prod(state ** np.array(alpha), axis=1)
        Phi[:, -1] = 1.0
        return Phi

    def standardize(self, state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (state - mean) / std


@dataclass(frozen=True)
class WeightedLaguerreBasis:
    """Tensor-product weighted Laguerre functions, total degree <= `degree`,
    INCLUDING cross terms across dimensions."""


    n_dims: int
    degree: int
    K: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "exponents", _monomial_exponents(self.n_dims, self.degree))

    @property
    def n_features(self) -> int:
        return len(self.exponents) + 1

    def build_features(self, state: np.ndarray) -> np.ndarray:
        scaled = state / self.K
        Phi = np.empty((state.shape[0], self.n_features))
        for col, alpha in enumerate(self.exponents):
            alpha_arr = np.array(alpha)
            active = (alpha_arr > 0).astype(float)             # 1 for dims IN this term, 0 otherwise
            weight = np.exp(-(scaled * active).sum(axis=1) / 2.0)   # only active dims decay
            shape = np.prod(eval_laguerre(alpha_arr, scaled), axis=1)   # L_0=1 handles inactive dims
            Phi[:, col] = weight * shape
        Phi[:, -1] = 1.0
        return Phi

    def standardize(self, state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return state / std


@dataclass(frozen=True, eq=False)
class RandomFeaturesBasis:
    """Single-hidden-layer random feature map: sigma(Theta @ x + b), constant
    appended (CLAUDE.md's RNN formula). Theta/b are drawn once (outside this
    class, by make_random_features_basis) and fixed for the life of the
    object -- never resample per time step ("fix the RNN weights once" gate).

    eq=False: dataclass equality on numpy-array fields raises ("truth value of
    an array is ambiguous"), and we never need to compare two instances."""

    Theta: np.ndarray   # (n_hidden, n_dims)
    b: np.ndarray        # (n_hidden,)
    activation: Callable[[np.ndarray], np.ndarray] = np.tanh

    @property
    def n_features(self) -> int:
        return self.Theta.shape[0] + 1

    def build_features(self, state: np.ndarray) -> np.ndarray:
        Phi = np.empty((state.shape[0], self.n_features))
        Phi[:, :-1] = self.activation(state @ self.Theta.T + self.b)
        Phi[:, -1] = 1.0
        return Phi

    def standardize(self, state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (state - mean) / std


def make_random_features_basis(
    rng: np.random.Generator,
    n_dims: int,
    n_hidden: int = 20,
    activation: Callable[[np.ndarray], np.ndarray] = np.tanh,
    theta_scale: float | None = None,
    b_scale: float = 1.0,
) -> RandomFeaturesBasis:
    """Default theta_scale=None means 1/sqrt(n_dims), i.e. Theta_jl ~ N(0, 1/n_dims)
    while b_j ~ N(0,1) unscaled -- keeps Theta @ x's variance roughly
    independent of state dimension (a sum of n_dims terms, each shrunk by
    1/n_dims in variance), while b has no such fan-in to compensate for."""
    if theta_scale is None:
        theta_scale = 1.0 / np.sqrt(n_dims)
    Theta = rng.standard_normal((n_hidden, n_dims)) * theta_scale
    b = rng.standard_normal(n_hidden) * b_scale
    return RandomFeaturesBasis(Theta=Theta, b=b, activation=activation)
