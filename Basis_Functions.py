"""Feature maps for the LSM regression: build_features(state) -> Phi.

Every family follows the same convention: `state` has shape (n_samples, n_dims)
(n_dims = regression state dimension, e.g. 2 for (Z, I) or 1 for (S,)), and
`build_features` returns Phi of shape (n_samples, n_features) with the constant
feature always in the LAST column. Regression.py's ridge solver relies on this to know
which column to exempt from the ridge penalty.
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
    """Total-degree <= `degree` tensor-product monomials in `n_dims` variables,
    including cross terms (e.g. degree=2, n_dims=2: x1, x2, x1^2, x1*x2, x2^2,
    plus the constant). Feature count = C(degree + n_dims, n_dims); grows
    combinatorially with n_dims -- this is the "preselection cost" CLAUDE.md
    contrasts against the RNN family."""

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


@dataclass(frozen=True)
class WeightedLaguerreBasis:
    """Per-dimension weighted Laguerre functions, degree 0..`degree`: one
    shared unweighted constant, plus exp(-x/(2K)) * L_d(x/K) for d=0..degree
    in each dimension (no cross-dimension terms -- feature count =
    1 + n_dims*(degree+1), linear in n_dims, unlike PolynomialBasis's
    combinatorial growth). K rescales the input before evaluating (K=1
    reproduces the reference implementation's un-rescaled form).

    L_d is the standard Laguerre polynomial (scipy.special.eval_laguerre),
    generated via its three-term recurrence rather than hand-derived closed
    forms -- verified against the previous hardcoded degree-0/1/2 formulas in
    Basis_Functions_Test.py, so this is a trusted generalization, not new math."""

    n_dims: int
    degree: int
    K: float = 1.0

    @property
    def n_features(self) -> int:
        return 1 + self.n_dims * (self.degree + 1)

    def build_features(self, state: np.ndarray) -> np.ndarray:
        scaled = state / self.K
        weight = np.exp(-scaled / 2.0)
        terms = [weight * eval_laguerre(d, scaled) for d in range(self.degree + 1)]
        return np.concatenate(terms + [np.ones((state.shape[0], 1))], axis=1)


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


def make_random_features_basis(
    rng: np.random.Generator,
    n_dims: int,
    n_hidden: int = 20,
    activation: Callable[[np.ndarray], np.ndarray] = np.tanh,
    scale: float = 1.0,
) -> RandomFeaturesBasis:
    """Draw Theta ~ N(0, scale^2) (n_hidden, n_dims) and b ~ N(0, scale^2)
    (n_hidden,) once from `rng`, then fix them into a RandomFeaturesBasis."""
    Theta = rng.standard_normal((n_hidden, n_dims)) * scale
    b = rng.standard_normal(n_hidden) * scale
    return RandomFeaturesBasis(Theta=Theta, b=b, activation=activation)
