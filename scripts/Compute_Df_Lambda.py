"""Effective degrees of freedom df(lambda) for the Laguerre and RNN
ridge-lambda grids, averaged across ALL backward-induction steps n=1..N-1
(not a single arbitrary step -- df(lambda) is fit separately per step, and
its value can vary step to step since each step's Phi is built from that
step's own training-row distribution).

df(lambda) = 1 + sum_i nu_i/(nu_i + lambda*S_n), where nu_i are eigenvalues
of the Gram matrix of the CENTERED non-constant features (constant is the
LAST column, per Basis_Functions.py's convention) and S_n = Phi.shape[0]
(= M_t for this project's settled per-level mode). The leading "1 +" and the
centering are NOT optional simplifications -- ridge_regression's actual
penalty is lambda*S_n*I_0 with I_0 leaving the constant unpenalized, so the
constant direction is always retained in full (df(lambda) >= 1 for every
lambda), and centering the other features is what makes their contribution
exactly separable from the constant's (verified: this formula reproduces
tr(A^-1 G) from the ACTUAL A = G + lambda*S_n*I_0 to machine precision; the
naive sum_i s_i/(s_i+lambda*S_n) over ALL K raw eigenvalues -- this script's
own earlier version -- implicitly assumes I_0=I, a different, more heavily
penalized estimator than what's actually fit, and can (and did) drop below 1
at small df(lambda), which is impossible for the real estimator).

Matches Laguerre_Ridge_Lambda_Experiment.py / RNN_Ridge_Lambda_Experiment.py's
own seeds exactly, so results are directly comparable to those tables.
Writes one row per (dim, [K,] lambda) with mean/std/min/max df across steps.

Run: python -m scripts.Compute_Df_Lambda
"""
import os

import numpy as np
import pandas as pd

from core.Basis_Functions import WeightedLaguerreBasis, make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000

LAGUERRE_LAMBDAS = {
    1: [0.001, 0.01, 0.1, 1.0],
    10: [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    25: [0.001, 0.01, 0.1, 1.0, 10.0, 15.0, 20.0],
}
RNN_LAMBDAS = {
    1: [0.001, 0.01, 0.1, 1.0, 5.0],
    10: [0.001, 0.01, 0.1, 1.0, 5.0],
    25: [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 15.0, 20.0],
}
RNN_K_VALUES = [10, 20, 30, 40, 50]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def eigs_per_step(basis, train_paths_S):
    """Per step: eigenvalues of the CENTERED non-constant features' Gram
    matrix (constant is Phi's last column), plus that step's S_n. Centering
    uses this step's own training rows only, same discipline as the
    project's own standardize() convention elsewhere."""
    out = []
    for n in range(1, N_STEPS):
        state = train_paths_S[n]
        mean = state.mean(axis=0)
        std = state.std(axis=0)
        standardized = basis.standardize(state, mean, std)
        Phi = basis.build_features(standardized)
        Phi_rest = Phi[:, :-1]
        Phi_rest_centered = Phi_rest - Phi_rest.mean(axis=0, keepdims=True)
        G_centered = Phi_rest_centered.T @ Phi_rest_centered
        nu = np.clip(np.linalg.eigvalsh(G_centered), 0, None)
        out.append((nu, Phi.shape[0]))
    return out


def df_stats(eigs, lam):
    dfs = np.array([1.0 + np.sum(nu / (nu + lam * n_samples)) for nu, n_samples in eigs])
    return dfs.mean(), dfs.std(), dfs.min(), dfs.max()


if __name__ == "__main__":
    laguerre_rows = []
    for d, lambdas in LAGUERRE_LAMBDAS.items():
        train_rng = np.random.default_rng(1000 * d + 0)
        train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=d)
        basis = WeightedLaguerreBasis(n_dims=d, degree=2, K=1.0)
        eigs = eigs_per_step(basis, train_paths.S)
        for lam in lambdas:
            mean, std, lo, hi = df_stats(eigs, lam)
            laguerre_rows.append({
                "n_dims": d, "ridge_lambda": lam,
                "df_mean": mean, "df_std": std, "df_min": lo, "df_max": hi,
            })
        print(f"laguerre d={d} done")

    laguerre_df = pd.DataFrame(laguerre_rows)
    laguerre_path = os.path.join(OUT_DIR, "df_lambda_laguerre.csv")
    laguerre_df.to_csv(laguerre_path, index=False)
    print(f"wrote {os.path.abspath(laguerre_path)}")

    rnn_rows = []
    for d, lambdas in RNN_LAMBDAS.items():
        train_rng = np.random.default_rng(1000 * d + 0)
        train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=d)
        for k in RNN_K_VALUES:
            weight_rng = np.random.default_rng(100_000 + d * 1000 + 0)
            basis = make_random_features_basis(weight_rng, n_dims=d, n_hidden=k)
            eigs = eigs_per_step(basis, train_paths.S)
            for lam in lambdas:
                mean, std, lo, hi = df_stats(eigs, lam)
                rnn_rows.append({
                    "n_dims": d, "basis_n_hidden": k, "ridge_lambda": lam,
                    "df_mean": mean, "df_std": std, "df_min": lo, "df_max": hi,
                })
        print(f"rnn d={d} done")

    rnn_df = pd.DataFrame(rnn_rows)
    rnn_path = os.path.join(OUT_DIR, "df_lambda_rnn.csv")
    rnn_df.to_csv(rnn_path, index=False)
    print(f"wrote {os.path.abspath(rnn_path)}")
