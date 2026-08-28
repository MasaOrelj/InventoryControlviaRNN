"""df(lambda) for the spikes-off RNN arms (see conversation) -- same formula
and per-step averaging as Compute_Df_Lambda.py, just with MARKET_PARAMS'
lam_up=lam_down=0 and the exact lambda grid / seeds used by
RNN_K30_D{1,10,25}_NoSpikes_Experiment.py (rep 0, K-1=30), so these df values
are directly comparable to the spikes-off price/SD tables.

Run: python -m scripts.Compute_Df_Lambda_NoSpikes
"""
import os

import numpy as np
import pandas as pd

from core.Basis_Functions import make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=0.0, mu_up=0.6, lam_down=0.0, mu_down=0.4)
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
N_HIDDEN = 30
LAMBDA_VALUES = [0.001, 0.01, 0.1, 0.3, 0.7, 1.0, 5.0]
DIMS = [1, 10, 25]

WEIGHT_SEED_BASE = 100_000
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def eigs_per_step(basis, train_paths_S):
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
    rows = []
    for d in DIMS:
        train_rng = np.random.default_rng(1000 * d + 0)
        train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=d)
        weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + d * 1000 + 0)
        basis = make_random_features_basis(weight_rng, n_dims=d, n_hidden=N_HIDDEN)
        eigs = eigs_per_step(basis, train_paths.S)
        for lam in LAMBDA_VALUES:
            mean, std, lo, hi = df_stats(eigs, lam)
            rows.append({
                "n_dims": d, "basis_n_hidden": N_HIDDEN, "ridge_lambda": lam,
                "df_mean": mean, "df_std": std, "df_min": lo, "df_max": hi,
            })
        print(f"d={d} done")

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "df_lambda_rnn_nospikes.csv")
    df.to_csv(path, index=False)
    print(f"wrote {os.path.abspath(path)}")
