"""Per-cell significance flags for the RNN K-1=30 L-sweep table (see
conversation) -- one table per dimension, L in {1,5,10,25,40} as columns,
lambda as rows. Same exact sign-flip methodology and "bold means ties both
best-price and best-SD" convention as Compute_Ridge_Lambda_Significance.py,
just with the reference cell chosen WITHIN EACH (n_dims, L) BLOCK separately
(not pooled across L), since L is a genuinely different contract, not a
comparable axis like K-1 was.

Sources: L=1 from rnn_ridge_lambda_L1_experiment.csv, L=5/25/40 from
rnn_ridge_lambda_L_sweep_experiment.csv (filtered by L), L=10 from
rnn_ridge_lambda_experiment.csv (filtered to basis_n_hidden==30). All at
K-1=30 only.

Run: python -m scripts.Compute_RNN_L_Sweep_K30_Significance
"""
import itertools
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "rnn_ridge_lambda_L_sweep_K30_significance.csv")

ALPHA = 0.05
K_HIDDEN = 30
EXCLUDED_LAMBDAS = {10.0, 15.0, 20.0}


def _sign_flip_price_p(x: np.ndarray, y: np.ndarray) -> float:
    if np.array_equal(x, y):
        return 1.0
    D = x - y
    n = len(D)
    D_obs = D.mean()
    count = total = 0
    for signs in itertools.product([-1, 1], repeat=n):
        eps = np.array(signs)
        if abs((eps * D).mean()) >= abs(D_obs) - 1e-9:
            count += 1
        total += 1
    return count / total


def _sign_flip_sd_p(x: np.ndarray, y: np.ndarray) -> float:
    if np.array_equal(x, y):
        return 1.0
    S = x + y
    D = x - y
    n = len(D)

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a**2).sum() * (b**2).sum())
        return 0.0 if denom == 0 else (a * b).sum() / denom

    r_obs = corr(S, D)
    count = total = 0
    for signs in itertools.product([-1, 1], repeat=n):
        eps = np.array(signs)
        if abs(corr(S, eps * D)) >= abs(r_obs) - 1e-9:
            count += 1
        total += 1
    return count / total


def _significance(cells: dict) -> dict:
    means = {key: arr.mean() for key, arr in cells.items()}
    sds = {key: arr.std(ddof=1) for key, arr in cells.items()}
    sd_ref = cells[min(sds, key=sds.get)]
    price_ref = cells[max(means, key=means.get)]
    return {
        key: (_sign_flip_sd_p(arr, sd_ref), _sign_flip_price_p(arr, price_ref))
        for key, arr in cells.items()
    }


def _load(L: int) -> pd.DataFrame:
    if L == 1:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "rnn_ridge_lambda_L1_experiment.csv"))
    elif L == 10:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "rnn_ridge_lambda_experiment.csv"))
    else:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "rnn_ridge_lambda_L_sweep_experiment.csv"))
        df = df[df.L == L]
    df = df[df.basis_n_hidden == K_HIDDEN]
    return df[~df.ridge_lambda.isin(EXCLUDED_LAMBDAS)]


if __name__ == "__main__":
    rows = []
    for L in [1, 5, 10, 25, 40]:
        df = _load(L)
        for n_dims, dim_df in df.groupby("n_dims"):
            cells = {}
            for (fit_type, lam), cell_df in dim_df.groupby(["fit_type", "ridge_lambda"], dropna=False):
                cells[(fit_type, lam)] = cell_df.sort_values("rep")["price"].to_numpy()

            sig = _significance(cells)

            for key, arr in cells.items():
                fit_type, lam = key
                p_sd, p_price = sig[key]
                pass_sd, pass_price = p_sd >= ALPHA, p_price >= ALPHA
                rows.append({
                    "n_dims": n_dims, "L": L, "fit_type": fit_type, "ridge_lambda": lam,
                    "price": arr.mean(), "sd": arr.std(ddof=1), "n_reps": len(arr),
                    "p_sd_vs_min_sd": p_sd, "p_price_vs_max_price": p_price,
                    "pass_sd": pass_sd, "pass_price": pass_price, "bold": pass_sd and pass_price,
                })
        print(f"L={L} done")

    out = pd.DataFrame(rows).sort_values(["n_dims", "L", "ridge_lambda"], na_position="first")
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {os.path.abspath(OUT_PATH)}")
