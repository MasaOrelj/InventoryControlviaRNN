"""Per-cell significance flags for the Laguerre L-sweep table (see
conversation) -- mirrors Compute_RNN_L_Sweep_K30_Significance.py exactly
(same exact/Monte-Carlo sign-flip methodology, same "bold means ties both
best-price and best-SD" convention, reference chosen within each (n_dims, L)
block separately), for the Laguerre (degree 2) basis instead of RNN.

Rep counts are NOT uniform across cells here (several one-off backfills this
session left some cells at n=20, n=13, or n=40 while their neighbors stayed
at n=10 -- see conversation), so every pairwise comparison is restricted to
the two cells' COMMON rep indices, not their full (possibly mismatched)
sample, and the sign-flip test switches from exact enumeration to a 200,000-
sample Monte Carlo approximation once that common n exceeds 20 (2^n exact
enumeration is infeasible beyond that -- established earlier this session).

Run: python -m scripts.Compute_Laguerre_L_Sweep_Significance
"""
import itertools
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "laguerre_ridge_lambda_L_sweep_significance.csv")

ALPHA = 0.05
MC_SAMPLES = 200_000
MC_SEED = 0
EXCLUDED_LAMBDAS_BY_DIM = {10: {0.5}, 25: {0.5, 15.0, 20.0}}


def _sign_flip_price_p(x: np.ndarray, y: np.ndarray) -> float:
    if np.array_equal(x, y):
        return 1.0
    D = x - y
    n = len(D)
    D_obs = D.mean()
    if n <= 20:
        count = total = 0
        for signs in itertools.product([-1, 1], repeat=n):
            eps = np.array(signs)
            if abs((eps * D).mean()) >= abs(D_obs) - 1e-9:
                count += 1
            total += 1
        return count / total
    rng = np.random.default_rng(MC_SEED)
    eps = rng.choice([-1, 1], size=(MC_SAMPLES, n))
    stats = (eps * D).mean(axis=1)
    return (np.abs(stats) >= abs(D_obs) - 1e-9).mean()


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
    if n <= 20:
        count = total = 0
        for signs in itertools.product([-1, 1], repeat=n):
            eps = np.array(signs)
            if abs(corr(S, eps * D)) >= abs(r_obs) - 1e-9:
                count += 1
            total += 1
        return count / total
    rng = np.random.default_rng(MC_SEED)
    count = 0
    for _ in range(MC_SAMPLES):
        eps = rng.choice([-1, 1], size=n)
        if abs(corr(S, eps * D)) >= abs(r_obs) - 1e-9:
            count += 1
    return count / MC_SAMPLES


def _significance(cells: dict) -> dict:
    """cells: {key: pd.Series indexed by rep}. Comparisons use each pair's
    COMMON rep index only."""
    means = {key: s.mean() for key, s in cells.items()}
    sds = {key: s.std(ddof=1) for key, s in cells.items()}
    sd_ref_key = min(sds, key=sds.get)
    price_ref_key = max(means, key=means.get)
    sd_ref, price_ref = cells[sd_ref_key], cells[price_ref_key]

    out = {}
    for key, s in cells.items():
        common_sd = sorted(set(s.index) & set(sd_ref.index))
        common_price = sorted(set(s.index) & set(price_ref.index))
        p_sd = _sign_flip_sd_p(s.loc[common_sd].to_numpy(), sd_ref.loc[common_sd].to_numpy())
        p_price = _sign_flip_price_p(s.loc[common_price].to_numpy(), price_ref.loc[common_price].to_numpy())
        out[key] = (p_sd, p_price)
    return out


def _load(L: int) -> pd.DataFrame:
    fname = "laguerre_ridge_lambda_experiment.csv" if L == 10 else f"laguerre_ridge_lambda_L{L}_experiment.csv"
    return pd.read_csv(os.path.join(RESULTS_DIR, fname))


if __name__ == "__main__":
    rows = []
    for L in [1, 5, 10, 25, 40]:
        df = _load(L)
        for n_dims, dim_df in df.groupby("n_dims"):
            excluded = EXCLUDED_LAMBDAS_BY_DIM.get(n_dims, set())
            dim_df = dim_df[~dim_df.ridge_lambda.isin(excluded)]
            cells = {}
            for (fit_type, lam), cell_df in dim_df.groupby(["fit_type", "ridge_lambda"], dropna=False):
                cells[(fit_type, lam)] = cell_df.sort_values("rep").set_index("rep")["price"]

            sig = _significance(cells)

            for key, s in cells.items():
                fit_type, lam = key
                p_sd, p_price = sig[key]
                pass_sd, pass_price = p_sd >= ALPHA, p_price >= ALPHA
                rows.append({
                    "n_dims": n_dims, "L": L, "fit_type": fit_type, "ridge_lambda": lam,
                    "price": s.mean(), "sd": s.std(ddof=1), "n_reps": len(s),
                    "p_sd_vs_min_sd": p_sd, "p_price_vs_max_price": p_price,
                    "pass_sd": pass_sd, "pass_price": pass_price, "bold": pass_sd and pass_price,
                })
        print(f"L={L} done")

    out = pd.DataFrame(rows).sort_values(["n_dims", "L", "ridge_lambda"], na_position="first")
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {os.path.abspath(OUT_PATH)}")
