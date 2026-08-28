"""Per-cell significance flags for the Laguerre ridge-lambda table, mirroring
Compute_Ridge_Lambda_Significance.py's RNN version -- same exact sign-flip
permutation tests (see that script's docstring for the full justification:
the parametric paired t-test / Pitman-Morgan test's t-distribution step is
only an approximation, and a validation check showed it overstates
significance for the SD test in particular at small n).

Laguerre never had a bold/significance column at all before this -- unlike
the RNN table, this is a new addition, not a swap of an existing test. Makes
equal sense here: the paired structure comes from the SAME seed convention
(train_rng depends only on (n_dims, rep), shared across every lambda), so
the exact test's assumptions are identically satisfied.

Within EACH dimension's own lambda grid (Laguerre has no K-1 sweep --
degree=2 is fixed, so there's no second axis to grid over the way RNN's
K-1 does), find the cell with the smallest SD and the cell with the largest
Price, then test every other cell against those two references.

Run: python -m scripts.Compute_Laguerre_Ridge_Lambda_Significance
"""
import itertools
import os

import numpy as np
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "laguerre_ridge_lambda_experiment.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "laguerre_ridge_lambda_significance.csv")

ALPHA = 0.05


def _sign_flip_price_p(x: np.ndarray, y: np.ndarray) -> float:
    if np.array_equal(x, y):
        return 1.0
    D = x - y
    n = len(D)
    D_obs = D.mean()
    count = 0
    total = 0
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
    count = 0
    total = 0
    for signs in itertools.product([-1, 1], repeat=n):
        eps = np.array(signs)
        if abs(corr(S, eps * D)) >= abs(r_obs) - 1e-9:
            count += 1
        total += 1
    return count / total


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    rows = []
    for n_dims, dim_df in df.groupby("n_dims"):
        cells = {}
        for (fit_type, lam), cell_df in dim_df.groupby(["fit_type", "ridge_lambda"], dropna=False):
            cells[(fit_type, lam)] = cell_df.sort_values("rep")["price"].to_numpy()

        means = {key: arr.mean() for key, arr in cells.items()}
        sds = {key: arr.std(ddof=1) for key, arr in cells.items()}
        sd_ref = cells[min(sds, key=sds.get)]
        price_ref = cells[max(means, key=means.get)]

        for key, arr in cells.items():
            fit_type, lam = key
            p_sd = _sign_flip_sd_p(arr, sd_ref)
            p_price = _sign_flip_price_p(arr, price_ref)
            pass_sd, pass_price = p_sd >= ALPHA, p_price >= ALPHA
            rows.append({
                "n_dims": n_dims, "fit_type": fit_type, "ridge_lambda": lam,
                "price": arr.mean(), "sd": arr.std(ddof=1),
                "p_sd_vs_min_sd": p_sd, "p_price_vs_max_price": p_price,
                "pass_sd": pass_sd, "pass_price": pass_price, "bold": pass_sd and pass_price,
            })

    out = pd.DataFrame(rows).sort_values(["n_dims", "ridge_lambda"], na_position="first")
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {os.path.abspath(OUT_PATH)}")
    print(out.to_string(index=False))
