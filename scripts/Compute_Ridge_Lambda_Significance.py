"""Per-cell significance flags for the RNN ridge-lambda tables
(rnn_ridge_lambda_table_d*.tex): within EACH dimension's table (plain + the
lambda grid, all 5 K-1 values -- 30 cells), find the cell with the smallest
SD and the cell with the largest Price, then test every other cell against
those two references, on the SAME reps (train/eval paths are shared across
every (K-1, lambda) cell at a given dimension and rep -- see
RNN_Ridge_Lambda_Experiment.py -- so these are paired, not independent,
samples).

EXACT sign-flip permutation tests (not the parametric paired t-test /
Pitman-Morgan test used in an earlier version of this script -- see
conversation): with only n=10 (or 5) reps, the t-distribution step in both
parametric tests is only an approximation, justified by assuming the
underlying differences are close to normal. A validation check against real
data here showed the SD test in particular was NOT robust to that assumption
-- e.g. RNN d=10, K-1=10, lambda=0.1 vs. the table's best-SD cell: parametric
Pitman-Morgan gave p=0.0072 (significant), the exact test gives p=0.469 (not
even close) -- a reversed conclusion, not a minor correction. Price held up
much better under the same check (CLT gives means better small-sample
robustness than a correlation-derived statistic gets from its
t-approximation), but is now tested exactly too for consistency.

For each pair (cell X, reference Y), let D_i = X_i - Y_i, S_i = X_i + Y_i
(i = 1..n reps). Enumerate ALL 2^n sign vectors eps in {-1,+1}^n (exact, not
Monte Carlo -- trivial at n=10 or n=5):

- Price: Dbar(eps) = mean(eps_i * D_i). p = fraction of the 2^n vectors with
  |Dbar(eps)| >= |Dbar_obs| (eps = all +1, i.e. the actual observed mean
  difference).
- SD: r(eps) = corr(S, eps*D) -- S held fixed, D sign-flipped. r(eps=all +1)
  is exactly the Pitman-Morgan r already used elsewhere in this project;
  flipping D_i's sign while holding S_i fixed is exactly "swap which of the
  pair is X vs Y for that rep", the correct symmetry under H0: equal
  variance. p = fraction of |r(eps)| >= |r_obs|.

A cell is flagged "bold" only if BOTH p-values are >= 0.05, i.e. that cell's
SD is not distinguishable from the table's best (lowest) SD AND its price is
not distinguishable from the table's best (highest) price. The reference
cells themselves trivially pass (p=1, no test run -- comparing a sample to
itself isn't a meaningful test).

pass_sd / pass_price expose each individual criterion (p >= ALPHA) on its
own, for tables (e.g. d=25, where NO cell is ever "bold" against the FULL
lambda grid) where it's still useful to show which single criterion a cell
ties on -- colored rather than bolded, since bold is reserved for cells
passing both at once.

bold_adjusted / pass_sd_adjusted / pass_price_adjusted repeat the exact same
two tests, but with the reference cells (and every comparison) recomputed
from a RESTRICTED row set that excludes ridge_lambda in EXCLUDED_LAMBDAS
(only present at d=25) -- i.e. "if the lambda=10,15,20 rows didn't exist,
which cells would tie on both criteria". For d=1/d=10 (which never have
those lambda rows to begin with) this is identical to the unrestricted
bold/pass_sd/pass_price, so it's harmless to compute uniformly.

d=25 uses 5 reps (all others 10), matching RNN_Ridge_Lambda_Experiment.py's
DIMS_AND_REPS -- deflates the tests' power there, so expect fewer bolded
cells to actually mean fewer TRUE distinctions, not just less data.

Run: python -m scripts.Compute_Ridge_Lambda_Significance
"""
import itertools
import os

import numpy as np
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_ridge_lambda_experiment.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_ridge_lambda_significance.csv")

ALPHA = 0.05
EXCLUDED_LAMBDAS = {10.0, 15.0, 20.0}


def _sign_flip_price_p(x: np.ndarray, y: np.ndarray) -> float:
    """Exact sign-flip test on the paired mean difference. Replaces the
    paired t-test -- see module docstring."""
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
    """Exact sign-flip test on corr(S, D) -- replaces the Pitman-Morgan
    (parametric t-distribution) test -- see module docstring."""
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


def _significance(cells: dict) -> dict:
    """cells: {(k, fit_type, lam): price_array}. Returns {key: (p_sd, p_price)}."""
    means = {key: arr.mean() for key, arr in cells.items()}
    sds = {key: arr.std(ddof=1) for key, arr in cells.items()}
    sd_ref = cells[min(sds, key=sds.get)]
    price_ref = cells[max(means, key=means.get)]
    return {
        key: (_sign_flip_sd_p(arr, sd_ref), _sign_flip_price_p(arr, price_ref))
        for key, arr in cells.items()
    }


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    rows = []
    for n_dims, dim_df in df.groupby("n_dims"):
        cells = {}
        for (k, fit_type, lam), cell_df in dim_df.groupby(["basis_n_hidden", "fit_type", "ridge_lambda"], dropna=False):
            cells[(k, fit_type, lam)] = cell_df.sort_values("rep")["price"].to_numpy()

        cells_adjusted = {key: arr for key, arr in cells.items() if key[2] not in EXCLUDED_LAMBDAS or pd.isna(key[2])}

        sig_full = _significance(cells)
        sig_adjusted = _significance(cells_adjusted)

        for key, arr in cells.items():
            k, fit_type, lam = key
            p_sd, p_price = sig_full[key]
            pass_sd, pass_price = p_sd >= ALPHA, p_price >= ALPHA

            if key in sig_adjusted:
                p_sd_adj, p_price_adj = sig_adjusted[key]
                pass_sd_adj, pass_price_adj = p_sd_adj >= ALPHA, p_price_adj >= ALPHA
            else:
                p_sd_adj = p_price_adj = np.nan
                pass_sd_adj = pass_price_adj = False

            rows.append({
                "n_dims": n_dims, "basis_n_hidden": k, "fit_type": fit_type, "ridge_lambda": lam,
                "price": arr.mean(), "sd": arr.std(ddof=1),
                "p_sd_vs_min_sd": p_sd, "p_price_vs_max_price": p_price,
                "pass_sd": pass_sd, "pass_price": pass_price, "bold": pass_sd and pass_price,
                "p_sd_adjusted": p_sd_adj, "p_price_adjusted": p_price_adj,
                "pass_sd_adjusted": pass_sd_adj, "pass_price_adjusted": pass_price_adj,
                "bold_adjusted": pass_sd_adj and pass_price_adj,
            })

    out = pd.DataFrame(rows).sort_values(["n_dims", "basis_n_hidden", "ridge_lambda"], na_position="first")
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {os.path.abspath(OUT_PATH)}")
    print(out.to_string(index=False))
