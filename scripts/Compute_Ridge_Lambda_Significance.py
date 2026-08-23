"""Per-cell significance flags for the RNN ridge-lambda tables
(rnn_ridge_lambda_table_d*.tex): within EACH dimension's table (plain + the
lambda grid, all 5 K-1 values -- 30 cells), find the cell with the smallest
SD and the cell with the largest Price, then test every other cell against
those two references, on the SAME reps (train/eval paths are shared across
every (K-1, lambda) cell at a given dimension and rep -- see
RNN_Ridge_Lambda_Experiment.py -- so these are paired, not independent,
samples):

- SD vs. the table's minimum SD: Pitman-Morgan paired test for equal
  variances, via r = corr(X+Y, X-Y), t = r*sqrt(n-2)/sqrt(1-r^2), df=n-2.
- Price vs. the table's maximum price: paired t-test on the price
  differences, df=n-1.

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
import os

import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_ridge_lambda_experiment.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_ridge_lambda_significance.csv")

ALPHA = 0.05
EXCLUDED_LAMBDAS = {10.0, 15.0, 20.0}


def _pitman_morgan_p(x: np.ndarray, y: np.ndarray) -> float:
    if np.array_equal(x, y):
        return 1.0
    n = len(x)
    r, _ = stats.pearsonr(x + y, x - y)
    t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
    return float(2 * stats.t.sf(abs(t), n - 2))


def _paired_t_p(x: np.ndarray, y: np.ndarray) -> float:
    if np.array_equal(x, y):
        return 1.0
    _, p = stats.ttest_rel(x, y)
    return float(p)


def _significance(cells: dict) -> dict:
    """cells: {(k, fit_type, lam): price_array}. Returns {key: (p_sd, p_price)}."""
    means = {key: arr.mean() for key, arr in cells.items()}
    sds = {key: arr.std(ddof=1) for key, arr in cells.items()}
    sd_ref = cells[min(sds, key=sds.get)]
    price_ref = cells[max(means, key=means.get)]
    return {
        key: (_pitman_morgan_p(arr, sd_ref), _paired_t_p(arr, price_ref))
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
