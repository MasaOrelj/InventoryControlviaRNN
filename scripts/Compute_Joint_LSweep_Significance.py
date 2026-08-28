"""Paired-difference significance for the joint L-sweep table (see
conversation): for each (n_dims, L) cell, RNN and Laguerre share the SAME
40 training seeds (Joint_LSweep_40Rep_Experiment.py's train_rng depends only
on (n_dims, rep), not basis_type -- "compare on identical paths"), so
RNN_price[rep] - Laguerre_price[rep] is a genuine paired difference, not two
independent samples.

Same exact/Monte-Carlo sign-flip test as
Compute_Ridge_Lambda_Significance.py / the RNN L-sweep significance scripts:
D_i = RNN_i - Laguerre_i, D_obs = mean(D). Enumerate all 2^n sign vectors
(exact, n=40 -> infeasible, so Monte Carlo with 200,000 samples instead,
matching every other n=40 comparison this session). p = fraction of
|mean(eps*D)| >= |D_obs|.

Run: python -m scripts.Compute_Joint_LSweep_Significance
"""
import itertools
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "joint_lsweep_40rep_experiment.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "joint_lsweep_significance.csv")

MC_SAMPLES = 200_000
MC_SEED = 0


def sign_flip_price_p(D: np.ndarray) -> float:
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


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    rows = []
    for d in [1, 10, 25]:
        for L in [1, 5, 10, 25, 40]:
            lag = df[(df.n_dims == d) & (df.L == L) & (df.basis_type == "laguerre")].sort_values("rep")["price"].to_numpy()
            rnn = df[(df.n_dims == d) & (df.L == L) & (df.basis_type == "rnn")].sort_values("rep")["price"].to_numpy()
            D = rnn - lag
            p = sign_flip_price_p(D)
            rows.append({
                "n_dims": d, "L": L,
                "mean_diff": D.mean(), "sd_diff": D.std(ddof=1),
                "n": len(D), "p_value": p, "significant": p < 0.05,
            })
            print(f"d={d:2d} L={L:2d}: mean_diff={D.mean():+8.2f}  sd_diff={D.std(ddof=1):7.2f}  p={p:.4f}  {'*' if p < 0.05 else ''}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {os.path.abspath(OUT_PATH)}")
