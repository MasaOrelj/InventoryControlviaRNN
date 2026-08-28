"""Combine the original 10 reps (rnn_ridge_lambda_experiment.csv, K-1=30
subset) with the 30 new reps (10-39) from the two stability-check scripts
(RNN_K20_Stability_Check.py's K-1=30/d=10 portion, and
RNN_K30_Stability_Check_D1_D25.py's d=1/d=25 portion), plus the 40-rep
lambda=0.3/0.7 spot checks (RNN_K30_D10_Lambda03_07_Experiment.py,
RNN_K30_D25_Lambda03_07_Experiment.py -- see conversation: added to pin down
where d=10's true optimum sits between the existing 0.1/1.0 grid points, and
to give d=25 the same two points for comparison), into one 40-rep-per-cell
CSV, ready for a K-1=30 table across all three dimensions.

Run: python -m scripts.Combine_K30_40Rep_CSV
"""
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "rnn_ridge_lambda_K30_40rep_experiment.csv")

orig = pd.read_csv(os.path.join(RESULTS_DIR, "rnn_ridge_lambda_experiment.csv"))
orig = orig[orig.basis_n_hidden == 30]

d10_new = pd.read_csv(os.path.join(RESULTS_DIR, "rnn_k20_stability_check_experiment.csv"))
d10_new = d10_new[d10_new.basis_n_hidden == 30]

d1_d25_new = pd.read_csv(os.path.join(RESULTS_DIR, "rnn_k30_stability_check_d1_d25_experiment.csv"))

d10_lam03_07 = pd.read_csv(os.path.join(RESULTS_DIR, "rnn_k30_d10_lambda03_07_experiment.csv"))
d25_lam03_07 = pd.read_csv(os.path.join(RESULTS_DIR, "rnn_k30_d25_lambda03_07_experiment.csv"))

combined = pd.concat([orig, d10_new, d1_d25_new, d10_lam03_07, d25_lam03_07], ignore_index=True)
combined = combined.sort_values(["n_dims", "fit_type", "ridge_lambda", "rep"], na_position="first")

if __name__ == "__main__":
    combined.to_csv(OUT_PATH, index=False)
    print(f"wrote {os.path.abspath(OUT_PATH)}")
    print(combined.groupby(["n_dims", "fit_type", "ridge_lambda"], dropna=False).rep.nunique())
