"""RNN ridge-lambda story as figures instead of a 5x9x2 table grid (see
conversation): relative difference of RNN price vs. a FIXED Laguerre
benchmark (that dimension's own chosen regularization: d=1 plain, d=10
lambda=0.1, d=25 lambda=1) on the y-axis, df(lambda) on the x-axis (log
scale), one line per K-1 (hidden units), one SEPARATE figure per dimension,
with a shaded error band from RNN's own across-repetition SD.

Run: python -m scripts.Ridge_Df_Comparison_Plot
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
RNN_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_ridge_lambda_experiment.csv")
LAGUERRE_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "laguerre_ridge_lambda_experiment.csv")
DF_RNN_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "df_lambda_rnn.csv")

# That dimension's own chosen Laguerre regularization, used as the fixed
# benchmark price RNN is compared against (see conversation).
LAGUERRE_BENCHMARK_CONFIG = {1: ("plain", None), 10: ("ridge", 0.1), 25: ("ridge", 1.0)}
DIMS = [1, 10, 25]
K_VALUES = [10, 20, 30, 40, 50]
COLORS = {10: "tab:blue", 20: "tab:orange", 30: "tab:green", 40: "tab:red", 50: "tab:purple"}

if __name__ == "__main__":
    rnn = pd.read_csv(RNN_CSV)
    laguerre = pd.read_csv(LAGUERRE_CSV)
    df_rnn = pd.read_csv(DF_RNN_CSV)

    # Fixed Laguerre benchmark price per dimension.
    benchmark = {}
    for d, (fit_type, lam) in LAGUERRE_BENCHMARK_CONFIG.items():
        if fit_type == "plain":
            sub = laguerre[(laguerre.n_dims == d) & (laguerre.fit_type == "plain")]
        else:
            sub = laguerre[(laguerre.n_dims == d) & (laguerre.fit_type == "ridge") & (laguerre.ridge_lambda == lam)]
        benchmark[d] = sub.price.mean()

    os.makedirs(OUT_DIR, exist_ok=True)

    for d in DIMS:
        bench = benchmark[d]
        rnn_d = rnn[(rnn.n_dims == d) & (rnn.fit_type == "ridge")]
        df_d = df_rnn[df_rnn.n_dims == d]

        fig, ax = plt.subplots(figsize=(7, 5.5))

        rnn_all = rnn[rnn.n_dims == d]   # includes plain rows, needed for the df=K point below

        for k in K_VALUES:
            g = rnn_d[rnn_d.basis_n_hidden == k].groupby("ridge_lambda")["price"].agg(["mean", "std"]).reset_index()
            g = g.merge(df_d[df_d.basis_n_hidden == k][["ridge_lambda", "df_mean"]], on="ridge_lambda")
            g = g.sort_values("df_mean")

            rel_diff = (g["mean"] - bench) / bench * 100
            rel_sd = g["std"] / bench * 100

            ax.plot(g["df_mean"], rel_diff, marker="o", markersize=4, linewidth=1.2,
                    color=COLORS[k], label=f"$K-1={k}$")
            ax.fill_between(g["df_mean"], rel_diff - rel_sd, rel_diff + rel_sd,
                             color=COLORS[k], alpha=0.15)

            # Plain (lambda=0) fit: df = K = (K-1) hidden units + 1 constant,
            # exactly -- unregularized OLS uses every feature fully, by
            # definition, no need to estimate it. Plotted as an open marker
            # connected by a dashed segment, so it reads as a separate
            # regime from the ridge-regularized curve (see conversation).
            plain_row = rnn_all[(rnn_all.basis_n_hidden == k) & (rnn_all.fit_type == "plain")]
            plain_price_mean = plain_row.price.mean()
            plain_price_sd = plain_row.price.std()
            plain_rel_diff = (plain_price_mean - bench) / bench * 100
            plain_df = k + 1

            last_df, last_rel = g["df_mean"].iloc[-1], rel_diff.iloc[-1]
            ax.plot([last_df, plain_df], [last_rel, plain_rel_diff],
                    linestyle="--", linewidth=1.0, color=COLORS[k])
            ax.plot(plain_df, plain_rel_diff, marker="o", markersize=6,
                    markerfacecolor="none", markeredgecolor=COLORS[k], markeredgewidth=1.5)

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=100))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.set_xlabel(r"df($\lambda$)")
        ax.set_ylabel("RNN price relative to Laguerre benchmark (%)")
        ax.set_title(f"RNN Regularization Analysis ($d={d}$)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"ridge_df_comparison_d{d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {os.path.abspath(out_path)}  (Laguerre benchmark price = {bench:.2f})")
