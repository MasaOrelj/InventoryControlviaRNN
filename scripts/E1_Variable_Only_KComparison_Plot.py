"""E1 follow-up (see conversation): does Variable/joint inventory mode's own
convergence rate improve as K-1 grows (20 -> 50 -> 100)? If parameter count
were the main reason Variable outpaces Fixed at small M_t, growing K-1 should
show up as a visible shift in Variable's own convergence curve. One panel per
dimension, Variable only, one line per K-1, plus a dotted red reference line
for Fixed (per-level) at K-1=20 -- at M_t=10000, Fixed sits above every
Variable K-1 for d=10/25 (see conversation), so the reference line shows how
much of that gap survives at each capacity and each M_t.

Run: python -m scripts.E1_Variable_Only_KComparison_Plot
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SOURCES = {
    20: os.path.join(RESULTS_DIR, "e1_inventory_mode_rnn_experiment.csv"),
    50: os.path.join(RESULTS_DIR, "e1_inventory_mode_rnn_K50_experiment.csv"),
    100: os.path.join(RESULTS_DIR, "e1_inventory_mode_rnn_K100_experiment.csv"),
}
FIXED_SOURCE = SOURCES[20]   # K-1=20 CSV also contains the per-level (Fixed) rows
DIMS = [1, 10, 25]
COLORS = {20: "tab:blue", 50: "tab:orange", 100: "tab:green"}

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    dfs = {k: pd.read_csv(path) for k, path in SOURCES.items()}
    fixed_df = pd.read_csv(FIXED_SOURCE)

    for d in DIMS:
        fig, ax = plt.subplots(figsize=(7, 5.5))

        for k, df in dfs.items():
            sub = df[(df.n_dims == d) & (df.regression_mode == "joint")]
            g = sub.groupby("n_paths_train")["price"].agg(["mean", "std"]).reset_index().sort_values("n_paths_train")
            ax.errorbar(
                g["n_paths_train"], g["mean"], yerr=g["std"],
                marker="o", markersize=5, linewidth=1.5, capsize=4,
                color=COLORS[k], label=f"K-1={k}",
            )

        fixed_sub = fixed_df[(fixed_df.n_dims == d) & (fixed_df.regression_mode == "per-level")]
        fixed_g = fixed_sub.groupby("n_paths_train")["price"].mean().reset_index().sort_values("n_paths_train")
        ax.plot(
            fixed_g["n_paths_train"], fixed_g["price"],
            linestyle=":", color="red", linewidth=2,
            label="Fixed K-1=20",
        )

        ax.set_xscale("log")
        ax.set_xlabel(r"$M_t$")
        ax.set_ylabel("Price")
        ax.set_title(f"Variable Inventory Convergence by Capacity ($d={d}$)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"e1_variable_only_kcomparison_d{d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {os.path.abspath(out_path)}")
