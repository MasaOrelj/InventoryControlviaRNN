"""E1 convergence plot: RNN price vs. M_t, separate (A) vs. joint (B)
inventory formulation, one panel per dimension, with a shaded error band
from the across-repetition SD. Mirrors the "one figure carries the argument"
lesson already applied to the ridge-lambda comparison (see conversation).

Run: python -m scripts.E1_Inventory_Mode_Plot
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "e1_inventory_mode_rnn_experiment.csv")

DIMS = [1, 10, 25]
COLORS = {"per-level": "tab:blue", "joint": "tab:orange"}
LABELS = {"per-level": "Fixed", "joint": "Variable"}
BASIS_LABEL = "RNN"

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    os.makedirs(OUT_DIR, exist_ok=True)

    for d in DIMS:
        sub = df[df.n_dims == d]
        fig, ax = plt.subplots(figsize=(7, 5.5))

        for mode in ["per-level", "joint"]:
            g = sub[sub.regression_mode == mode].groupby("n_paths_train")["price"].agg(["mean", "std"]).reset_index()
            g = g.sort_values("n_paths_train")
            ax.errorbar(
                g["n_paths_train"], g["mean"], yerr=g["std"],
                marker="o", markersize=5, linewidth=1.5, capsize=4,
                color=COLORS[mode], label=LABELS[mode],
            )

        ax.set_xscale("log")
        ax.set_xlabel(r"$M_t$")
        ax.set_ylabel("Price")
        ax.set_title(f"{BASIS_LABEL} Training Sample Sensitivity and Convergence ($d={d}$)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"e1_inventory_mode_rnn_d{d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {os.path.abspath(out_path)}")
