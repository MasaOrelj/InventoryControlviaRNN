"""Plot for Beta_Spike_Sensitivity_Experiment_D10.py: same as
Beta_Spike_Sensitivity_Plot.py but for the d=10 follow-up, reusing its
styling helpers directly.

Run: python -m scripts.Beta_Spike_Sensitivity_D10_Plot
"""
import os

import pandas as pd

from scripts.Beta_Spike_Sensitivity_Plot import OUT_DIR, _style_axis, COLORS, LABELS
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "beta_spike_sensitivity_experiment_d10.csv")


def plot_absolute(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for basis_type in ["laguerre", "rnn"]:
        g = df[df.basis_type == basis_type].groupby("half_life_over_delta")["diff"].agg(["mean", "std"]).reset_index()
        g = g.sort_values("half_life_over_delta")
        ax.errorbar(
            g["half_life_over_delta"], g["mean"], yerr=g["std"],
            marker="o", markersize=4, linewidth=1.2, capsize=3,
            color=COLORS[basis_type], label=LABELS[basis_type],
        )
    _style_axis(
        ax, df,
        ylabel="spike contribution to price  " r"$\tilde V_0^{\mathrm{with\ spikes}} - \tilde V_0^{\mathrm{no\ spikes}}$",
        title="Spike Contribution to Swing Option Value vs. " r"$\beta$" " (d=10)",
    )
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "beta_spike_sensitivity_d10.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.abspath(out_path)}")


def plot_relative(df: pd.DataFrame) -> None:
    df = df.copy()
    df["rel_diff_pct"] = df["diff"] / df["price_no_spikes"] * 100.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for basis_type in ["laguerre", "rnn"]:
        g = df[df.basis_type == basis_type].groupby("half_life_over_delta")["rel_diff_pct"].agg(["mean", "std"]).reset_index()
        g = g.sort_values("half_life_over_delta")
        ax.errorbar(
            g["half_life_over_delta"], g["mean"], yerr=g["std"],
            marker="o", markersize=4, linewidth=1.2, capsize=3,
            color=COLORS[basis_type], label=LABELS[basis_type],
        )
    _style_axis(
        ax, df,
        ylabel="Relative price difference (%)",
        title="Relative Swing Option Price Difference Between\nSpiking and Non-Spiking Electricity Markets vs. " r"$\beta$" " (d=10)",
    )
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "beta_spike_sensitivity_d10_relative.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    plot_absolute(df)
    plot_relative(df)
