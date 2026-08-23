"""Plot for Beta_Spike_Sensitivity_Experiment.py: spike contribution to the
swing option value vs. spike half-life / exercise interval, both bases on one
plot. Not part of the LSM pipeline -- a standalone plotting script, mirroring
Electricity_Market_Plots.py's conventions.

Run: python -m scripts.Beta_Spike_Sensitivity_Plot
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "beta_spike_sensitivity_experiment.csv")

COLORS = {"laguerre": "tab:blue", "rnn": "tab:orange"}
LABELS = {"laguerre": "Laguerre (deg 2)", "rnn": "RNN"}


def _style_axis(ax, df, ylabel, title):
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.text(1.0, 0.96, "Ratio = 1", transform=ax.get_xaxis_transform(),
            color="gray", fontsize=9, ha="left", va="top",
            rotation=90, backgroundcolor="white")

    ax.set_xscale("log")

    ratios = sorted(df["half_life_over_delta"].unique())
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r:.2f}" for r in ratios], rotation=45, ha="right", fontsize=8)
    ax.minorticks_off()

    ax.set_xlabel(r"Ratio  $(\ln 2/\beta)/\Delta$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


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
        title="Spike contribution to swing option value vs. spike half-life (d=1)",
    )
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "beta_spike_sensitivity.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.abspath(out_path)}")


def plot_relative(df: pd.DataFrame) -> None:
    df = df.copy()
    df["rel_diff_pct"] = df["diff"] / df["price_no_spikes"] * 100.0   # per-rep relative contribution

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
        title="Relative Swing Option Price Difference Between\nSpiking and Non-Spiking Electricity Markets vs. " r"$\beta$" " (d=1)",
    )
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "beta_spike_sensitivity_relative.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    plot_absolute(df)
    plot_relative(df)
