"""Diagnostic plots for market.py: sample paths + mean/variance vs. theory for Z, Y, f, S.

Not part of the LSM pipeline -- a standalone script to visually sanity-check the
HHK simulator, mirroring the checks in the old reference `model_tests.py`. Run:

    python -m scripts.plot_market_sanity_checks
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from Electricity_Market_Model import DEFAULT_PARAMS, HHKParams, seasonality, simulate_hhk

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def run_checks(
    params: HHKParams, n_paths: int, n_steps: int, maturity: float, tag: str, z0: float = 1.0,
) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = simulate_hhk(rng, params, n_paths=n_paths, n_steps=n_steps, maturity=maturity, z0=z0)

    t = paths.t
    Z = paths.Z[:, :, 0]   # (n_steps+1, n_paths) -- squeeze the single market dimension
    Y = paths.Y[:, :, 0]
    S = paths.S[:, :, 0]
    f = seasonality(t, params)

    # 1) One sample path: Z and Y together.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, Z[:, 0], linewidth=1.0, alpha=0.8, label="Z (OU)")
    ax.plot(t, Y[:, 0], linewidth=1.0, alpha=0.8, label="Y (spikes)")
    ax.set_title(f"[{tag}] Z and Y: one sample path")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("Z(t), Y(t)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_sample_path_ZY.png"))
    plt.close(fig)

    # 2) One sample path: exp(f) and S together.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, np.exp(f), linewidth=1.0, alpha=0.8, label="exp(f)")
    ax.plot(t, S[:, 0], linewidth=1.0, alpha=0.8, label="S")
    ax.set_title(f"[{tag}] S and exp(f): one sample path")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("exp(f(t)), S(t)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_sample_path_S.png"))
    plt.close(fig)

    # 3) Mean of Z vs. theory: E[Z_t] = z0 * exp(-kappa*t).
    z_mean_theory = z0 * np.exp(-params.kappa * t)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, Z.mean(axis=1), label="sample mean Z(t)")
    ax.plot(t, z_mean_theory, linestyle="--", label="theory")
    ax.set_title(f"[{tag}] Mean of Z(t) vs. theory")
    ax.set_xlabel("time (years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_mean_Z.png"))
    plt.close(fig)

    # 4) Mean of Y vs. theory: E[Y_t] = (lam_up*mu_up - lam_down*mu_down)/beta * (1 - exp(-beta t)).
    up_rate = params.lam_up * params.mu_up
    down_rate = params.lam_down * params.mu_down
    y_mean_theory = (up_rate - down_rate) / params.beta * (1 - np.exp(-params.beta * t))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, Y.mean(axis=1), label="sample mean Y(t)")
    ax.plot(t, y_mean_theory, linestyle="--", label="theory")
    ax.set_title(f"[{tag}] Mean of Y(t) vs. theory")
    ax.set_xlabel("time (years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_mean_Y.png"))
    plt.close(fig)

    # 5) Variance of Z vs. theory (standard OU variance formula).
    z_var_theory = params.sigma**2 / (2 * params.kappa) * (1 - np.exp(-2 * params.kappa * t))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, Z.var(axis=1, ddof=1), label="empirical Var[Z(t)]")
    ax.plot(t, z_var_theory, linestyle="--", label="theory")
    ax.set_title(f"[{tag}] Variance of Z(t) vs. theory")
    ax.set_xlabel("time (years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_var_Z.png"))
    plt.close(fig)

    # 6) Variance of Y vs. theory: Var[compound Poisson OU] with Exp(mean mu) jump sizes,
    #    second moment of Exp(mean mu) is 2*mu^2. Up and down jumps add independently.
    up_var_rate = params.lam_up * (2.0 * params.mu_up**2)
    down_var_rate = params.lam_down * (2.0 * params.mu_down**2)
    y_var_theory = (up_var_rate + down_var_rate) / (2 * params.beta) * (1 - np.exp(-2 * params.beta * t))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, Y.var(axis=1, ddof=1), label="empirical Var[Y(t)]")
    ax.plot(t, y_var_theory, linestyle="--", label="theory")
    ax.set_title(f"[{tag}] Variance of Y(t) vs. theory")
    ax.set_xlabel("time (years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_var_Y.png"))
    plt.close(fig)

    # 7) Distribution of Y at t=1 (log-scale histogram, shows the jump-driven right tail).
    k = int(np.argmin(np.abs(t - 1.0)))
    y_samples = Y[k]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_samples, bins=80, density=True)
    ax.set_yscale("log")
    ax.set_title(f"[{tag}] Y at t={t[k]:.3f} (log y-scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_hist_Y.png"))
    plt.close(fig)

    print(f"[{tag}] Y at t={t[k]:.3f}: mean={y_samples.mean():.4f}, var={y_samples.var(ddof=1):.4f}, "
          f"min/max={y_samples.min():.4f}/{y_samples.max():.4f}")


if __name__ == "__main__":
    # Single-sided jumps, thesis reference parameters, z0=1.0 to make the Z-mean check visible.
    run_checks(DEFAULT_PARAMS, n_paths=20000, n_steps=400, maturity=2.0, tag="single_jump", z0=1.0)

    # Double-sided jumps (up + down spikes), for later swing-option low-swing behaviour.
    double_jump_params = HHKParams(
        kappa=7.0, sigma=1.4, beta=200.0,
        lam_up=5.0, mu_up=0.4, lam_down=5.0, mu_down=0.6,
    )
    run_checks(double_jump_params, n_paths=20000, n_steps=200, maturity=2.0, tag="double_jump")

    print(f"Plots written to {os.path.abspath(OUT_DIR)}")
