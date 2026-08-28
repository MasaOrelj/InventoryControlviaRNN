"""Exploratory analysis (see conversation): for the MAIN calibrated market
(lam_up=5.0, lam_down=3.0 -- the standard, not the "rare spikes" diagnostic),
at each dimension d in {1,10,25}, fit a policy (RNN K-1=30 and Laguerre deg
2, each at their established L=10 chosen lambda) and forward-simulate it on
the evaluation sample, tracking each path's ACTUAL exercise trajectory
(starting at inventory L, decrementing only when that path's OWN fitted
decision at its OWN current inventory level says swing) -- something
evaluate_policy itself doesn't expose, since its backward recursion computes
cash flows for every possible inventory level in parallel rather than
following one real trajectory per path.

For every actual exercise, classifies it "up" (the (S-K)(q_max-q_tilde) term
of Swing.swing_gain was the larger, i.e. price above strike -> increase
consumption) or "down" (the (K-S)(q_tilde-q_min) term was larger, price
below strike -> decrease consumption), reading the per-dimension gain
BEFORE aggregation so at d>1 the classification uses whichever dimension
actually won max_aggregation's max at that exercise event.

Single rep (rep=0) per (basis, d) -- exploratory, not a reps-averaged
statistic.

L is taken from the command line (default 10). Chosen lambda per (basis, d,
L) is imported directly from Joint_LSweep_40Rep_Experiment.py's own
RNN_LAMBDA/LAGUERRE_LAMBDA tables, so this never risks drifting out of sync
with the settled regularization scheme.

Run: python -m scripts.Swing_UpDown_Classification [L]
"""
import sys

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis, make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import fit_policy, least_squares, ridge_regression
from core.Swing import SwingContract, inventory_grid
from scripts.Joint_LSweep_40Rep_Experiment import LAGUERRE_LAMBDA as _LAGUERRE_LAMBDA_BY_DL
from scripts.Joint_LSweep_40Rep_Experiment import RNN_LAMBDA as _RNN_LAMBDA_BY_DL
import functools

SWING_L = int(sys.argv[1]) if len(sys.argv) > 1 else 10

MARKET_PARAMS = HHKParams(kappa=7.0, sigma=1.4, beta=40.0, lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=SWING_L)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
N_SAMPLES = 10_000
ALPHA = np.exp(-R * MATURITY / N_STEPS)

N_HIDDEN = 30
LAGUERRE_DEGREE = 2
LAGUERRE_K = 1.0

# Established chosen lambda per basis/dimension at THIS L (see conversation).
RNN_LAMBDA = {d: _RNN_LAMBDA_BY_DL[(d, SWING_L)] for d in (1, 10, 25)}
LAGUERRE_LAMBDA = {d: _LAGUERRE_LAMBDA_BY_DL[(d, SWING_L)] for d in (1, 10, 25)}

EVAL_SEED = 500_000
WEIGHT_SEED_BASE = 100_000
REP = 0


def make_fit(lam):
    if lam == "plain":
        return least_squares
    return functools.partial(ridge_regression, ridge_lambda=lam)


def classify_and_simulate(basis_type: str, n_dims: int, lam) -> dict:
    train_rng = np.random.default_rng(1000 * n_dims + REP)
    train_paths = simulate_hhk(train_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)
    eval_rng = np.random.default_rng(EVAL_SEED + n_dims)
    eval_paths = simulate_hhk(eval_rng, MARKET_PARAMS, n_paths=N_SAMPLES, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims)

    if basis_type == "laguerre":
        basis = WeightedLaguerreBasis(n_dims=n_dims, degree=LAGUERRE_DEGREE, K=LAGUERRE_K)
    else:
        weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + n_dims * 1000 + REP)
        basis = make_random_features_basis(weight_rng, n_dims=n_dims, n_hidden=N_HIDDEN)

    fit = make_fit(lam)
    policy = fit_policy(
        S_train=train_paths.S, regression_state_train=train_paths.S, contract=CONTRACT,
        aggregate=max_aggregation, basis=basis, fit=fit, alpha=ALPHA, train_itm_only=False,
    )

    S_eval = eval_paths.S   # (N+1, M_e, d)
    M_e = S_eval.shape[1]
    L = CONTRACT.L

    # Per-dimension gain BEFORE aggregation, so the winning dimension at each
    # exercise event can be identified.
    up = (S_eval - CONTRACT.K) * (CONTRACT.q_max - CONTRACT.q_tilde)
    down = (CONTRACT.K - S_eval) * (CONTRACT.q_tilde - CONTRACT.q_min)
    Q_per_dim = np.maximum(np.maximum(up, down), 0.0)
    Q_total = Q_per_dim.max(axis=-1)   # matches max_aggregation

    current_i = np.full(M_e, L, dtype=int)
    up_count = 0
    down_count = 0
    # Index N_STEPS holds the terminal, automatic collection (see below) --
    # arrays sized N_STEPS+1 so it has its own slot alongside the N_STEPS
    # interior decision dates (indices 1..N_STEPS-1).
    up_by_step = np.zeros(N_STEPS + 1, dtype=int)
    down_by_step = np.zeros(N_STEPS + 1, dtype=int)

    for n in range(1, N_STEPS):
        grid = inventory_grid(n, L)
        state = S_eval[n]
        mean, std = policy.standardize_stats[n]
        Phi = basis.build_features(basis.standardize(state, mean, std))
        immediate = Q_total[n]

        for i in grid:
            if i == 0:
                continue
            mask = current_i == i
            if not mask.any():
                continue
            beta0, beta1 = policy.betas[(n, i)]
            c0 = Phi @ beta0
            c1 = Phi @ beta1
            swing = (immediate > 0) & (immediate + c1 > c0) & mask

            if swing.any():
                winning_dim = Q_per_dim[n][swing].argmax(axis=-1)
                up_wins = up[n][swing, :][np.arange(swing.sum()), winning_dim] >= \
                    down[n][swing, :][np.arange(swing.sum()), winning_dim]
                n_up = int(up_wins.sum())
                n_down = int((~up_wins).sum())
                up_count += n_up
                down_count += n_down
                up_by_step[n] += n_up
                down_by_step[n] += n_down
                current_i[swing] -= 1

    # Terminal date (n=N_STEPS): g_N(x) = 1{i>=1} * Q_total(s) -- automatic,
    # not a discretionary decision like the interior dates (no Swing.decide,
    # no Q_total>0 guard), so every path that still has inventory left
    # collects it. Still a real exercise-equivalent event and gets classified
    # up/down and counted the same way (see conversation).
    terminal_mask = current_i >= 1
    if terminal_mask.any():
        winning_dim = Q_per_dim[N_STEPS][terminal_mask].argmax(axis=-1)
        up_wins = up[N_STEPS][terminal_mask, :][np.arange(terminal_mask.sum()), winning_dim] >= \
            down[N_STEPS][terminal_mask, :][np.arange(terminal_mask.sum()), winning_dim]
        n_up = int(up_wins.sum())
        n_down = int((~up_wins).sum())
        up_count += n_up
        down_count += n_down
        up_by_step[N_STEPS] += n_up
        down_by_step[N_STEPS] += n_down

    # Volume deviation from baseline q_tilde per swing: +（q_max-q_tilde) for
    # an up swing (consume at q_max that date), -(q_tilde-q_min) for a down
    # swing (consume at q_min) -- constant per swing given the contract, so
    # total volume is just counts x these magnitudes.
    up_unit = CONTRACT.q_max - CONTRACT.q_tilde
    down_unit = CONTRACT.q_tilde - CONTRACT.q_min

    return {
        "up": up_count, "down": down_count, "total": up_count + down_count,
        "up_by_step": up_by_step, "down_by_step": down_by_step,
        "up_volume": up_count * up_unit, "down_volume": down_count * down_unit,
        "net_volume": up_count * up_unit - down_count * down_unit,
        "up_volume_by_step": up_by_step * up_unit, "down_volume_by_step": down_by_step * down_unit,
    }


QUARTER_EDGES = [(1, 13), (13, 25), (25, 37), (37, N_STEPS + 1)]   # last quarter includes the terminal (index N_STEPS) collection


def print_result(label: str, r: dict) -> None:
    total = max(r["total"], 1)
    print(f"  {label}: up={r['up']:6d}  down={r['down']:6d}  total={r['total']:6d}  ({100*r['up']/total:.1f}% up)")
    print(f"    joint purchased energy: up_volume={r['up_volume']:9.1f}  down_volume={r['down_volume']:9.1f}  net_volume={r['net_volume']:+9.1f}")
    quarter_strs = []
    for lo, hi in QUARTER_EDGES:
        qu = r["up_by_step"][lo:hi].sum()
        qd = r["down_by_step"][lo:hi].sum()
        qt = max(qu + qd, 1)
        qnetv = r["up_volume_by_step"][lo:hi].sum() - r["down_volume_by_step"][lo:hi].sum()
        quarter_strs.append(f"[{lo}-{hi-1}]: up={qu} down={qd} ({100*qu/qt:.0f}% up, net_vol={qnetv:+.0f})")
    print("    by quarter: " + "   ".join(quarter_strs))
    tu, td = r["up_by_step"][N_STEPS], r["down_by_step"][N_STEPS]
    print(f"    of which terminal (automatic, i>=1 at expiry): up={tu} down={td}")


if __name__ == "__main__":
    print(f"Swing up/down classification, main calibration (lam_up={MARKET_PARAMS.lam_up}, lam_down={MARKET_PARAMS.lam_down}), L={SWING_L}, rep={REP}\n")

    for d in [1, 10, 25]:
        r_rnn = classify_and_simulate("rnn", d, RNN_LAMBDA[d])
        r_lag = classify_and_simulate("laguerre", d, LAGUERRE_LAMBDA[d])
        print(f"d={d:2d}  RNN(lam={RNN_LAMBDA[d]}):")
        print_result("RNN", r_rnn)
        print(f"       Laguerre(lam={LAGUERRE_LAMBDA[d]}):")
        print_result("Laguerre", r_lag)
        print()
