"""Training-sample-size sensitivity: RNN vs Laguerre convergence speed as a
function of M_t, at d in {1, 10, 25}, state=S, plain fit (no ridge), with the
SAME fixed evaluation sample (M_e=10000, per dimension) throughout.
Mentor-suggested follow-up:
Evaluation_Sample_Size_Experiment.py freezes ONE already-fitted policy and
only varies M_e -- no refitting happens there, so it can never show a
convergence-with-M_t effect. This experiment isolates exactly that: how fast
each architecture's fitted policy converges as training data grows.

M_T_VALUES = [250, 1000, 5000, 10000], N_REPS=10. NESTED training samples
(matching the Evaluation_Sample_Size_Experiment.py fix): for each
(n_dims, rep), ONE training draw at the largest size (10000) is simulated,
and each smaller M_t uses a PREFIX of that SAME draw -- not a fresh
independent sample per M_t. So within one rep, growing M_t means "reveal
more of the same underlying paths", giving a genuinely converging sequence
per rep rather than four unrelated independent estimates. RNN's random
feature weights are tied to (n_dims, rep) only (not M_t), for the same
reason -- one rep's fit uses one consistent random projection across all
four M_t levels drawn from its training sample.

Run: python -m scripts.Training_Sample_Sensitivity_Experiment
"""
import os
import time

import numpy as np

from core.Basis_Functions import WeightedLaguerreBasis, make_random_features_basis
from core.Electricity_Market_Model import HHKParams, simulate_hhk
from core.Payoff_Aggregation import max_aggregation
from core.Regression import evaluate_policy, fit_policy, least_squares
from core.Swing import SwingContract
from scripts.Experiment_Log import append_result_row

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "training_sample_sensitivity_experiment.csv")

MARKET_PARAMS = HHKParams(
    kappa=7.0, sigma=1.4, beta=40.0,
    lam_up=5.0, mu_up=0.6, lam_down=3.0, mu_down=0.4,
)
CONTRACT = SwingContract(K=100.0, q_min=0.0, q_max=50.0, q_tilde=25.0, L=10)
R = 0.02
MATURITY = 1.0
N_STEPS = 50
ALPHA = np.exp(-R * MATURITY / N_STEPS)

DIMS = [1, 10, 25]
BASIS_TYPES = ["laguerre", "rnn"]
LAGUERRE_DEGREE = 2
LAGUERRE_K = 1.0
RNN_N_HIDDEN = 20

M_T_VALUES = [250, 1_000, 5_000, 10_000]
N_REPS = 10
M_E = 10_000

EVAL_SEED_BASE = 700_000     # fixed per dimension, independent of M_t/rep/basis
TRAIN_SEED_BASE = 300_000    # train seed = TRAIN_SEED_BASE + n_dims*1_000_000 + rep (no m_t -- nested)
WEIGHT_SEED_BASE = 400_000   # RNN weight seed, same (n_dims, rep) key -> tied to that training draw


def make_basis(basis_type: str, n_dims: int, weight_rng: np.random.Generator):
    if basis_type == "laguerre":
        return WeightedLaguerreBasis(n_dims=n_dims, degree=LAGUERRE_DEGREE, K=LAGUERRE_K), LAGUERRE_DEGREE
    return make_random_features_basis(weight_rng, n_dims=n_dims, n_hidden=RNN_N_HIDDEN), None


if __name__ == "__main__":
    print(f"M_t in {M_T_VALUES}, M_e={M_E}, {N_REPS} reps, dims={DIMS}, basis={BASIS_TYPES}, state=S, fit=plain")
    print(f"logging to {os.path.abspath(CSV_PATH)}\n")

    max_m_t = max(M_T_VALUES)

    for n_dims in DIMS:
        eval_rng = np.random.default_rng(EVAL_SEED_BASE + n_dims)
        eval_paths = simulate_hhk(
            eval_rng, MARKET_PARAMS, n_paths=M_E, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
        )

        t0 = time.time()
        for rep in range(N_REPS):
            seed_key = n_dims * 1_000_000 + rep   # no m_t component -- shared across the nested M_t levels

            # ONE training draw at the largest M_t, per (n_dims, rep); each
            # smaller M_t below is a PREFIX of this same draw.
            train_rng = np.random.default_rng(TRAIN_SEED_BASE + seed_key)
            train_paths_full = simulate_hhk(
                train_rng, MARKET_PARAMS, n_paths=max_m_t, n_steps=N_STEPS, maturity=MATURITY, n_dims=n_dims,
            )

            for basis_type in BASIS_TYPES:
                # RNN weights fixed for this (n_dims, rep) -- shared across
                # all four nested M_t levels below, redrawn next rep.
                weight_rng = np.random.default_rng(WEIGHT_SEED_BASE + seed_key)
                basis, basis_degree = make_basis(basis_type, n_dims, weight_rng)

                for m_t in M_T_VALUES:
                    S_train = train_paths_full.S[:, :m_t, :]

                    t_run = time.time()
                    policy = fit_policy(
                        S_train=S_train, regression_state_train=S_train, contract=CONTRACT,
                        aggregate=max_aggregation, basis=basis, fit=least_squares, alpha=ALPHA,
                        train_itm_only=False,
                    )
                    result = evaluate_policy(
                        policy, S_eval=eval_paths.S, regression_state_eval=eval_paths.S,
                        contract=CONTRACT, aggregate=max_aggregation, basis=basis, alpha=ALPHA,
                    )
                    duration_sec = time.time() - t_run

                    append_result_row(CSV_PATH, {
                        "rep": rep,
                        "kappa": MARKET_PARAMS.kappa, "sigma": MARKET_PARAMS.sigma, "beta": MARKET_PARAMS.beta,
                        "lam_up": MARKET_PARAMS.lam_up, "mu_up": MARKET_PARAMS.mu_up,
                        "lam_down": MARKET_PARAMS.lam_down, "mu_down": MARKET_PARAMS.mu_down,
                        "f_level": MARKET_PARAMS.f_level, "f_amp": MARKET_PARAMS.f_amp,
                        "f_period": MARKET_PARAMS.f_period,
                        "discount_rate": R, "maturity": MATURITY, "n_steps": N_STEPS,
                        "n_paths_train": m_t, "n_paths_eval": M_E, "n_dims": n_dims,
                        "K": CONTRACT.K, "q_min": CONTRACT.q_min, "q_max": CONTRACT.q_max,
                        "q_tilde": CONTRACT.q_tilde, "L": CONTRACT.L,
                        "regression_mode": "per-level", "state_input": "S",
                        "basis_type": basis_type, "basis_degree": basis_degree,
                        "basis_n_hidden": RNN_N_HIDDEN if basis_type == "rnn" else "",
                        "basis_activation": "tanh" if basis_type == "rnn" else "",
                        "basis_K": LAGUERRE_K if basis_type == "laguerre" else "",
                        "fit_type": "plain", "ridge_lambda": "",
                        "price": result["v0"], "duration_sec": duration_sec,
                    })

            print(f"  d={n_dims:2d} rep {rep + 1}/{N_REPS} done ({time.time() - t0:.1f}s elapsed)")
        print()
