"""CSV logging for experiment runs: one row per single repetition, with every
parameter that run used, so results can be exported (e.g. to R) for further
analysis without needing to re-derive the config from code.
"""

import csv
import os

FIELDNAMES = [
    "rep",
    # market model parameters (HHK electricity)
    "kappa", "sigma", "beta", "lam_up", "mu_up", "lam_down", "mu_down",
    "f_level", "f_amp", "f_period", "discount_rate", "maturity", "n_steps",
    "n_paths_train", "n_paths_eval", "n_dims",
    # swing option parameters
    "K", "q_min", "q_max", "q_tilde", "L",
    # regression configuration
    "state_input",       # "S" or "ZY"
    "regression_mode",   # "per-level" (inventory fixed per regression) or "joint" (inventory as a regressor)
    "basis_type",        # "poly", "laguerre", or "rnn"
    "basis_degree",      # poly, laguerre
    "basis_n_hidden",    # rnn only
    "basis_activation",  # rnn only
    "basis_K",           # laguerre only -- its own input-rescaling constant, unrelated to "K" (the strike) above
    "fit_type",          # "plain" or "ridge"
    "ridge_lambda",      # ridge only
    # results
    "price",
    "duration_sec",
]


def append_result_row(csv_path: str, row: dict) -> None:
    """Append one row to csv_path, writing the header first if the file is new
    (does not exist yet). `row` may omit keys that don't apply to that run
    (e.g. basis_degree when basis_type="rnn") -- they're written as ''."""
    is_new_file = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="raise")
        if is_new_file:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
