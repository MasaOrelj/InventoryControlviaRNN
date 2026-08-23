library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

# Joint (B) counterpart to training_sample_sensitivity_table.R -- same shape
# (Dim / M_t rows, Laguerre vs. RNN price/SD side by side, Rel. Diff. column),
# but sourced from the two E1 joint-mode experiments instead of the original
# separate (A) one. Laguerre joint: d=1,10 at 10 reps, d=25 at 5 reps (cost
# reasons, see E1_Inventory_Mode_Experiment_Laguerre.py). RNN joint: 10 reps
# at every dimension, filtered out of the RNN E1 experiment (which also has
# the separate-mode rows, unlike the Laguerre one).
laguerre <- read_csv("results/e1_inventory_mode_laguerre_experiment.csv") %>%
  mutate(basis_type = "laguerre")

rnn <- read_csv("results/e1_inventory_mode_rnn_experiment.csv") %>%
  filter(regression_mode == "joint") %>%
  mutate(basis_type = "rnn")

df <- bind_rows(laguerre, rnn)

result_table <- df %>%
  group_by(n_dims, n_paths_train, basis_type) %>%
  summarise(
    sd    = sd(price),
    price = mean(price),
    .groups = "drop"
  ) %>%
  arrange(n_dims, n_paths_train, basis_type)

print(result_table)

wide <- result_table %>%
  pivot_wider(
    id_cols = c(n_dims, n_paths_train),
    names_from = basis_type,
    values_from = c(price, sd),
    names_glue = "{basis_type}_{.value}"
  ) %>%
  mutate(rel_diff = (rnn_price - laguerre_price) * 100 / laguerre_price) %>%
  arrange(n_dims, n_paths_train) %>%
  select(n_dims, n_paths_train, laguerre_price, laguerre_sd, rnn_price, rnn_sd, rel_diff)

n_dims_vec  <- wide$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- wide %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rrccccc",
    digits = c(0, 0, 1, 1, 1, 1, 2),
    col.names = c("Dim", "$M_t$", "Price", "SD", "Price", "SD", "Rel. Diff (\\%)"),
    caption = "Training-sample-size sensitivity under the JOINT (variable-inventory) formulation: Laguerre (degree 2) vs.\\ RNN, plain fit, state $S$, fixed evaluation sample per dimension ($M_e=10{,}000$); 10 repetitions per (dimension, $M_t$) cell except Laguerre at $d=25$ (5 repetitions). Rel.\\ Diff.\\ (\\%) is $(\\text{Price}_{RNN} - \\text{Price}_{Laguerre})/\\text{Price}_{Laguerre} \\times 100$: positive means RNN prices higher than Laguerre. Companion to Table~\\ref{tab:training_sample_sensitivity} (there: fixed/separate inventory).",
    label = "e1_joint_sensitivity",
    linesep = linesep_vec,
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 2, "Laguerre (deg 2)" = 2, "RNN" = 2, " " = 1), escape = FALSE)

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/e1_joint_sensitivity_table.tex")
