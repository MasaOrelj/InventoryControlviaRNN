library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/e1_inventory_mode_rnn_experiment.csv", show_col_types = FALSE)

# Same E1 RNN data as e1_inventory_mode_rnn_table.R, reshaped wide -- Dim/M_t
# rows, Fixed / Variable as side-by-side column pairs, matching
# training_sample_sensitivity_table.R's layout (basis side by side) rather
# than mode-as-rows.
result_table <- df %>%
  group_by(n_dims, n_paths_train, regression_mode) %>%
  summarise(
    sd    = sd(price),      # computed before price is overwritten by its own mean below
    price = mean(price),
    .groups = "drop"
  )

wide <- result_table %>%
  pivot_wider(
    id_cols = c(n_dims, n_paths_train),
    names_from = regression_mode,
    values_from = c(price, sd),
    names_glue = "{regression_mode}_{.value}"
  ) %>%
  arrange(n_dims, n_paths_train) %>%
  select(n_dims, n_paths_train, `per-level_price`, `per-level_sd`, joint_price, joint_sd)

print(wide)

n_dims_vec  <- wide$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- wide %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rrcccc",
    digits = c(0, 0, 1, 1, 1, 1),
    col.names = c("Dim", "$M_t$", "Price", "SD", "Price", "SD"),
    caption = "E1: RNN training-sample convergence, fixed vs.\\ variable inventory formulation, side by side (plain fit, $K-1=20$, $M_e=10{,}000$, 10 repetitions per cell).",
    label = "e1_inventory_mode_rnn_wide",
    linesep = linesep_vec,
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 2, "Fixed" = 2, "Variable" = 2))

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/e1_inventory_mode_rnn_wide_table.tex")
