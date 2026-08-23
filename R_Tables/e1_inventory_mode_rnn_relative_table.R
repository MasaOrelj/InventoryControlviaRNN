library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/e1_inventory_mode_rnn_experiment.csv", show_col_types = FALSE)

# Per (dimension, mode, M_t): mean price across the 10 reps, then expressed
# as a % difference from that SAME (dimension, mode)'s own price at
# M_t=10000 -- i.e. how close does each smaller training sample get to the
# largest one, per mode, per dimension. Mirrors
# training_sample_sensitivity_relative_table.R (there: per basis; here: per
# inventory mode).
result_table <- df %>%
  group_by(n_dims, n_paths_train, regression_mode) %>%
  summarise(price = mean(price), .groups = "drop") %>%
  group_by(n_dims, regression_mode) %>%
  mutate(price_at_10000 = price[n_paths_train == 10000]) %>%
  ungroup() %>%
  mutate(rel_diff = (price - price_at_10000) * 100 / price_at_10000) %>%
  arrange(n_dims, n_paths_train, regression_mode)

print(result_table)

wide <- result_table %>%
  select(n_dims, n_paths_train, regression_mode, rel_diff) %>%
  pivot_wider(
    id_cols = c(n_dims, n_paths_train),
    names_from = regression_mode,
    values_from = rel_diff,
    names_glue = "{regression_mode}_rel_diff"
  ) %>%
  arrange(n_dims, n_paths_train) %>%
  select(n_dims, n_paths_train, `per-level_rel_diff`, joint_rel_diff)

n_dims_vec  <- wide$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- wide %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rrcc",
    digits = c(0, 0, 2, 2),
    col.names = c("Dim", "$M_t$", "Fixed", "Variable"),
    caption = "E1: training-sample convergence, price at each $M_t$ expressed as a percentage difference from that same (dimension, mode)'s own price at $M_t=10{,}000$, i.e.\\ $(\\text{Price}_{M_t} - \\text{Price}_{M_t=10{,}000})/\\text{Price}_{M_t=10{,}000} \\times 100$. Same underlying data as Table~\\ref{tab:e1_inventory_mode_rnn_wide}; the $M_t=10{,}000$ row is $0$ by construction.",
    label = "e1_inventory_mode_rnn_relative",
    linesep = linesep_vec,
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 2, "Rel.\\\\ Diff.\\\\ vs.\\\\ $M_t=10{,}000$ (\\\\%)" = 2), escape = FALSE)

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/e1_inventory_mode_rnn_relative_table.tex")
