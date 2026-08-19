library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/training_sample_sensitivity_experiment.csv", show_col_types = FALSE)

# Per (dimension, basis, M_t): mean price across the 10 reps, then expressed
# as a % difference from that SAME (dimension, basis)'s own M_t=10000 price
# -- i.e. how close does each smaller training sample get to the largest one,
# per architecture, per dimension.
result_table <- df %>%
  group_by(n_dims, n_paths_train, basis_type) %>%
  summarise(price = mean(price), .groups = "drop") %>%
  group_by(n_dims, basis_type) %>%
  mutate(price_at_10000 = price[n_paths_train == 10000]) %>%
  ungroup() %>%
  mutate(rel_diff = (price - price_at_10000) * 100 / price_at_10000) %>%
  arrange(n_dims, n_paths_train, basis_type)

print(result_table)

wide <- result_table %>%
  select(n_dims, n_paths_train, basis_type, rel_diff) %>%
  pivot_wider(
    id_cols = c(n_dims, n_paths_train),
    names_from = basis_type,
    values_from = rel_diff,
    names_glue = "{basis_type}_rel_diff"
  ) %>%
  arrange(n_dims, n_paths_train) %>%
  select(n_dims, n_paths_train, laguerre_rel_diff, rnn_rel_diff)

# Blank spacing line before each new Dim block, computed from where n_dims
# actually changes -- not a fixed row count.
n_dims_vec  <- wide$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- wide %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rrcc",
    digits = c(0, 0, 2, 2),
    col.names = c("Dim", "$M_t$", "Laguerre (deg 2)", "RNN"),
    caption = "Training-sample-size convergence: price at each $M_t$ expressed as a percentage difference from that same (dimension, basis)'s own price at $M_t=10{,}000$, i.e.\\ $(\\text{Price}_{M_t} - \\text{Price}_{M_t=10{,}000})/\\text{Price}_{M_t=10{,}000} \\times 100$. Same underlying data as Table~\\ref{tab:training_sample_sensitivity}; the $M_t=10{,}000$ row is $0$ by construction.",
    label = "training_sample_sensitivity_relative",
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

writeLines(latex_str, "R_Tables/training_sample_sensitivity_relative_table.tex")
