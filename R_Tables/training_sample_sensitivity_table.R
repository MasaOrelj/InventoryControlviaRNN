library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/training_sample_sensitivity_experiment.csv")

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

# Blank spacing line before each new Dim block, computed from where n_dims
# actually changes -- not a fixed row count, so it stays correct regardless
# of how many M_t rows end up in each dimension's block.
n_dims_vec  <- wide$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- wide %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rrccccc",
    digits = c(0, 0, 1, 1, 1, 1, 2),
    col.names = c("Dim", "$M_t$", "Price", "SD", "Price", "SD", "Rel. Diff (\\%)"),
    caption = "Training-sample-size sensitivity: Laguerre (degree 2) vs.\\ RNN, plain fit, state $S$, fixed evaluation sample per dimension ($M_e=10{,}000$), 10 repetitions per (dimension, $M_t$) cell. Rel.\\ Diff.\\ (\\%) is $(\\text{Price}_{RNN} - \\text{Price}_{Laguerre})/\\text{Price}_{Laguerre} \\times 100$: positive means RNN prices higher than Laguerre.",
    label = "training_sample_sensitivity",
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

writeLines(latex_str, "R_Tables/training_sample_sensitivity_table.tex")
