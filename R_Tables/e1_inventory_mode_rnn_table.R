library(readr)
library(dplyr)
library(kableExtra)

df <- read_csv("results/e1_inventory_mode_rnn_experiment.csv", show_col_types = FALSE)

# E1: fixed (separate/per-level) vs. variable (joint) inventory formulation,
# RNN, plain fit. K = n_hidden+1 = 21 regardless of inventory_mode for RNN
# specifically -- unlike Laguerre/polynomial, RNN's feature count doesn't
# grow with input dimension (only Theta's SHAPE does), so it's included here
# for completeness but is not the "K grows" story the task brief's Laguerre
# side is about.
result_table <- df %>%
  mutate(k_features = basis_n_hidden + 1) %>%
  group_by(n_dims, n_paths_train, regression_mode) %>%
  summarise(
    mean_price = mean(price),
    sd_price   = sd(price),
    n_reps     = n(),
    mean_duration = mean(duration_sec),
    k_features = first(k_features),
    .groups = "drop"
  ) %>%
  mutate(
    se       = sd_price / sqrt(n_reps),
    ci_lower = mean_price - 1.96 * se,
    ci_upper = mean_price + 1.96 * se,
    mode_lab = ifelse(regression_mode == "per-level", "Fixed", "Variable"),
  ) %>%
  arrange(n_dims, n_paths_train, regression_mode == "joint")

print(result_table)

n_dims_vec <- result_table$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- result_table %>%
  select(n_dims, n_paths_train, mode_lab, mean_price, sd_price, ci_lower, ci_upper, k_features, mean_duration) %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rrlccccc",
    digits = c(0, 0, 0, 1, 1, 1, 1, 0, 2),
    col.names = c("Dim", "$M_t$", "Mode", "Price", "SD", "CI lower", "CI upper", "$K$", "Duration (s)"),
    caption = "E1: RNN training-sample convergence, fixed vs.\\ variable inventory formulation, plain fit, $K-1=20$ hidden units, $M_e=10{,}000$, 10 repetitions per cell. CI is the 95\\% CLT interval across repetitions ($\\bar{\\text{Price}} \\pm 1.96\\cdot\\text{SD}/\\sqrt{10}$). Duration is fit+eval wall-clock time, averaged over repetitions.",
    label = "e1_inventory_mode_rnn",
    linesep = linesep_vec,
    escape = FALSE
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/e1_inventory_mode_rnn_table.tex")
