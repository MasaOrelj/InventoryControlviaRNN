library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/state_representation_experiment_n10000.csv")

wide_by_rep <- df %>%
  select(rep, n_dims, basis_type, fit_type, ridge_lambda, state_input, price, duration_sec) %>%
  pivot_wider(names_from = state_input, values_from = c(price, duration_sec))

result_table <- wide_by_rep %>%
  group_by(n_dims, basis_type, fit_type, ridge_lambda) %>%
  summarise(
    S_mean            = mean(price_S),
    ZY_mean           = mean(price_ZY),
    rel_diff          = (ZY_mean - S_mean) * 100 / S_mean,
    S_std             = sd(price_S),
    ZY_std            = sd(price_ZY),
    S_mean_duration   = mean(duration_sec_S),
    ZY_mean_duration  = mean(duration_sec_ZY),
    n_reps            = n(),
    .groups = "drop"
  ) %>%
  arrange(n_dims, basis_type, fit_type)

print(result_table)

# Blank spacing line right before each NEW n_dims block starts -- computed
# from where n_dims actually changes, not a fixed "every 4th row" pattern,
# since d=25 currently only has 2 rows (RNN only, poly not run there yet)
# while d=1/d=10 have 4 (poly+rnn x plain+ridge). A fixed-period pattern
# would misalign the spacing once block sizes aren't all equal.
n_dims_vec <- result_table$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- result_table %>%
  mutate(
    lambda = ifelse(is.na(ridge_lambda), "-", as.character(ridge_lambda)),
    alg    = case_when(
      basis_type == "poly" ~ "Pre-selection",
      basis_type == "rnn"  ~ "RNN",
      TRUE                 ~ basis_type
    )
  ) %>%
  select(n_dims, alg, lambda,
         S_mean, S_std, S_mean_duration,
         ZY_mean, ZY_std, ZY_mean_duration,
         rel_diff) %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    digits = c(0, 0, 0, 1, 1, 2, 1, 1, 2, 2),
    col.names = c("Dim", "Alg", "$\\lambda$",
                  "Price", "SD", "Duration",
                  "Price", "SD", "Duration",
                  "Rel. Diff (\\%)"),
    caption = "Comparison of state representations $(S_n,I_n)$ and \\ $(Z_n,Y_n,I_n)$ with $10$ repetitions per experiment.",
    label = "state_representation",
    linesep = linesep_vec,
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 3,
                     "$(S_n,I_n)$" = 3,
                     "$(Z_n,Y_n,I_n)$" = 3,
                     " " = 1), escape = FALSE)

# Inject \footnotesize and \setlength{\tabcolsep}{4pt} right after \caption{...},
# before \centering -- kableExtra's own styling API doesn't produce this exact
# preamble, so this is a direct, surgical string edit on the generated LaTeX
# rather than fighting kable_styling()'s higher-level (less precise) options.
latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/state_representation_table.tex")
