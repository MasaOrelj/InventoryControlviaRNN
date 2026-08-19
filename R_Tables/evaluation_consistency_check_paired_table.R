library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/evaluation_consistency_check.csv", show_col_types = FALSE)

# Paired RNN - Laguerre difference, same 100 shared draws per dimension:
# sharing the evaluation sample doesn't shrink either price's own marginal
# SE (see evaluation_consistency_check_table.R's empirical_sd) -- it makes
# the two errors correlated, so the SE of their DIFFERENCE collapses instead.
# The empirical mean/SD come directly from this table's own 100 draws (paired
# by draw); first_draw_se is the single-draw paired-cashflow CLT SE from
# Evaluation_Consistency_Check_Paired_SE.py's companion CSV -- these two
# should roughly agree, the same way each basis's own empirical_sd already
# agrees with its first_draw_se in the main consistency-check table.
paired_se <- read_csv("results/evaluation_consistency_check_paired_se.csv", show_col_types = FALSE)

wide_v0 <- df %>%
  select(n_dims, basis_type, draw, v0) %>%
  pivot_wider(names_from = basis_type, values_from = v0)

result_table <- wide_v0 %>%
  mutate(diff = rnn - laguerre) %>%
  group_by(n_dims) %>%
  summarise(
    empirical_mean = mean(diff),
    empirical_sd   = sd(diff),
    .groups = "drop"
  ) %>%
  left_join(paired_se %>% select(n_dims, first_draw_se = paired_first_draw_se), by = "n_dims") %>%
  mutate(ratio = empirical_sd / first_draw_se) %>%
  arrange(n_dims)

print(result_table)

latex_table <- result_table %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rcccc",
    digits = c(0, 1, 1, 1, 2),
    col.names = c("Dim", "Mean $\\tilde V_0^{RNN} - \\tilde V_0^{Lag}$",
                  "Empirical SD", "CLT SE", "Ratio"),
    caption = "Paired RNN $-$ Laguerre difference, same 100 shared evaluation draws per dimension ($M_t=10{,}000$, $M_e=10{,}000$). Sharing the evaluation sample does not shrink either price's own marginal SE (Table~\\ref{tab:evaluation_consistency_check}), but it makes the two errors correlated, so the SE of their difference is far smaller than either marginal SE alone.",
    label = "evaluation_consistency_check_paired",
    escape = FALSE
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/evaluation_consistency_check_paired_table.tex")
