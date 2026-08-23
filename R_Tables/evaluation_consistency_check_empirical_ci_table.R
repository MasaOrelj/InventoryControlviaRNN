library(readr)
library(dplyr)
library(kableExtra)

df <- read_csv("results/evaluation_consistency_check.csv", show_col_types = FALSE)

# Empirical (studentized-pivot) interval, replacing the symmetric CLT interval
# V0 +/- 1.96*SE with the asymmetric V0 - q_97.5%*SE <= V <= V0 - q_2.5%*SE,
# where q_2.5%/q_97.5% are the EMPIRICAL quantiles of T_k = (v0_k - Vbar)/SE_k
# across the 100 draws per (dimension, basis) cell -- Vbar is the mean over
# all 100 draws, SE_k is each draw's OWN CLT SE (path_level_sd_k/sqrt(M_e)).
# T is (approximately) pivotal, so its empirical distribution -- not the
# normal's -- gives an honestly-calibrated interval for a SINGLE future run's
# own V0/SE, capturing the asymmetry (fat left tail / thin right tail) found
# in Table~\ref{tab:evaluation_consistency_check} that a symmetric z=1.96
# interval can't represent.
result_table <- df %>%
  group_by(n_dims, basis_type) %>%
  mutate(vbar = mean(v0)) %>%
  ungroup() %>%
  mutate(se_k = path_level_sd / sqrt(n_paths_eval), t_k = (v0 - vbar) / se_k) %>%
  group_by(n_dims, basis_type) %>%
  summarise(
    q_025 = quantile(t_k, 0.025),
    q_975 = quantile(t_k, 0.975),
    .groups = "drop"
  ) %>%
  mutate(
    alg = case_when(
      basis_type == "rnn"      ~ "RNN",
      basis_type == "laguerre" ~ "Laguerre (deg 2)",
      TRUE                     ~ basis_type
    )
  ) %>%
  arrange(n_dims, alg) %>%
  select(n_dims, alg, q_025, q_975)

print(result_table)

n_dims_vec  <- result_table$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- result_table %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "clcc",
    digits = c(0, 0, 2, 2),
    col.names = c("Dim", "Alg", "$q_{2.5\\%}$", "$q_{97.5\\%}$"),
    caption = "Empirical quantiles of the studentized pivot $T_k=(\\tilde V_0^{(k)}-\\bar V)/\\widehat{SE}_k$ across the 100 draws underlying Table~\\ref{tab:evaluation_consistency_check}, compared to the normal distribution's $\\pm1.96$. The resulting empirical 95\\% interval for a single future run is $[\\tilde V_0 - q_{97.5\\%}\\cdot\\widehat{SE},\\ \\tilde V_0 - q_{2.5\\%}\\cdot\\widehat{SE}]$, replacing the symmetric CLT interval $\\tilde V_0 \\pm 1.96\\cdot\\widehat{SE}$. Every cell has $q_{97.5\\%}<1.96<|q_{2.5\\%}|$, reflecting the fat-left/thin-right asymmetry of $T_k$'s distribution (itself a consequence of the SE skew in Table~\\ref{tab:evaluation_consistency_check_dispersion}): the interval must reach further above $\\tilde V_0$ than below it to maintain 95\\% coverage.",
    label = "evaluation_consistency_check_empirical_ci",
    linesep = linesep_vec,
    escape = FALSE
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/evaluation_consistency_check_empirical_ci_table.tex")
