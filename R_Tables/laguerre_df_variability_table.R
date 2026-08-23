library(readr)
library(dplyr)
library(kableExtra)

df_lambda <- read_csv("results/df_lambda_laguerre.csv", show_col_types = FALSE)

# Step-to-step variability of df(lambda) across the N-1 backward-induction
# steps (companion to Tables~\ref{tab:laguerre_ridge_lambda}, which show only
# the mean): df(lambda) is fit separately per step, and each step's own Phi
# is built from that step's own training-row distribution, so df(lambda)
# itself is not a single fixed quantity throughout the contract.
result_table <- df_lambda %>%
  mutate(rel_std = df_std / df_mean * 100) %>%
  arrange(n_dims, ridge_lambda) %>%
  select(n_dims, ridge_lambda, df_mean, df_std, rel_std, df_min, df_max)

print(result_table)

n_dims_vec  <- result_table$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- result_table %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rrcccccc",
    digits = c(0, 3, 2, 2, 1, 2, 2),
    col.names = c("Dim", "$\\lambda$", "df mean", "df SD", "Rel.\\ SD (\\%)", "df min", "df max"),
    caption = "Step-to-step variability of df($\\lambda$) for the Laguerre ridge $\\lambda$ grid (companion to Table~\\ref{tab:laguerre_ridge_lambda}, which reports only the mean).",
    label = "laguerre_df_variability",
    linesep = linesep_vec,
    escape = FALSE
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/laguerre_df_variability_table.tex")
