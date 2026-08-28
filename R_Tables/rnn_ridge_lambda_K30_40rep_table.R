library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/rnn_ridge_lambda_K30_40rep_experiment.csv")

# df(lambda) is provably L-invariant (see conversation) and precomputed once
# by scripts/Compute_Df_Lambda.py, not recomputed here.
df_lambda <- read_csv("results/df_lambda_rnn.csv") %>%
  filter(basis_n_hidden == 30) %>%
  select(n_dims, ridge_lambda, df_mean)

result_table <- df %>%
  group_by(n_dims, fit_type, ridge_lambda) %>%
  summarise(
    sd    = sd(price),    # computed before price is overwritten by its own mean below
    price = mean(price),
    n_reps = n(),
    .groups = "drop"
  ) %>%
  group_by(n_dims) %>%
  mutate(plain_price = price[fit_type == "plain"]) %>%
  ungroup() %>%
  mutate(
    rel_diff = (price - plain_price) * 100 / plain_price,   # vs. that dimension's own plain baseline
    lambda_sort = ifelse(fit_type == "plain", -1, ridge_lambda),
  ) %>%
  left_join(df_lambda, by = c("n_dims", "ridge_lambda")) %>%
  # plain (lambda=0) uses every feature fully: df = K = n_hidden+1 = 31.
  mutate(df_mean = ifelse(fit_type == "plain", 31, df_mean)) %>%
  arrange(n_dims, lambda_sort)

print(result_table)

# Mirrors laguerre_ridge_lambda_table.R's layout: one minipage tabular per
# dimension (only ITS OWN lambda rows), 2 above (d=1, d=10) + 1 below (d=25).
make_dim_tabular <- function(d) {
  sub_df <- result_table %>%
    filter(n_dims == d) %>%
    # d=25 only: drop lambda=10,15,20 rows -- those are still stuck at 10
    # reps (never backfilled to 40), so they'd be inconsistent in a 40-rep
    # table. lambda=5 now has 40 reps too, so it stays.
    filter(!(n_dims == 25 & ridge_lambda %in% c(10, 15, 20))) %>%
    mutate(lambda_lab = ifelse(fit_type == "plain", "plain", as.character(ridge_lambda))) %>%
    arrange(lambda_sort) %>%
    select(lambda_lab, df_mean, price, sd, rel_diff)

  linesep_vec <- ifelse(sub_df$lambda_lab == "plain", "\\addlinespace", "")
  linesep_vec <- linesep_vec[-length(linesep_vec)]

  tbl <- sub_df %>%
    kbl(
      format = "latex",
      booktabs = TRUE,
      align = "lcccc",
      digits = c(0, 2, 1, 2, 2),
      col.names = c("$\\lambda$", "df($\\lambda$)", "Price", "SD", "\\shortstack{Rel.\\ Diff\\\\(\\%)}"),
      escape = FALSE,
      linesep = linesep_vec
    )

  as.character(tbl)
}

minipage_for <- function(d) {
  paste0(
    "\\begin{minipage}[t]{0.48\\textwidth}\n\\centering\n",
    "\\textbf{$d = ", d, "$}\\\\[2pt]\n",
    "\\scriptsize\n",
    "\\setlength{\\tabcolsep}{3pt}\n",
    make_dim_tabular(d),
    "\n\\end{minipage}"
  )
}

combined <- paste0(
  "\\begin{table}[h]\n\\centering\n",
  "\\caption{\\label{tab:rnn_ridge_lambda_K30_40rep}Plain least squares vs.\\ a ridge $\\lambda$ grid for the RNN basis ($K-1=30$), 40 repetitions per cell: one table per dimension, each with only the $\\lambda$ values tested there. Rel.\\ Diff.\\ (\\%) is $(\\text{Price}_\\lambda - \\text{Price}_{plain})/\\text{Price}_{plain} \\times 100$, within that dimension: positive means the ridge fit beats plain. df($\\lambda$) is averaged across all $N-1$ backward-induction steps.}\n",
  paste0(sapply(c(1, 10), minipage_for), collapse = "\\hfill\n"),
  "\n\\\\[6pt]\n",
  minipage_for(25),
  "\n\\end{table}\n"
)

writeLines(combined, "R_Tables/rnn_ridge_lambda_K30_40rep_table.tex")
