library(readr)
library(dplyr)
library(kableExtra)

df <- read_csv("results/rnn_ridge_lambda_L1_experiment.csv", show_col_types = FALSE)

# df(lambda) is provably L-invariant (see conversation: the ridge hat matrix
# depends only on Phi and lambda, never the target, and Phi never touches
# inventory/L in separate mode) -- verified exactly (0.000e+00 diff) against
# the real fit_policy pipeline. So the existing L=10 df(lambda) values for
# K-1=30 apply unchanged here; no new df computation needed.
dfl <- read_csv("results/df_lambda_rnn.csv", show_col_types = FALSE) %>%
  filter(basis_n_hidden == 30) %>%
  select(n_dims, ridge_lambda, df_mean)

result_table <- df %>%
  group_by(n_dims, fit_type, ridge_lambda) %>%
  summarise(
    sd    = sd(price),
    price = mean(price),
    .groups = "drop"
  ) %>%
  group_by(n_dims) %>%
  mutate(plain_price = price[fit_type == "plain"]) %>%
  ungroup() %>%
  mutate(
    rel_diff = (price - plain_price) * 100 / plain_price,
    lambda_sort = ifelse(fit_type == "plain", -1, ridge_lambda),
    lambda_lab = ifelse(fit_type == "plain", "plain", as.character(ridge_lambda)),
  ) %>%
  left_join(dfl, by = c("n_dims", "ridge_lambda")) %>%
  mutate(df_lab = ifelse(fit_type == "plain", "31 (=K)", sprintf("%.2f", df_mean))) %>%
  arrange(n_dims, lambda_sort) %>%
  select(n_dims, lambda_lab, price, sd, df_lab, rel_diff)

print(result_table)

n_dims_vec  <- result_table$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- result_table %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rrcccc",
    digits = c(0, 0, 1, 2, 0, 2),
    col.names = c("Dim", "$\\lambda$", "Price", "SD", "df($\\lambda$)", "Rel.\\ Diff.\\ (\\%)"),
    caption = "RNN, $L=1$ (one swing right), $K-1=30$: plain least squares vs.\\ the same ridge $\\lambda$ grid used at $L=10$. df($\\lambda$) reused from the $L=10$ computation -- provably L-invariant, verified exactly against the real fit\\_policy pipeline. Rel.\\ Diff.\\ (\\%) is $(\\text{Price}_\\lambda - \\text{Price}_{plain})/\\text{Price}_{plain}\\times 100$.",
    label = "rnn_ridge_lambda_L1",
    linesep = linesep_vec,
    escape = FALSE
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/rnn_ridge_lambda_L1_table.tex")
