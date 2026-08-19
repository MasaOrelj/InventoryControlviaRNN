library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/rnn_ridge_lambda_experiment.csv")

result_table <- df %>%
  group_by(n_dims, basis_n_hidden, fit_type, ridge_lambda) %>%
  summarise(
    sd    = sd(price),    # computed before price is overwritten by its own mean below
    price = mean(price),
    .groups = "drop"
  ) %>%
  mutate(lambda_sort = ifelse(fit_type == "plain", -1, ridge_lambda)) %>%
  arrange(n_dims, basis_n_hidden, lambda_sort)

print(result_table)

# One grid table per dimension: lambda in rows, n_hidden (K) in columns
# (Price/SD pair per K), so each dimension's regularization-vs-capacity
# picture can be read on its own page-width table.
make_dim_table <- function(d) {
  wide <- result_table %>%
    filter(n_dims == d) %>%
    mutate(lambda_lab = ifelse(fit_type == "plain", "plain", as.character(ridge_lambda))) %>%
    select(basis_n_hidden, lambda_sort, lambda_lab, price, sd) %>%
    pivot_wider(
      id_cols = c(lambda_sort, lambda_lab),
      names_from = basis_n_hidden,
      values_from = c(price, sd),
      names_glue = "K{basis_n_hidden}_{.value}"
    ) %>%
    arrange(lambda_sort) %>%
    select(lambda_lab,
           K10_price, K10_sd, K20_price, K20_sd, K30_price, K30_sd,
           K40_price, K40_sd, K50_price, K50_sd)

  linesep_vec <- ifelse(wide$lambda_lab == "plain", "\\addlinespace", "")
  linesep_vec <- linesep_vec[-length(linesep_vec)]

  latex_table <- wide %>%
    kbl(
      format = "latex",
      booktabs = TRUE,
      align = "lcccccccccc",
      digits = c(0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2),
      col.names = c("$\\lambda$", "Price", "SD", "Price", "SD", "Price", "SD", "Price", "SD", "Price", "SD"),
      caption = paste0(
        "Plain least squares vs.\\ a ridge $\\lambda$ grid for the RNN basis, by number of hidden units $K$, at $d=", d, "$."
      ),
      label = paste0("rnn_ridge_lambda_d", d),
      linesep = linesep_vec,
      escape = FALSE
    ) %>%
    add_header_above(c(" " = 1, "K = 10" = 2, "K = 20" = 2, "K = 30" = 2, "K = 40" = 2, "K = 50" = 2))

  latex_str <- as.character(latex_table)
  latex_str <- sub(
    "(\\\\caption\\{[^\n]*\\}\n)",
    "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
    latex_str
  )

  writeLines(latex_str, paste0("R_Tables/rnn_ridge_lambda_table_d", d, ".tex"))
}

for (d in c(1, 10, 25)) {
  make_dim_table(d)
}
