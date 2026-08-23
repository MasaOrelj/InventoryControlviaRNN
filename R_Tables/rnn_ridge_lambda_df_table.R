library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df_lambda <- read_csv("results/df_lambda_rnn.csv", show_col_types = FALSE)

# Companion to rnn_ridge_lambda_table.R: df(lambda) by (lambda, K), one small
# table per dimension. Kept separate from the Price/SD grid there -- adding
# df(lambda) for all 5 K values to that table would push it to 16 columns.
make_dim_table <- function(d) {
  wide <- df_lambda %>%
    filter(n_dims == d) %>%
    select(ridge_lambda, basis_n_hidden, df_mean) %>%
    pivot_wider(id_cols = ridge_lambda, names_from = basis_n_hidden, values_from = df_mean,
                names_glue = "K{basis_n_hidden}") %>%
    arrange(ridge_lambda) %>%
    select(ridge_lambda, K10, K20, K30, K40, K50)

  latex_table <- wide %>%
    kbl(
      format = "latex",
      booktabs = TRUE,
      align = "rccccc",
      digits = c(3, 2, 2, 2, 2, 2),
      col.names = c("$\\lambda$", "$K-1=10$", "$K-1=20$", "$K-1=30$", "$K-1=40$", "$K-1=50$"),
      caption = paste0(
        "Effective degrees of freedom df($\\lambda$) for the RNN ridge $\\lambda$ grid at $d=", d, "$ ",
        "(companion to Table~\\ref{tab:rnn_ridge_lambda_d", d, "})."
      ),
      label = paste0("rnn_ridge_lambda_df_d", d),
      escape = FALSE
    )

  latex_str <- as.character(latex_table)
  latex_str <- sub(
    "(\\\\caption\\{[^\n]*\\}\n)",
    "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
    latex_str
  )

  writeLines(latex_str, paste0("R_Tables/rnn_ridge_lambda_df_table_d", d, ".tex"))
}

for (d in c(1, 10, 25)) {
  make_dim_table(d)
}
