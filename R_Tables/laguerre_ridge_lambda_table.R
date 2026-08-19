library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/laguerre_ridge_lambda_experiment.csv")

result_table <- df %>%
  group_by(n_dims, fit_type, ridge_lambda) %>%
  summarise(
    sd    = sd(price),    # computed before price is overwritten by its own mean below
    price = mean(price),
    .groups = "drop"
  ) %>%
  group_by(n_dims) %>%
  mutate(plain_price = price[fit_type == "plain"]) %>%
  ungroup() %>%
  mutate(
    rel_diff = (price - plain_price) * 100 / plain_price,   # vs. that dimension's own plain baseline
    lambda_sort = ifelse(fit_type == "plain", -1, ridge_lambda),
  ) %>%
  arrange(n_dims, lambda_sort)

print(result_table)

# Three separate small tables (one per dimension, each with only ITS OWN
# lambda rows -- no NA rows from lambdas tested at other dimensions), placed
# side by side via minipages inside one outer table float. Each minipage
# gets a bold "d = X" heading rather than its own \caption, since LaTeX
# doesn't support multiple \caption calls inside one table float without the
# subcaption package (not assumed to be loaded in the thesis preamble).
make_dim_tabular <- function(d) {
  sub_df <- result_table %>%
    filter(n_dims == d) %>%
    mutate(lambda_lab = ifelse(fit_type == "plain", "plain", as.character(ridge_lambda))) %>%
    arrange(lambda_sort) %>%
    select(lambda_lab, price, sd, rel_diff)

  linesep_vec <- ifelse(sub_df$lambda_lab == "plain", "\\addlinespace", "")
  linesep_vec <- linesep_vec[-length(linesep_vec)]

  tbl <- sub_df %>%
    kbl(
      format = "latex",
      booktabs = TRUE,
      align = "lccc",
      digits = c(0, 1, 2, 2),
      col.names = c("$\\lambda$", "Price", "SD", "\\shortstack{Rel.\\ Diff\\\\(\\%)}"),
      escape = FALSE,
      linesep = linesep_vec
    )

  # kbl() without a caption skips the \begin{table}...\end{table} wrapper
  # entirely and returns just \begin{tabular}...\end{tabular} -- exactly
  # the fragment needed inside a minipage.
  as.character(tbl)
}

minipage_for <- function(d) {
  paste0(
    "\\begin{minipage}[t]{0.32\\textwidth}\n\\centering\n",
    "\\textbf{$d = ", d, "$}\\\\[2pt]\n",
    "\\scriptsize\n",
    "\\setlength{\\tabcolsep}{3pt}\n",
    make_dim_tabular(d),
    "\n\\end{minipage}"
  )
}

combined <- paste0(
  "\\begin{table}[h]\n\\centering\n",
  "\\caption{\\label{tab:laguerre_ridge_lambda}Plain least squares vs.\\ a ridge $\\lambda$ grid for the Laguerre (degree 2) basis: one table per dimension, each with only the $\\lambda$ values tested there (10 repetitions at $d=1,10$, 5 at $d=25$). Rel.\\ Diff.\\ (\\%) is $(\\text{Price}_\\lambda - \\text{Price}_{plain})/\\text{Price}_{plain} \\times 100$, within that dimension: positive means the ridge fit beats plain.}\n",
  paste0(sapply(c(1, 10, 25), minipage_for), collapse = "\\hfill\n"),
  "\n\\end{table}\n"
)

writeLines(combined, "R_Tables/laguerre_ridge_lambda_table.tex")
