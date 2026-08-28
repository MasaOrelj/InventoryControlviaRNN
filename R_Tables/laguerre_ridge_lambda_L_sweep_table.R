library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

L_VALUES <- c(1, 5, 10, 25, 40)

load_prices <- function(L) {
  fname <- if (L == 10) "results/laguerre_ridge_lambda_experiment.csv" else paste0("results/laguerre_ridge_lambda_L", L, "_experiment.csv")
  read_csv(fname, show_col_types = FALSE) %>% mutate(L = !!L)
}

raw <- bind_rows(lapply(L_VALUES, load_prices)) %>%
  filter(!(n_dims == 10 & ridge_lambda %in% c(0.5))) %>%
  filter(!(n_dims == 25 & ridge_lambda %in% c(0.5, 15, 20)))

sig <- read_csv("results/laguerre_ridge_lambda_L_sweep_significance.csv", show_col_types = FALSE) %>%
  select(n_dims, L, fit_type, ridge_lambda, bold)

result_table <- raw %>%
  group_by(n_dims, L, fit_type, ridge_lambda) %>%
  summarise(
    sd    = sd(price),
    price = mean(price),
    .groups = "drop"
  ) %>%
  left_join(sig, by = c("n_dims", "L", "fit_type", "ridge_lambda")) %>%
  mutate(
    lambda_sort = ifelse(fit_type == "plain", -1, ridge_lambda),
    price_fmt = ifelse(bold, sprintf("\\textbf{%.1f}", price), sprintf("%.1f", price)),
    sd_fmt    = ifelse(bold, sprintf("\\textbf{%.2f}", sd),    sprintf("%.2f", sd)),
  ) %>%
  arrange(n_dims, L, lambda_sort)

print(result_table)

# One tabular per dimension, L in {1,5,10,25,40} as header groups, lambda as
# rows -- mirrors rnn_ridge_lambda_L_sweep_K30_table.R exactly, for the
# Laguerre basis. Bold means that cell's price AND sd are both statistically
# tied with that (n_dims, L) block's own best price / best sd (see
# Compute_Laguerre_L_Sweep_Significance.py). Not every lambda was tested at
# every L (a few one-off backfills only touched specific L's -- see
# conversation), so missing cells render as "--".
make_dim_tabular <- function(d) {
  value_cols <- as.vector(rbind(paste0("L", L_VALUES, "_price_fmt"), paste0("L", L_VALUES, "_sd_fmt")))

  wide <- result_table %>%
    filter(n_dims == d) %>%
    mutate(lambda_lab = ifelse(fit_type == "plain", "plain", as.character(ridge_lambda))) %>%
    select(L, lambda_sort, lambda_lab, price_fmt, sd_fmt) %>%
    pivot_wider(
      id_cols = c(lambda_sort, lambda_lab),
      names_from = L,
      values_from = c(price_fmt, sd_fmt),
      names_glue = "L{L}_{.value}"
    ) %>%
    arrange(lambda_sort) %>%
    select(lambda_lab, all_of(value_cols)) %>%
    mutate(across(all_of(value_cols), ~ ifelse(is.na(.), "--", .)))

  linesep_vec <- ifelse(wide$lambda_lab == "plain", "\\addlinespace", "")
  linesep_vec <- linesep_vec[-length(linesep_vec)]

  n_l <- length(L_VALUES)
  header_above <- c(1, rep(2, n_l))
  names(header_above) <- c(" ", paste0("$L = ", L_VALUES, "$"))

  wide %>%
    kbl(
      format = "latex",
      booktabs = TRUE,
      align = paste0("l", strrep("cc", n_l)),
      col.names = c("$\\lambda$", rep(c("Price", "SD"), n_l)),
      linesep = linesep_vec,
      escape = FALSE
    ) %>%
    add_header_above(header_above, escape = FALSE) %>%
    as.character() %>%
    trimws()
}

make_dim_block <- function(d) {
  caption <- paste0(
    "Laguerre (degree 2) basis, $d=", d, "$: plain least squares vs.\\ a ridge $\\lambda$ grid, ",
    "across swing-rights counts $L \\in \\{1,5,10,25,40\\}$. ",
    "\\textbf{Bold} marks cells whose price and SD are both statistically indistinguishable ",
    "(sign-flip test, $\\alpha=0.05$) from that $(d,L)$ block's own best price and best SD. ",
    "``--'' marks $\\lambda$ values not tested at that $L$."
  )
  paste0(
    "\\begin{table}[htbp]\n",
    "\\caption{\\label{tab:laguerre_ridge_lambda_L_sweep_d", d, "}", caption, "}\n",
    "\\scriptsize\n",
    "\\setlength{\\tabcolsep}{3pt}\n",
    "\\centering\n",
    make_dim_tabular(d), "\n",
    "\\end{table}\n"
  )
}

combined <- paste0(sapply(c(1, 10, 25), make_dim_block), collapse = "\n")
writeLines(combined, "R_Tables/laguerre_ridge_lambda_L_sweep_table.tex")
