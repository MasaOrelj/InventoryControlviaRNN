library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/rnn_ridge_lambda_experiment.csv")

# Per-cell significance flags from scripts/Compute_Ridge_Lambda_Significance.py.
#
# d=1, d=10: NO coloring -- only "bold" (both Price and SD statistically tied
# with that table's best SD / best Price at once, tested against the FULL
# lambda grid) is shown, in plain black \textbf.
#
# d=25: coloring IS used, since no cell ever passes "bold" against the full
# grid there -- Price in priceorange where it alone ties with the table's
# best Price, SD in sdblue where it alone ties with the table's best SD (see
# pass_price / pass_sd). On top of that, "bold_adjusted" flags cells that
# would pass BOTH criteria if the lambda=10,15,20 rows didn't exist (rerun
# against just lambda in {plain,0.001,...,5}) -- those get bolded too, but
# the existing color is PRESERVED under the bold (e.g. a bold_adjusted cell
# whose Price was already priceorange renders as bold orange Price + bold
# black SD, since that SD was never colored in the first place).
sig <- read_csv("results/rnn_ridge_lambda_significance.csv") %>%
  select(n_dims, basis_n_hidden, fit_type, ridge_lambda,
         pass_sd, pass_price, bold, bold_adjusted)

result_table <- df %>%
  group_by(n_dims, basis_n_hidden, fit_type, ridge_lambda) %>%
  summarise(
    sd    = sd(price),    # computed before price is overwritten by its own mean below
    price = mean(price),
    .groups = "drop"
  ) %>%
  left_join(sig, by = c("n_dims", "basis_n_hidden", "fit_type", "ridge_lambda")) %>%
  group_by(n_dims, basis_n_hidden) %>%
  mutate(plain_price = price[fit_type == "plain"]) %>%   # that SAME K's own plain baseline
  ungroup() %>%
  mutate(
    rel_diff = (price - plain_price) * 100 / plain_price,
    lambda_sort = ifelse(fit_type == "plain", -1, ridge_lambda),
    # d=1/d=10 only ever use "bold" (color columns are FALSE there in
    # practice, but guard explicitly so this stays correct if that changes).
    price_fmt = case_when(
      n_dims == 25 & bold_adjusted & pass_price ~ sprintf("\\textbf{\\textcolor{priceorange}{%.1f}}", price),
      n_dims == 25 & bold_adjusted              ~ sprintf("\\textbf{%.1f}", price),
      n_dims == 25 & pass_price                 ~ sprintf("\\textcolor{priceorange}{%.1f}", price),
      bold                                      ~ sprintf("\\textbf{%.1f}", price),
      TRUE                                      ~ sprintf("%.1f", price)
    ),
    sd_fmt = case_when(
      n_dims == 25 & bold_adjusted & pass_sd ~ sprintf("\\textbf{\\textcolor{sdblue}{%.2f}}", sd),
      n_dims == 25 & bold_adjusted           ~ sprintf("\\textbf{%.2f}", sd),
      n_dims == 25 & pass_sd                 ~ sprintf("\\textcolor{sdblue}{%.2f}", sd),
      bold                                   ~ sprintf("\\textbf{%.2f}", sd),
      TRUE                                   ~ sprintf("%.2f", sd)
    ),
  ) %>%
  arrange(n_dims, basis_n_hidden, lambda_sort)

print(result_table)

# Two K-1 groups (10,20,30 then 40,50) stacked as two `tabular`s inside ONE
# `table` float/caption -- at scriptsize the 5-K-1-group table was already
# about as small as it could go; splitting the K-1 groups lets the text go
# back up to footnotesize and still stay page-width. The second tabular
# drops its own \toprule (the \vspace gap already separates the two blocks).
make_k_group_tabular <- function(d, ks) {
  value_cols <- as.vector(rbind(paste0("K", ks, "_price_fmt"), paste0("K", ks, "_sd_fmt"), paste0("K", ks, "_rel_diff")))

  wide <- result_table %>%
    filter(n_dims == d, basis_n_hidden %in% ks) %>%
    mutate(lambda_lab = ifelse(fit_type == "plain", "plain", as.character(ridge_lambda))) %>%
    select(basis_n_hidden, lambda_sort, lambda_lab, price_fmt, sd_fmt, rel_diff) %>%
    pivot_wider(
      id_cols = c(lambda_sort, lambda_lab),
      names_from = basis_n_hidden,
      values_from = c(price_fmt, sd_fmt, rel_diff),
      names_glue = "K{basis_n_hidden}_{.value}"
    ) %>%
    arrange(lambda_sort) %>%
    select(lambda_lab, all_of(value_cols))

  linesep_vec <- ifelse(wide$lambda_lab == "plain", "\\addlinespace", "")
  linesep_vec <- linesep_vec[-length(linesep_vec)]

  n_k <- length(ks)
  header_above <- c(1, rep(3, n_k))
  names(header_above) <- c(" ", paste0("$K-1 = ", ks, "$"))

  tabular_str <- wide %>%
    kbl(
      format = "latex",
      booktabs = TRUE,
      align = paste0("l", strrep("ccc", n_k)),
      digits = c(0, rep(c(NA, NA, 2), n_k)),
      col.names = c("$\\lambda$", rep(c("Price", "SD", "\\shortstack{Rel.\\ Diff\\\\(\\%)}"), n_k)),
      linesep = linesep_vec,
      escape = FALSE
    ) %>%
    add_header_above(header_above, escape = FALSE) %>%
    as.character() %>%
    trimws()

  tabular_str
}

make_dim_table <- function(d) {
  tabular_a <- make_k_group_tabular(d, c(10, 20, 30))
  tabular_b <- make_k_group_tabular(d, c(40, 50))

  # Drop tabular B's leading \toprule -- \vspace{1em} already separates it
  # from tabular A, so a second top rule would be redundant.
  tabular_b_lines <- strsplit(tabular_b, "\n")[[1]]
  tabular_b <- paste(tabular_b_lines[tabular_b_lines != "\\toprule"], collapse = "\n")

  caption <- paste0(
    "Plain least squares vs.\\ a ridge $\\lambda$ grid for the RNN basis, by number of hidden units $K-1$, at $d=", d, "$. ",
    "Rel.\\ Diff.\\ (\\%) is $(\\text{Price}_\\lambda - \\text{Price}_{plain})/\\text{Price}_{plain} \\times 100$ against that SAME $K-1$'s own plain baseline: positive means the ridge fit beats plain at that $K-1$."
  )

  # Requires \usepackage{xcolor} in the main document preamble.
  latex_str <- paste0(
    "\\definecolor{priceorange}{HTML}{9A5A12}\n",
    "\\definecolor{sdblue}{HTML}{0A6FD6}\n",
    "\\begin{table}\n\n",
    "\\caption{\\label{tab:rnn_ridge_lambda_d", d, "}", caption, "}\n",
    "\\footnotesize\n",
    "\\setlength{\\tabcolsep}{3pt}\n",
    "\\centering\n",
    tabular_a, "\n\n",
    "\\vspace{1em}\n\n",
    tabular_b, "\n",
    "\\end{table}\n"
  )

  writeLines(latex_str, paste0("R_Tables/rnn_ridge_lambda_table_d", d, ".tex"))
}

for (d in c(1, 10, 25)) {
  make_dim_table(d)
}
