library(readr)
library(dplyr)
library(kableExtra)

# Plain fit comes from the main basis-function run (filtered to degree 2, the
# only degree the ridge run also covers); ridge fit comes from the dedicated
# ridge comparison run. Combined so the plain-vs-ridge reversal at d=1/d=10
# (and its absence at d=25) is visible in adjacent rows.
plain <- read_csv("results/basis_function_experiment_n10000.csv") %>%
  filter(basis_degree == 2) %>%
  mutate(fit_type = "plain")

ridge <- read_csv("results/basis_function_experiment_ridge_n10000.csv") %>%
  mutate(fit_type = "ridge")

df <- bind_rows(plain, ridge)

result_table <- df %>%
  group_by(n_dims, basis_type, fit_type) %>%
  summarise(
    price = mean(price),
    sd    = sd(price),
    .groups = "drop"
  ) %>%
  mutate(
    alg = case_when(
      basis_type == "poly"     ~ "Poly (deg 2)",
      basis_type == "laguerre" ~ "Laguerre (deg 2)",
      TRUE                     ~ basis_type
    )
  ) %>%
  arrange(n_dims, alg, fit_type)

print(result_table)

wide <- result_table %>%
  select(n_dims, alg, fit_type, price, sd) %>%
  tidyr::pivot_wider(
    id_cols = c(alg, fit_type),
    names_from = n_dims,
    values_from = c(price, sd),
    names_glue = "d{n_dims}_{.value}"
  ) %>%
  arrange(alg, fit_type) %>%
  select(alg, fit_type,
         d1_price, d1_sd, d10_price, d10_sd, d25_price, d25_sd)

# Blank spacing line before each new Alg block (poly vs laguerre), computed
# from where `alg` actually changes -- not a fixed row count, since this
# stays correct however many fit types end up in each block.
alg_vec     <- wide$alg
linesep_vec <- ifelse(is.na(lead(alg_vec)) | alg_vec == lead(alg_vec), "", "\\addlinespace")

latex_table <- wide %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "llcccccc",
    digits = c(0, 0, 1, 1, 1, 1, 1, 1),
    col.names = c("Basis", "Fit", "Price", "SD", "Price", "SD", "Price", "SD"),
    caption = "Plain vs.\\ ridge ($\\lambda=1$) least squares for polynomial and Laguerre
      (degree 2), by dimension.",
    label = "basis_function_ridge",
    linesep = linesep_vec,
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 2, "d = 1" = 2, "d = 10" = 2, "d = 25" = 2), escape = FALSE)

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/basis_function_ridge_table.tex")
