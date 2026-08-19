#BASIS FUNCTIONS
library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/basis_function_experiment_n10000.csv")

summary_long <- df %>%
  mutate(algo = paste0(basis_type, "_deg", basis_degree)) %>%
  group_by(n_dims, algo) %>%
  summarise(
    mean_price = mean(price),
    sd_price   = sd(price),
    sd         = sd_price,
    duration   = mean(duration_sec),
    .groups    = "drop"
  ) %>%
  rename(price = mean_price) %>%
  select(n_dims, algo, price, sd, duration)

wide <- summary_long %>%
  pivot_wider(
    id_cols = algo,
    names_from = n_dims,
    values_from = c(price, sd, duration),
    names_glue = "d{n_dims}_{.value}"
  )

algo_order   <- c("poly_deg2", "poly_deg3", "laguerre_deg2", "laguerre_deg3")
dim_order    <- c(1, 10, 25)
metric_order <- c("price", "sd", "duration")
ordered_cols <- as.vector(sapply(dim_order, function(d) paste0("d", d, "_", metric_order)))

label_map <- c(
  poly_deg2     = "Poly (deg 2)",
  poly_deg3     = "Poly (deg 3)",
  laguerre_deg2 = "Laguerre (deg 2)",
  laguerre_deg3 = "Laguerre (deg 3)"
)

wide <- wide %>%
  mutate(algo = factor(algo, levels = algo_order)) %>%
  arrange(algo) %>%
  mutate(basis = label_map[as.character(algo)]) %>%
  select(basis, all_of(ordered_cols))

# Price/SD -> 1 decimal, Duration -> 2 decimals (matches the target table);
# NA (skipped combos, e.g. poly deg3 / laguerre deg3 at d=25) -> "-" either way.
formatted <- wide %>%
  mutate(across(matches("_price$|_sd$"),
                ~ ifelse(is.na(.x), "-", formatC(.x, format = "f", digits = 1)))) %>%
  mutate(across(matches("_duration$"),
                ~ ifelse(is.na(.x), "-", formatC(.x, format = "f", digits = 2))))

latex_table <- formatted %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "c",
    col.names = c("Basis", rep(c("Price", "SD", "Duration"), 3)),
    caption = "Comparison of basis function families and degrees, by dimension.",
    label = "basis_functions",
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 1, "d = 1" = 3, "d = 10" = 3, "d = 25" = 3), escape = FALSE)

# Same surgical injection as the first table: \footnotesize and
# \setlength{\tabcolsep}{4pt} right after \caption{...}, before \centering.
latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/basis_function_table.tex")
print(latex_str)
