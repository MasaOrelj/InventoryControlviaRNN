library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/state_representation_experiment.csv")

wide_by_rep <- df %>%
  select(rep, n_dims, basis_type, fit_type, ridge_lambda, state_input, price, duration_sec) %>%
  pivot_wider(names_from = state_input, values_from = c(price, duration_sec))

result_table <- wide_by_rep %>%
  group_by(n_dims, basis_type, fit_type, ridge_lambda) %>%
  summarise(
    S_mean       = mean(price_S),
    ZY_mean      = mean(price_ZY),
    rel_diff     = (ZY_mean - S_mean)*100 / S_mean,
    S_std          = sd(price_S),    
    ZY_std          = sd(price_ZY),    # std of the paired difference
    S_mean_duration = mean(duration_sec_S),
    ZY_mean_duration = mean(duration_sec_ZY),
    n_reps = n(),
    .groups = "drop"
  )

print(result_table)

latex_table <- result_table %>%
  mutate(lambda = ifelse(is.na(ridge_lambda), "-", as.character(ridge_lambda))) %>%
  select(n_dims, basis_type, fit_type, lambda,
         S_mean, S_std, S_mean_duration,
         ZY_mean, ZY_std, ZY_mean_duration,
         rel_diff) %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    digits = c(0, 0, 0, 0, 2, 2, 3, 2, 2, 3, 3),
    col.names = c("Dim", "Basis", "Fit", "$\\lambda$",
                  "Price", "SD", "Duration",
                  "Price", "SD", "Duration",
                  "Rel.\\ Diff (\\%)"),
    caption = "Comparison of state representations $(S_n,I_n)$ vs.\\ $(Z_n,Y_n,I_n)$.",
    label = "tab:state_representation"
  ) %>%
  add_header_above(c(" " = 4,
                     "State Representation $(S_n,I_n)$" = 3,
                     "State Representation $(Z_n,Y_n,I_n)$" = 3,
                     " " = 1))

writeLines(latex_table, "state_representation_table.tex")


df <- read_csv("results/basis_function_experiment.csv")

# One combined label per algorithm (family + degree kept together, as before).
summary_long <- df %>%
  mutate(algo = paste0(basis_type, "_deg", basis_degree)) %>%
  group_by(n_dims, algo) %>%
  summarise(
    mean_price    = mean(price),
    sd      = sd(price),
    duration = mean(duration_sec),
    .groups  = "drop"
  )

wide <- summary_long %>%
  pivot_wider(id_cols = n_dims, names_from = algo, values_from = c(mean_price, sd, duration))

# Fixed column order: algorithm-major (all 3 metrics per algorithm grouped
# together), not metric-major.
algo_order   <- c("poly_deg2", "poly_deg3", "laguerre_deg2", "laguerre_deg3")
metric_order <- c("price", "var", "duration")
ordered_cols <- as.vector(sapply(algo_order, function(a) paste0(a, "_", metric_order)))
wide <- wide %>% select(n_dims, all_of(ordered_cols)) %>% arrange(n_dims)

# poly_deg3 has no d=25 row (that cell was skipped) -> pivot_wider fills it
# with NA. Format everything as fixed-decimal strings first and turn NA into
# "-" explicitly, so the table shows a clean dash instead of a literal "NA".
formatted <- wide %>%
  mutate(across(-n_dims, ~ ifelse(is.na(.x), "-", formatC(.x, format = "f", digits = 2))))

latex_table <- formatted %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "c",
    col.names = c("Dim", rep(c("Price", "Var", "Duration"), 4)),
    caption = "Comparison of basis function families and degrees.",
    label = "tab:basis_functions"
  ) %>%
  add_header_above(c(" " = 1,
                     "Poly deg 2" = 3, "Poly deg 3" = 3,
                     "Laguerre deg 2" = 3, "Laguerre deg 3" = 3))

writeLines(latex_table, "basis_function_table.tex")
print(latex_table)