library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/state_representation_experiment_n10000.csv")

wide_by_rep <- df %>%
  select(rep, n_dims, basis_type, fit_type, ridge_lambda, state_input, price, duration_sec) %>%
  pivot_wider(names_from = state_input, values_from = c(price, duration_sec))

result_table <- wide_by_rep %>%
  group_by(n_dims, basis_type, fit_type, ridge_lambda) %>%
  summarise(
    S_mean            = mean(price_S),
    ZY_mean           = mean(price_ZY),
    rel_diff          = (ZY_mean - S_mean) * 100 / S_mean,
    S_std             = sd(price_S),
    ZY_std            = sd(price_ZY),
    S_mean_duration   = mean(duration_sec_S),
    ZY_mean_duration  = mean(duration_sec_ZY),
    n_reps            = n(),
    .groups = "drop"
  ) %>%
  arrange(n_dims, basis_type, fit_type)   # explicit order: linesep below depends on exactly 4 rows/dim

print(result_table)

n_dims_vec <- result_table$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- result_table %>%
  mutate(
    lambda = ifelse(is.na(ridge_lambda), "-", as.character(ridge_lambda)),
    alg    = case_when(
      basis_type == "poly" ~ "Pre-selection",
      basis_type == "rnn"  ~ "RNN",
      TRUE                 ~ basis_type
    )
  ) %>%
  select(n_dims, alg, fit_type, lambda,
         S_mean, S_std, S_mean_duration,
         ZY_mean, ZY_std, ZY_mean_duration,
         rel_diff) %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    digits = c(0, 0, 0, 0, 1, 1, 2, 1, 1, 2, 2),
    col.names = c("Dim", "Alg", "Fit", "$\\lambda$",
                  "Price", "SD", "Duration",
                  "Price", "SD", "Duration",
                  "Rel. Diff (\\%)"),
    caption = "Comparison of state representations $(S_n,I_n)$ vs.\\ $(Z_n,Y_n,I_n)$.",
    label = "tab:state_representation",
    linesep = linesep_vec
  ) %>%
  add_header_above(c(" " = 4,
                     "$(S_n,I_n)$" = 3,
                     "$(Z_n,Y_n,I_n)$" = 3,
                     " " = 1))

# Inject \footnotesize and \setlength{\tabcolsep}{4pt} right after \caption{...},
# before \centering -- kableExtra's own styling API doesn't produce this exact
# preamble, so this is a direct, surgical string edit on the generated LaTeX
# rather than fighting kable_styling()'s higher-level (less precise) options.
latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/state_representation_table.tex")



#BASIS FUNCTIONS
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
    label = "tab:basis_functions"
  ) %>%
  add_header_above(c(" " = 1, "d = 1" = 3, "d = 10" = 3, "d = 25" = 3))

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

df2 <- read_csv("results/basis_function_experiment_ridge_n10000.csv")

summary_long <- df2 %>%
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

algo_order   <- c("poly_deg2", "laguerre_deg2")
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



#####################################
# Evaluation sample consistency check
#####################################

df <- read_csv("results/evaluation_consistency_check.csv")

# Per (dimension, basis) cell: empirical mean/SD across all draws vs. the
# CLT-predicted SE from the FIRST draw's own path_level_sd alone -- the
# "honest" comparison (what a single real run would actually have access to),
# not a pooled-across-draws SE, which would be unrealistically optimistic
# (see conversation).
result_table <- df %>%
  group_by(n_dims, basis_type) %>%
  summarise(
    m_e             = first(n_paths_eval),
    n_draws         = n(),
    empirical_mean  = mean(v0),
    empirical_sd    = sd(v0),
    first_draw_se   = path_level_sd[which.min(draw)] / sqrt(first(n_paths_eval)),
    .groups = "drop"
  ) %>%
  mutate(
    ratio = empirical_sd / first_draw_se,
    alg   = case_when(
      basis_type == "rnn"      ~ "RNN",
      basis_type == "laguerre" ~ "Laguerre (deg 2)",
      TRUE                     ~ basis_type
    )
  ) %>%
  arrange(n_dims, alg) %>%
  select(n_dims, alg, m_e, n_draws, empirical_mean, empirical_sd, first_draw_se, ratio)

print(result_table)

# Blank spacing line before each new Dim block, computed from where n_dims
# actually changes (robust to however many basis rows land in each block --
# e.g. once/if d=25 is added here with only a subset of bases run).
n_dims_vec  <- result_table$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- result_table %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "clcccccc",
    digits = c(0, 0, 0, 0, 1, 1, 1, 2),
    col.names = c("Dim", "Alg", "$M_e$", "\\# draws", "Mean $\\tilde V_0$",
                  "Empirical SD", "CLT SE", "Ratio"),
    caption = "Consistency check: empirical spread of $\\tilde V_0$ across
      independent evaluation samples vs.\\ the CLT-predicted standard error
      from a single evaluation, for one fixed policy per cell
      ($M_t=10{,}000$, $M_e=10{,}000$).",
    label = "tab:evaluation_consistency_check",
    linesep = linesep_vec
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/evaluation_consistency_check_table.tex")



####################################
# Evaluation sample size experiment
####################################

df <- read_csv("results/evaluation_sample_size_experiment.csv")

result_table <- df %>%
  mutate(
    alg      = case_when(
      basis_type == "rnn"      ~ "RNN",
      basis_type == "laguerre" ~ "Laguerre (deg 2)",
      TRUE                     ~ basis_type
    ),
    ci_lower = v0 - 1.96 * mc_se,
    ci_upper = v0 + 1.96 * mc_se
  ) %>%
  arrange(n_dims, basis_type, n_paths_eval) %>%
  select(n_dims, alg, n_paths_eval, v0, path_level_sd, mc_se, ci_lower, ci_upper)

print(result_table)

# Blank spacing line before each new (Dim, Alg) block -- computed from where
# the (n_dims, alg) pair actually changes, not a fixed row-count pattern, so
# it stays correct regardless of how many M_e values end up in each block.
group_key   <- paste(result_table$n_dims, result_table$alg)
linesep_vec <- ifelse(is.na(lead(group_key)) | group_key == lead(group_key), "", "\\addlinespace")

latex_table <- result_table %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    digits = c(0, 0, 0, 1, 1, 1, 1, 1),
    col.names = c("Dim", "Alg", "$M_e$", "Price", "Path SD", "SE",
                  "95\\% CI lower", "95\\% CI upper"),
    caption = "Effect of the evaluation sample size $M_e$ on the CLT standard
      error and 95\\% confidence interval, for one fixed policy per
      (dimension, basis) cell ($M_t=10{,}000$).",
    label = "tab:evaluation_sample_size",
    linesep = linesep_vec
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/evaluation_sample_size_table.tex")

