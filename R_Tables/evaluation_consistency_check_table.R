library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/evaluation_consistency_check.csv")

# Per (dimension, basis) cell: empirical mean/SD across all draws vs. the
# CLT-predicted SE from the FIRST draw's own path_level_sd alone -- the
# "honest" comparison (what a single real run would actually have access to),
# not a pooled-across-draws SE, which would be unrealistically optimistic
# (see conversation).
# Empirical coverage: for EACH of the 100 draws, form that draw's own CLT
# interval (v0 +/- 1.96*path_level_sd/sqrt(M_e)) and check whether it
# contains the empirical mean across all 100 draws (our best available proxy
# for the true V(pi)) -- an assumption-free measurement of what the nominal
# 95% interval is actually delivering, rather than inferring it from the
# empirical-SD/CLT-SE ratio.
per_basis <- df %>%
  group_by(n_dims, basis_type) %>%
  mutate(empirical_mean_grp = mean(v0)) %>%
  ungroup() %>%
  mutate(
    se_draw   = path_level_sd / sqrt(n_paths_eval),
    covers    = abs(v0 - empirical_mean_grp) <= 1.96 * se_draw,
  ) %>%
  group_by(n_dims, basis_type) %>%
  summarise(
    m_e             = first(n_paths_eval),
    n_draws         = n(),
    empirical_mean  = mean(v0),
    empirical_sd    = sd(v0),
    first_draw_se   = path_level_sd[which.min(draw)] / sqrt(first(n_paths_eval)),
    coverage        = mean(covers) * 100,
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
  select(n_dims, alg, m_e, n_draws, empirical_mean, empirical_sd, first_draw_se, ratio, coverage)

result_table <- per_basis %>%
  arrange(n_dims, alg) %>%
  select(n_dims, alg, empirical_mean, empirical_sd, first_draw_se, ratio, coverage)

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
    align = "clccccc",
    digits = c(0, 0, 1, 1, 1, 2, 0),
    col.names = c("Dim", "Alg", "Mean $\\tilde V_0$",
                  "Empirical SD", "CLT SE", "Ratio", "Coverage (\\%)"),
    caption = "Consistency check: empirical spread of $\\tilde V_0$ across independent evaluation samples vs.\\ the CLT-predicted standard error from a single evaluation, for one fixed policy per cell ($M_t=10{,}000$, $M_e=10{,}000$, 100 draws). Coverage is the assumption-free measurement: the fraction of the 100 draws' own nominal 95\\% CLT intervals ($\\tilde V_0 \\pm 1.96 \\cdot \\hat\\sigma_{M_e}/\\sqrt{M_e}$) that contain the empirical mean across all 100 draws.",
    label = "evaluation_consistency_check",
    linesep = linesep_vec,
    escape = FALSE
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/evaluation_consistency_check_table.tex")
