library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/evaluation_consistency_check.csv")

# Per (dimension, basis) cell: empirical mean/SD across all draws vs. the
# CLT-predicted SE, combined across all 100 draws via RMS (sqrt of the mean
# of se_draw^2), NOT a plain arithmetic mean of se_draw -- se_draw^2 is a
# variance estimate, and averaging variances (then taking one final sqrt) is
# the correct way to combine them; averaging the SDs directly is biased
# downward by Jensen's inequality (sqrt is concave), and that bias is exactly
# what made a plain average look like a persistent miscalibration at d=25
# (see conversation: RMS brings the ratio to ~1 everywhere, arithmetic mean
# didn't). Coverage moved to the companion dispersion table (Table~\ref{tab:evaluation_consistency_check_dispersion}):
# it's a per-draw reliability question, not an aggregate-accuracy one, so it
# belongs there alongside the per-draw SE instability metrics instead of here.
per_basis <- df %>%
  mutate(se_draw = path_level_sd / sqrt(n_paths_eval)) %>%
  group_by(n_dims, basis_type) %>%
  summarise(
    m_e             = first(n_paths_eval),
    n_draws         = n(),
    empirical_mean  = mean(v0),
    empirical_sd    = sd(v0),
    rms_se          = sqrt(mean(se_draw^2)),
    .groups = "drop"
  ) %>%
  mutate(
    ratio = empirical_sd / rms_se,
    alg   = case_when(
      basis_type == "rnn"      ~ "RNN",
      basis_type == "laguerre" ~ "Laguerre (deg 2)",
      TRUE                     ~ basis_type
    )
  ) %>%
  select(n_dims, alg, m_e, n_draws, empirical_mean, empirical_sd, rms_se, ratio)

result_table <- per_basis %>%
  arrange(n_dims, alg) %>%
  select(n_dims, alg, empirical_mean, empirical_sd, rms_se, ratio)

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
    align = "clcccc",
    digits = c(0, 0, 1, 1, 1, 2),
    col.names = c("Dim", "Alg", "Mean $\\tilde V_0$",
                  "Empirical SD", "CLT SE (RMS)", "Ratio"),
    caption = "Consistency check: empirical spread of $\\tilde V_0$ across independent evaluation samples vs.\\ the CLT-predicted standard error, for one fixed policy per cell ($M_t=10{,}000$, $M_e=10{,}000$, 100 draws). CLT SE (RMS) combines $\\hat\\sigma_{M_e}/\\sqrt{M_e}$ across all 100 draws via root-mean-square (the correct way to average variance estimates, unlike a plain arithmetic mean of SEs, which is biased downward by Jensen's inequality); it brings the Ratio to approximately 1 at every (dim, alg) cell.",
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
