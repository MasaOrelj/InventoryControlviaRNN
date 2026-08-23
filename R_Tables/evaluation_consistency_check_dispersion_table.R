library(readr)
library(dplyr)
library(kableExtra)

df <- read_csv("results/evaluation_consistency_check.csv", show_col_types = FALSE)

# Dispersion of the per-draw CLT SE itself, across the 100 draws: even though
# the RMS-combined SE is well-calibrated on average (Table~\ref{tab:evaluation_consistency_check}),
# any ONE real run's own se_draw can be far from that combined value -- this
# table quantifies how far. CV = std(se_draw)/mean(se_draw) (standard
# coefficient-of-variation definition, using the plain arithmetic mean, not
# the RMS used elsewhere for the accuracy comparison). Coverage lives HERE,
# not in the accuracy table: it's the practical payoff of this per-draw
# instability question ("does it actually break things?"), not the aggregate
# accuracy question that table answers -- for EACH of the 100 draws, form
# that draw's OWN CLT interval (v0 +/- 1.96*se_draw) and check whether it
# contains the empirical mean across all 100 draws (our best available proxy
# for the true V(pi)).
result_table <- df %>%
  group_by(n_dims, basis_type) %>%
  mutate(empirical_mean_grp = mean(v0)) %>%
  ungroup() %>%
  mutate(
    se_draw = path_level_sd / sqrt(n_paths_eval),
    covers  = abs(v0 - empirical_mean_grp) <= 1.96 * se_draw,
  ) %>%
  group_by(n_dims, basis_type) %>%
  summarise(
    se_min       = min(se_draw),
    se_max       = max(se_draw),
    max_min      = max(se_draw) / min(se_draw),
    cv           = sd(se_draw) / mean(se_draw) * 100,
    median_ratio = median(se_draw) / sqrt(mean(se_draw^2)),   # median / RMS -- shows the skew, not just the spread
    coverage     = mean(covers) * 100,
    .groups = "drop"
  ) %>%
  mutate(
    alg = case_when(
      basis_type == "rnn"      ~ "RNN",
      basis_type == "laguerre" ~ "Laguerre (deg 2)",
      TRUE                     ~ basis_type
    )
  ) %>%
  arrange(n_dims, alg) %>%
  select(n_dims, alg, se_min, se_max, max_min, cv, median_ratio, coverage)

print(result_table)

n_dims_vec  <- result_table$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- result_table %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "clcccccc",
    digits = c(0, 0, 1, 1, 1, 0, 2, 0),
    col.names = c("Dim", "Alg", "SE min", "SE max", "Max/Min", "CV (\\%)", "Median/RMS", "Coverage (\\%)"),
    caption = "Dispersion of the per-draw CLT SE $\\hat\\sigma_{M_e}^{(k)}/\\sqrt{M_e}$ across the 100 draws underlying Table~\\ref{tab:evaluation_consistency_check}. Even though the RMS-combined SE there is well-calibrated on average, a single real run's own SE is highly unstable, especially at higher dimension: at $d=25$ it can be off by a factor of 16 depending on which evaluation sample was drawn. Median/RMS $<1$ at every cell shows this instability is right-skewed, not just wide: a typical single run understates the RMS-combined SE (down to $\\approx$0.6 at $d=25$), while only a minority of runs that catch a rare large-jump path overstate it -- this same skew is what produces the asymmetric multipliers in Table~\\ref{tab:evaluation_consistency_check_empirical_ci}. Coverage is the practical payoff of that instability: the fraction of the 100 draws' own nominal 95\\% CLT intervals ($\\tilde V_0 \\pm 1.96 \\cdot \\hat\\sigma_{M_e}/\\sqrt{M_e}$) that contain the empirical mean across all 100 draws; with only 100 draws its own standard error is $\\approx$2.2\\%, so the values shown are not meaningfully distinguishable from each other or from nominal 95\\%, but do show the instability above does not translate into badly broken coverage.",
    label = "evaluation_consistency_check_dispersion",
    linesep = linesep_vec,
    escape = FALSE
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/evaluation_consistency_check_dispersion_table.tex")
