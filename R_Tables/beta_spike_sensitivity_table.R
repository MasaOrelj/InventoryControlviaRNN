library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/beta_spike_sensitivity_experiment.csv", show_col_types = FALSE)

# Per (beta, basis): mean and SD of the spike contribution (price_with_spikes
# - price_no_spikes) across the 20 reps. half_life_over_delta is identical
# across reps for a given beta (deterministic), so `first()` is fine.
result_table <- df %>%
  group_by(beta, basis_type) %>%
  summarise(
    half_life_over_delta = first(half_life_over_delta),
    mean_diff = mean(diff),
    sd_diff   = sd(diff),
    rel_diff  = mean(diff / price_no_spikes) * 100,   # relative to that REP's own no-spike baseline, then averaged
    .groups = "drop"
  ) %>%
  arrange(beta, basis_type)

print(result_table)

wide <- result_table %>%
  pivot_wider(
    id_cols = c(beta, half_life_over_delta),
    names_from = basis_type,
    values_from = c(mean_diff, sd_diff, rel_diff),
    names_glue = "{basis_type}_{.value}"
  ) %>%
  arrange(beta) %>%
  select(beta, half_life_over_delta,
         laguerre_mean_diff, laguerre_sd_diff, laguerre_rel_diff,
         rnn_mean_diff, rnn_sd_diff, rnn_rel_diff)

latex_table <- wide %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "rrccccc",
    digits = c(0, 2, 1, 1, 2, 1, 1, 2),
    col.names = c("$\\beta$", "$\\frac{\\ln 2/\\beta}{\\Delta}$",
                  "Price Diff.", "SD", "Rel.\\ Diff.\\ (\\%)",
                  "Price Diff.", "SD", "Rel.\\ Diff.\\ (\\%)"),
    caption = "Spike contribution to the swing option value, $\\tilde V_0^{\\text{with spikes}}(\\beta) - \\tilde V_0^{\\text{no spikes}}$, across a $\\beta$ sweep at $d=1$ (20 repetitions, same underlying paths reused across every $\\beta$ within a repetition -- see conversation). $\\frac{\\ln 2/\\beta}{\\Delta}$ is the spike half-life divided by the exercise interval $\\Delta=T/N\\approx7.3$ days; values below 1 mean a typical spike has already decayed away before the next decision date. Rel.\\ Diff.\\ (\\%) is the spike contribution as a percentage of that same repetition's own no-spike baseline price, averaged over the 20 repetitions.",
    label = "beta_spike_sensitivity",
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 2, "Laguerre (deg 2)" = 3, "RNN" = 3))

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/beta_spike_sensitivity_table.tex")
