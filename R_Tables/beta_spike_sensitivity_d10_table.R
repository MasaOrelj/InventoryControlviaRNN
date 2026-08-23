library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/beta_spike_sensitivity_experiment_d10.csv", show_col_types = FALSE)

result_table <- df %>%
  group_by(beta, basis_type) %>%
  summarise(
    half_life_over_delta = first(half_life_over_delta),
    mean_diff = mean(diff),
    sd_diff   = sd(diff),
    rel_diff  = mean(diff / price_no_spikes) * 100,
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
    caption = "Spike contribution to the swing option value at $d=10$, $\\tilde V_0^{\\text{with spikes}}(\\beta) - \\tilde V_0^{\\text{no spikes}}$ (10 repetitions, same underlying paths reused across every $\\beta$ within a repetition). Beta grid narrowed to $d=1$'s own transition zone (Table~\\ref{tab:beta_spike_sensitivity}) plus two points further out, to test whether the crossover shifts to larger $\\beta$ once \\texttt{max\\_aggregation} only needs ONE of $d$ independent dimensions to have a still-fresh spike. Rel.\\ Diff.\\ (\\%) is the spike contribution as a percentage of that repetition's own no-spike baseline price, averaged over the 10 repetitions.",
    label = "beta_spike_sensitivity_d10",
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 2, "Laguerre (deg 2)" = 3, "RNN" = 3))

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/beta_spike_sensitivity_d10_table.tex")
