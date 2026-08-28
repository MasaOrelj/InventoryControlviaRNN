library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

df <- read_csv("results/joint_lsweep_40rep_experiment.csv", show_col_types = FALSE)

summary_long <- df %>%
  group_by(n_dims, L, basis_type) %>%
  summarise(
    sd    = sd(price),    # computed before price is overwritten by its own mean below
    price = mean(price),
    .groups = "drop"
  )

# Component-B CLT interval (evaluation Monte Carlo noise for ONE fixed,
# already-fitted policy -- rep=0 -- see Joint_LSweep_CLT_CI.py and
# conversation), distinct from `sd` above (Component A, the between-rep
# policy-fit spread).
ci <- read_csv("results/joint_lsweep_clt_ci.csv", show_col_types = FALSE) %>%
  select(n_dims, L, basis_type, ci_lower, ci_upper)

# Paired-difference significance (RNN vs Laguerre share the SAME 40 training
# seeds per (n_dims, L) -- see Compute_Joint_LSweep_Significance.py and
# conversation): exact/Monte-Carlo sign-flip test, same methodology as the
# RNN ridge-lambda tables.
sig <- read_csv("results/joint_lsweep_significance.csv", show_col_types = FALSE) %>%
  select(n_dims, L, sd_diff, p_value, significant)

wide <- summary_long %>%
  left_join(ci, by = c("n_dims", "L", "basis_type")) %>%
  mutate(ci_fmt = sprintf("[%.0f, %.0f]", ci_lower, ci_upper)) %>%
  pivot_wider(id_cols = c(n_dims, L), names_from = basis_type, values_from = c(price, sd, ci_fmt)) %>%
  mutate(rel_diff = (price_rnn - price_laguerre) * 100 / price_laguerre) %>%
  left_join(sig, by = c("n_dims", "L")) %>%
  arrange(L, n_dims) %>%
  select(L, n_dims, price_laguerre, sd_laguerre, ci_fmt_laguerre, price_rnn, sd_rnn, ci_fmt_rnn, rel_diff, sd_diff, p_value, significant)

print(wide)

# Blank line between each NEW L block.
linesep_vec <- ifelse(is.na(lead(wide$L)) | wide$L == lead(wide$L), "", "\\addlinespace")

latex_table <- wide %>%
  mutate(
    L_lab = as.character(L),
    price_laguerre = sprintf("%.1f", price_laguerre),
    sd_laguerre    = sprintf("%.2f", sd_laguerre),
    price_rnn      = sprintf("%.1f", price_rnn),
    sd_rnn         = sprintf("%.2f", sd_rnn),
    rel_diff_fmt   = ifelse(significant, sprintf("\\textbf{%.2f*}", rel_diff), sprintf("%.2f", rel_diff)),
    sd_diff        = sprintf("%.2f", sd_diff),
  ) %>%
  select(L_lab, n_dims, price_laguerre, sd_laguerre, ci_fmt_laguerre, price_rnn, sd_rnn, ci_fmt_rnn, rel_diff_fmt, sd_diff) %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    align = "c",
    col.names = c("$L$", "$d$", "Price", "SD", "95\\% CI", "Price", "SD", "95\\% CI", "\\shortstack{Rel.\\ Diff\\\\(\\%)}", "\\shortstack{SD of\\\\Paired Diff}"),
    caption = "Laguerre (degree 2) vs.\\ RNN ($K-1=30$), 40 repetitions per cell, across swing-rights counts $L$ and dimensions $d$, each at its own settled ridge $\\lambda$. SD is the between-repetition spread (policy-fit variability, Component A); the 95\\% CI is the CLT interval $\\hat v_0 \\pm z \\cdot s/\\sqrt{M_e}$ from a single fixed policy's evaluation cashflows (Component B, evaluation Monte Carlo noise -- see conversation), not a resampling of the SD column. Rel.\\ Diff.\\ (\\%) is $(\\text{Price}_{RNN} - \\text{Price}_{Laguerre})/\\text{Price}_{Laguerre} \\times 100$; \\textbf{bold*} marks cells where the paired difference (RNN and Laguerre share the same 40 training seeds) is statistically significant by an exact/Monte-Carlo sign-flip test ($\\alpha=0.05$). SD of Paired Diff is the standard deviation of $\\text{Price}_{RNN,i}-\\text{Price}_{Laguerre,i}$ across the 40 matched repetitions $i$.",
    label = "joint_lsweep",
    linesep = linesep_vec,
    escape = FALSE
  ) %>%
  add_header_above(c(" " = 2, "Laguerre" = 3, "RNN" = 3, " " = 2), escape = FALSE)

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/joint_lsweep_table.tex")
print(latex_str)
