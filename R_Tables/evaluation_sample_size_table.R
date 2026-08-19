library(readr)
library(dplyr)
library(kableExtra)

df <- read_csv("results/evaluation_sample_size_experiment.csv")

# RNN only: the policy is frozen across every M_e row here (this experiment
# never refits), so the CLT-SE-vs-M_e story is about evaluation Monte Carlo
# noise scaling, not about which basis was used -- both bases converge
# similarly once M_t=10000, so showing both would just duplicate the same
# pattern twice. See conversation.
result_table <- df %>%
  filter(basis_type == "rnn") %>%
  mutate(
    ci_lower = v0 - 1.96 * mc_se,
    ci_upper = v0 + 1.96 * mc_se
  ) %>%
  arrange(n_dims, n_paths_eval) %>%
  select(n_dims, n_paths_eval, v0, path_level_sd, mc_se, ci_lower, ci_upper)

print(result_table)

# Blank spacing line before each new Dim block -- computed from where n_dims
# actually changes, not a fixed row-count pattern, so it stays correct
# regardless of how many M_e values end up in each block.
n_dims_vec  <- result_table$n_dims
linesep_vec <- ifelse(is.na(lead(n_dims_vec)) | n_dims_vec == lead(n_dims_vec), "", "\\addlinespace")

latex_table <- result_table %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    digits = c(0, 0, 1, 1, 1, 1, 1),
    col.names = c("Dim", "$M_e$", "Price", "Path SD", "SE",
                  "95\\% CI lower", "95\\% CI upper"),
    caption = "Effect of the evaluation sample size $M_e$ on the CLT standard error and 95\\% confidence interval for one fixed policy per dimension.",
    label = "evaluation_sample_size",
    linesep = linesep_vec,
    escape = FALSE
  )

latex_str <- as.character(latex_table)
latex_str <- sub(
  "(\\\\caption\\{[^\n]*\\}\n)",
  "\\1\\\\footnotesize\n\\\\setlength{\\\\tabcolsep}{4pt}\n",
  latex_str
)

writeLines(latex_str, "R_Tables/evaluation_sample_size_table.tex")
