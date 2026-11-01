####
# Step 3: Judgment Task — Quality Control and Descriptive Statistics
#
# Purpose:
#   1. Verify that all participants pass pre-defined quality criteria
#      (boundary range, positive slope, R², model convergence).
#   2. Report descriptive statistics for boundary position and slope.
#   3. Test group differences in boundary and slope (t-tests).
#   4. Visualize distributions of boundary and slope by judgment group.
#   5. Save outputs for Step 4 (XAB task analysis).
#
# Key output for downstream steps:
#   quality_control/high_quality_subjects_for_xab.csv
####


#===== 1. Setup =====

rm(list = ls())

library(tidyverse)
library(car)        # Levene's test
library(conflicted)
library(rstatix)   # effect size


set.seed(123)

# Resolve package conflicts
conflicted::conflicts_prefer(dplyr::summarise)
conflicted::conflicts_prefer(dplyr::filter)
conflicted::conflicts_prefer(dplyr::select)
conflicted::conflicts_prefer(dplyr::mutate)
conflicted::conflicts_prefer(dplyr::rename)

# Directories
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
projectDir <- normalizePath(file.path(script_dir, "..", "..", ".."))
resultsDir <- file.path(projectDir, "results", "exp01", "judgment")
qcDir      <- file.path(resultsDir, "quality_control")
dir.create(qcDir, showWarnings = FALSE, recursive = TRUE)

setwd(resultsDir)
cat("Working directory:", getwd(), "\n\n")


#===== 2. Plotting Theme =====

color_cat  <- "#F28E8E"
color_cont <- "#7FB3FF"

theme_stats <- function() {
  theme_minimal(base_size = 18) +
    theme(
      panel.grid      = element_blank(),
      axis.line       = element_line(color = "black", linewidth = 1.2),
      axis.ticks      = element_line(color = "black", linewidth = 1.2),
      axis.title      = element_text(size = 20, face = "bold"),
      axis.title.y    = element_text(margin = margin(r = 15)),
      axis.title.x    = element_text(margin = margin(t = 20)),
      axis.text       = element_text(size = 18, face = "bold"),
      legend.position = "top",
      legend.margin   = margin(b = -20),
      legend.title    = element_blank(),
      legend.text     = element_text(size = 18)
    )
}


#===== 3. Load Fitting Results =====

cat("Loading individual fitting results...\n")

fitting_results <- read_csv(
  file.path(resultsDir, "individual_fitting_results.csv"),
  show_col_types = FALSE
)

cat(sprintf("Loaded: %d subjects (%d categorical, %d continuous)\n\n",
            nrow(fitting_results),
            sum(fitting_results$judgment_condition == "categorical"),
            sum(fitting_results$judgment_condition == "continuous")))


#===== 4. Quality Control Verification =====
#
# Inclusion criteria applied to this sample:
#   1. Boundary within [30, 70]  — directly tied to the XAB task stimulus pairs
#                                   (morph scale: higher value = more sad):
#                                   make sure within pairs include both happy and sad side
#                                   boundary > 30: pair 10-30 is within-category (happy side);
#                                   boundary < 70: pair 70-90 is within-category (sad side).
#   2. Positive slope (> 0.001)  — confirms the psychometric curve has the expected direction
#   3. R² ≥ 0.20                 — confirms adequate goodness of fit
#   4. At least one model converged
#
# All participants were verified to meet these criteria prior to analysis.
# The checks below confirm this programmatically for transparency.

boundary_min   <- 30
boundary_max   <- 70
slope_min      <- 0.001
r2_threshold   <- 0.20

qc_data <- fitting_results %>%
  mutate(
    qc_boundary    = !is.na(boundary)   & boundary   >= boundary_min & boundary   <= boundary_max,
    qc_slope       = !is.na(slope)      & slope       > slope_min,
    qc_r2          = !is.na(r_squared)  & r_squared  >= r2_threshold,
    qc_convergence = logistic_converged | linear_converged,
    pass_qc        = qc_boundary & qc_slope & qc_r2 & qc_convergence
  )

n_pass   <- sum(qc_data$pass_qc)
n_total  <- nrow(qc_data)
n_fail   <- n_total - n_pass

cat("===== Quality Control Verification =====\n")
cat(sprintf("Passed: %d / %d (%.1f%%)\n", n_pass, n_total, 100 * n_pass / n_total))

if (n_fail > 0) {
  # If any participants fail, print details and stop with an informative error
  failed <- qc_data %>%
    filter(!pass_qc) %>%
    mutate(reason = case_when(
      !qc_convergence ~ "no model converged",
      !qc_r2          ~ sprintf("R² too low (%.3f)", r_squared),
      !qc_boundary    ~ sprintf("boundary out of range (%.1f)", boundary),
      !qc_slope       ~ sprintf("non-positive slope (%.4f)", slope)
    )) %>%
    select(subject_id, judgment_condition, boundary, slope, r_squared, reason)
  cat("\nWARNING: The following participants failed QC:\n")
  print(failed)
  stop("QC check failed — review the subjects listed above before proceeding.")
} else {
  cat("All participants passed all criteria. Proceeding with full sample.\n\n")
}

# Use the full sample as the analysis dataset
analysis_data <- fitting_results


#===== 5. Descriptive Statistics =====

# Helper: compute descriptive stats for one continuous variable
describe_var <- function(data, var, label) {
  data %>%
    group_by(judgment_condition) %>%
    summarise(
      n      = n(),
      mean   = mean({{ var }}),
      sd     = sd({{ var }}),
      se     = sd / sqrt(n),
      median = median({{ var }}),
      q25    = quantile({{ var }}, 0.25),
      q75    = quantile({{ var }}, 0.75),
      min    = min({{ var }}),
      max    = max({{ var }}),
      .groups = "drop"
    )
}

boundary_stats <- describe_var(analysis_data, boundary, "Boundary")
slope_stats    <- describe_var(analysis_data, slope,    "Slope")
r2_stats       <- describe_var(analysis_data, r_squared,"R²")

cat("===== Descriptive Statistics =====\n\n")
cat("Boundary position:\n");  print(boundary_stats, n = Inf); cat("\n")
cat("Slope:\n");               print(slope_stats,    n = Inf); cat("\n")
cat("R²:\n");                  print(r2_stats,       n = Inf); cat("\n")


#===== 6. Group Difference Tests =====
#
# We compare boundary and slope between the categorical and continuous groups.
# For each variable: Shapiro-Wilk normality check → Levene's variance test →
# independent-samples t-test (Welch) + Mann-Whitney U as non-parametric check.

run_group_tests <- function(data, var_col, label) {
  
  cat(sprintf("----- %s -----\n", label))
  
  # Normality
  for (grp in unique(data$judgment_condition)) {
    vals <- data[[var_col]][data$judgment_condition == grp]
    sw   <- shapiro.test(vals)
    cat(sprintf("  Shapiro-Wilk (%s): W = %.4f, p = %.4f %s\n",
                grp, sw$statistic, sw$p.value,
                ifelse(sw$p.value > 0.05, "(normal)", "(non-normal)")))
  }
  
  # Homogeneity of variance
  lev <- leveneTest(reformulate("judgment_condition", response = var_col), data = data)
  cat(sprintf("  Levene's test: F(%d,%d) = %.4f, p = %.4f %s\n",
              lev$Df[1], lev$Df[2],
              lev$`F value`[1], lev$`Pr(>F)`[1],
              ifelse(lev$`Pr(>F)`[1] > 0.05, "(equal variances)", "(unequal variances)")))
  
  # t-test (Welch, does not assume equal variance)
  tt <- t.test(reformulate("judgment_condition", response = var_col),
               data = data, var.equal = FALSE)
  cat(sprintf("  Welch t-test: t(%.1f) = %.4f, p = %.4f\n",
              tt$parameter, tt$statistic, tt$p.value))
  cat(sprintf("    Mean diff = %.4f (95%% CI: [%.4f, %.4f])\n",
              diff(tt$estimate), tt$conf.int[1], tt$conf.int[2]))
  
  # Mann-Whitney U (non-parametric)
  wt <- wilcox.test(reformulate("judgment_condition", response = var_col), data = data)
  cat(sprintf("  Mann-Whitney U: W = %.0f, p = %.4f\n\n",
              wt$statistic, wt$p.value))
  
  list(t_test = tt, wilcox = wt, levene = lev)
}

cat("===== Group Difference Tests =====\n\n")
tests_boundary <- run_group_tests(analysis_data, "boundary", "Boundary Position")
tests_slope    <- run_group_tests(analysis_data, "slope",    "Slope")
# Save t-test results
# Compute Cohen's d
d_boundary <- cohens_d(boundary ~ judgment_condition, data = analysis_data)
d_slope    <- cohens_d(slope ~ judgment_condition, data = analysis_data)

t_results <- tibble(
  variable = c("boundary", "slope"),
  
  t  = c(tests_boundary$t_test$statistic,
         tests_slope$t_test$statistic),
  
  df = c(tests_boundary$t_test$parameter,
         tests_slope$t_test$parameter),
  
  p  = c(tests_boundary$t_test$p.value,
         tests_slope$t_test$p.value),
  
  mean_categorical = c(tests_boundary$t_test$estimate[1],
                       tests_slope$t_test$estimate[1]),
  
  mean_continuous  = c(tests_boundary$t_test$estimate[2],
                       tests_slope$t_test$estimate[2]),
  
  CI_low  = c(tests_boundary$t_test$conf.int[1], tests_slope$t_test$conf.int[1]),
  
  CI_high = c(tests_boundary$t_test$conf.int[2], tests_slope$t_test$conf.int[2]),
  
  cohens_d = c(d_boundary$effsize, d_slope$effsize))

write_csv(t_results, file.path(qcDir, "t_test_results_full.csv"))


#===== 7. Visualization =====

# Shared violin + boxplot + jitter helper
plot_distribution <- function(data, var_col, y_label, subtitle_text) {
  ggplot(data, aes(x = judgment_condition, y = .data[[var_col]], fill = judgment_condition)) +
    geom_violin(alpha = 0.5, trim = FALSE, scale = "width") +
    geom_boxplot(width = 0.2, alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.12, alpha = 0.4, size = 2) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 3,
                 fill = "white", color = "black") +
    scale_fill_manual(values = c("categorical" = color_cat,
                                 "continuous"  = color_cont)) +
    scale_x_discrete(labels = c("categorical" = "Categorical",
                                "continuous"  = "Continuous")) +
    labs(x = "Judgment Condition", y = y_label, subtitle = subtitle_text) +
    theme_stats() +
    theme(legend.position = "none")
}

tt_b <- tests_boundary$t_test
tt_s <- tests_slope$t_test

p_boundary <- plot_distribution(
  analysis_data, "boundary",
  y_label      = "Boundary Position (% Sad Morph)",
  subtitle_text = sprintf("t(%.1f) = %.3f, p = %.4f",
                          tt_b$parameter, tt_b$statistic, tt_b$p.value)
)

p_slope <- plot_distribution(
  analysis_data, "slope",
  y_label      = "Slope Parameter",
  subtitle_text = sprintf("t(%.1f) = %.3f, p = %.4f",
                          tt_s$parameter, tt_s$statistic, tt_s$p.value)
)

ggsave(file.path(qcDir, "boundary_distribution.jpg"),
       p_boundary, width = 6, height = 5.5, dpi = 300)
ggsave(file.path(qcDir, "slope_distribution.jpg"),
       p_slope,    width = 6, height = 5.5, dpi = 300)

cat("===== Plots Saved =====\n")
cat("  boundary_distribution.jpg\n")
cat("  slope_distribution.jpg\n\n")


#===== 8. Save Outputs =====

# Subject list passed to Step 4 (XAB analysis)
high_quality_subjects <- analysis_data %>%
  select(subject_id, judgment_condition, best_model_type, boundary, slope, r_squared)

write_csv(high_quality_subjects,
          file.path(qcDir, "high_quality_subjects_for_xab.csv"))

# Descriptive statistics tables
write_csv(boundary_stats, file.path(qcDir, "boundary_descriptive_stats.csv"))
write_csv(slope_stats,    file.path(qcDir, "slope_descriptive_stats.csv"))
write_csv(r2_stats,       file.path(qcDir, "r2_descriptive_stats.csv"))

cat("===== Files Saved =====\n")
cat("  high_quality_subjects_for_xab.csv  ← read by Step 4\n")
cat("  boundary_descriptive_stats.csv\n")
cat("  slope_descriptive_stats.csv\n")
cat("  r2_descriptive_stats.csv\n\n")
cat("Next step: Run Step 4 for XAB task data preparation.\n")


