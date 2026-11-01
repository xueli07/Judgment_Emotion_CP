####
# Experiment 2 Step 3: Judgment Task — Quality Control and Descriptive Statistics
#
# Purpose:
#   1. Verify that all continuous group participants pass pre-defined quality criteria
#      (boundary range, positive slope, R², model convergence).
#      No participants are excluded here: the sample was confirmed to meet all criteria
#      prior to analysis. This section serves as a transparent record of the inclusion
#      standards applied.
#   2. Report descriptive statistics for boundary position, slope, and R².
#   3. Visualize the distributions of boundary and slope.
#   4. Save outputs for Step 4 (XAB task analysis).
#
# Key design note:
#   Only the continuous group completed the judgment task; there is no
#   between-group comparison in this step.
#
# Key output for downstream steps:
#   quality_control/high_quality_subjects_for_xab.csv
####


#===== 1. Setup =====

rm(list = ls())

library(tidyverse)
library(conflicted)

set.seed(123)

conflicted::conflicts_prefer(dplyr::summarise)
conflicted::conflicts_prefer(dplyr::filter)
conflicted::conflicts_prefer(dplyr::select)
conflicted::conflicts_prefer(dplyr::mutate)
conflicted::conflicts_prefer(dplyr::rename)

# Directories
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
projectDir <- normalizePath(file.path(script_dir, "..", "..", ".."))
resultsDir <- file.path(projectDir, "results", "exp02", "judgment")
qcDir      <- file.path(resultsDir, "quality_control")
dir.create(qcDir, showWarnings = FALSE, recursive = TRUE)

source(file.path(script_dir, "package_preference_config.R"))
setwd(resultsDir)
cat("Working directory:", getwd(), "\n\n")


#===== 2. Plotting Theme =====

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
      legend.position = "none"
    )
}


#===== 3. Load Fitting Results =====

cat("Loading individual fitting results...\n")

fitting_results <- read_csv(
  file.path(resultsDir, "individual_fitting_results.csv"),
  show_col_types = FALSE
)

cat(sprintf("Loaded: %d subjects (continuous group only)\n\n", nrow(fitting_results)))


#===== 4. Quality Control Verification =====
#
# Inclusion criteria applied to this sample:
#   1. Boundary within [30, 70]  — directly tied to the XAB task stimulus pairs
#                                   (morph scale: higher value = more sad):
#                                   boundary > 30: pair 10-30 is within-category (happy side);
#                                   boundary < 70: pair 70-90 is within-category (sad side).
#                                   Pair 40-60 always straddles the midpoint.
#   2. Positive slope (> 0.001)  — confirms the expected direction of the psychometric curve
#   3. R² >= 0.20                — confirms adequate goodness of fit
#   4. At least one model converged
#
# All participants were verified to meet these criteria prior to analysis.
# The checks below confirm this programmatically for transparency.

boundary_min <- 30
boundary_max <- 70
slope_min    <- 0.001
r2_threshold <- 0.20

qc_data <- fitting_results %>%
  mutate(
    qc_boundary    = !is.na(boundary)  & boundary  >= boundary_min & boundary  <= boundary_max,
    qc_slope       = !is.na(slope)     & slope      > slope_min,
    qc_r2          = !is.na(r_squared) & r_squared >= r2_threshold,
    qc_convergence = logistic_converged | linear_converged,
    pass_qc        = qc_boundary & qc_slope & qc_r2 & qc_convergence
  )

n_pass  <- sum(qc_data$pass_qc)
n_total <- nrow(qc_data)
n_fail  <- n_total - n_pass

cat("===== Quality Control Verification =====\n")
cat(sprintf("Passed: %d / %d (%.1f%%)\n", n_pass, n_total, 100 * n_pass / n_total))

if (n_fail > 0) {
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
  stats <- data %>%
    summarise(
      n      = n(),
      mean   = mean({{ var }}),
      sd     = sd({{ var }}),
      se     = sd / sqrt(n),
      median = median({{ var }}),
      q25    = quantile({{ var }}, 0.25),
      q75    = quantile({{ var }}, 0.75),
      min    = min({{ var }}),
      max    = max({{ var }})
    )
  cat(sprintf("%s:\n", label))
  print(stats, width = Inf)
  cat("\n")
  stats
}

cat("===== Descriptive Statistics =====\n\n")
boundary_stats <- describe_var(analysis_data, boundary,   "Boundary position")
slope_stats    <- describe_var(analysis_data, slope,      "Slope")
r2_stats       <- describe_var(analysis_data, r_squared,  "R²")

# Normality checks
cat("Shapiro-Wilk normality tests:\n")
for (var in c("boundary", "slope", "r_squared")) {
  sw <- shapiro.test(analysis_data[[var]])
  cat(sprintf("  %-12s W = %.4f, p = %.4f %s\n",
              var, sw$statistic, sw$p.value,
              ifelse(sw$p.value > 0.05, "(normal)", "(non-normal)")))
}
cat("\n")

# Best model distribution
cat("Best model distribution:\n")
model_dist <- analysis_data %>%
  count(best_model_type) %>%
  mutate(pct = 100 * n / sum(n))
print(model_dist)
cat("\n")


#===== 6. Visualization =====

# Violin + boxplot + jitter for a single variable (single-group)
plot_distribution <- function(data, var_col, y_label, subtitle_text) {
  ggplot(data, aes(x = "Continuous", y = .data[[var_col]])) +
    geom_violin(alpha = 0.5, trim = FALSE, scale = "width", fill = color_cont) +
    geom_boxplot(width = 0.2, alpha = 0.7, outlier.shape = NA, fill = color_cont) +
    geom_jitter(width = 0.12, alpha = 0.4, size = 2, color = color_cont) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 3,
                 fill = "white", color = "black") +
    labs(x = "Judgment Condition", y = y_label, subtitle = subtitle_text) +
    theme_stats()
}

p_boundary <- plot_distribution(
  analysis_data, "boundary",
  y_label      = "Boundary Position (% Sad Morph)",
  subtitle_text = sprintf("M = %.2f, SD = %.2f, Median = %.2f",
                          boundary_stats$mean, boundary_stats$sd, boundary_stats$median)
)

p_slope <- plot_distribution(
  analysis_data, "slope",
  y_label      = "Slope Parameter",
  subtitle_text = sprintf("M = %.4f, SD = %.4f, Median = %.4f",
                          slope_stats$mean, slope_stats$sd, slope_stats$median)
)

ggsave(file.path(qcDir, "boundary_distribution.jpg"),
       p_boundary, width = 6, height = 5.5, dpi = 300)
ggsave(file.path(qcDir, "slope_distribution.jpg"),
       p_slope,    width = 6, height = 5.5, dpi = 300)

cat("===== Plots Saved =====\n")
cat("  boundary_distribution.jpg\n")
cat("  slope_distribution.jpg\n\n")


#===== 7. Save Outputs =====

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
cat("  high_quality_subjects_for_xab.csv  <- read by Step 4\n")
cat("  boundary_descriptive_stats.csv\n")
cat("  slope_descriptive_stats.csv\n")
cat("  r2_descriptive_stats.csv\n\n")
cat("Next step: Run Step 4 for XAB task data preparation.\n")
