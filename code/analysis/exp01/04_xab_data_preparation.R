####
# Step 4: XAB Task - Data Preparation
#
# Purpose:
#   Prepare XAB discrimination task data for the main analysis and supplementary analysis.
#
#   Main analysis:
#     Both groups use their own group-level model-fitted psychometric boundary to
#     classify stimulus pairs as between- or within-category.
#
#   Supplementary analysis:
#     Fixed boundary at the physical midpoint (morph 50), using only three
#     theoretically diagnostic pairs: 10-30, 40-60, 70-90.
#
# Output:
#   - Trial-level data for statistical modeling (2 CSV files in data_for_modeling/)
#   - Subject-level accuracy and RT statistics (2 CSV files, read by Step 5)
####


#===== 1. Setup =====

rm(list = ls())

library(tidyverse)
library(conflicted)
library(Rmisc)

set.seed(123)

# Project directories
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
projectDir <- normalizePath(file.path(script_dir, "..", "..", ".."))
dataDir    <- file.path(projectDir, "data", "processed", "exp01")
identDir   <- file.path(projectDir, "results", "exp01", "judgment")
xabDir     <- file.path(projectDir, "results", "exp01", "xab")

# Create output directories
dir.create(file.path(xabDir, "data_for_modeling"), showWarnings = FALSE, recursive = TRUE)

# Resolve package conflicts
source(file.path(script_dir, "package_preference_config.R"))
setwd(xabDir)
cat("Working directory:", getwd(), "\n\n")


#===== 2. Load Data =====

cat("Loading data...\n")

# Raw trial data (all tasks combined)
combined_data <- read_csv(
  file.path(dataDir, "exp01_trial_data.csv"),
  show_col_types = FALSE
)

# High-quality subjects from judgment task QC
# Contains individual psychometric boundary and slope per subject
high_quality_subjects <- read_csv(
  file.path(identDir, "quality_control/high_quality_subjects_for_xab.csv"),
  show_col_types = FALSE
)

# Group-level model-fitted boundaries (from group-level psychometric curve fitting)
# Each group's trials are classified using their own group boundary (main analysis)
group_model_boundaries <- read_csv(
  file.path(identDir, "group_fitting_results.csv"),
  show_col_types = FALSE
) %>%
  select(judgment_condition, boundary) %>%
  rename(model_fitted_boundary = boundary)

cat("Group model-fitted boundaries:\n")
print(group_model_boundaries)
cat("\n")


#===== 3. Extract and Filter XAB Trials =====

cat("Extracting XAB trials...\n")

# Keep only XAB trials and relevant columns
xab_data <- combined_data %>%
  filter(!is.na(xab_rt)) %>%
  select(subject_id, judgment_condition, xab_pairs, xab_x_emotion,
         xab_block, xab_trial, xab_resp, xab_rt)

# Retain only high-quality subjects (passed judgment task QC)
xab_data <- xab_data %>%
  filter(subject_id %in% high_quality_subjects$subject_id)

cat(sprintf("Retained: %d trials from %d subjects (%d categorical, %d continuous)\n\n",
            nrow(xab_data),
            n_distinct(xab_data$subject_id),
            n_distinct(xab_data$subject_id[xab_data$judgment_condition == "categorical"]),
            n_distinct(xab_data$subject_id[xab_data$judgment_condition == "continuous"])))


#===== 4. Helper Functions =====

# Classify a stimulus pair as "between" or "within" category given a boundary.
# "Between": the two morph levels straddle the boundary.
# "Within":  both morph levels fall on the same side.
# Pair string format: 4-digit string, e.g. "4060" = morph 40 vs morph 60.
classify_pair <- function(pair_string, boundary) {
  pair_char <- as.character(pair_string)
  m1 <- as.numeric(substr(pair_char, 1, 2))
  m2 <- as.numeric(substr(pair_char, 3, 4))
  ifelse(min(m1, m2) < boundary & max(m1, m2) >= boundary, "between", "within")
}

classify_pairs_vectorized <- function(pair_strings, boundary) {
  sapply(pair_strings, classify_pair, boundary = boundary)
}

# Compute subject-level accuracy and RT descriptives for a given dataset.
# Returns both a subject-level table and a group-level summary.
compute_descriptives <- function(data, label) {
  
  # Subject-level: accuracy and mean RT per subject × pair type
  subject_stats <- data %>%
    group_by(subject_id, judgment_condition, pair_type) %>%
    summarise(
      n_trials  = n(),
      accuracy  = mean(xab_resp == 1),
      mean_rt   = mean(xab_rt),
      sd_rt     = sd(xab_rt),
      .groups   = "drop"
    )
  
  # Group-level summary across subjects
  group_stats <- subject_stats %>%
    group_by(judgment_condition, pair_type) %>%
    summarise(
      n_subjects   = n(),
      mean_acc     = mean(accuracy),
      se_acc       = sd(accuracy) / sqrt(n_subjects),
      mean_rt      = mean(mean_rt),
      se_rt        = sd(mean_rt) / sqrt(n_subjects),
      .groups      = "drop"
    )
  
  # CP effect (between − within) per subject
  cp_effect <- subject_stats %>%
    select(subject_id, judgment_condition, pair_type, accuracy) %>%
    pivot_wider(names_from = pair_type, values_from = accuracy) %>%
    mutate(cp_effect = between - within)
  
  # RT difference (between − within) per subject
  rt_diff <- subject_stats %>%
    select(subject_id, judgment_condition, pair_type, mean_rt) %>%
    pivot_wider(names_from = pair_type, values_from = mean_rt) %>%
    mutate(rt_diff = between - within)
  
  # Print brief summary
  cat(sprintf("\n--- %s ---\n", label))
  cat("Group-level accuracy and RT:\n")
  print(group_stats, n = Inf)
  
  # Warn if any subject has fewer than 10 trials in a condition
  imbalanced <- subject_stats %>%
    pivot_wider(names_from = pair_type, values_from = n_trials, values_fill = 0) %>%
    filter(between < 10 | within < 10)
  if (nrow(imbalanced) > 0) {
    cat("\nWARNING: subjects with <10 trials in at least one condition:\n")
    print(imbalanced)
  }
  
  list(subject_stats = subject_stats,
       group_stats   = group_stats,
       cp_effect     = cp_effect,
       rt_diff       = rt_diff)
}


#===== 5. Main Analysis =====
#
# Both groups use their own group-level model-fitted psychometric boundary to
# classify stimulus pairs. This avoids confounding individual variability in
# boundary location with discrimination performance, giving a cleaner group-level
# test of categorical perception.

cat("========================================\n")
cat("MAIN ANALYSIS\n")
cat("Both groups use their group-level model-fitted boundary\n")
cat("========================================\n")

# Classify each trial's pair type using that group's model-fitted boundary
xab_main <- xab_data %>%
  left_join(group_model_boundaries, by = "judgment_condition") %>%
  mutate(
    boundary_used   = model_fitted_boundary,
    boundary_source = "group_model_fitted"
  ) %>%
  rowwise() %>%
  mutate(pair_type = classify_pair(xab_pairs, boundary_used)) %>%
  ungroup()

# Compute descriptive statistics
desc_main <- compute_descriptives(xab_main, "Main Analysis")

# Save subject-level stats (accuracy + RT) for Step 5
write_csv(desc_main$subject_stats,
          file.path(xabDir, "subject_stats_main.csv"))

# Save trial-level data for Bayesian modeling
xab_main %>%
  select(subject_id, judgment_condition, xab_pairs, xab_x_emotion,
         xab_block, xab_trial, xab_resp, xab_rt,
         boundary_used, boundary_source, pair_type) %>%
  mutate(success  = as.integer(xab_resp == 1),
         analysis = "main") %>%
  write_csv(file.path(xabDir, "data_for_modeling/xab_main_trial_level.csv"))

cat(sprintf("\nSaved: xab_main_trial_level.csv (%d trials)\n", nrow(xab_main)))
cat("Saved: subject_stats_main.csv\n")


#===== 6. Supplementary Analysis =====
#
# A boundary-assumption-free test of categorical perception using:
#   - A fixed boundary at the physical midpoint (morph level 50)
#   - Only three theoretically diagnostic stimulus pairs:
#       10-30: within-category (happy side)
#       40-60: between-category (straddles the midpoint)
#       70-90: within-category (sad side)

cat("\n========================================\n")
cat("SUPPLEMENTARY ANALYSIS\n")
cat("Fixed boundary at 50, key pairs only (10-30, 40-60, 70-90)\n")
cat("========================================\n")

key_pairs <- c(1030, 4060, 7090)

xab_supp <- xab_data %>%
  filter(xab_pairs %in% key_pairs) %>%
  mutate(
    boundary_used   = 50,
    boundary_source = "fixed_50"
  ) %>%
  rowwise() %>%
  mutate(pair_type = classify_pair(xab_pairs, boundary_used)) %>%
  ungroup()

# Verify pair classification
cat("\nPair classifications (boundary = 50):\n")
xab_supp %>%
  distinct(xab_pairs, pair_type) %>%
  mutate(label = paste0(substr(as.character(xab_pairs), 1, 2), "-",
                        substr(as.character(xab_pairs), 3, 4))) %>%
  arrange(xab_pairs) %>%
  print()

cat(sprintf("\nRetained: %d trials (%d categorical, %d continuous)\n",
            nrow(xab_supp),
            sum(xab_supp$judgment_condition == "categorical"),
            sum(xab_supp$judgment_condition == "continuous")))

# Compute descriptive statistics
desc_supp <- compute_descriptives(xab_supp, "Supplementary Analysis")

# Save subject-level stats for Step 5
write_csv(desc_supp$subject_stats,
          file.path(xabDir, "subject_stats_supp.csv"))

# Save trial-level data for Bayesian modeling
xab_supp %>%
  select(subject_id, judgment_condition, xab_pairs, xab_x_emotion,
         xab_block, xab_trial, xab_resp, xab_rt,
         boundary_used, boundary_source, pair_type) %>%
  mutate(success  = as.integer(xab_resp == 1),
         analysis = "supp") %>%
  write_csv(file.path(xabDir, "data_for_modeling/xab_supp_trial_level.csv"))

cat(sprintf("\nSaved: xab_supp_trial_level.csv (%d trials)\n", nrow(xab_supp)))
cat("Saved: subject_stats_supp.csv\n")


#===== 7. Summary =====

cat("\n========================================\n")
cat("DATA PREPARATION COMPLETE\n")
cat("========================================\n\n")
cat("Trial-level data saved to data_for_modeling/:\n")
cat("  xab_main_trial_level.csv   (main analysis)\n")
cat("  xab_supp_trial_level.csv   (supplementary analysis)\n\n")
cat("Subject-level stats saved (read by Step 5):\n")
cat("  subject_stats_main.csv\n")
cat("  subject_stats_supp.csv\n\n")
cat("Next step: Run Step 5 for frequentist statistical tests.\n")

