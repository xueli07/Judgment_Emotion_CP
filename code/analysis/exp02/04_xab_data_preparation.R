####
# Experiment 2 Step 4: XAB Task — Data Preparation
#
# Purpose:
#   Prepare XAB discrimination task data for the main analysis and supplementary analysis.
#
#   Main analysis:
#     Both groups (control and continuous) use the continuous group's model-fitted
#     psychometric boundary to classify stimulus pairs as between- or within-category.
#     (Mirrors Exp5 main analysis logic, adapted for control vs. continuous comparison.)
#
#   Supplementary analysis:
#     Fixed boundary at the physical midpoint (morph 50), using only three
#     theoretically diagnostic pairs: 10-30, 40-60, 70-90.
#
# Key difference from Exp5:
#   - Groups: "control" vs. "continuous" 
#   - Control group has no judgment task, so no QC filtering applies to them.
#     Only continuous group subjects are filtered by judgment task QC.
#   - Both groups use the CONTINUOUS group's model-fitted boundary,
#     since the control group has no psychometric boundary.
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
dataDir    <- file.path(projectDir, "data", "processed", "exp02")
identDir   <- file.path(projectDir, "results", "exp02", "judgment")
xabDir     <- file.path(projectDir, "results", "exp02", "xab")

dir.create(file.path(xabDir, "data_for_modeling"), showWarnings = FALSE, recursive = TRUE)

source(file.path(script_dir, "package_preference_config.R"))
setwd(xabDir)
cat("Working directory:", getwd(), "\n\n")


#===== 2. Load Data =====

cat("Loading data...\n")

combined_data <- read_csv(
  file.path(dataDir, "exp02_trial_data.csv"),
  show_col_types = FALSE
)

# High-quality subjects from judgment task QC (continuous group only)
high_quality_subjects <- read_csv(
  file.path(identDir, "quality_control/high_quality_subjects_for_xab.csv"),
  show_col_types = FALSE
)

# Group-level model-fitted boundaries (continuous group boundary used for both groups)
group_model_boundaries <- read_csv(
  file.path(identDir, "group_fitting_results.csv"),
  show_col_types = FALSE
) %>%
  select(judgment_condition, boundary) %>%
  rename(model_fitted_boundary = boundary)

cont_model_boundary <- group_model_boundaries %>%
  filter(judgment_condition == "continuous") %>%
  pull(model_fitted_boundary)

cat(sprintf("Continuous group model-fitted boundary: %.2f\n\n", cont_model_boundary))


#===== 3. Extract and Filter XAB Trials =====

cat("Extracting XAB trials...\n")

xab_data <- combined_data %>%
  filter(!is.na(xab_rt)) %>%
  select(subject_id, judgment_condition, xab_pairs, xab_x_emotion,
         xab_block, xab_trial, xab_resp, xab_rt)

# Control group: keep all (no judgment task → no QC filter)
# Continuous group: keep only high-quality subjects
xab_data <- xab_data %>%
  filter(judgment_condition == "control" |
           subject_id %in% high_quality_subjects$subject_id)

cat(sprintf("Retained: %d trials from %d subjects (%d control, %d continuous)\n\n",
            nrow(xab_data),
            n_distinct(xab_data$subject_id),
            n_distinct(xab_data$subject_id[xab_data$judgment_condition == "control"]),
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
compute_descriptives <- function(data, label) {
  
  subject_stats <- data %>%
    group_by(subject_id, judgment_condition, pair_type) %>%
    summarise(
      n_trials = n(),
      accuracy = mean(xab_resp == 1),
      mean_rt  = mean(xab_rt),
      sd_rt    = sd(xab_rt),
      .groups  = "drop"
    )
  
  group_stats <- subject_stats %>%
    group_by(judgment_condition, pair_type) %>%
    summarise(
      n_subjects = n(),
      mean_acc   = mean(accuracy),
      se_acc     = sd(accuracy) / sqrt(n_subjects),
      mean_rt    = mean(mean_rt),
      se_rt      = sd(mean_rt) / sqrt(n_subjects),
      .groups    = "drop"
    )
  
  cp_effect <- subject_stats %>%
    select(subject_id, judgment_condition, pair_type, accuracy) %>%
    pivot_wider(names_from = pair_type, values_from = accuracy) %>%
    mutate(cp_effect = between - within)
  
  rt_diff <- subject_stats %>%
    select(subject_id, judgment_condition, pair_type, mean_rt) %>%
    pivot_wider(names_from = pair_type, values_from = mean_rt) %>%
    mutate(rt_diff = between - within)
  
  cat(sprintf("\n--- %s ---\n", label))
  cat("Group-level accuracy and RT:\n")
  print(group_stats, n = Inf)
  
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
# Both groups use the continuous group's model-fitted psychometric boundary.
# Since the control group underwent no judgment task, they have no
# group-level boundary of their own. Applying the continuous group's boundary
# to both groups ensures a common reference frame for pair classification,
# directly analogous to Analysis 1a in the original Exp6 code.

cat("========================================\n")
cat("MAIN ANALYSIS\n")
cat("Both groups use continuous group model-fitted boundary\n")
cat(sprintf("Boundary: %.2f\n", cont_model_boundary))
cat("========================================\n")

xab_main <- xab_data %>%
  mutate(
    boundary_used   = cont_model_boundary,
    boundary_source = "cont_group_model_fitted"
  ) %>%
  rowwise() %>%
  mutate(pair_type = classify_pair(xab_pairs, boundary_used)) %>%
  ungroup()

desc_main <- compute_descriptives(xab_main, "Main Analysis")

write_csv(desc_main$subject_stats,
          file.path(xabDir, "subject_stats_main.csv"))

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
# Boundary-assumption-free test using:
#   - Fixed boundary at the physical midpoint (morph level 50)
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

cat("\nPair classifications (boundary = 50):\n")
xab_supp %>%
  distinct(xab_pairs, pair_type) %>%
  mutate(label = paste0(substr(as.character(xab_pairs), 1, 2), "-",
                        substr(as.character(xab_pairs), 3, 4))) %>%
  arrange(xab_pairs) %>%
  print()

cat(sprintf("\nRetained: %d trials (%d control, %d continuous)\n",
            nrow(xab_supp),
            sum(xab_supp$judgment_condition == "control"),
            sum(xab_supp$judgment_condition == "continuous")))

desc_supp <- compute_descriptives(xab_supp, "Supplementary Analysis")

write_csv(desc_supp$subject_stats,
          file.path(xabDir, "subject_stats_supp.csv"))

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
