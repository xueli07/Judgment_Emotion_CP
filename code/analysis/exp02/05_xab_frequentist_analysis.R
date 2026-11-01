####
# Step 5 (exp02): XAB Task - Frequentist Statistical Analysis
#
# Purpose:
#   Perform 2(Judgment Condition: control/continuous) × 2(Pair Type: between/within)
#   mixed ANOVA and follow-up tests on XAB discrimination data, for both accuracy
#   and reaction time (RT).
#
# Analyses:
#   Main analysis:          Both groups use continuous group model-fitted boundary
#   Supplementary analysis: Fixed boundary at 50, key pairs only (10-30, 40-60, 70-90)
#
# For each analysis × dependent variable (accuracy / RT):
#   1. Assumption checks (Shapiro-Wilk normality; Levene's homogeneity of variance)
#   2. Mixed ANOVA with partial η²
#   3. Post-hoc simple effects tests (t-tests with Cohen's d)
#   4. Visualization: bar chart (group mean ± SE) + CP/RT effect plot
#
# Outputs (saved to statistical_tests/):
#   - ANOVA results CSV per analysis
#   - Post-hoc results CSV per analysis
#   - Plots (accuracy & RT bar chart, CP/RT effect) per analysis
#   - Master ANOVA summary table across both analyses
####

#===== 1. Setup =====

rm(list = ls())

library(tidyverse)
library(ggplot2)
library(scales)
library(ez)          # Mixed ANOVA with repeated measures
library(car)         # Levene's test
library(rstatix)     # Assumption tests, Cohen's d
library(conflicted)


set.seed(123)

# Project directories
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
projectDir <- normalizePath(file.path(script_dir, "..", "..", ".."))
xabDir     <- file.path(projectDir, "results", "exp02", "xab")
statsDir   <- file.path(xabDir, "statistical_tests")

dir.create(file.path(statsDir, "plots"), showWarnings = FALSE, recursive = TRUE)

source(file.path(script_dir, "package_preference_config.R"))
setwd(xabDir)
cat("Working directory:", getwd(), "\n\n")


#===== 2. Plotting Theme and Color Scheme =====

color_control  <- "#F28E8E"   # Control group
color_cont <- "#7FB3FF"   # Continuous group

# Custom theme
# Clean publication-style theme
# Custom theme
theme_stats <- function(base_size = 12) {
  theme_minimal(base_size = 18) +
    theme(
      
      panel.grid = element_blank(),
      
      axis.line = element_line(color = "black", linewidth = 1.2),
      axis.ticks = element_line(color = "black", linewidth = 1.2),
      
      axis.title = element_text(size = 20, face = "bold"),
      axis.title.y = element_text(size = 26, face = "bold", margin = margin(r = 15)),
      axis.title.x = element_text(size = 26, face = "bold",margin = margin(t = 20)),
      axis.text.x = element_text(color = "black"),
      axis.text.y = element_text(color = "black"),
      axis.text = element_text(size = 22,color = "black",
                               margin = margin(t = 15)),
      
      legend.position = "top",
      legend.margin = margin(b = -15),
      legend.title = element_blank(),
      legend.text = element_text(size = 18)
    )
}



#===== 3. Analysis Functions =====
# ---- 3.1 Assumption Checks ----

# Shapiro-Wilk normality test and Levene's homogeneity test for a given DV.
# dv_col: column name as string ("accuracy" or "mean_rt")
check_assumptions <- function(data, dv_col, label) {
  
  cat(sprintf("\n--- Assumption Checks: %s (%s) ---\n", label, toupper(dv_col)))
  
  # Normality per group × pair type
  norm_results <- data %>%
    group_by(judgment_condition, pair_type) %>%
    summarise(
      n         = n(),
      W         = shapiro.test(.data[[dv_col]])$statistic,
      p_shapiro = shapiro.test(.data[[dv_col]])$p.value,
      normal    = p_shapiro > 0.05,
      .groups   = "drop"
    )
  cat("Shapiro-Wilk normality:\n")
  print(norm_results)
  
  # Levene's test per pair type
  lev_between <- leveneTest(
    reformulate("judgment_condition", response = dv_col),
    data = filter(data, pair_type == "between")
  )
  lev_within <- leveneTest(
    reformulate("judgment_condition", response = dv_col),
    data = filter(data, pair_type == "within")
  )
  
  cat(sprintf("\nLevene's test - Between pairs: F(%d,%d) = %.3f, p = %.4f\n",
              lev_between$Df[1], lev_between$Df[2],
              lev_between$`F value`[1], lev_between$`Pr(>F)`[1]))
  cat(sprintf("Levene's test - Within pairs:   F(%d,%d) = %.3f, p = %.4f\n",
              lev_within$Df[1], lev_within$Df[2],
              lev_within$`F value`[1], lev_within$`Pr(>F)`[1]))
  
  list(normality      = norm_results,
       levene_between = lev_between,
       levene_within  = lev_within)
}


# ---- 3.2 Mixed ANOVA ----

# 2(Judgment Condition: between-subjects) × 2(Pair Type: within-subjects) mixed ANOVA.
# Returns ANOVA table with partial η² and effect size interpretation.
run_mixed_anova <- function(data, dv_col, label) {
  
  cat(sprintf("\n--- Mixed ANOVA: %s (%s) ---\n", label, toupper(dv_col)))
  
  # ezANOVA uses base-R NSE and does not accept !!sym() or string arguments for dv.
  # Renaming the target column to a fixed name "dv_value" before calling ezANOVA
  # is the simplest reliable workaround.
  data <- data %>%
    rename(dv_value = all_of(dv_col)) %>%
    mutate(subject_id    = as.factor(subject_id),
           judgment_condition = as.factor(judgment_condition),
           pair_type     = as.factor(pair_type))
  
  anova_res <- ezANOVA(
    data     = data,
    dv       = dv_value,
    wid      = subject_id,
    between  = judgment_condition,
    within   = pair_type,
    type     = 3,
    detailed = TRUE
  )
  
  # Add partial η² and interpretation
  anova_table <- anova_res$ANOVA %>%
    mutate(
      partial_eta_sq = SSn / (SSn + SSd),
      effect_size    = case_when(
        partial_eta_sq < 0.01 ~ "negligible",
        partial_eta_sq < 0.06 ~ "small",
        partial_eta_sq < 0.14 ~ "medium",
        TRUE                  ~ "large"
      )
    )
  
  cat("\nANOVA results:\n")
  print(anova_table %>% select(Effect, F, p, partial_eta_sq, effect_size))
  
  list(anova = anova_res, table = anova_table)
}


# ---- 3.3 Post-hoc Tests ----

# Helper: safely extract Cohen's d from rstatix::cohens_d() output.
# rstatix returns a data frame whose column name varies by version and call type
# (e.g. "Cohens_d", "effsize", "d"). We try all known names defensively.
.extract_d <- function(d_obj) {
  if (is.null(d_obj) || isTRUE(is.na(d_obj))) return(NA_real_)
  if (is.numeric(d_obj) && length(d_obj) == 1)  return(d_obj)
  if (is.data.frame(d_obj) || is.list(d_obj)) {
    for (nm in c("Cohens_d", "cohens_d", "effsize", "estimate", "d")) {
      # Use [[ only if the name actually exists to avoid out-of-bounds error
      if (nm %in% names(d_obj)) {
        val <- d_obj[[nm]]
        if (!is.null(val) && length(val) > 0) return(as.numeric(val[1]))
      }
    }
  }
  NA_real_
}

# Post-hoc simple effects tests for a given DV.
# Tests:
#   1. Judgment Condition effect within each pair type (independent t-tests)
#   2. Pair Type effect within each group (paired t-tests → CP/RT effect)
#   3. CP/RT difference comparison between groups (independent t-test)
run_posthoc <- function(data, dv_col, label) {
  
  cat(sprintf("\n--- Post-hoc Tests: %s (%s) ---\n", label, toupper(dv_col)))
  
  is_paired_dv <- dv_col == "mean_rt"   # label for printing only
  
  # 1. Judgment Condition effect within each pair type -----
  
  cat("\n1. Judgment Condition effect by pair type:\n")
  posthoc_by_pair <- map(c("between", "within"), function(pt) {
    d <- filter(data, pair_type == pt)
    tt <- t.test(reformulate("judgment_condition", response = dv_col),
                 data = d, var.equal = FALSE)
    cd_raw <- rstatix::cohens_d(reformulate("judgment_condition", response = dv_col),
                                data = d)
    cd <- .extract_d(cd_raw)
    stats <- d %>%
      group_by(judgment_condition) %>%
      summarise(M = mean(.data[[dv_col]]), SE = sd(.data[[dv_col]]) / sqrt(n()),
                .groups = "drop")
    cat(sprintf("  %s pairs: t(%.2f) = %.3f, p = %.4f, d = %.3f\n",
                pt, tt$parameter, tt$statistic, tt$p.value, cd))
    cat(sprintf("    Control M = %.4f (SE = %.4f), Continuous M = %.4f (SE = %.4f)\n",
                stats$M[stats$judgment_condition == "control"],
                stats$SE[stats$judgment_condition == "control"],
                stats$M[stats$judgment_condition == "continuous"],
                stats$SE[stats$judgment_condition == "continuous"]))
    list(pair_type = pt, test = tt, cohens_d = cd, stats = stats)
  })
  names(posthoc_by_pair) <- c("between", "within")
  
  # 2. Pair Type effect within each group (paired t-test) -----
  
  cat("\n2. Pair Type effect by group (paired t-test):\n")
  posthoc_by_group <- map(c("control", "continuous"), function(grp) {
    d_wide <- data %>%
      filter(judgment_condition == grp) %>%
      select(subject_id, pair_type, all_of(dv_col)) %>%
      pivot_wider(names_from = pair_type, values_from = all_of(dv_col))
    stats <- data %>%
      filter(judgment_condition == grp) %>%
      group_by(pair_type) %>%
      summarise(M = mean(.data[[dv_col]]), SE = sd(.data[[dv_col]]) / sqrt(n()),
                .groups = "drop")
    tt <- t.test(d_wide$between, d_wide$within, paired = TRUE)
    cd_raw <- tryCatch(rstatix::cohens_d(d_wide$between, d_wide$within, paired = TRUE),
                       error = function(e) NA)
    cd <- .extract_d(cd_raw)
    diff_mean <- mean(d_wide$between - d_wide$within, na.rm = TRUE)
    cat(sprintf("  %s: t(%d) = %.3f, p = %.4f, d = %s\n",
                grp, tt$parameter, tt$statistic, tt$p.value,
                ifelse(is.na(cd), "NA", sprintf("%.3f", cd))))
    cat(sprintf("    Between M = %.4f (SE = %.4f), Within M = %.4f (SE = %.4f), Diff = %.4f\n",
                stats$M[stats$pair_type == "between"],
                stats$SE[stats$pair_type == "between"],
                stats$M[stats$pair_type == "within"],
                stats$SE[stats$pair_type == "within"],
                diff_mean))
    list(group = grp, test = tt, cohens_d = cd, stats = stats, diff = diff_mean)
  })
  names(posthoc_by_group) <- c("control", "continuous")
  
  # 3. CP / RT-difference comparison between groups -----
  
  cat("\n3. Effect size (between − within) comparison between groups:\n")
  diff_data <- data %>%
    select(subject_id, judgment_condition, pair_type, all_of(dv_col)) %>%
    pivot_wider(names_from = pair_type, values_from = all_of(dv_col)) %>%
    mutate(diff = between - within)
  
  diff_stats <- diff_data %>%
    group_by(judgment_condition) %>%
    summarise(M = mean(diff), SE = sd(diff) / sqrt(n()), .groups = "drop")
  
  diff_test <- t.test(diff ~ judgment_condition, data = diff_data, var.equal = FALSE)
  diff_d_raw <- rstatix::cohens_d(diff ~ judgment_condition, data = diff_data)
  diff_d <- .extract_d(diff_d_raw)
  
  # 提取95% CI
  diff_ci_low  <- round(diff_test$conf.int[1], 4)
  diff_ci_high <- round(diff_test$conf.int[2], 4)
  
  cat(sprintf("  Independent t-test: t(%.2f) = %.3f, p = %.4f, d = %.3f, 95%% CI [%.4f, %.4f]\n",
              diff_test$parameter, diff_test$statistic, diff_test$p.value, diff_d,
              diff_ci_low, diff_ci_high))
  cat(sprintf("    Control M_diff = %.4f (SE = %.4f), Continuous M_diff = %.4f (SE = %.4f)\n",
              diff_stats$M[diff_stats$judgment_condition == "control"],
              diff_stats$SE[diff_stats$judgment_condition == "control"],
              diff_stats$M[diff_stats$judgment_condition == "continuous"],
              diff_stats$SE[diff_stats$judgment_condition == "continuous"]))
  
  list(by_pair_type  = posthoc_by_pair,
       by_group      = posthoc_by_group,
       diff_between_groups = list(test = diff_test, cohens_d = diff_d,
                                  ci = c(diff_ci_low, diff_ci_high),
                                  stats = diff_stats))
}


# ---- 3.4 Visualization ----

# Bar chart (group mean ± SE) for accuracy or RT, by pair type and judgment group.
create_plot <- function(data, dv_col, y_label, y_limits, y_breaks,
                        label_fn = NULL, analysis_label) {
  
  plot_data <- data %>%
    group_by(judgment_condition, pair_type) %>%
    summarise(M  = mean(.data[[dv_col]]),
              SE = sd(.data[[dv_col]]) / sqrt(n()),
              .groups = "drop")
  
  ggplot(plot_data,
         aes(x = pair_type, y = M, fill = judgment_condition)) +
    geom_bar(stat = "identity", position = position_dodge(0.9),
             alpha = 1, width = 0.7) +
    geom_errorbar(aes(ymin = M - SE, ymax = M + SE),
                  position = position_dodge(0.9),
                  width = 0.25, linewidth = 0.75, color = "grey25") +
    scale_fill_manual(
      values = c("control" = color_control, "continuous" = color_cont),
      labels = c("control" = "Control", "continuous" = "Continuous")
    ) +
    scale_x_discrete(
      labels = c("between" = "Between-Category", "within" = "Within-Category")
    ) +
    scale_y_continuous(
      breaks = y_breaks,
      labels = if (!is.null(label_fn)) label_fn else waiver(),
      expand = c(0, 0)
    ) +
    coord_cartesian(ylim = y_limits) +
    labs(x = "Pair Type", y = y_label) +
    theme_stats()
}

# CP effect plot: individual data points + boxplot + group mean diamond
create_cp_effect_plot <- function(subject_data, dv_col, y_label, y_limits, y_breaks, analysis_label) {
  
  cp_data <- subject_data %>%
    select(subject_id, judgment_condition, pair_type, all_of(dv_col)) %>%
    pivot_wider(names_from = pair_type, values_from = all_of(dv_col)) %>%
    mutate(effect = between - within)
  
  ggplot(cp_data, aes(x = judgment_condition, y = effect, fill = judgment_condition)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray60", linewidth = 0.8) +
    geom_line(aes(group = subject_id), color = "gray70", alpha = 0.5, linewidth = 0.5) +
    geom_jitter(aes(color = judgment_condition), width = 0.05, alpha = 0.5, size = 2) +
    geom_boxplot(width = 0.2, alpha = 1, outlier.shape = NA, color = "grey25") +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 4, fill = "white", color = "black", stroke = 1.2) +
    scale_fill_manual(values  = c("control" = "#F28E8E", "continuous" = "#7FB3FF")) +
    scale_color_manual(values = c("control" = "#F28E8E", "continuous" = "#7FB3FF")) +
    scale_x_discrete(labels = c("control" = "Control", "continuous" = "Continuous")) +
    scale_y_continuous(breaks = y_breaks, labels = function(x) format(x, scientific = FALSE, drop0trailing = TRUE)) +
    coord_cartesian(ylim = y_limits) +
    labs(x = "Judgment Condition", y = y_label) +
    theme_stats() +
    theme(legend.position = "none",
          axis.title.y = element_text(size = 28, face = "bold", margin = margin(r = 15)),
          axis.title.x = element_text(size = 32, face = "bold",margin = margin(t = 20)),
          axis.text = element_text(size = 25,color = "black",margin = margin(t = 15)),)
}


# ---- 3.5 Save Results ----

# Save ANOVA table, post-hoc summary, and plots for one analysis × DV combination.
# analysis_id: short file suffix, e.g. "main" or "supp"
# analysis_label: human-readable label for printed output
save_results <- function(anova_res, posthoc_res, plot_bar, plot_cp,
                         analysis_id, dv_label, analysis_label) {
  
  out_dir <- file.path(statsDir, analysis_id)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # ANOVA table
  write_csv(anova_res$table,
            file.path(out_dir, paste0(dv_label, "_anova_table.csv")))
  
  # Post-hoc summary table
  posthoc_tbl <- tibble(
    comparison = c(
      "Judgment Condition - Between pairs",
      "Judgment Condition - Within pairs",
      "Pair Type - Control group",
      "Pair Type - Continuous group",
      "Effect difference (Cat vs Cont)"
    ),
    t   = c(
      posthoc_res$by_pair_type$between$test$statistic,
      posthoc_res$by_pair_type$within$test$statistic,
      posthoc_res$by_group$control$test$statistic,
      posthoc_res$by_group$continuous$test$statistic,
      posthoc_res$diff_between_groups$test$statistic
    ),
    df  = c(
      posthoc_res$by_pair_type$between$test$parameter,
      posthoc_res$by_pair_type$within$test$parameter,
      posthoc_res$by_group$control$test$parameter,
      posthoc_res$by_group$continuous$test$parameter,
      posthoc_res$diff_between_groups$test$parameter
    ),
    p   = c(
      posthoc_res$by_pair_type$between$test$p.value,
      posthoc_res$by_pair_type$within$test$p.value,
      posthoc_res$by_group$control$test$p.value,
      posthoc_res$by_group$continuous$test$p.value,
      posthoc_res$diff_between_groups$test$p.value
    ),
    cohens_d = c(
      posthoc_res$by_pair_type$between$cohens_d,
      posthoc_res$by_pair_type$within$cohens_d,
      posthoc_res$by_group$control$cohens_d,
      posthoc_res$by_group$continuous$cohens_d,
      posthoc_res$diff_between_groups$cohens_d
    )
  )
  write_csv(posthoc_tbl,
            file.path(out_dir, paste0(dv_label, "_posthoc_tests.csv")))
  
  # Plots
  ggsave(file.path(statsDir, "plots",
                   paste0(dv_label, "_barplot_", analysis_id, ".jpg")),
         plot_bar, width = 8, height = 7, dpi = 300)
  ggsave(file.path(statsDir, "plots",
                   paste0(dv_label, "_cp_effect_", analysis_id, ".jpg")),
         plot_cp, width = 8, height = 8.5, dpi = 300)
  
  cat(sprintf("Saved: %s results for %s\n", dv_label, analysis_label))
  
  posthoc_tbl
}


# ---- 3.6 Full Pipeline for One Analysis ----

# Runs the complete accuracy + RT analysis pipeline for one analysis.
# analysis_id:    short file suffix matching Step 4 output, e.g. "main" or "supp"
# analysis_label: human-readable name printed in headings and output messages
run_analysis <- function(trial_data, analysis_id, analysis_label) {
  
  cat("\n")
  cat(strrep("=", 60), "\n")
  cat(sprintf("%s\n", analysis_label))
  cat(strrep("=", 60), "\n")
  
  subject_data <- read_csv(
    file.path(xabDir, paste0("subject_stats_", analysis_id, ".csv")),
    show_col_types = FALSE
  ) %>%
    mutate(judgment_condition = factor(judgment_condition, levels = c("control", "continuous")))
  
  cat(sprintf("\nData: %d subjects, %d trials\n", n_distinct(subject_data$subject_id), nrow(trial_data)))
  
  results <- list()
  
  for (dv in c("accuracy", "mean_rt")) {
    
    dv_label  <- if (dv == "accuracy") "acc" else "rt"
    y_lab     <- if (dv == "accuracy") "Mean Accuracy" else "Mean RT (s)"
    y_lims    <- if (dv == "accuracy") c(0.5, 0.9) else c(0.8, 2.2)
    y_brks    <- if (dv == "accuracy") c(0.5, 0.6, 0.7, 0.8, 0.9) else seq(0.8, 2.2, 0.2)
    lbl_fn    <- if (dv == "accuracy") percent_format(accuracy = 1) else NULL
    
    cp_y_lab  <- if (dv == "accuracy") "CP Effect (Accuracy: Between \u2212 Within)" else "RT Difference (s: Between \u2212 Within)"
    cp_y_lims <- if (dv == "accuracy") c(-0.2, 0.2) else c(-0.5, 0.5)
    cp_y_brks <- if (dv == "accuracy") round(seq(-0.3, 0.3, by = 0.1), 2) else round(seq(-0.6, 0.6, by = 0.1), 2)
    
    assumptions <- check_assumptions(subject_data, dv, analysis_label)
    anova_res <- run_mixed_anova(subject_data, dv, analysis_label)
    posthoc_res <- run_posthoc(subject_data, dv, analysis_label)
    
    plot_bar <- create_plot(subject_data, dv, y_lab, y_lims, y_brks, lbl_fn, analysis_label)
    plot_cp  <- create_cp_effect_plot(subject_data, dv, cp_y_lab, cp_y_lims, cp_y_brks, analysis_label)
    
    posthoc_tbl <- save_results(anova_res, posthoc_res, plot_bar, plot_cp, analysis_id, dv_label, analysis_label)
    
    results[[dv]] <- list(
      assumptions = assumptions,
      anova       = anova_res,
      posthoc     = posthoc_res,
      posthoc_tbl = posthoc_tbl
    )
  }
  
  results$subject_data <- subject_data
  results
}

#===== 4. Load Trial-Level Data =====

cat("Loading trial-level data...\n")

data_main <- read_csv(
  file.path(xabDir, "data_for_modeling/xab_main_trial_level.csv"),
  show_col_types = FALSE
)
data_supp <- read_csv(
  file.path(xabDir, "data_for_modeling/xab_supp_trial_level.csv"),
  show_col_types = FALSE
)

cat(sprintf("Main analysis:          %d trials, %d subjects\n",
            nrow(data_main), n_distinct(data_main$subject_id)))
cat(sprintf("Supplementary analysis: %d trials, %d subjects\n",
            nrow(data_supp), n_distinct(data_supp$subject_id)))


#===== 5. Run Analyses =====

# --- Main Analysis ---
res_main <- run_analysis(
  trial_data     = data_main,
  analysis_id    = "main",
  analysis_label = "Main Analysis"
)

# --- Supplementary Analysis ---
res_supp <- run_analysis(
  trial_data     = data_supp,
  analysis_id    = "supp",
  analysis_label = "Supplementary Analysis"
)


#===== 6. Master Summary Tables =====

cat("\n========================================\n")
cat("Master ANOVA Summary\n")
cat("========================================\n\n")

# Helper: extract F, p, and partial η² for all three effects from one ANOVA result
extract_anova_summary <- function(res_dv, analysis_label, dv_label) {
  tbl <- res_dv$anova$table
  get_row <- function(effect) tbl %>% filter(Effect == effect) %>%
    select(F, p, partial_eta_sq)
  
  bind_cols(
    tibble(analysis = analysis_label, dv = dv_label),
    get_row("judgment_condition") %>%
      rename(F_judgment = F, p_judgment = p, eta2_judgment = partial_eta_sq),
    get_row("pair_type") %>%
      rename(F_pair = F, p_pair = p, eta2_pair = partial_eta_sq),
    get_row("judgment_condition:pair_type") %>%
      rename(F_interaction = F, p_interaction = p, eta2_interaction = partial_eta_sq)
  )
}

master_summary <- bind_rows(
  extract_anova_summary(res_main$accuracy, "Main Analysis",          "accuracy"),
  extract_anova_summary(res_main$mean_rt,  "Main Analysis",          "RT"),
  extract_anova_summary(res_supp$accuracy, "Supplementary Analysis", "accuracy"),
  extract_anova_summary(res_supp$mean_rt,  "Supplementary Analysis", "RT")
)

cat("ANOVA summary (both analyses, both DVs):\n")
print(master_summary, width = Inf)

write_csv(master_summary, file.path(statsDir, "master_anova_summary.csv"))
cat("\nSaved: master_anova_summary.csv\n")


#===== 7. Save Workspace =====

save(res_main, res_supp, master_summary,
     file = file.path(statsDir, "xab_statistical_results.RData"))

cat("\n========================================\n")
cat("ALL ANALYSES COMPLETE\n")
cat("========================================\n\n")
cat("Results saved to:", statsDir, "\n\n")
cat("  master_anova_summary.csv\n")
cat("  main/    Main Analysis          — acc + RT anova, posthoc\n")
cat("  supp/    Supplementary Analysis — acc + RT anova, posthoc\n")
cat("  plots/   bar charts + CP/RT effect plots for both analyses\n")
cat("  xab_statistical_results.RData\n")




