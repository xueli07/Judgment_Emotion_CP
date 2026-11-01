# ==============================================================================
# MASTER PIPELINE: Empirical vs. HDDM Predicted Overlay Visualization
# Description: Generates publication-ready visualizations overlaying empirical 
#              data with HDDM posterior predictive simulations. Executes formal 
#              frequentist MANOVA and post-hoc simple effects using bruceR.
# ==============================================================================

rm(list = ls())

# Load required packages
library(tidyverse)
library(ggplot2)
library(scales)
library(conflicted)
library(bruceR)    # For Empirical ANOVA and post-hoc simple effects analyses

# Resolve namespace conflicts in favor of dplyr
conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::select)

# Establish global seed for reproducibility
set.seed(123)

# --- 1. Global Parameters & Directories ---
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
projectDir <- normalizePath(file.path(script_dir, "..", ".."))
setwd(projectDir)

models_to_run <- c("M4_v_judgment_x_pair")

# Critical requirement: Overlay plotting mandates the use of Aggregated simulation data 
sim_csv_filename <- "Master_PPC_Simulations_Aggregated.csv" 

# Create and verify output directories
plotsDir <- file.path(projectDir, "results", "hddm", "plots_overlay_combined")
statsDir <- file.path(projectDir, "results", "hddm", "stats_empirical")
dir.create(plotsDir, showWarnings = FALSE, recursive = TRUE)
dir.create(statsDir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("[INFO] Current working directory initialized at: %s\n", getwd()))

# --- 2. Plotting Theme and Color Palette ---
color_cat  <- "#F28E8E"   # Categorical group palette
color_cont <- "#7FB3FF"   # Continuous group palette

#' Publication-Ready Minimalist Theme
#'
#' Defines a clean, grid-free theme optimized for academic journals.
#'
#' @param base_size The base font size for the plot.
#' @return A ggplot2 theme object.
theme_stats <- function(base_size = 20) {
  theme_minimal(base_size = 20) +
    theme(
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", linewidth = 1.2),
      axis.ticks = element_line(color = "black", linewidth = 1.2),
      axis.title = element_text(size = 20, face = "bold"),
      axis.title.y = element_text(size = 26, face = "bold", margin = margin(r = 15)),
      axis.title.x = element_text(size = 26, face = "bold", margin = margin(t = 20)),
      axis.text = element_text(size = 22, color = "black", margin = margin(t = 15)),
      
      # Clean legend formulation (color mappings only)
      legend.position = "top",
      legend.margin = margin(b = -15),
      legend.title = element_blank(),
      legend.text = element_text(size = 18)
    )
}

# --- 3. Overlay Plotting Functions ---

#' Create Overlay Bar Plot
#'
#' Generates a dodged bar plot representing empirical data, overlaid with 
#' simulated posterior predictive means (diamonds) and standard errors.
#'
#' @param obs_data Data frame containing observed empirical data.
#' @param sim_data Data frame containing simulated PPC data.
#' @param dv_obs Character string specifying the dependent variable in obs_data.
#' @param dv_sim Character string specifying the dependent variable in sim_data.
#' @param y_label Character string for the Y-axis label.
#' @param y_limits Numeric vector specifying Y-axis bounds.
#' @param y_breaks Numeric vector specifying Y-axis tick intervals.
#' @param label_fn Optional scale formatting function (e.g., percent_format).
#' @return A ggplot2 object.
create_overlay_barplot <- function(obs_data, sim_data, dv_obs, dv_sim, y_label, y_limits, y_breaks, label_fn = NULL) {
  
  obs_agg <- obs_data %>%
    group_by(judgment_condition, pair_type) %>%
    summarise(M_obs  = mean(.data[[dv_obs]], na.rm = TRUE),
              SE_obs = sd(.data[[dv_obs]], na.rm = TRUE) / sqrt(n()),
              .groups = "drop")
  
  sim_agg <- sim_data %>%
    group_by(judgment_condition, pair_type) %>%
    summarise(M_sim  = mean(.data[[dv_sim]], na.rm = TRUE),
              SE_sim = sd(.data[[dv_sim]], na.rm = TRUE) / sqrt(n()),
              .groups = "drop")
  
  # Align coordinates for dodge offsets to prevent overlap
  plot_data <- left_join(obs_agg, sim_agg, by = c("judgment_condition", "pair_type")) %>%
    mutate(
      x_num = as.numeric(pair_type),
      x_bar = ifelse(judgment_condition == "categorical", x_num - 0.225, x_num + 0.225),
      x_sim = x_bar + 0.14 
    )
  
  ggplot(plot_data) +
    # 1. Base Layer: Observed Bars & Error Bars
    geom_bar(aes(x = x_bar, y = M_obs, fill = judgment_condition), stat = "identity", width = 0.35, alpha = 1) +
    geom_errorbar(aes(x = x_bar, ymin = M_obs - SE_obs, ymax = M_obs + SE_obs),
                  width = 0.125, linewidth = 0.75, color = "grey25") +
    
    # 2. Overlay Layer: Simulated Diamonds & Error Bars (Shape mapping omitted to preserve clean legend)
    geom_errorbar(aes(x = x_sim, ymin = M_sim - SE_sim, ymax = M_sim + SE_sim),
                  width = 0.05, linewidth = 0.75, color = "black") + 
    geom_point(aes(x = x_sim, y = M_sim, fill = judgment_condition),
               shape = 23, size = 4.5, color = "black", stroke = 1.2) +
    
    # Aesthetics & Legends
    scale_fill_manual(values = c("categorical" = color_cat, "continuous" = color_cont),
                      labels = c("categorical" = "Categorical", "continuous" = "Continuous")) +
    scale_x_continuous(breaks = c(1, 2), labels = c("Between-Category", "Within-Category"), limits = c(0.4, 2.6)) +
    scale_y_continuous(breaks = y_breaks, labels = if (!is.null(label_fn)) label_fn else waiver(), expand = c(0, 0)) +
    coord_cartesian(ylim = y_limits) +
    labs(x = "Pair Type", y = y_label) + 
    theme_stats() +
    guides(fill = guide_legend(override.aes = list(shape = NA))) 
}

#' Create Overlay Categorical Perception (CP) Plot
#'
#' Generates a detailed interaction plot displaying individual empirical trajectories, 
#' boxplots, and aggregated simulated CP effects.
#'
#' @param obs_data Data frame containing observed empirical data.
#' @param sim_data Data frame containing simulated PPC data.
#' @param dv_obs Character string specifying the dependent variable in obs_data.
#' @param dv_sim Character string specifying the dependent variable in sim_data.
#' @param cp_y_lab Character string for the Y-axis label.
#' @param cp_y_lims Numeric vector specifying Y-axis bounds.
#' @param cp_y_brks Numeric vector specifying Y-axis tick intervals.
#' @return A ggplot2 object.
create_overlay_cp_plot <- function(obs_data, sim_data, dv_obs, dv_sim, cp_y_lab, cp_y_lims, cp_y_brks) {
  
  obs_cp <- obs_data %>%
    select(subj_idx, judgment_condition, pair_type, all_of(dv_obs)) %>%
    pivot_wider(names_from = pair_type, values_from = all_of(dv_obs)) %>%
    mutate(effect = between - within,
           x_num = as.numeric(judgment_condition))
  
  sim_cp_agg <- sim_data %>%
    select(subj_idx, judgment_condition, pair_type, all_of(dv_sim)) %>%
    pivot_wider(names_from = pair_type, values_from = all_of(dv_sim)) %>%
    mutate(effect = between - within) %>%
    group_by(judgment_condition) %>%
    summarise(M_sim_effect = mean(effect, na.rm = TRUE), 
              SE_sim_effect = sd(effect, na.rm = TRUE) / sqrt(n()),
              .groups = "drop") %>%
    mutate(x_num = as.numeric(judgment_condition),
           x_sim = x_num + 0.05)
  
  ggplot() +
    # 1. Base Layer: Observed Trajectories, Jitter, & Boxplots
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray60", linewidth = 0.8) +
    geom_line(data = obs_cp, aes(x = x_num, y = effect, group = subj_idx), color = "gray70", alpha = 0.5, linewidth = 0.5) +
    geom_jitter(data = obs_cp, aes(x = x_num, y = effect, color = judgment_condition), width = 0.05, alpha = 0.5, size = 2) +
    geom_boxplot(data = obs_cp, aes(x = x_num, y = effect, fill = judgment_condition, group = x_num), width = 0.2, alpha = 1, outlier.shape = NA, color = "grey25") +
    
    # Empirical Group Means
    stat_summary(data = obs_cp, aes(x = x_num, y = effect), fun = mean, geom = "point", shape = 21, size = 4, fill = "white", color = "black", stroke = 1.2) +
    
    # 2. Overlay Layer: Simulated Diamonds & Standard Errors
    geom_errorbar(data = sim_cp_agg, aes(x = x_sim, ymin = M_sim_effect - SE_sim_effect, ymax = M_sim_effect + SE_sim_effect), width = 0.05, linewidth = 0.75, color = "black") +
    geom_point(data = sim_cp_agg, aes(x = x_sim, y = M_sim_effect, fill = judgment_condition), shape = 23, size = 4.5, color = "black", stroke = 1.2) +
    
    # Aesthetics & Scales
    scale_fill_manual(
      values = c("categorical" = color_cat, "continuous" = color_cont),
      labels = c("categorical" = "Categorical", "continuous" = "Continuous") 
    ) +
    scale_color_manual(values = c("categorical" = color_cat, "continuous" = color_cont), guide = "none") +
    scale_x_continuous(breaks = c(1, 2), labels = c("Categorical", "Continuous"), limits = c(0.5, 2.5)) +
    scale_y_continuous(breaks = cp_y_brks, labels = function(x) format(x, scientific = FALSE, drop0trailing = TRUE)) +
    coord_cartesian(ylim = cp_y_lims) +
    labs(x = "Judgment Condition", y = cp_y_lab) +
    
    theme_stats() +
    
    # Custom overlay theme parameters for enhanced legibility
    theme(
      axis.title.y = element_text(size = 32, face = "bold", margin = margin(r = 15)), 
      axis.title.x = element_text(size = 32, face = "bold", margin = margin(t = 20)), 
      axis.text = element_text(size = 28, color = "black", margin = margin(t = 15)),  
      legend.text = element_text(size = 24),
      legend.key.size = unit(1.2, "cm")  # Force enlargement of legend keys
    ) +
    
    # Purge extraneous visual artifacts from the legend
    guides(fill = guide_legend(override.aes = list(shape = NA, linetype = 0, color = NA, alpha = 1))) 
}

# --- 4. Load & Prepare Empirical Data (Observed) ---
cat("\n[INFO] Phase 1/3: Loading Empirical Data...\n")
raw_data <- read_csv("data/xab_main_trial_level_combined.csv", show_col_types = FALSE)

clean_data <- raw_data %>% filter(xab_rt <= 3.0)

obs_subject_data <- clean_data %>%
  group_by(subj_idx, judgment_condition, pair_type) %>%
  summarise(
    accuracy = mean(success, na.rm = TRUE),
    mean_rt  = mean(xab_rt[success == 1], na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  mutate(
    subj_idx = as.factor(subj_idx),
    judgment_condition = factor(judgment_condition, levels = c("categorical", "continuous")),
    pair_type = factor(pair_type, levels = c("between", "within"))
  )


# --- 5. Process Models (Overlay Plotting) ---
for (model_name in models_to_run) {
  cat(sprintf("\n[INFO] Phase 2/3: Generating Overlay Plots for Model: %s\n", model_name))
  
  sim_csv_path <- file.path(projectDir, "reference_outputs", "hddm", "result", "formal_model", model_name, sim_csv_filename)
  
  if (!file.exists(sim_csv_path)) {
    cat(sprintf("[ERROR] Simulation data not found: %s. Skipping...\n", sim_csv_path))
    next
  }
  
  sim_raw <- read_csv(sim_csv_path, show_col_types = FALSE)
  
  sim_subject_data <- sim_raw %>%
    group_by(subj_idx, judgment_condition, pair_type) %>%
    summarise(
      sim_accuracy = mean(Simulated_Acc_Mean, na.rm = TRUE),
      sim_mean_rt  = mean(Simulated_RT_Mean_Correct, na.rm = TRUE),
      .groups      = "drop"
    ) %>%
    mutate(
      subj_idx = as.factor(subj_idx),
      judgment_condition = factor(judgment_condition, levels = c("categorical", "continuous")),
      pair_type = factor(pair_type, levels = c("between", "within"))
    )
  
  # --- Visualization Loop ---
  for (dv in c("accuracy", "mean_rt")) {
    
    dv_obs <- dv
    dv_sim <- if(dv == "accuracy") "sim_accuracy" else "sim_mean_rt"
    
    dv_label  <- if (dv == "accuracy") "acc" else "rt"
    y_lab     <- if (dv == "accuracy") "Mean Accuracy" else "Mean Correct RT (s)"
    y_lims    <- if (dv == "accuracy") c(0.5, 0.9) else c(0.8, 2.2)
    y_brks    <- if (dv == "accuracy") c(0.5, 0.6, 0.7, 0.8, 0.9) else seq(0.8, 2.2, 0.2)
    lbl_fn    <- if (dv == "accuracy") percent_format(accuracy = 1) else NULL
    
    cp_y_lab  <- if (dv == "accuracy") "CP Effect (Acc)" else "RT Difference (s)"
    cp_y_lims <- if (dv == "accuracy") c(-0.2, 0.2) else c(-0.5, 0.5)
    cp_y_brks <- if (dv == "accuracy") round(seq(-0.3, 0.3, by = 0.1), 2) else round(seq(-0.6, 0.6, by = 0.1), 2)
    
    plot_bar <- create_overlay_barplot(obs_subject_data, sim_subject_data, dv_obs, dv_sim, y_lab, y_lims, y_brks, lbl_fn)
    plot_cp  <- create_overlay_cp_plot(obs_subject_data, sim_subject_data, dv_obs, dv_sim, cp_y_lab, cp_y_lims, cp_y_brks)
    
    bar_file <- file.path(plotsDir, paste0("Overlay_", model_name, "_", dv_label, "_barplot.jpg"))
    cp_file  <- file.path(plotsDir, paste0("Overlay_", model_name, "_", dv_label, "_cp_effect.jpg"))
    
    ggsave(bar_file, plot_bar, width = 9, height = 7, dpi = 300)
    ggsave(cp_file, plot_cp, width = 12, height = 8.5, dpi = 300)
  }
}

# --- 6. Run Empirical ANOVA (Only on Observed Data) ---
cat("\n[INFO] Phase 3/3: Executing Formal Empirical ANOVA...\n")

for (dv in c("accuracy", "mean_rt")) {
  dv_label <- if (dv == "accuracy") "acc" else "rt"
  doc_file_anova <- file.path(statsDir, sprintf("Empirical_ANOVA_%s.doc", dv_label))
  
  bruceR_anova <- NULL
  tryCatch({
    bruceR_anova <- MANOVA(
      data = obs_subject_data,
      subID = "subj_idx",
      dv = dv,
      within = "pair_type",
      between = "judgment_condition",
      file = doc_file_anova
    )
  }, error = function(e) { cat("[ERROR] ANOVA Computation Failed:\n"); print(e) })
  
  if (!is.null(bruceR_anova)) {
    doc_file_posthoc_A <- file.path(statsDir, sprintf("Empirical_PostHoc_Pair_within_Train_%s.doc", dv_label))
    doc_file_posthoc_B <- file.path(statsDir, sprintf("Empirical_PostHoc_Train_within_Pair_%s.doc", dv_label))
    
    tryCatch({
      EMMEANS(model = bruceR_anova, effect = "pair_type", by = "judgment_condition", file = doc_file_posthoc_A)
      EMMEANS(model = bruceR_anova, effect = "judgment_condition", by = "pair_type", file = doc_file_posthoc_B)
    }, error = function(e) { cat("[ERROR] Post-Hoc Analysis Failed:\n"); print(e) })
  }
}

cat("\n[SUCCESS] Pipeline Complete! Final publication-ready plots and statistics generated.\n")
