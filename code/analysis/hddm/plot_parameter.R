# ==============================================================================
# HDDM Posterior Parameters Visualization Pipeline
# Description: Generates publication-ready bar plots with Bayesian standard error
#              bars for HDDM posterior estimates (e.g., drift rate, decision threshold).
# ==============================================================================

rm(list = ls())

# Load required libraries
library(tidyverse)
library(ggplot2)
library(scales)

# --- 1. Directory Configuration & Data Loading ---
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
package_root <- normalizePath(file.path(script_dir, "..", ".."))
base_dir <- file.path(package_root, "reference_outputs", "hddm", "result", "formal_model")
setwd(base_dir)

# Create a dedicated directory for parameter plots
plots_dir <- file.path("M4_v_judgment_x_pair", "plots_parameters")
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

# Load posterior parameter data exported from the Python pipeline
csv_file <- file.path("M4_v_judgment_x_pair", "M4_v_judgment_x_pair_v_plot.csv")
param_data <- read_csv(csv_file, show_col_types = FALSE)

# Enforce factor levels to ensure correct plotting order on the x-axis and legend
param_data$judgment_condition <- factor(param_data$judgment_condition, levels = c("categorical", "continuous"))
param_data$pair_type <- factor(param_data$pair_type, levels = c("between", "within"))

# --- 2. Visualization Aesthetics Configuration ---
color_cat  <- "#F28E8E"   # Categorical group
color_cont <- "#7FB3FF"   # Continuous group

#' Custom Publication-Ready Theme for Statistical Plots
#'
#' @param base_size Base font size for the plot.
#' @return A ggplot2 theme object.
theme_stats <- function(base_size = 12) {
  theme_minimal(base_size = 18) +
    theme(
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", linewidth = 1.2),
      axis.ticks = element_line(color = "black", linewidth = 1.2),
      axis.title = element_text(size = 20, face = "bold"),
      axis.title.y = element_text(size = 24, face = "bold", margin = margin(r = 15)),
      axis.title.x = element_text(size = 26, face = "bold", margin = margin(t = 20)),
      axis.text = element_text(size = 22, color = "black", margin = margin(t = 15)),
      legend.position = "top",
      legend.margin = margin(b = -15),
      legend.title = element_blank(),
      legend.text = element_text(size = 18)
    )
}

# --- 3. Core Plotting Function ---

#' Plot Posterior Parameter Distributions
#'
#' Generates a dodged bar plot with error bars representing the posterior mean
#' and Bayesian standard error (standard deviation of the posterior samples).
#'
#' @param data A data frame containing the parameter estimates.
#' @param param_prefix Character string specifying the target parameter (e.g., "v", "a").
#' @param y_label Character string for the Y-axis label.
#' @param y_limits Numeric vector of length 2 specifying the Y-axis limits.
#' @param y_breaks Numeric vector specifying the Y-axis tick marks.
#' @return A ggplot2 object.
plot_posterior_param <- function(data, param_prefix, y_label, y_limits, y_breaks) {
  
  # Dynamically construct column names (e.g., v_mean, v_std)
  mean_col <- paste0(param_prefix, "_mean")
  err_col  <- paste0(param_prefix, "_std") # In Bayesian inference, posterior std represents SE
  
  ggplot(data, aes(x = pair_type, y = .data[[mean_col]], fill = judgment_condition)) +
    # Main bar geometry
    geom_bar(stat = "identity", position = position_dodge(0.9), alpha = 1, width = 0.7) +
    # Error bars mapping to Bayesian standard error
    geom_errorbar(aes(ymin = .data[[mean_col]] - .data[[err_col]], 
                      ymax = .data[[mean_col]] + .data[[err_col]]),
                  position = position_dodge(0.9), width = 0.25, linewidth = 0.75, color = "grey25") +
    # Color palette and legend mapping
    scale_fill_manual(
      values = c("categorical" = color_cat, "continuous" = color_cont),
      labels = c("categorical" = "Categorical", "continuous" = "Continuous")
    ) +
    # X-axis label mapping
    scale_x_discrete(labels = c("between" = "Between-Category", "within" = "Within-Category")) +
    # Y-axis scaling and limits
    scale_y_continuous(breaks = y_breaks, expand = c(0, 0)) +
    coord_cartesian(ylim = y_limits) +
    labs(x = "Pair Type", y = y_label) +
    theme_stats()
}

# --- 4. Execution and Export ---

cat("[INFO] Initiating posterior parameter distribution plotting...\n")

# A. Plot Drift Rate (v)
plot_v <- plot_posterior_param(
  data = param_data, 
  param_prefix = "v", 
  y_label = "Drift Rate (v)", 
  y_limits = c(0.4, 0.8), # Note: Adjust according to the empirical range of v
  y_breaks = seq(0.4, 0.8, 0.1)
)
v_save_path <- file.path(plots_dir, "Posterior_Param_v_DriftRate.jpg")
ggsave(v_save_path, plot_v, width = 8, height = 7, dpi = 300)
cat(sprintf("[SUCCESS] Saved: %s\n", basename(v_save_path)))

# B. Plot Decision Threshold (a)
plot_a <- plot_posterior_param(
  data = param_data, 
  param_prefix = "a", 
  y_label = "Decision Threshold (a)", 
  y_limits = c(1.5, 2.0), # Note: Adjust according to the empirical range of a
  y_breaks = seq(1.5, 2.0, 0.1)
)
a_save_path <- file.path(plots_dir, "Posterior_Param_a_Threshold.jpg")
ggsave(a_save_path, plot_a, width = 8, height = 7, dpi = 300)
cat(sprintf("[SUCCESS] Saved: %s\n", basename(a_save_path)))

cat("\n[SUCCESS] All parameter plotting tasks completed successfully.\n")
