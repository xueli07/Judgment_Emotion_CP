# ==============================================================================
# Regenerate manuscript-ready HDDM figures for manuscript page 8
#
# This script only reads existing HDDM result files and empirical data. It does
# not refit any HDDM model or rerun posterior predictive simulations.
# ==============================================================================

rm(list = ls())

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(grid)

set.seed(42)

find_project_dir <- function() {
  candidates <- unique(c(
    normalizePath(getwd(), winslash = "/", mustWork = FALSE),
    normalizePath(file.path(getwd(), ".."), winslash = "/", mustWork = FALSE),
    normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = FALSE),

  ))

  for (candidate in candidates) {
    hddm_data <- file.path(candidate, "data/processed", "hddm", "xab_trial_level_exp01_exp02.csv")
    if (file.exists(hddm_data)) {
      return(normalizePath(candidate, winslash = "/", mustWork = TRUE))
    }
  }

  stop("Could not locate the project root. Please set project_dir manually.")
}

project_dir <- find_project_dir()
hddm_dir <- project_dir
model_name <- "M4_v_judgment_x_pair"
model_dir <- file.path(project_dir, "reference_outputs", "hddm", "tables")
out_dir <- file.path(project_dir, "results", "figure4")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

save_plot_png_pdf <- function(filename_base, plot, width, height, dpi = 600) {
  png_path <- file.path(out_dir, paste0(filename_base, ".png"))
  pdf_path <- file.path(out_dir, paste0(filename_base, ".pdf"))

  ggsave(png_path, plot, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(pdf_path, plot, width = width, height = height, device = grDevices::cairo_pdf, bg = "white")

  message("[saved] ", png_path)
  message("[saved] ", pdf_path)
}

color_cat <- "#F28E8E"
color_cont <- "#7FB3FF"

theme_nc <- function(base_size = 18) {
  theme_classic(base_size = base_size) +
    theme(
      text = element_text(family = "sans", color = "black"),
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", linewidth = 1.2),
      axis.ticks = element_line(color = "black", linewidth = 1.2),
      axis.title = element_text(size = 20, face = "bold", color = "black"),
      axis.title.y = element_text(size = 24, face = "bold", color = "black", margin = margin(r = 15)),
      axis.title.x = element_text(size = 24, face = "bold", color = "black", margin = margin(t = 16)),
      axis.text = element_text(size = 22, color = "black", margin = margin(t = 12)),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(size = 18, color = "black"),
      legend.key = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )
}

# ==============================================================================
# Figure 8a-b: schematic diffusion paths
# ==============================================================================

non_decision_time <- 0.2
n_trials <- 5
noise_std <- 0.12
dt <- 0.01

simulate_path <- function(v, a, z, max_time) {
  time_steps <- floor(max_time / dt)
  path <- numeric(time_steps + 1)
  path[1] <- z

  for (i in seq_len(time_steps)) {
    step <- v * dt + rnorm(1, mean = 0, sd = noise_std)
    new_val <- path[i] + step

    if (new_val >= a) {
      path[i + 1] <- a
      path <- path[seq_len(i + 1)]
      break
    }

    if (new_val <= 0) {
      path[i + 1] <- 0
      path <- path[seq_len(i + 1)]
      break
    }

    path[i + 1] <- new_val
  }

  path
}

schematic_levels <- c(
  "Categorical, between-category",
  "Continuous, between-category",
  "Categorical, within-category",
  "Continuous, within-category"
)

schematic_colors <- c(
  "Categorical, between-category" = color_cat,
  "Continuous, between-category" = color_cont,
  "Categorical, within-category" = color_cat,
  "Continuous, within-category" = color_cont
)

schematic_linetypes <- c(
  "Categorical, between-category" = "solid",
  "Continuous, between-category" = "solid",
  "Categorical, within-category" = "dashed",
  "Continuous, within-category" = "dashed"
)

theme_schematic <- theme_classic(base_size = 20) +
  theme(
    text = element_text(family = "sans", color = "black"),
    plot.title = element_blank(),
    axis.title = element_text(face = "bold", size = 20, color = "black"),
    axis.line = element_line(linewidth = 1.2, color = "black"),
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_text(size = 18, color = "black"),
    legend.position = c(0.95, 0.05),
    legend.justification = c(1, 0),
    legend.title = element_blank(),
    legend.text = element_text(size = 15, color = "black"),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key.width = unit(3, "line"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  )

make_schematic_v <- function() {
  a_fixed <- 2.0
  z_fixed <- 1.0
  max_time <- 2.5
  offset_y <- 0.02

  conditions <- list(
    list(v = 1.40, lty = "solid", y_off = offset_y * 1.5, label = "Categorical, between-category"),
    list(v = 1.25, lty = "solid", y_off = offset_y * 0.5, label = "Continuous, between-category"),
    list(v = 1.15, lty = "dashed", y_off = -offset_y * 0.5, label = "Continuous, within-category"),
    list(v = 0.70, lty = "dashed", y_off = -offset_y * 1.5, label = "Categorical, within-category")
  )

  path_df <- data.frame()
  arrow_df <- data.frame()
  group_id <- 1

  for (cond in conditions) {
    for (i in seq_len(n_trials)) {
      path_value <- simulate_path(cond$v, a_fixed, z_fixed, max_time)
      time_value <- (seq_along(path_value) - 1) * dt + non_decision_time
      path_df <- bind_rows(path_df, data.frame(
        time = time_value,
        value = path_value + cond$y_off,
        condition = factor(cond$label, levels = schematic_levels),
        group = group_id,
        alpha = ifelse(cond$lty == "dashed", 0.15, 0.10)
      ))
      group_id <- group_id + 1
    }

    t_end <- (a_fixed - z_fixed) / cond$v + non_decision_time
    arrow_df <- bind_rows(arrow_df, data.frame(
      x = non_decision_time,
      y = z_fixed,
      xend = t_end,
      yend = a_fixed,
      condition = factor(cond$label, levels = schematic_levels)
    ))
  }

  ggplot() +
    geom_hline(yintercept = a_fixed, linewidth = 1.2, color = "#333333") +
    geom_hline(yintercept = z_fixed, linewidth = 1, color = "lightgray") +
    geom_path(
      data = path_df,
      aes(x = time, y = value, group = group, color = condition, linetype = condition, alpha = alpha),
      linewidth = 0.4
    ) +
    geom_segment(
      data = arrow_df,
      aes(x = x, y = y, xend = xend, yend = yend, color = condition, linetype = condition),
      arrow = arrow(length = unit(0.35, "cm"), type = "closed"),
      linewidth = 1.2,
      alpha = 0.9
    ) +
    scale_color_manual(values = schematic_colors, labels = function(x) sub("Categorical", "CP-Present", x, fixed = TRUE)) +
    scale_linetype_manual(values = schematic_linetypes, labels = function(x) sub("Categorical", "CP-Present", x, fixed = TRUE)) +
    scale_alpha_identity() +
    scale_y_continuous(
      breaks = c(0, z_fixed, a_fixed),
      labels = expression("0", italic(z), italic(a)),
      limits = c(0, a_fixed + 0.2)
    ) +
    coord_cartesian(xlim = c(0, max_time), expand = FALSE) +
    labs(x = "Reaction time", y = "Decision variable") +
    theme_schematic
}

make_schematic_a <- function() {
  v_fixed <- 1.0
  z_fixed <- 1.0
  max_time <- 2.5

  a_cat_between <- 2.00
  a_cont_between <- 2.06
  a_cont_within <- 2.12
  a_cat_within <- 2.35

  conditions <- list(
    list(a = a_cat_within, label = "Categorical, within-category"),
    list(a = a_cont_within, label = "Continuous, within-category"),
    list(a = a_cont_between, label = "Continuous, between-category"),
    list(a = a_cat_between, label = "Categorical, between-category")
  )

  path_df <- data.frame()
  arrow_df <- data.frame()
  group_id <- 1

  for (cond in conditions) {
    for (i in seq_len(n_trials)) {
      path_value <- simulate_path(v_fixed, cond$a, z_fixed, max_time)
      time_value <- (seq_along(path_value) - 1) * dt + non_decision_time
      path_df <- bind_rows(path_df, data.frame(
        time = time_value,
        value = path_value,
        condition = factor(cond$label, levels = schematic_levels),
        group = group_id,
        alpha = ifelse(grepl("within", cond$label), 0.15, 0.10)
      ))
      group_id <- group_id + 1
    }

    t_end <- (cond$a - z_fixed) / v_fixed + non_decision_time
    arrow_df <- bind_rows(arrow_df, data.frame(
      x = non_decision_time,
      y = z_fixed,
      xend = t_end,
      yend = cond$a,
      condition = factor(cond$label, levels = schematic_levels)
    ))
  }

  boundary_df <- data.frame(
    yint = c(a_cat_between, a_cont_between, a_cat_within, a_cont_within),
    condition = factor(c(
      "Categorical, between-category",
      "Continuous, between-category",
      "Categorical, within-category",
      "Continuous, within-category"
    ), levels = schematic_levels)
  )

  ggplot() +
    geom_hline(yintercept = 0, linewidth = 1, color = "#333333") +
    geom_hline(yintercept = z_fixed, linewidth = 1, color = "lightgray") +
    geom_hline(
      data = boundary_df,
      aes(yintercept = yint, color = condition, linetype = condition),
      linewidth = 2,
      alpha = 0.8
    ) +
    geom_path(
      data = path_df,
      aes(x = time, y = value, group = group, color = condition, linetype = condition, alpha = alpha),
      linewidth = 0.4,
      show.legend = FALSE
    ) +
    geom_segment(
      data = arrow_df,
      aes(x = x, y = y, xend = xend, yend = yend),
      color = "#222222",
      arrow = arrow(length = unit(0.35, "cm"), type = "closed"),
      linewidth = 1.2,
      alpha = 0.9,
      show.legend = FALSE
    ) +
    scale_color_manual(values = schematic_colors, labels = function(x) sub("Categorical", "CP-Present", x, fixed = TRUE)) +
    scale_linetype_manual(values = schematic_linetypes, labels = function(x) sub("Categorical", "CP-Present", x, fixed = TRUE)) +
    scale_alpha_identity() +
    scale_y_continuous(
      breaks = c(0, z_fixed, a_cat_between, a_cont_between, a_cont_within, a_cat_within),
      labels = c("0", "z", "", "", "", ""),
      limits = c(0, a_cat_within + 0.2)
    ) +
    coord_cartesian(xlim = c(0, max_time), expand = FALSE) +
    labs(x = "Reaction time", y = "Decision variable") +
    theme_schematic
}

plot_8a <- make_schematic_v()
plot_8b <- make_schematic_a()
save_plot_png_pdf("Fig4a_schematic_interaction_v", plot_8a, width = 8, height = 6)
save_plot_png_pdf("Fig4b_schematic_interaction_a", plot_8b, width = 8, height = 6)

# ==============================================================================
# Figure 8c-d: posterior parameter summaries
# ==============================================================================

param_csv <- file.path(model_dir, paste0(model_name, "_v_plot.csv"))
if (!file.exists(param_csv)) {
  stop("Missing posterior parameter summary file: ", param_csv)
}

param_data <- read_csv(param_csv, show_col_types = FALSE) %>%
  mutate(
    judgment_condition = factor(judgment_condition, levels = c("categorical", "continuous")),
    pair_type = factor(pair_type, levels = c("between", "within"))
  )

plot_posterior_param <- function(data, param_prefix, y_label, y_limits, y_breaks) {
  mean_col <- paste0(param_prefix, "_mean")
  err_col <- paste0(param_prefix, "_std")

  ggplot(data, aes(x = pair_type, y = .data[[mean_col]], fill = judgment_condition)) +
    geom_col(position = position_dodge(0.9), alpha = 1, width = 0.7) +
    geom_errorbar(
      aes(ymin = .data[[mean_col]] - .data[[err_col]], ymax = .data[[mean_col]] + .data[[err_col]]),
      position = position_dodge(0.9),
      width = 0.25,
      linewidth = 0.75,
      color = "grey25"
    ) +
    scale_fill_manual(
      values = c("categorical" = color_cat, "continuous" = color_cont),
      labels = c("categorical" = "CP-Present", "continuous" = "Continuous")
    ) +
    scale_x_discrete(labels = c("between" = "Between-category", "within" = "Within-category")) +
    scale_y_continuous(breaks = y_breaks, expand = c(0, 0)) +
    coord_cartesian(ylim = y_limits) +
    labs(x = "Pair type", y = y_label) +
    theme_nc()
}

plot_8c <- plot_posterior_param(
  data = param_data,
  param_prefix = "v",
  y_label = "Drift rate (v)",
  y_limits = c(0.4, 0.8),
  y_breaks = seq(0.4, 0.8, 0.1)
)

plot_8d <- plot_posterior_param(
  data = param_data,
  param_prefix = "a",
  y_label = "Decision threshold (a)",
  y_limits = c(1.5, 2.0),
  y_breaks = seq(1.5, 2.0, 0.1)
)

# Preserve significance annotations shown in the supplied manuscript Fig. 4c.
# These are display annotations, not newly calculated statistical results.
plot_8c <- plot_8c +
  annotate("segment", x = c(0.725, 1.725, 1), xend = c(1.175, 2.175, 2),
           y = c(0.745, 0.748, 0.773), yend = c(0.745, 0.748, 0.773), linewidth = 0.7) +
  annotate("text", x = c(0.95, 1.95, 1.5), y = c(0.759, 0.762, 0.786),
           label = c("ns", "***", "***"), size = 7, fontface = "bold")
save_plot_png_pdf("Fig4c_posterior_drift_rate", plot_8c, width = 8, height = 7)


# ==============================================================================
# Figure 8e-f: empirical vs. HDDM posterior predictive checks for accuracy
# ==============================================================================

raw_data_path <- file.path(project_dir, "data/processed", "hddm", "xab_trial_level_exp01_exp02.csv")
sim_data_path <- file.path(model_dir, "Master_PPC_Simulations_Aggregated.csv")

if (!file.exists(raw_data_path)) {
  stop("Missing empirical trial-level data: ", raw_data_path)
}

if (!file.exists(sim_data_path)) {
  stop("Missing posterior predictive simulation data: ", sim_data_path)
}

obs_subject_data <- read_csv(raw_data_path, show_col_types = FALSE) %>%
  filter(xab_rt <= 3.0) %>%
  group_by(subj_idx, judgment_condition, pair_type) %>%
  summarise(
    accuracy = mean(success, na.rm = TRUE),
    mean_rt = mean(xab_rt[success == 1], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    subj_idx = factor(subj_idx),
    judgment_condition = factor(judgment_condition, levels = c("categorical", "continuous")),
    pair_type = factor(pair_type, levels = c("between", "within"))
  )

sim_subject_data <- read_csv(sim_data_path, show_col_types = FALSE) %>%
  group_by(subj_idx, judgment_condition, pair_type) %>%
  summarise(
    sim_accuracy = mean(Simulated_Acc_Mean, na.rm = TRUE),
    sim_mean_rt = mean(Simulated_RT_Mean_Correct, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    subj_idx = factor(subj_idx),
    judgment_condition = factor(judgment_condition, levels = c("categorical", "continuous")),
    pair_type = factor(pair_type, levels = c("between", "within"))
  )

create_overlay_barplot <- function(obs_data, sim_data) {
  obs_agg <- obs_data %>%
    group_by(judgment_condition, pair_type) %>%
    summarise(
      m_obs = mean(accuracy, na.rm = TRUE),
      se_obs = sd(accuracy, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  sim_agg <- sim_data %>%
    group_by(judgment_condition, pair_type) %>%
    summarise(
      m_sim = mean(sim_accuracy, na.rm = TRUE),
      se_sim = sd(sim_accuracy, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  plot_data <- left_join(obs_agg, sim_agg, by = c("judgment_condition", "pair_type")) %>%
    mutate(
      x_num = as.numeric(pair_type),
      x_bar = ifelse(judgment_condition == "categorical", x_num - 0.225, x_num + 0.225),
      x_sim = x_bar + 0.14
    )

  ggplot(plot_data) +
    geom_col(aes(x = x_bar, y = m_obs, fill = judgment_condition), width = 0.35, alpha = 1) +
    geom_errorbar(
      aes(x = x_bar, ymin = m_obs - se_obs, ymax = m_obs + se_obs),
      width = 0.125,
      linewidth = 0.75,
      color = "grey25"
    ) +
    geom_errorbar(
      aes(x = x_sim, ymin = m_sim - se_sim, ymax = m_sim + se_sim),
      width = 0.05,
      linewidth = 0.75,
      color = "black"
    ) +
    geom_point(
      aes(x = x_sim, y = m_sim, fill = judgment_condition),
      shape = 23,
      size = 4.5,
      color = "black",
      stroke = 1.2
    ) +
    scale_fill_manual(
      values = c("categorical" = color_cat, "continuous" = color_cont),
      labels = c("categorical" = "CP-Present", "continuous" = "Continuous")
    ) +
    scale_x_continuous(
      breaks = c(1, 2),
      labels = c("Between-category", "Within-category"),
      limits = c(0.4, 2.6)
    ) +
    scale_y_continuous(
      breaks = c(0.5, 0.6, 0.7, 0.8, 0.9),
      labels = percent_format(accuracy = 1),
      expand = c(0, 0)
    ) +
    coord_cartesian(ylim = c(0.5, 0.9)) +
    labs(x = "Pair type", y = "Mean accuracy") +
    theme_nc() +
    guides(fill = guide_legend(override.aes = list(shape = NA)))
}

create_overlay_cp_plot <- function(obs_data, sim_data) {
  obs_cp <- obs_data %>%
    select(subj_idx, judgment_condition, pair_type, accuracy) %>%
    pivot_wider(names_from = pair_type, values_from = accuracy) %>%
    mutate(
      effect = between - within,
      x_num = as.numeric(judgment_condition)
    )

  sim_cp_agg <- sim_data %>%
    select(subj_idx, judgment_condition, pair_type, sim_accuracy) %>%
    pivot_wider(names_from = pair_type, values_from = sim_accuracy) %>%
    mutate(effect = between - within) %>%
    group_by(judgment_condition) %>%
    summarise(
      m_sim_effect = mean(effect, na.rm = TRUE),
      se_sim_effect = sd(effect, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      x_num = as.numeric(judgment_condition),
      x_sim = x_num + 0.05
    )

  ggplot() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray60", linewidth = 0.8) +
    geom_line(data = obs_cp, aes(x = x_num, y = effect, group = subj_idx), color = "gray70", alpha = 0.5, linewidth = 0.5) +
    geom_jitter(data = obs_cp, aes(x = x_num, y = effect, color = judgment_condition), width = 0.05, alpha = 0.5, size = 2) +
    geom_boxplot(
      data = obs_cp,
      aes(x = x_num, y = effect, fill = judgment_condition, group = x_num),
      width = 0.2,
      alpha = 1,
      outlier.shape = NA,
      color = "grey25"
    ) +
    stat_summary(
      data = obs_cp,
      aes(x = x_num, y = effect),
      fun = mean,
      geom = "point",
      shape = 21,
      size = 4,
      fill = "white",
      color = "black",
      stroke = 1.2
    ) +
    geom_errorbar(
      data = sim_cp_agg,
      aes(x = x_sim, ymin = m_sim_effect - se_sim_effect, ymax = m_sim_effect + se_sim_effect),
      width = 0.05,
      linewidth = 0.75,
      color = "black"
    ) +
    geom_point(
      data = sim_cp_agg,
      aes(x = x_sim, y = m_sim_effect, fill = judgment_condition),
      shape = 23,
      size = 4.5,
      color = "black",
      stroke = 1.2
    ) +
    scale_fill_manual(
      values = c("categorical" = color_cat, "continuous" = color_cont),
      labels = c("categorical" = "CP-Present", "continuous" = "Continuous")
    ) +
    scale_color_manual(values = c("categorical" = color_cat, "continuous" = color_cont), guide = "none") +
    scale_x_continuous(breaks = c(1, 2), labels = c("CP-Present", "Continuous"), limits = c(0.5, 2.5)) +
    scale_y_continuous(breaks = round(seq(-0.3, 0.3, by = 0.1), 2)) +
    coord_cartesian(ylim = c(-0.2, 0.2)) +
    labs(x = "Judgment format", y = "CP effect (accuracy)") +
    theme_nc(base_size = 20) +
    theme(
      axis.title.y = element_text(size = 32, face = "bold", color = "black", margin = margin(r = 15)),
      axis.title.x = element_text(size = 32, face = "bold", color = "black", margin = margin(t = 20)),
      axis.text = element_text(size = 28, color = "black", margin = margin(t = 15)),
      legend.text = element_text(size = 24, color = "black"),
      legend.key.size = unit(1.2, "cm")
    ) +
    guides(fill = guide_legend(override.aes = list(shape = NA, linetype = 0, color = NA, alpha = 1)))
}

plot_8e <- create_overlay_barplot(obs_subject_data, sim_subject_data)
plot_8f <- create_overlay_cp_plot(obs_subject_data, sim_subject_data)

save_plot_png_pdf("Fig4e_ppc_accuracy_barplot", plot_8e, width = 9, height = 7)
save_plot_png_pdf("Fig4f_ppc_accuracy_cp_effect", plot_8f, width = 12, height = 8.5)

message("Done. Regenerated page 8 figures are in: ", out_dir)
