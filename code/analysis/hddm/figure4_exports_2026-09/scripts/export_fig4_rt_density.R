# ==============================================================================
# Regenerate manuscript-ready HDDM PPC reaction-time density figure
#
# This script only reads existing posterior predictive output. It does not refit
# HDDM models or regenerate posterior predictive samples.
# ==============================================================================

rm(list = ls())

library(ggplot2)
library(dplyr)
library(readr)
library(grid)

find_project_dir <- function() {
  candidates <- unique(c(
    normalizePath(getwd(), winslash = "/", mustWork = FALSE),
    normalizePath(file.path(getwd(), ".."), winslash = "/", mustWork = FALSE),
    normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = FALSE),

  ))

  for (candidate in candidates) {
    target <- file.path(
      candidate,
      "scripts", "HDDM", "output", "result", "formal_model",
      "M4_v_judgment_x_pair", "Master_PPC_Simulations_Single.csv"
    )
    if (file.exists(target)) {
      return(normalizePath(candidate, winslash = "/", mustWork = TRUE))
    }
  }

  stop("Could not locate the project root. Please set project_dir manually.")
}

project_dir <- find_project_dir()
hddm_dir <- project_dir
model_dir <- file.path(project_dir, "reference_outputs", "hddm", "tables")
out_dir <- file.path(project_dir, "results", "figure4")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ppc_csv <- file.path(model_dir, "Master_PPC_Simulations_Single.csv")
if (!file.exists(ppc_csv)) {
  stop("Missing posterior predictive data: ", ppc_csv)
}

save_plot_png_pdf <- function(filename_base, plot, width, height, dpi = 600) {
  png_path <- file.path(out_dir, paste0(filename_base, ".png"))
  pdf_path <- file.path(out_dir, paste0(filename_base, ".pdf"))

  ggsave(png_path, plot, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(pdf_path, plot, width = width, height = height, device = grDevices::cairo_pdf, bg = "white")

  message("[saved] ", png_path)
  message("[saved] ", pdf_path)
}

ppc_data <- read_csv(ppc_csv, show_col_types = FALSE) %>%
  mutate(
    observed_rt = abs(Observed_RT),
    predicted_rt = abs(Simulated_RT_Mean_Correct),
    judgment_condition = factor(judgment_condition, levels = c("categorical", "continuous")),
    pair_type = factor(pair_type, levels = c("within", "between")),
    condition = factor(
      paste(judgment_condition, pair_type, sep = "_"),
      levels = c("categorical_within", "categorical_between", "continuous_between", "continuous_within"),
      labels = c("CP-Present\nwithin", "CP-Present\nbetween", "Continuous\nbetween", "Continuous\nwithin")
    )
  ) %>%
  filter(
    Observed_Acc == 1,
    is.finite(observed_rt),
    is.finite(predicted_rt),
    observed_rt <= 3.5,
    predicted_rt <= 3.5
  )

plot_rt_density <- ggplot(ppc_data, aes(x = observed_rt)) +
  geom_histogram(
    aes(y = after_stat(density), fill = "Observed data"),
    bins = 30,
    color = "white",
    linewidth = 0.25,
    alpha = 0.85
  ) +
  geom_density(
    aes(x = predicted_rt, color = "Predicted data"),
    linewidth = 1.0,
    adjust = 1.0,
    inherit.aes = FALSE,
    key_glyph = "path"
  ) +
  facet_wrap(~ condition, nrow = 2) +
  scale_fill_manual(
    values = c("Observed data" = "#9AAED6"),
    name = NULL
  ) +
  scale_color_manual(
    values = c("Predicted data" = "#FF7F50"),
    name = NULL
  ) +
  scale_x_continuous(
    breaks = seq(0, 3, 1),
    limits = c(0, 3.5),
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_y_continuous(
    breaks = seq(0, 1.0, 0.5),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(x = "Reaction time (s)", y = "Density") +
  theme_classic(base_size = 14) +
  theme(
    text = element_text(family = "sans", color = "black"),
    axis.line = element_line(color = "black", linewidth = 0.8),
    axis.ticks = element_line(color = "black", linewidth = 0.8),
    axis.title = element_text(size = 16, face = "bold", color = "black"),
    axis.title.x = element_text(margin = margin(t = 18)),
    axis.title.y = element_text(margin = margin(r = 14)),
    axis.text = element_text(size = 11, color = "black"),
    strip.background = element_blank(),
    strip.text = element_text(size = 9.5, face = "bold", color = "black", hjust = 1),
    legend.position = "top",
    legend.justification = "left",
    legend.box = "horizontal",
    legend.text = element_text(size = 11, face = "bold", color = "black"),
    legend.key.width = unit(1.0, "cm"),
    legend.spacing.x = unit(0.5, "cm"),
    panel.spacing = unit(0.35, "lines"),
    plot.margin = margin(t = 6, r = 8, b = 6, l = 8),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  guides(
    fill = guide_legend(order = 1, override.aes = list(color = NA)),
    color = guide_legend(order = 2, override.aes = list(linewidth = 1.0, fill = NA))
  )

save_plot_png_pdf("Fig4d_ppc_reaction_time_density", plot_rt_density, width = 5.2, height = 4.25)

message("Done. RT density PPC figure is in: ", out_dir)
