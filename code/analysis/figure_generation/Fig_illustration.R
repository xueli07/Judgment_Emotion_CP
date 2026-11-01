# Figure 1 theoretical illustration code
# Run from the root of the reproducibility package.
output_dir <- file.path("results", "figure1")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

#install.packages("dplyr")
library(dplyr)
#install.packages("tidyverse")
library(tidyverse)
# Optional: ensure Arial is available on Windows
windowsFonts(Arial = windowsFont("Arial"))

set.seed(1)

# ====================================================
# Common settings
# ====================================================

cols <- c("Happy" = "#E64B35",
          "Sad"   = "#4DBBD5")

xlim <- c(-2.0, 2.0)
ylim <- c(-1.8, 1.6)

# ====================================================
# Panel B: Representation
# ====================================================

n_per_cluster <- 30

mu_left  <- c(-0.55,  0.10)
mu_right <- c( 0.55, -0.10)

sd_x <- 0.32
sd_y <- 0.38

# Boundary closer to vertical
a <- 2.5
b <- 0.0

rep_df <- bind_rows(
  tibble(
    x = rnorm(n_per_cluster, mu_left[1],  sd_x),
    y = rnorm(n_per_cluster, mu_left[2],  sd_y),
    group = "Happy"
  ),
  tibble(
    x = rnorm(n_per_cluster, mu_right[1], sd_x),
    y = rnorm(n_per_cluster, mu_right[2], sd_y),
    group = "Sad"
  )
)

boundary_df <- tibble(
  #x = xlim,
  # = a * xlim + b
  x = c(-0.45, 0.45),
  y = a * c(-0.45, 0.45) + b
)

pB <- ggplot(rep_df, aes(x, y, color = group)) +
  geom_line(
    data = boundary_df,
    aes(x, y),
    inherit.aes = FALSE,
    linetype = "dashed",
    linewidth = 1.1,
    alpha = 0.6,
    color = "grey30"
  ) +
  geom_point(size = 2.4, alpha = 0.95) +
  # annotate("text", x = -1.1, y = 1.35,
  #          label = "Happy-category\nrepresentation",
  #          size = 6.5, family = "Arial") +
  # annotate("text", x = 1.1, y = 1.35,
  #          label = "Sad-category\nrepresentation",
  #          size = 6.5, family = "Arial") +
  annotate("text", x = 0.05, y = -1.55,
           label = "Category Boundary",
           size = 6.2, family = "Arial") +
  scale_color_manual(values = cols) +
  coord_cartesian(xlim = xlim, ylim = ylim, expand = FALSE) +
  theme_void() +
  theme(
    legend.position = "none",
    text = element_text(family = "Arial"),
    plot.margin = margin(5, 5, 5, 5)
  )

pB

ggsave(file.path(output_dir, "Fig_illustration_representation.jpg"), pB, width = 4.2, height = 4)

# ====================================================
# Panel A: Emotion Continuum bar
# ====================================================

grad <- data.frame(x = seq(0, 1, length.out = 400), y = 1)

p_grad <- ggplot(grad, aes(x, y, fill = x)) +
  geom_tile(height = 1) +
  scale_fill_gradientn(
    colours = c(cols["Happy"], "#F3EFE6", cols["Sad"])
  ) +
  coord_cartesian(
    xlim = c(0, 1),
    ylim = c(0.6, 1.4),
    expand = FALSE
  ) +
  theme_void() +
  theme(
    legend.position = "none",
    text = element_text(family = "Arial")
  )

p_grad

ggsave(file.path(output_dir, "Fig_illustration_emotion_continuum_bar.jpg"), p_grad, width = 6, height = 0.7)

# ====================================================
# Panel C: Behavior
# ====================================================

pse <- 50
slope <- 7

x_min <- 0
x_max <- 100

x <- seq(x_min, x_max, length.out = 400)
p_sad <- 1 / (1 + exp(-(x - pse) / slope))
beh <- data.frame(x = x, p = p_sad)

dpdx <- 1 / (4 * slope)
dx <- 10

tangent_df <- data.frame(
  x = c(pse - dx, pse + dx),
  y = 0.5 + dpdx * c(-dx, dx)
)

pC <- ggplot(beh, aes(x, p)) +
  # [修改点 1] 替换为渐变线段
  geom_segment(aes(xend = dplyr::lead(x), yend = dplyr::lead(p), color = x),
               linewidth = 1.4, na.rm = TRUE) +
  # [修改点 2] 匹配你 Panel A 设定的三色渐变
  scale_color_gradientn(colours = c(cols["Happy"], "#F3EFE6", cols["Sad"])) +
  
  geom_hline(yintercept = 0.5, linetype = "dotted",
             linewidth = 0.8, color = "grey35") +
  geom_vline(xintercept = pse, linetype = "dashed",
             linewidth = 1.0, color = "grey30") +
  geom_line(
    data = tangent_df,
    aes(x, y),
    inherit.aes = FALSE,
    linewidth = 1.2,
    color = "black"
  ) +
 
  scale_x_continuous(
    limits = c(x_min, x_max),
    breaks = c(0, 20, 40, 60, 80, 100),
    labels = c("0%", "20%", "40%", "60%", "80%", "100%")
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = c(0, 0.5, 1)
  ) +
  labs(
    x = "Emotion Continuum (Happy \u2192 Sad)",
    y = 'P(judged as "Sad")'
  ) +
  theme_classic() +
  theme(
    legend.position = "none", # [修改点 3] 隐藏图例
    text = element_text(family = "Arial"),
    axis.title = element_text(size = 16),
    axis.line  = element_line(linewidth = 1.0),
    plot.margin = margin(6, 6, 6, 6),
    axis.text  = element_text(size = 14,color = "black")
  )

pC

ggsave(file.path(output_dir, "Fig_illustration_behavior.jpg"), pC, width = 4.2, height = 4)

# ====================================================
# Extra versions for Panel B and Panel C
#   1) strong category separation
#   2) weak category separation
# ====================================================

set.seed(123)

# ====================================================
# Part 1. Stronger category separation
# ====================================================

# ---------- Panel B strong ----------
n_per_cluster_strong <- 30

mu_left_strong  <- c(-0.95,  0.18)
mu_right_strong <- c( 0.95, -0.18)

sd_x_strong <- 0.24
sd_y_strong <- 0.30

a_strong <- 2.5
b_strong <- 0.0

rep_df_strong <- bind_rows(
  tibble(
    x = rnorm(n_per_cluster_strong, mu_left_strong[1],  sd_x_strong),
    y = rnorm(n_per_cluster_strong, mu_left_strong[2],  sd_y_strong),
    group = "Happy"
  ),
  tibble(
    x = rnorm(n_per_cluster_strong, mu_right_strong[1], sd_x_strong),
    y = rnorm(n_per_cluster_strong, mu_right_strong[2], sd_y_strong),
    group = "Sad"
  )
)

boundary_df_strong <- tibble(
  #x = xlim,
  #y = a_strong * xlim + b_strong
  x = c(-0.45, 0.45),
  y = a_strong * c(-0.45, 0.45) + b_strong
)

pB_strong <- ggplot(rep_df_strong, aes(x, y, color = group)) +
  geom_line(
    data = boundary_df_strong,
    aes(x, y),
    inherit.aes = FALSE,
    linetype = "dashed",
    linewidth = 1.1,
    alpha = 0.6,
    color = "grey30"
  ) +
  geom_point(size = 2.5, alpha = 0.95) +
  # annotate("text", x = -1, y = 1.35,
  #          label = "Happy-category\nrepresentation",
  #          size = 6.5, family = "Arial") +
  # annotate("text", x = 1, y = 1.35,
  #          label = "Sad-category\nrepresentation",
  #          size = 6.5, family = "Arial") +
  annotate("text", x = 0.07, y = -1.55,
           label = "Category Boundary",
           size = 6.2, family = "Arial") +
  scale_color_manual(values = cols) +
  coord_cartesian(xlim = xlim, ylim = ylim, expand = FALSE) +
  theme_void() +
  theme(
    legend.position = "none",
    text = element_text(family = "Arial"),
    plot.margin = margin(5, 5, 5, 5),
    axis.text = element_blank(),  # 去掉刻度标签
    axis.ticks = element_blank(), # 去掉刻度线
    axis.title = element_blank()  # 可选：去掉轴标题（如果有的话）
  )

pB_strong

ggsave(file.path(output_dir, "Fig_illustration_representation_strong.jpg"), pB_strong, width = 4.2, height = 4)

# ---------- Panel C strong ----------
pse_strong <- 50
slope_strong <- 3.2

x_strong <- seq(x_min, x_max, length.out = 400)
p_sad_strong <- 1 / (1 + exp(-(x_strong - pse_strong) / slope_strong))
beh_strong <- data.frame(x = x_strong, p = p_sad_strong)

dpdx_strong <- 1 / (4 * slope_strong)

dx_max <- 0.5 / dpdx_strong
dx_strong <- min(4.5, dx_max * 0.92)

tangent_df_strong <- data.frame(
  x = c(pse_strong - dx_strong, pse_strong + dx_strong),
  y = 0.5 + dpdx_strong * c(-dx_strong, dx_strong)
)

slope_label_x <- pse_strong + dx_strong + 2.5
slope_label_y <- 0.5 + dpdx_strong * dx_strong + 0.02

pC_strong <- ggplot(beh_strong, aes(x, p)) +
  # [修改点] Panel C strong 渐变
  geom_segment(aes(xend = dplyr::lead(x), yend = dplyr::lead(p), color = x),
               linewidth = 1.4, na.rm = TRUE) +
  scale_color_gradientn(colours = c(cols["Happy"], "#F3EFE6", cols["Sad"])) +
  
  geom_hline(yintercept = 0.5, linetype = "dotted",
             linewidth = 0.8, color = "grey35") +
  geom_vline(xintercept = pse_strong, linetype = "dashed",
             linewidth = 1.0, color = "grey30") +
  geom_line(
    data = tangent_df_strong,
    aes(x, y),
    inherit.aes = FALSE,
    linewidth = 1.4,
    color = "black"
  ) +
  # annotate("text",
  #          x = slope_label_x,
  #          y = slope_label_y,
  #          label = "Slope",
  #          family = "Arial",
  #          size = 5,
  #          hjust = 0,
  #          vjust = 0) +
  # annotate("text",
  #          x = pse_strong + 2, y = 0.10,
  #          label = "Category Boundary \n(PSE)",
  #          family = "Arial",
  #          size = 4.6,
  #          hjust = 0) +
  scale_x_continuous(
    limits = c(x_min, x_max),
    breaks = c(0, 20, 40, 60, 80, 100),
    labels = c("0%", "20%", "40%", "60%", "80%", "100%")
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = c(0, 0.5, 1)
  ) +
  labs(
    x = "Emotion Continuum (Happy \u2192 Sad)",
    y = 'P(judged as "Sad")'
  ) +
  theme_classic() +
  theme(
    legend.position = "none", # [修改点] 隐藏图例
    text = element_text(family = "Arial"),
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 14),
    axis.line  = element_line(linewidth = 1.0),
    plot.margin = margin(6, 6, 6, 6)
  )

pC_strong

ggsave(file.path(output_dir, "Fig_illustration_behavior_strong.jpg"), pC_strong, width = 4.2, height = 4)

# ====================================================
# Part 2. Weaker category separation
# ====================================================

# ---------- Panel B weak ----------
n_weak <- 260

x_weak <- rnorm(n_weak, mean = 0, sd = 0.62)
y_weak <- -0.22 * x_weak + rnorm(n_weak, mean = 0, sd = 0.34)

mix_raw <- (x_weak - min(x_weak)) / (max(x_weak) - min(x_weak))

rep_df_weak <- tibble(
  x = x_weak,
  y = y_weak,
  mix = mix_raw
)

a_weak <- 2.5
b_weak <- 0.0

boundary_df_weak <- tibble(
  #x = xlim,
  #y = a_weak * xlim + b_weak
  x = c(-0.45, 0.45),
  y = a_weak * c(-0.45, 0.45) + b_weak
)

pB_weak <- ggplot(rep_df_weak, aes(x, y, color = mix)) +
  geom_line(
    data = boundary_df_weak,
    aes(x, y),
    inherit.aes = FALSE,
    linetype = "dashed",
    linewidth = 1.05,
    alpha = 0.5,
    color = "grey30"
  ) +
  geom_point(size = 2.35, alpha = 0.95) +
  annotate("text", x = -0.95, y = 1.32,
           label = "Happy-category\nrepresentation",color=cols["happy"],
           size = 6.5, family = "Arial") +
  annotate("text", x = 0.95, y = -1.32,
           label = "Sad-category\nrepresentation",color=cols["sad"],
           size = 6.5, family = "Arial") +
  annotate("text", x = 0.05, y = -1.55,
           label = "Category Boundary",
           size = 6.2, family = "Arial") +
  scale_color_gradientn(
    colours = c(cols["Happy"], "#F3EFE6", cols["Sad"])
  ) +
  coord_cartesian(xlim = xlim, ylim = ylim, expand = FALSE) +
  theme_void() +
  theme(
    legend.position = "none",
    text = element_text(family = "Arial"),
    plot.margin = margin(5, 5, 5, 5)
  )

pB_weak

ggsave(file.path(output_dir, "Fig_illustration_representation_weak.jpg"), pB_weak, width = 4.2, height = 4)

# ---------- Panel C weak ----------
pse_weak <- 50
slope_weak <- 18

x_weak_curve <- seq(x_min, x_max, length.out = 400)
p_sad_weak <- 1 / (1 + exp(-(x_weak_curve - pse_weak) / slope_weak))
beh_weak <- data.frame(x = x_weak_curve, p = p_sad_weak)

dpdx_weak <- 1 / (4 * slope_weak)
dx_weak <- 13

tangent_df_weak <- data.frame(
  x = c(pse_weak - dx_weak, pse_weak + dx_weak),
  y = 0.5 + dpdx_weak * c(-dx_weak, dx_weak)
)

pC_weak <- ggplot(beh_weak, aes(x, p)) +
  # [修改点] Panel C weak 渐变
  geom_segment(aes(xend = dplyr::lead(x), yend = dplyr::lead(p), color = x),
               linewidth = 1.4, na.rm = TRUE) +
  scale_color_gradientn(colours = c(cols["Happy"], "#F3EFE6", cols["Sad"])) +
  
  geom_hline(yintercept = 0.5, linetype = "dotted",
             linewidth = 0.8, color = "grey35") +
  geom_vline(xintercept = pse_weak, linetype = "dashed",
             linewidth = 1.0, color = "grey30") +
  geom_line(
    data = tangent_df_weak,
    aes(x, y),
    inherit.aes = FALSE,
    linewidth = 1.2,
    color = "black"
  ) +
  # annotate("text",
  #          x = pse_weak + dx_weak + 3,
  #          y = 0.5 + dpdx_weak * dx_weak,
  #          label = "Slope",
  #          family = "Arial",
  #          size = 5,
  #          hjust = 0) +
 
  # annotate("text",
  #          x = pse_weak + 2, y = 0.10,
  #          label = "Category Boundary (PSE)",
  #          family = "Arial",
  #          size = 4.6,
  #          hjust = 0) +
  scale_x_continuous(
    limits = c(x_min, x_max),
    breaks = c(0, 20, 40, 60, 80, 100),
    labels = c("0%", "20%", "40%", "60%", "80%", "100%")
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = c(0, 0.5, 1)
  ) +
  labs(
    x = "Emotion Continuum (Happy \u2192 Sad)",
    y = 'P(judged as "Sad")'
  ) +
  theme_classic() +
  theme(
    legend.position = "none", # [修改点] 隐藏图例
    text = element_text(family = "Arial"),
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 14),
    axis.line  = element_line(linewidth = 1.0),
    plot.margin = margin(6, 6, 6, 6)
  )

pC_weak

ggsave(file.path(output_dir, "Fig_illustration_behavior_weak.jpg"), pC_weak, width = 4.2, height = 4)