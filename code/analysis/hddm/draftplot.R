# 加载必要的包
library(ggplot2)
library(dplyr)
library(grid)

# ==========================================
# 1. 基础设置与视觉规范
# ==========================================
out_dir <- "output_schematics"
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

color_cat <- '#d95f5f'  # Categorical: 莫兰迪红
color_con <- '#5c8ab8'  # Continuous: 莫兰迪蓝

non_decision_time <- 0.2
n_trials <- 5
noise_std <- 0.12
dt <- 0.01

simulate_path <- function(v, a, z, max_time) {
  time_steps <- floor(max_time / dt)
  path <- numeric(time_steps + 1)
  path[1] <- z
  for (i in 1:time_steps) {
    step <- v * dt + rnorm(1, mean = 0, sd = noise_std)
    new_val <- path[i] + step
    if (new_val >= a) {
      path[i + 1] <- a
      path <- path[1:(i + 1)] 
      break
    } else if (new_val <= 0) {
      path[i + 1] <- 0
      path <- path[1:(i + 1)]
      break
    }
    path[i + 1] <- new_val
  }
  return(path)
}

# 🌟 优化后的 APA 主题 (整体字号放大)
my_theme <- theme_classic(base_size = 20) + # 基础字号放大到 20
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 22, margin = margin(b = 15)), # 标题变大
    axis.title = element_text(face = "bold", size = 20), # 轴标题变大
    axis.line = element_line(linewidth = 1.2, color = "black"), # 坐标轴线稍微加粗以匹配大字号
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_text(size = 18, color = "black"), # Y轴刻度数字变大
    legend.position = c(0.95, 0.05),
    legend.justification = c(1, 0),
    legend.title = element_blank(),
    legend.text = element_text(size = 16), # 图例文字变大
    legend.background = element_rect(fill = "transparent"),
    legend.key.width = unit(3, "line")
  )

cond_levels <- c("Categorical & Between", "Continuous & Between", 
                 "Categorical & Within", "Continuous & Within")
color_map <- c("Categorical & Between" = color_cat, "Continuous & Between" = color_con,
               "Categorical & Within" = color_cat, "Continuous & Within" = color_con)
lty_map <- c("Categorical & Between" = "solid", "Continuous & Between" = "solid",
             "Categorical & Within" = "dashed", "Continuous & Within" = "dashed")

set.seed(42) 

# ==========================================================
# 绘图 1：交互作用作用于 漂移率 v (边界 a 保持不变)
# ==========================================================
a_fixed <- 2.0
z_fixed <- 1.0
max_time_1 <- 2.5
offset_y <- 0.02 

conditions_v <- list(
  list(v = 1.4, col = color_cat, lty = "solid",  y_off = offset_y * 1.5,  label = "Categorical & Between"),
  list(v = 1.25, col = color_con, lty = "solid",  y_off = offset_y * 0.5,  label = "Continuous & Between"),
  list(v = 1.15, col = color_con, lty = "dashed", y_off = -offset_y * 0.5, label = "Continuous & Within"),
  list(v = 0.7, col = color_cat, lty = "dashed", y_off = -offset_y * 1.5, label = "Categorical & Within")
)

path_df_v <- data.frame()
arrow_df_v <- data.frame()
group_id <- 1

for (cond in conditions_v) {
  for (i in 1:n_trials) {
    p_val <- simulate_path(cond$v, a_fixed, z_fixed, max_time_1)
    t_val <- (0:(length(p_val)-1)) * dt + non_decision_time
    path_df_v <- rbind(path_df_v, data.frame(
      time = t_val, value = p_val + cond$y_off, 
      Condition = factor(cond$label, levels = cond_levels),
      group = group_id, alpha = ifelse(cond$lty == "dashed", 0.15, 0.10)
    ))
    group_id <- group_id + 1
  }
  
  t_end <- (a_fixed - z_fixed) / cond$v + non_decision_time
  arrow_df_v <- rbind(arrow_df_v, data.frame(
    x = non_decision_time, y = z_fixed,
    xend = t_end, yend = a_fixed,
    Condition = factor(cond$label, levels = cond_levels)
  ))
}

# 高级 Y 轴标签 (数学斜体)
labels_v <- expression("0", italic(z), italic(a))

p1 <- ggplot() +
  geom_hline(yintercept = a_fixed, linewidth = 1.2, color = "#333333") +
  geom_hline(yintercept = z_fixed, linewidth = 1, color = "lightgray") +
  geom_path(data = path_df_v, aes(x = time, y = value, group = group, 
                                  color = Condition, linetype = Condition, alpha = alpha), 
            linewidth = 0.4) +
  geom_segment(data = arrow_df_v, aes(x = x, y = y, xend = xend, yend = yend, 
                                      color = Condition, linetype = Condition),
               arrow = arrow(length = unit(0.35, "cm"), type = "closed"), 
               linewidth = 1.2, alpha = 0.9) +
  scale_color_manual(values = color_map) +
  scale_linetype_manual(values = lty_map) +
  scale_alpha_identity() + 
  scale_y_continuous(breaks = c(0, z_fixed, a_fixed), labels = labels_v, limits = c(0, a_fixed + 0.2)) +
  coord_cartesian(xlim = c(0, max_time_1), expand = FALSE) +
  labs(x = "Reaction Time", y = "Decision Variable", title = "") +
  my_theme

# 🌟 保存为 4:3 比例 (8x6)
ggsave(file.path(out_dir, "Schematic_Interaction_v.png"), plot = p1, width = 8, height = 6, dpi = 300)


# ==========================================================
# 绘图 2：假设交互作用作用于 边界 a (优化边界间距版)
# ==========================================================
v_fixed <- 1.0
max_time_2 <- 2.5
z_fixed <- 1.0

a_cat_bet <- 2.00  
a_con_bet <- 2.06  
a_con_wit <- 2.12  
a_cat_wit <- 2.35  

conditions_a <- list(
  list(a = a_cat_wit, col = color_cat, lty = "dashed", label = "Categorical & Within"),
  list(a = a_con_wit, col = color_con, lty = "dashed", label = "Continuous & Within"),
  list(a = a_con_bet, col = color_con, lty = "solid",  label = "Continuous & Between"),
  list(a = a_cat_bet, col = color_cat, lty = "solid",  label = "Categorical & Between")
)

path_df_a <- data.frame()
arrow_df_a <- data.frame()
group_id <- 1

for (cond in conditions_a) {
  for (i in 1:n_trials) {
    p_val <- simulate_path(v_fixed, cond$a, z_fixed, max_time_2)
    t_val <- (0:(length(p_val)-1)) * dt + non_decision_time
    path_df_a <- rbind(path_df_a, data.frame(
      time = t_val, value = p_val, 
      Condition = factor(cond$label, levels = cond_levels),
      group = group_id, alpha = ifelse(cond$lty == "dashed", 0.15, 0.10)
    ))
    group_id <- group_id + 1
  }
  
  t_end <- (cond$a - z_fixed) / v_fixed + non_decision_time
  arrow_df_a <- rbind(arrow_df_a, data.frame(
    x = non_decision_time, y = z_fixed,
    xend = t_end, yend = cond$a,
    Condition = factor(cond$label, levels = cond_levels)
  ))
}

p2 <- ggplot() +
  geom_hline(yintercept = 0, linewidth = 1, color = "#333333") +
  geom_hline(yintercept = z_fixed, linewidth = 1, color = "lightgray") +
  
  geom_hline(data = data.frame(
    yint = c(a_cat_bet, a_con_bet, a_cat_wit, a_con_wit),
    Condition = factor(c("Categorical & Between", "Continuous & Between", 
                         "Categorical & Within", "Continuous & Within"), levels = cond_levels)
  ), 
  aes(yintercept = yint, color = Condition, linetype = Condition), 
  linewidth = 2, alpha = 0.8) +
  
  geom_path(data = path_df_a, aes(x = time, y = value, group = group, 
                                  color = Condition, linetype = Condition, alpha = alpha), 
            linewidth = 0.4, show.legend = FALSE) +
  
  geom_segment(data = arrow_df_a, aes(x = x, y = y, xend = xend, yend = yend),
               color = "#222222", linetype = "solid",
               arrow = arrow(length = unit(0.35, "cm"), type = "closed"), 
               linewidth = 1.2, alpha = 0.9, show.legend = FALSE) +
  
  scale_color_manual(values = color_map) +
  scale_linetype_manual(values = lty_map) +
  scale_alpha_identity() + 
  
  scale_y_continuous(breaks = c(0, z_fixed, a_cat_bet, a_con_bet, a_con_wit, a_cat_wit), 
                     labels = c("0", expression(italic(z)),"","","",""),  limits = c(0, a_cat_wit + 0.2)) +
  coord_cartesian(xlim = c(0, max_time_2), expand = FALSE) +
  labs(x = "Reaction Time", y = "Decision Variable", title = "") +
  my_theme

# 🌟 保存为 4:3 比例 (8x6)
ggsave(file.path(out_dir, "Schematic_Interaction_a.png"), plot = p2, width = 8, height = 6, dpi = 300)

print("✅ 图表 1 & 2 已按 4:3 比例和放大字号重新生成。")