rm(list = ls())

# Set current directory
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
package_root <- normalizePath(file.path(script_dir, "..", ".."))
dataDir <- file.path(package_root, "data", "exp04")
analysisDir <- file.path(package_root, "results", "exp04")
dir.create(analysisDir, recursive = TRUE, showWarnings = FALSE)
setwd(analysisDir)

library(readxl)
library(ggplot2)
library(reshape2)
library(broom)
library(gridExtra)
library(dplyr)
library(RColorBrewer)
library(sjPlot)
library(psych)
library(lme4)
library(devtools)
# library(lmSupport)
library(tidyverse)
library(car)
library(ggpubr)

scriptDir <- dirname(rstudioapi::getSourceEditorContext()$path)
source(paste(scriptDir, "helper_functions.R", sep = "\\"))

plot_color = c("#F28E8E", "#7FB3FF")

##################################
# Load data (exp04)
##################################
data_final <- read.csv(file.path(dataDir, "exp04_trial_data.csv"))

# Center emotion_level (and divide by 10 to make units meaningful)
data_final$emotion_level <- (data_final$emotion_level - 50) / 10
data_final$participant   <- as.factor(data_final$participant)

# Code thinking condition (categorical = -0.5, continuous = 0.5)
data_final$thinking <- ifelse(data_final$thinking == "categorical", -0.5, 0.5)

# Code task awareness (Informed/block = -0.5, Uninformed/mix = 0.5)
data_final$task_awareness <- ifelse(data_final$task_awareness == "Informed", -0.5, 0.5)

# Extract specific face stimulus ID
data_final <- data_final %>%
  mutate(face_id = sub(".*/(m_[^_/]+)_.*", "\\1", face_stim))

##################################
# Aggregate across participants (subj-level means)
##################################
temp1 <- aggregate(resp ~ participant + thinking + task_awareness + emotion_level,
                   data = data_final, FUN = "mean")
temp2 <- aggregate(RT ~ participant + thinking + task_awareness + emotion_level,
                   data = data_final, FUN = "mean")

data_final_subj <- data.frame(temp1, rt = temp2$RT)

# Order the data frame
data_final_subj <- data_final_subj[order(data_final_subj$participant,
                                         data_final_subj$thinking,
                                         data_final_subj$task_awareness,
                                         data_final_subj$emotion_level), ]

##################################
# Aggregate across stimuli (stim-level means)
##################################
temp1 <- aggregate(resp ~ face_stim + thinking + task_awareness + emotion_level,
                   data = data_final, FUN = "mean")
temp2 <- aggregate(RT ~ face_stim + thinking + task_awareness + emotion_level,
                   data = data_final, FUN = "mean")

data_final_stim <- data.frame(temp1, rt = temp2$RT)

# Order the data frame
data_final_stim <- data_final_stim[order(data_final_stim$face_stim,
                                         data_final_stim$thinking,
                                         data_final_stim$task_awareness,
                                         data_final_stim$emotion_level), ]

##################################
# Fit individual psychometric functions (4 conditions per participant)
##################################
library(purrr)

participants <- unique(data_final$participant)

# Initialize results data frame
results <- data.frame(
  participant = character(),
  task_awareness = character(),
  condition = character(),
  PSE = numeric(),
  slope = numeric(),
  stringsAsFactors = FALSE
)

# Ensure output folder exists
if (!dir.exists("fig7_onebyone")) dir.create("fig7_onebyone", recursive = TRUE)

# Plot helper: more robust prediction + no title
plot_psychometric_4cond <- function(data, model, model_type, cond_label, participant_id) {
  
  pred_data <- data.frame(
    emotion_level = seq(min(data$emotion_level), max(data$emotion_level), length.out = 200)
  )
  
  # Robust prediction based on model type
  if (model_type == "glm") {
    pred_data$pred <- predict(model, newdata = pred_data, type = "response")
  } else if (model_type == "nls") {
    # nls prediction can fail/NA if exp overflows; do a safe manual logistic calculation
    co <- coef(model)
    b0 <- co["b0"]; b1 <- co["b1"]
    eta <- b0 + b1 * pred_data$emotion_level
    eta <- pmax(pmin(eta, 35), -35)
    pred_data$pred <- 1 / (1 + exp(-eta))
  } else {
    pred_data$pred <- NA_real_
  }
  
  pred_data <- pred_data %>% filter(is.finite(pred))
  
  p <- ggplot() +
    geom_point(
      data = data,
      aes(x = emotion_level, y = resp),
      position = position_jitter(width = 0.3, height = 0.02),
      color = plot_color[2],
      alpha = 0.4,
      size = 2
    ) +
    geom_line(
      data = pred_data,
      aes(x = emotion_level, y = pred),
      color = plot_color[1],
      linewidth = 1.5
    ) +
    scale_x_continuous(
      name = "Morph: % Fear",
      breaks = c(-4, -2, 0, 2, 4),
      labels = c("10%", "30%", "50%", "70%", "90%"),
      limits = c(-5, 4.2),
      expand = expansion(mult = c(0, 0.05))) + 
    scale_y_continuous(
      name = "Response",
      breaks = c(0, 0.5, 1),
      limits = c(0, 1.1),
      expand = expansion(mult = c(0, 0))) +
    
    annotate("text", x = -4, y = 0, label = "Calm",
             angle = 90, hjust = 0.8, vjust = -6,
             size = 5, fontface = "italic",color = "black") +
    annotate("text", x = -4, y = 1, label = "Fear",
             angle = 90, hjust = 0.2, vjust = -6,
             size = 5, fontface = "italic",color = "black") +
    
    theme_minimal(base_size = 14) +
    theme(
      legend.box.spacing = unit(-20, "pt"),
      plot.margin = margin(t = 10, r = 10, b = 10, l = 10),
      axis.line.x = element_line(color = "black", linewidth = 1),
      axis.line.y = element_line(color = "black", linewidth = 1),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black"),
      axis.text = element_text(color = "black"),
      axis.title.y = element_text(color = "black", size = 22, face = "bold", margin = margin(r = 15)),
      axis.title.x = element_text(color = "black",size = 22, face = "bold"), 
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    coord_cartesian(clip = "off")
  
  ggsave(
    paste0("fig7_onebyone\\fig7_emotion_categorization_", cond_label, "_", participant_id, ".jpg"),
    p, width = 4.6, height = 4, dpi = 150
  )
}

# Safe nls wrapper to handle convergence errors
safe_nls <- function(formula, data, start) {
  tryCatch(
    nls(formula, data = data, start = start,
        control = nls.control(maxiter = 200, warnOnly = TRUE)),
    error = function(e) NULL
  )
}

##################################
# Loop through participants to fit individual psychometric functions
##################################
for (p in participants) {
  
  df <- data_final %>% filter(participant == p)
  
  # Subsetting data for the 4 conditions
  categorical_informed_data   <- df %>% filter(thinking == -0.5 & task_awareness == -0.5)
  continuous_informed_data    <- df %>% filter(thinking ==  0.5 & task_awareness == -0.5)
  categorical_uninformed_data <- df %>% filter(thinking == -0.5 & task_awareness ==  0.5)
  continuous_uninformed_data  <- df %>% filter(thinking ==  0.5 & task_awareness ==  0.5)
  
  # ----- 1) Categorical + Informed (glm)
  logit_model <- glm(resp ~ emotion_level, data = categorical_informed_data,
                     family = binomial(link = "logit"))
  logit_params <- coef(logit_model)
  logit_slope <- unname(logit_params["emotion_level"])
  logit_PSE   <- -unname(logit_params["(Intercept)"]) / logit_slope
  
  results <- rbind(results, data.frame(
    participant = p,
    task_awareness = "Informed",
    condition = "categorical",
    PSE = logit_PSE,
    slope = logit_slope
  ))
  
  plot_psychometric_4cond(categorical_informed_data, logit_model, "glm",
                          "categorical_Informed", participant_id = p)
  
  # ----- 2) Continuous + Informed (nls)
  nls_model <- safe_nls(resp ~ 1 / (1 + exp(-(b0 + b1 * emotion_level))),
                        data = continuous_informed_data, start = list(b0 = 0, b1 = 1))
  if (!is.null(nls_model)) {
    nls_params <- coef(nls_model)
    nls_slope <- unname(nls_params["b1"])
    nls_PSE   <- -unname(nls_params["b0"]) / nls_slope
  } else {
    nls_PSE <- NA_real_
    nls_slope <- NA_real_
  }
  
  results <- rbind(results, data.frame(
    participant = p,
    task_awareness = "Informed",
    condition = "continuous",
    PSE = nls_PSE,
    slope = nls_slope
  ))
  
  if (!is.null(nls_model)) {
    plot_psychometric_4cond(continuous_informed_data, nls_model, "nls",
                            "continuous_Informed", participant_id = p)
  }
  
  # ----- 3) Categorical + Uninformed (glm)
  logit_model2 <- glm(resp ~ emotion_level, data = categorical_uninformed_data,
                      family = binomial(link = "logit"))
  logit_params2 <- coef(logit_model2)
  logit_slope2 <- unname(logit_params2["emotion_level"])
  logit_PSE2   <- -unname(logit_params2["(Intercept)"]) / logit_slope2
  
  results <- rbind(results, data.frame(
    participant = p,
    task_awareness = "Uninformed",
    condition = "categorical",
    PSE = logit_PSE2,
    slope = logit_slope2
  ))
  
  plot_psychometric_4cond(categorical_uninformed_data, logit_model2, "glm",
                          "categorical_Uninformed", participant_id = p)
  
  # ----- 4) Continuous + Uninformed (nls)
  nls_model2 <- safe_nls(resp ~ 1 / (1 + exp(-(b0 + b1 * emotion_level))),
                         data = continuous_uninformed_data, start = list(b0 = 0, b1 = 1))
  if (!is.null(nls_model2)) {
    nls_params2 <- coef(nls_model2)
    nls_slope3 <- unname(nls_params2["b1"])
    nls_PSE3   <- -unname(nls_params2["b0"]) / nls_slope3
  } else {
    nls_PSE3 <- NA_real_
    nls_slope3 <- NA_real_
  }
  
  results <- rbind(results, data.frame(
    participant = p,
    task_awareness = "Uninformed",
    condition = "continuous",
    PSE = nls_PSE3,
    slope = nls_slope3
  ))
  
  if (!is.null(nls_model2)) {
    plot_psychometric_4cond(continuous_uninformed_data, nls_model2, "nls",
                            "continuous_Uninformed", participant_id = p)
  }
  
  print(paste("Processed participant:", p))
}

##################################
# Clean and Save Results
##################################
results_clean <- results %>% filter(!is.na(PSE) & !is.na(slope))
results_clean$participant <- as.factor(results_clean$participant)

write.csv(results_clean, "psychometric_PSE_slope_4conditions.csv", row.names = FALSE)

# ============================================================
# 2x2 repeated-measures ANOVA + pairwise comparisons
# ============================================================
library(bruceR)
library(effectsize)

# Ensure factor levels
results_clean <- results_clean %>%
  mutate(
    condition      = factor(condition,      levels = c("categorical", "continuous")),
    task_awareness = factor(task_awareness, levels = c("Informed", "Uninformed"))
  )

# ---- ANOVA for PSE ----
cat("=== ANOVA for PSE ===\n")
MANOVA(results_clean,
       subID  = "participant",
       dv     = "PSE",
       within = c("task_awareness", "condition"),
       file   = "exp2_pse.doc")

# ---- ANOVA for Slope ----
cat("\n=== ANOVA for Slope ===\n")
MANOVA(results_clean,
       subID  = "participant",
       dv     = "slope",
       within = c("task_awareness", "condition"),
       file   = "exp2_slope.doc") %>%
  EMMEANS("condition", by = "task_awareness", p.adjust = "bonferroni")

# ---- Pairwise t-tests (Bonferroni) ----
cat("\n=== Pairwise t-tests for PSE (Bonferroni corrected) ===\n")
pairwise_PSE <- results_clean %>%
  dplyr::group_by(task_awareness) %>%
  rstatix::pairwise_t_test(PSE ~ condition, paired = TRUE,
                           p.adjust.method = "bonferroni")
print(pairwise_PSE)

cat("\n=== Pairwise t-tests for Slope (Bonferroni corrected) ===\n")
pairwise_slope <- results_clean %>%
  dplyr::group_by(task_awareness) %>%
  rstatix::pairwise_t_test(slope ~ condition, paired = TRUE,
                           p.adjust.method = "bonferroni")
print(pairwise_slope)

# ---- Cohen's d (paired, consistent with Experiment 3) ----
informed_data   <- results_clean %>% dplyr::filter(task_awareness == "Informed")
uninformed_data <- results_clean %>% dplyr::filter(task_awareness == "Uninformed")

cat("\n=== Cohen's d and 95% CI for PSE ===\n")
d_PSE_informed <- effectsize::repeated_measures_d(
  informed_data$PSE[informed_data$condition == "categorical"],
  informed_data$PSE[informed_data$condition == "continuous"],
  paired = TRUE, ci = 0.95
)
cat("PSE - Informed:\n"); print(d_PSE_informed)

d_PSE_uninformed <- effectsize::repeated_measures_d(
  uninformed_data$PSE[uninformed_data$condition == "categorical"],
  uninformed_data$PSE[uninformed_data$condition == "continuous"],
  paired = TRUE, ci = 0.95
)
cat("PSE - Uninformed:\n"); print(d_PSE_uninformed)

cat("\n=== Cohen's d and 95% CI for Slope ===\n")
d_slope_informed <- effectsize::repeated_measures_d(
  informed_data$slope[informed_data$condition == "categorical"],
  informed_data$slope[informed_data$condition == "continuous"],
  paired = TRUE, ci = 0.95
)
cat("Slope - Informed:\n"); print(d_slope_informed)

d_slope_uninformed <- effectsize::repeated_measures_d(
  uninformed_data$slope[uninformed_data$condition == "categorical"],
  uninformed_data$slope[uninformed_data$condition == "continuous"],
  paired = TRUE, ci = 0.95
)
cat("Slope - Uninformed:\n"); print(d_slope_uninformed)

# ---- 95% CI for mean differences (consistent with Experiment 3) ----
cat("\n=== 95% CI for mean differences: PSE ===\n")
ci_PSE_informed <- t.test(
  informed_data$PSE[informed_data$condition == "categorical"],
  informed_data$PSE[informed_data$condition == "continuous"],
  paired = TRUE
)
cat("PSE Informed - mean diff:", round(ci_PSE_informed$estimate, 3),
    "95% CI: [", round(ci_PSE_informed$conf.int[1], 3), ",",
    round(ci_PSE_informed$conf.int[2], 3), "]\n")

ci_PSE_uninformed <- t.test(
  uninformed_data$PSE[uninformed_data$condition == "categorical"],
  uninformed_data$PSE[uninformed_data$condition == "continuous"],
  paired = TRUE
)
cat("PSE Uninformed - mean diff:", round(ci_PSE_uninformed$estimate, 3),
    "95% CI: [", round(ci_PSE_uninformed$conf.int[1], 3), ",",
    round(ci_PSE_uninformed$conf.int[2], 3), "]\n")

cat("\n=== 95% CI for mean differences: Slope ===\n")
ci_slope_informed <- t.test(
  informed_data$slope[informed_data$condition == "categorical"],
  informed_data$slope[informed_data$condition == "continuous"],
  paired = TRUE
)
cat("Slope Informed - mean diff:", round(ci_slope_informed$estimate, 3),
    "95% CI: [", round(ci_slope_informed$conf.int[1], 3), ",",
    round(ci_slope_informed$conf.int[2], 3), "]\n")

ci_slope_uninformed <- t.test(
  uninformed_data$slope[uninformed_data$condition == "categorical"],
  uninformed_data$slope[uninformed_data$condition == "continuous"],
  paired = TRUE
)
cat("Slope Uninformed - mean diff:", round(ci_slope_uninformed$estimate, 3),
    "95% CI: [", round(ci_slope_uninformed$conf.int[1], 3), ",",
    round(ci_slope_uninformed$conf.int[2], 3), "]\n")

##################################
# Bar plots: PSE
##################################
datac_PSE <- summarySE(data=results_clean, measurevar="PSE",
                       groupvars=c("task_awareness", "condition"),
                       na.rm=FALSE, conf.interval=.95, .drop=TRUE)

# Ensure consistent factor ordering for the plot
datac_PSE$task_awareness <- factor(datac_PSE$task_awareness, levels = c("Informed", "Uninformed"))

p1 <- ggplot(datac_PSE, aes(x = task_awareness, y = PSE, fill = condition)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = PSE - se, ymax = PSE + se),
                position = position_dodge(width = 0.8),
                width = 0.3, linewidth = 1) +
  geom_hline(yintercept = 0, color = "black", linewidth = 1.2) +
  scale_fill_manual(
    values = plot_color,
    labels = c("Categorical", "Continuous")
  ) +
  
  # 1. 移除默认的 X 轴文本标签
  scale_x_discrete(labels = NULL) + 
  
  # 2. 使用 annotate 添加自定义文本
  # x = 1 代表第一个组别，x = 2 代表第二个组别
  # y = 0.05 代表放在 y=0 横线的稍上方（你可以根据需要微调这个值）
  annotate("text", x = 1, y = 0.2, label = "Informed", size = 8) +
  annotate("text", x = 2, y = 0.2, label = "Uninformed", size = 8) +
  
  theme_classic(base_size = 24) +
  labs(y = "PSE", x = "") +
  scale_y_continuous(
    breaks = seq(-1, 0, by = 0.5),
    limits = c(-1.35, 0.2) # 你的上限是 0.1，所以 annotate 的 y 放 0.05 刚好在区域内
  ) +
  theme(
    aspect.ratio = 0.9,
    legend.position = "none",
    legend.box.spacing = unit(-5, "pt"),
    axis.line.x = element_blank(),
    axis.line.y = element_line(color = "black", linewidth = 1.2),
    
    # 彻底隐藏原本的 X 轴刻度线和文本位置
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(), 
    
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    legend.title = element_blank(),
    legend.box.margin = margin(t = 0, b = 0, l = 0, r = 0), 
    plot.margin = margin(t = 30, b = 1, l = 1, r = 1),
    axis.title = element_text(size = 26, face = "bold"),
    axis.title.y = element_text(size = 26, face = "bold", margin = margin(r = 15))
  ) +

  coord_cartesian(clip = "off") 

print(p1)
ggsave("fig8_emotion_categorization_psychometric_PSE_4cond_nolabel.jpg",
       p1, width = 6, height = 6, dpi = 300)


##################################
# Bar plots: slope
##################################
datac_slope <- summarySE(data=results_clean, measurevar="slope",
                         groupvars=c("task_awareness", "condition"),
                         na.rm=FALSE, conf.interval=.95, .drop=TRUE)

# Ensure consistent factor ordering for the plot
datac_slope$task_awareness <- factor(datac_slope$task_awareness, levels = c("Informed", "Uninformed"))

p2 <- ggplot(datac_slope, aes(x = task_awareness, y = slope, fill = condition)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = slope - se, ymax = slope + se),
                position = position_dodge(width = 0.8),
                width = 0.3, linewidth = 1) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.8) +
  scale_fill_manual(
    values = plot_color,
    labels = c("Categorical", "Continuous")
  ) +
  # scale_x_discrete(labels = c("Informed"   = "Informed",
  #                             "Uninformed" = "Uninformed")) +
  theme_classic(base_size = 24) +
  labs(y = "Slope", x = "") +
  scale_y_continuous(breaks = 0:3, expand = expansion(mult = 0), minor_breaks = seq(0.5, 2.5, by = 0.5), limits = c(0, 3.5)) +
  theme(
    aspect.ratio = 0.9,
    legend.position = "none",
    legend.box.spacing = unit(-20, "pt"),
    axis.line.x = element_line(color = "black", linewidth = 1.2),
    axis.line.y = element_line(color = "black", linewidth = 1.2),
    axis.ticks.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    legend.title = element_blank(),
    legend.box.margin = margin(t = 0, b = 0, l = 0, r = 0), 
    plot.margin = margin(t = 30, b = 1, l = 1, r = 1),
    axis.title = element_text(size = 20, face = "bold"),
    axis.title.y = element_text(size = 26, face = "bold", margin = margin(r = 15)),
    axis.title.x = element_text(size = 26),
    axis.text.x = element_text(size = 24, margin = margin(t = 15))
  )

print(p2)
ggsave("fig9_emotion_categorization_psychometric_slope_4cond_nolabel.jpg",
       p2, width = 6, height = 6, dpi = 300)



