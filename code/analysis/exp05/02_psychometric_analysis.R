rm(list = ls())

# Set current directory
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
package_root <- normalizePath(file.path(script_dir, "..", ".."))
dataDir <- file.path(package_root, "data", "exp05")
analysisDir <- file.path(package_root, "results", "exp05")
dir.create(analysisDir, recursive = TRUE, showWarnings = FALSE)
setwd(analysisDir)

plot_color = c( "#F28E8E", "#7FB3FF")
# ============================================================
# 0) Packages: keep minimal + avoid masking bugs
#    - Do NOT load psych / plyr / tidyverse (they can mask sd/filter/summarise)
#    - If you MUST use them elsewhere, load them AFTER this script finishes
# ============================================================
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tibble)
  library(rstatix)
})

# If plyr got loaded indirectly somewhere, detach it
if ("package:plyr" %in% search()) detach("package:plyr", unload = TRUE)

# (Optional) If psych got loaded somewhere, detach it too (it can mask sd())
if ("package:psych" %in% search()) detach("package:psych", unload = TRUE)

# ============================================================
# 1) Load data (exp05 NEW)
# ============================================================
data_final <- read.csv(file.path(dataDir, "exp05_trial_data.csv"), stringsAsFactors = FALSE)

# Center emotion_level: 10%->-4, 90%->4
data_final$emotion_level <- (data_final$emotion_level - 50) / 10
data_final$participant   <- as.factor(data_final$participant)

# Thinking: categorical=-0.5, continuous=0.5 (from column "thinking")
data_final$thinking <- ifelse(tolower(data_final$thinking) == "categorical", -0.5, 0.5)

# Task Awareness: Informed=-0.5, Uninformed=0.5 (from column "task_awareness")
data_final$task_awareness <- ifelse(tolower(data_final$task_awareness) == "informed", -0.5, 0.5)

# RT -> rt
if ("RT" %in% names(data_final)) data_final$rt <- as.numeric(data_final$RT)

# face_id
data_final <- dplyr::mutate(
  data_final,
  face_id = sub(".*/(m_[^_/]+)_.*", "\\1", face_stim)
)

# ============================================================
# 2) Aggregate across participants (subj-level means)
# ============================================================
temp1 <- stats::aggregate(resp ~ participant + thinking + task_awareness + emotion_level,
                          data = data_final, FUN = mean)
temp2 <- stats::aggregate(rt ~ participant + thinking + task_awareness + emotion_level,
                          data = data_final, FUN = mean)

data_final_subj <- data.frame(temp1, rt = temp2$rt)
data_final_subj <- data_final_subj[order(data_final_subj$participant,
                                         data_final_subj$thinking,
                                         data_final_subj$task_awareness,
                                         data_final_subj$emotion_level), ]

# ============================================================
# 3) Aggregate across stimuli (stim-level means)
# ============================================================
temp1 <- stats::aggregate(resp ~ face_stim + thinking + task_awareness + emotion_level,
                          data = data_final, FUN = mean)
temp2 <- stats::aggregate(rt ~ face_stim + thinking + task_awareness + emotion_level,
                          data = data_final, FUN = mean)

data_final_stim <- data.frame(temp1, rt = temp2$rt)
data_final_stim <- data_final_stim[order(data_final_stim$face_stim,
                                         data_final_stim$thinking,
                                         data_final_stim$task_awareness,
                                         data_final_stim$emotion_level), ]

# ============================================================
# 4) Fit individual psychometric functions (4 conditions per participant)
# ============================================================
participants <- unique(data_final$participant)

results <- data.frame(
  participant    = character(),
  task_awareness = character(),
  condition      = character(),
  PSE            = numeric(),
  slope          = numeric(),
  stringsAsFactors = FALSE
)

# Output folder
if (!dir.exists("fig7_onebyone")) dir.create("fig7_onebyone", recursive = TRUE)

# ---- Plot helper: robust prediction + no title ----
plot_psychometric_4cond <- function(data, model, model_type, cond_label, participant_id) {
  pred_data <- data.frame(
    emotion_level = seq(min(data$emotion_level), max(data$emotion_level), length.out = 200)
  )
  
  if (model_type == "glm") {
    pred_data$pred <- stats::predict(model, newdata = pred_data, type = "response")
  } else if (model_type == "nls") {
    co <- stats::coef(model)
    b0 <- unname(co["b0"]); b1 <- unname(co["b1"])
    eta <- b0 + b1 * pred_data$emotion_level
    eta <- pmax(pmin(eta, 35), -35)
    pred_data$pred <- 1 / (1 + exp(-eta))
  } else {
    pred_data$pred <- NA_real_
  }
  
  pred_data <- dplyr::filter(pred_data, is.finite(pred))
  
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
    file.path("fig7_onebyone",
              paste0("fig7_emotion_categorization_", cond_label, "_", participant_id, ".jpg")),
    p, width = 4.6, height = 4, dpi = 150
  )
}

# ---- Safe nls wrapper ----
safe_nls <- function(formula, data, start) {
  tryCatch(
    nls(formula, data = data, start = start,
        control = nls.control(maxiter = 200, warnOnly = TRUE)),
    error = function(e) NULL
  )
}

# ============================================================
# 5) Loop participants
# ============================================================
for (p in participants) {
  
  df <- dplyr::filter(data_final, participant == p)
  
  categorical_informed_data   <- dplyr::filter(df, thinking == -0.5 & task_awareness == -0.5)
  continuous_informed_data    <- dplyr::filter(df, thinking ==  0.5 & task_awareness == -0.5)
  categorical_uninformed_data <- dplyr::filter(df, thinking == -0.5 & task_awareness ==  0.5)
  continuous_uninformed_data  <- dplyr::filter(df, thinking ==  0.5 & task_awareness ==  0.5)
  
  # 1) Categorical + Informed (glm)
  logit_model <- stats::glm(resp ~ emotion_level, data = categorical_informed_data,
                            family = stats::binomial(link = "logit"))
  logit_params <- stats::coef(logit_model)
  logit_slope  <- unname(logit_params["emotion_level"])
  logit_PSE    <- -unname(logit_params["(Intercept)"]) / logit_slope
  
  results <- rbind(results, data.frame(
    participant    = as.character(p),
    task_awareness = "Informed",
    condition      = "categorical",
    PSE            = logit_PSE,
    slope          = logit_slope
  ))
  
  plot_psychometric_4cond(categorical_informed_data, logit_model, "glm",
                          "categorical_Informed", participant_id = p)
  
  # 2) Continuous + Informed (nls)
  nls_model <- safe_nls(resp ~ 1 / (1 + exp(-(b0 + b1 * emotion_level))),
                        data = continuous_informed_data, start = list(b0 = 0, b1 = 1))
  if (!is.null(nls_model)) {
    nls_params <- stats::coef(nls_model)
    nls_slope  <- unname(nls_params["b1"])
    nls_PSE    <- -unname(nls_params["b0"]) / nls_slope
  } else {
    nls_PSE   <- NA_real_
    nls_slope <- NA_real_
  }
  
  results <- rbind(results, data.frame(
    participant    = as.character(p),
    task_awareness = "Informed",
    condition      = "continuous",
    PSE            = nls_PSE,
    slope          = nls_slope
  ))
  
  if (!is.null(nls_model)) {
    plot_psychometric_4cond(continuous_informed_data, nls_model, "nls",
                            "continuous_Informed", participant_id = p)
  }
  
  # 3) Categorical + Uninformed (glm)
  logit_model2 <- stats::glm(resp ~ emotion_level, data = categorical_uninformed_data,
                             family = stats::binomial(link = "logit"))
  logit_params2 <- stats::coef(logit_model2)
  logit_slope2  <- unname(logit_params2["emotion_level"])
  logit_PSE2    <- -unname(logit_params2["(Intercept)"]) / logit_slope2
  
  results <- rbind(results, data.frame(
    participant    = as.character(p),
    task_awareness = "Uninformed",
    condition      = "categorical",
    PSE            = logit_PSE2,
    slope          = logit_slope2
  ))
  
  plot_psychometric_4cond(categorical_uninformed_data, logit_model2, "glm",
                          "categorical_Uninformed", participant_id = p)
  
  # 4) Continuous + Uninformed (nls)
  nls_model2 <- safe_nls(resp ~ 1 / (1 + exp(-(b0 + b1 * emotion_level))),
                         data = continuous_uninformed_data, start = list(b0 = 0, b1 = 1))
  if (!is.null(nls_model2)) {
    nls_params2 <- stats::coef(nls_model2)
    nls_slope3  <- unname(nls_params2["b1"])
    nls_PSE3    <- -unname(nls_params2["b0"]) / nls_slope3
  } else {
    nls_PSE3   <- NA_real_
    nls_slope3 <- NA_real_
  }
  
  results <- rbind(results, data.frame(
    participant    = as.character(p),
    task_awareness = "Uninformed",
    condition      = "continuous",
    PSE            = nls_PSE3,
    slope          = nls_slope3
  ))
  
  if (!is.null(nls_model2)) {
    plot_psychometric_4cond(continuous_uninformed_data, nls_model2, "nls",
                            "continuous_Uninformed", participant_id = p)
  }
  
  message("Processed participant: ", p)
}

# ============================================================
# 6) Clean + save results
# ============================================================
results_clean <- dplyr::filter(results, !is.na(PSE) & !is.na(slope))
results_clean$participant <- as.factor(results_clean$participant)

write.csv(results_clean, "psychometric_PSE_slope_4conditions.csv", row.names = FALSE)

# ============================================================
# 7) 2x2 repeated-measures ANOVA + pairwise
# ============================================================
library(bruceR)

anova_PSE_ez <- MANOVA(results_clean,subID = 'participant', dv = "PSE", within = c("task_awareness", "condition"),file = "exp3_pse.doc" )
anova_slope_ez <- MANOVA(results_clean,subID = 'participant', dv = "slope", within = c("task_awareness", "condition"),file = "exp3_slope.doc")

# anova_PSE <- rstatix::anova_test(
#   data   = results_clean,
#   dv     = PSE,
#   wid    = participant,
#   within = c(task_awareness, condition)
# )
# print(anova_PSE)
# 
# anova_slope <- rstatix::anova_test(
#   data   = results_clean,
#   dv     = slope,
#   wid    = participant,
#   within = c(task_awareness, condition)
# )
# print(anova_slope)
# 
# pairwise_PSE <- results_clean %>%
#   dplyr::group_by(task_awareness) %>%
#   rstatix::pairwise_t_test(PSE ~ condition, paired = TRUE, p.adjust.method = "bonferroni")
# print(pairwise_PSE)
# 
# pairwise_slope <- results_clean %>%
#   dplyr::group_by(task_awareness) %>%
#   rstatix::pairwise_t_test(slope ~ condition, paired = TRUE, p.adjust.method = "bonferroni")
# print(pairwise_slope)

# ============================================================
# 8) Bar plots (use base::aggregate to avoid summarise() weird NA)
# ============================================================

# Ensure variables are purely numeric (double check)
results_clean$PSE   <- as.numeric(results_clean$PSE)
results_clean$slope <- as.numeric(results_clean$slope)

# ---- helper: aggregate mean + se, then unpack ----
agg_mean_se <- function(df, value_col){
  out <- stats::aggregate(
    df[[value_col]],
    by = list(task_awareness = df$task_awareness, condition = df$condition),
    FUN = function(x){
      x <- x[is.finite(x)]
      c(n = length(x),
        mean = base::mean(x),
        se = stats::sd(x) / base::sqrt(length(x)))
    }
  )
  out <- cbind(out[, 1:2], out$x)
  out <- as.data.frame(out, stringsAsFactors = FALSE)
  colnames(out) <- c("task_awareness", "condition", "n", value_col, "se")
  out$n  <- as.integer(out$n)
  out[[value_col]] <- as.numeric(out[[value_col]])
  out$se <- as.numeric(out$se)
  out
}

datac_PSE   <- agg_mean_se(results_clean, "PSE")
datac_slope <- agg_mean_se(results_clean, "slope")

print(datac_PSE)
print(datac_slope)


# ---------------- Plot: PSE ----------------
p1 <- ggplot(datac_PSE, aes(x = task_awareness, y = PSE, fill = condition)) +

  scale_x_discrete(labels = c("Informed"   = "",
                              "Uninformed" = ""))+
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = PSE - se, ymax = PSE + se),
                position = position_dodge(width = 0.8),
                width = 0.3, linewidth = 1) +
  geom_hline(yintercept = 0, color = "black", linewidth = 1.2) +
  scale_fill_manual(values = plot_color,
                    labels = c("Categorical", "Continuous")) +
  
  # === 新增：用 annotate 添加自定义标签 ===
  # x=1 对应 Informed，x=2 对应 Uninformed
  # y=0.05 让字浮在 y=0 黑线的正上方一点点
  annotate("text", x = 1, y = 0.2, label = "Informed", size = 8) +
  annotate("text", x = 2, y = 0.2, label = "Uninformed", size = 8) +
  # ========================================

theme_classic(base_size = 24) +
  labs(y = "PSE", x = "") +
  scale_y_continuous(
    breaks = seq(-1, 0, by = 0.5),
    limits = c(-1.35, 0.2) 
  ) +
  theme(
    aspect.ratio = 0.9,
    legend.position = "none",
    legend.box.spacing = unit(-5, "pt"),
    axis.line.x = element_blank(),
    axis.line.y = element_line(color = "black", linewidth = 1.2),
    axis.ticks.x = element_blank(),
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

# ---------------- Plot: slope ----------------
p2 <- ggplot(datac_slope, aes(x = task_awareness, y = slope, fill = condition)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = slope - se, ymax = slope + se),
                position = position_dodge(width = 0.8),
                width = 0.3, linewidth = 1) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.8) +
  scale_fill_manual(values = plot_color,
                    labels = c("Categorical", "Continuous")) +
  scale_x_discrete(labels = c("Informed"   = "Informed",
                              "Uninformed" = "Uninformed"))+
  theme_classic(base_size = 24) +
  labs(y = "Slope", x = "") +
  scale_y_continuous(breaks = 0:3, expand = expansion(mult = 0), minor_breaks = seq(0.5, 2.5, by = 0.5), limits = c(0, 3.5))+
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
    axis.title.x = element_text(size = 26, face = "bold"),

    axis.text.x = element_text(size = 24, margin = margin(t = 15))
  )
print(p2)
ggsave("fig9_emotion_categorization_psychometric_slope_4cond_nolabel.jpg",
       p2, width = 6, height = 6, dpi = 300)

# ============================================================
# 9) Quick sanity checks (optional)
# ============================================================
print(datac_PSE)
print(datac_slope)





