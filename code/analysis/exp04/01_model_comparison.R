rm(list = ls())

# Set current directory
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
package_root <- normalizePath(file.path(script_dir, "..", ".."))
dataDir <- file.path(package_root, "data", "exp04")
analysisDir <- file.path(package_root, "results", "exp04")
dir.create(analysisDir, recursive = TRUE, showWarnings = FALSE)
setwd(analysisDir)

plot_color = c( "#F28E8E", "#7FB3FF")
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

##################################
# Load data
data_final <- read.csv(file.path(dataDir, "exp04_trial_data.csv"), stringsAsFactors = FALSE)
str(data_final)

##################################
# Preprocess
data_final <- data_final %>%
  dplyr::mutate(
    # Scale emotion level: 10% -> -4, 90% -> 4
    emotion_level = (emotion_level - 50) / 10,           
    participant   = as.factor(participant),
    
    # Robust coding for task types (adapt to your real labels)
    thinking = ifelse(tolower(thinking) == "categorical", -0.5, 0.5),
    # Replace 'anticipation' with 'task_awareness' and map values to numeric coding
    task_awareness = ifelse(tolower(task_awareness) == "informed", -0.5, 0.5),
    
    # Labels for plotting
    thinking_label = ifelse(thinking == -0.5, "Categorical", "Continuous"),
    # Replace labels: -0.5 is 'Informed' (block), 0.5 is 'Uninformed' (mix)
    task_awareness_label = ifelse(task_awareness == -0.5, "Informed", "Uninformed"),
    
    # Extract face ID (optional)
    face_id = sub(".*/(m_[^_/]+)_.*", "\\1", face_stim)
  )

##################################
##################################

# Force 'resp' to be numeric
data_final <- data_final %>%
  dplyr::mutate(resp = as.numeric(resp))

# Participant means (average for each subject under each condition x emotion level)
data_subj <- data_final %>%
  dplyr::group_by(participant, thinking_label, task_awareness_label, emotion_level) %>%
  dplyr::summarise(
    resp = mean(resp, na.rm = TRUE),
    .groups = "drop"
  )

# Condition mean + SE across participants (Fix: avoid overwriting 'resp')
datac_fig2 <- data_subj %>%
  dplyr::group_by(thinking_label, task_awareness_label, emotion_level) %>%
  dplyr::summarise(
    n_subj    = dplyr::n_distinct(participant),
    n_obs     = sum(is.finite(resp)),
    mean_resp = mean(resp, na.rm = TRUE),
    sd_resp   = stats::sd(resp, na.rm = TRUE),
    se        = sd_resp / sqrt(n_obs),
    .groups   = "drop"
  ) %>%
  dplyr::transmute(
    thinking_label,
    task_awareness_label,
    emotion_level,
    n_subj,
    n_obs,
    resp = mean_resp,
    sd_resp,
    se,
    emotion_scaled = emotion_level / 2
  )

# Plot Figure 2: Emotion Categorization
p_fig2 <- ggplot(datac_fig2,
                 aes(x = emotion_scaled, y = resp,
                     color = thinking_label,
                     linetype = task_awareness_label,
                     group = interaction(thinking_label, task_awareness_label))) +
  
  # Errorbar and lines (stand out)
  geom_line(
    linewidth = 1.25,
    position = position_dodge(width = 0.35)
  ) +
  geom_errorbar(
    aes(ymin = resp - se, ymax = resp + se),
    width = 0.4,
    linewidth = 0.5,
    color = "black",
    linetype = "solid",
    position = position_dodge(width = 0.35)
  ) +
  geom_point(
    size = 0.05,
    position = position_dodge(width = 0.35)
  ) +
  
  # Axes settings
  scale_x_continuous(
    name = "Morph: % Fear",
    breaks = c(-2, -1, 0, 1, 2),
    labels = c("10%", "30%", "50%", "70%", "90%"),
    expand = expansion(mult = c(0.1, 0.05))
  ) +
  scale_y_continuous(
    name = "Response",
    breaks = c(0, 0.5, 1),
    limits = c(0, 1.05),
    expand = expansion(mult = 0)
  ) +
  
  # Annotations for 'Calm' and 'Fear'
  annotate("text",
           x = -2.5, y = 0, label = "Calm",
           angle = 90, hjust = 0.5, vjust = -3.5,
           size = 6, fontface = "italic", color = "black") +
  annotate("text",
           x = -2.5, y = 1, label = "Fear",
           angle = 90, hjust = 0.5, vjust = -3.5,
           size = 6, fontface = "italic", color = "black") +
  
  # Custom colors and line types mapping
  scale_color_manual(values = c("Categorical" = plot_color[1],
                                "Continuous"  = plot_color[2])) +
  scale_linetype_manual(values = c("Informed"   = "solid",
                                   "Uninformed" = "dotted"),
                        labels = c("Informed" = "Informed",
                                   "Uninformed" = "Uninformed")) +
  
  # Theme and appearance settings
  theme_minimal(base_size = 22) +
  theme(
    aspect.ratio = 0.8,
    legend.position = "none",
    legend.box.spacing = unit(10, "pt"),
    legend.title = element_blank(),
    axis.line = element_line(color = "black", linewidth = 1.2),
    axis.ticks = element_line(color = "black", linewidth = 1.2),
    panel.grid = element_blank(),
    plot.margin = margin(t = 30, r = 10, b = 10, l = 10),
    axis.title = element_text(size = 20, face = "bold"),
    axis.title.y = element_text(size = 22, face = "bold", margin = margin(r = 15)),
    axis.title.x = element_text(size = 22, face = "bold"),
    axis.text = element_text(color = "black"),
    legend.box = "vertical",
    legend.margin = margin(t = 2, b = 2),
    legend.key.height = unit(0.8, "lines"),
    legend.spacing.y = unit(1, "pt")
  ) +
  coord_cartesian(clip = "off", xlim = c(-2, 2)) +
  guides(
    color    = guide_legend(nrow = 1, byrow = TRUE),
    linetype = guide_legend(nrow = 1, byrow = TRUE)
  )

print(p_fig2)
ggsave("fig2_emotion_categorization_allsubject_errorbar.jpg",
       p_fig2, width = 6, height = 5, dpi = 300)

##################################
# Model comparison (Fig3–Fig6): aggregate across stimuli within each 2x2 condition
data_stim <- data_final %>%
  dplyr::group_by(face_stim, thinking, task_awareness, emotion_level) %>%
  dplyr::summarise(resp = mean(resp, na.rm = TRUE), .groups = "drop")

# Model functions
my_logistic_fun <- function(x, p1, p2) 1 / (1 + exp(-(p1 + p2 * x)))
my_linear_fun   <- function(x, p1, p2) p1 * x + p2
my_step_fun     <- function(x, p1, p2) ifelse(x < 0, p1, p2)

rss_fun <- function(p, x, y, fun) sum((y - fun(x, p[1], p[2]))^2)
calc_AIC_from_RSS <- function(rss, n, k) n * log(rss / n) + 2 * k

# Define condition map with updated task_awareness labels
cond_map <- tibble::tribble(
  ~thinking, ~task_awareness, ~cond_name, ~fig_no,
  -0.5,      -0.5,            "categorical_Informed",      3,
  0.5,      -0.5,            "continuous_Informed",       4,
  -0.5,       0.5,            "categorical_Uninformed",    5,
  0.5,       0.5,            "continuous_Uninformed",     6
)

# Function to fit models for a single condition
fit_one_condition <- function(df_cond, cond_name, fig_no) {
  
  # Safe non-linear least squares fitting function
  safe_nls <- function(formula, data, start) {
    tryCatch(
      nls(formula, data = data, start = start,
          control = nls.control(maxiter = 200, warnOnly = TRUE)),
      error = function(e) NULL
    )
  }
  
  # Model fitting
  logistic_fit <- safe_nls(resp ~ my_logistic_fun(emotion_level, p1, p2),
                           data = df_cond, start = list(p1 = 0, p2 = 1))
  linear_fit   <- safe_nls(resp ~ my_linear_fun(emotion_level, p1, p2),
                           data = df_cond, start = list(p1 = 0.1, p2 = 0.5))
  
  step_optim  <- optim(par = c(0.2, 0.8),
                       fn = rss_fun,
                       x = df_cond$emotion_level,
                       y = df_cond$resp,
                       fun = my_step_fun)
  step_params <- step_optim$par
  
  # Calculate AIC
  n <- nrow(df_cond); k <- 2
  rss_logistic <- if (!is.null(logistic_fit)) sum(resid(logistic_fit)^2) else NA_real_
  rss_linear   <- if (!is.null(linear_fit))   sum(resid(linear_fit)^2)   else NA_real_
  rss_step     <- rss_fun(step_params, df_cond$emotion_level, df_cond$resp, my_step_fun)
  
  AIC_logistic <- if (!is.na(rss_logistic)) calc_AIC_from_RSS(rss_logistic, n, k) else NA_real_
  AIC_linear   <- if (!is.na(rss_linear))   calc_AIC_from_RSS(rss_linear,   n, k) else NA_real_
  AIC_step     <- calc_AIC_from_RSS(rss_step, n, k)
  
  aic_table <- data.frame(
    condition = cond_name,
    Model = c("Logistic", "Linear", "Step"),
    AIC   = c(AIC_logistic, AIC_linear, AIC_step),
    RSS   = c(rss_logistic, rss_linear, rss_step),
    n     = n,
    stringsAsFactors = FALSE
  )
  
  # Generate predictions for plotting
  x_seq <- seq(min(df_cond$emotion_level), max(df_cond$emotion_level), length.out = 200)
  
  logistic_pred <- if (!is.null(logistic_fit)) {
    p1 <- coef(logistic_fit)["p1"]
    p2 <- coef(logistic_fit)["p2"]
    eta <- p1 + p2 * x_seq
    eta <- pmax(pmin(eta, 35), -35)
    1 / (1 + exp(-eta))
  } else rep(NA_real_, length(x_seq))
  
  linear_pred <- if (!is.null(linear_fit)) {
    p1 <- coef(linear_fit)["p1"]
    p2 <- coef(linear_fit)["p2"]
    pred <- p1 * x_seq + p2
    pmin(pmax(pred, 0), 1)
  } else rep(NA_real_, length(x_seq))
  
  step_pred <- my_step_fun(x_seq, step_params[1], step_params[2])
  step_pred <- pmin(pmax(step_pred, 0), 1)
  
  df_preds <- dplyr::bind_rows(
    data.frame(emotion_level = x_seq, pred = logistic_pred, Model = "Logistic"),
    data.frame(emotion_level = x_seq, pred = linear_pred,   Model = "Linear"),
    data.frame(emotion_level = x_seq, pred = step_pred,     Model = "Step")
  ) %>% dplyr::filter(is.finite(pred))
  
  
  
  # 模型比较的画图函数
  # Plotting the model fits
  p <- ggplot(df_cond, aes(x = emotion_level, y = resp)) +
    geom_point(color = "gray40", size = 2, alpha = 0.85) +
    geom_line(data = df_preds, aes(y = pred, color = Model), linewidth = 1.2) +
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
      expand = expansion(mult = 0)
    ) +
    scale_color_manual(values = c("#1F77B4", "#FF7F0E", "#2CA02C")) +
    theme_minimal(base_size = 18) +
    theme(
      legend.position = "top",
      legend.box.spacing = unit(-10, "pt"),
      legend.title = element_blank(),
      axis.line.x = element_line(color = "black", linewidth = 1),
      axis.line.y = element_line(color = "black", linewidth = 1),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black"),
      axis.text = element_text(color = "black"),
      panel.grid = element_blank(),
      plot.margin = margin(t = 10, r = 10, b = 10, l = 10), 
      axis.title = element_text(size = 20, face = "bold"),
      axis.title.y = element_text(size = 22, face = "bold", margin = margin(r = 15)),
      axis.title.x = element_text(size = 22, face = "bold")
    ) +
    coord_cartesian(clip = "off") +
    annotate("text", x = -4, y = 0, label = "Calm",  # Position left of x-axis start
             angle = 90, hjust = 0.8, vjust = -6.5, size = 5, fontface = "italic",color = "black") +
    annotate("text", x = -4, y = 1, label = "Fear",
             angle = 90, hjust = 0.3, vjust = -6.5, size = 5, fontface = "italic",color = "black")
  
  fig_name <- paste0("fig", fig_no, "_emotion_categorization_model_comparison_", cond_name, ".jpg")
  ggsave(fig_name, p, width = 4.6, height = 4.0, dpi = 150)
  
  list(aic_table = aic_table, plot = p)
}

##################################
# Run all conditions
all_aic <- data.frame()

for (i in 1:nrow(cond_map)) {
  
  th  <- cond_map$thinking[i]
  # Using the updated variable name 'task_awareness'
  ant <- cond_map$task_awareness[i] 
  cn  <- cond_map$cond_name[i]
  fn  <- cond_map$fig_no[i]
  
  df_cond <- data_stim %>% dplyr::filter(thinking == th, task_awareness == ant)
  
  out <- fit_one_condition(df_cond, cn, fn)
  all_aic <- dplyr::bind_rows(all_aic, out$aic_table)
  
  print(out$plot)
  print(out$aic_table)
}

# Export results
write.csv(all_aic, "AIC_model_comparison_all_conditions.csv", row.names = FALSE)

best_model <- all_aic %>%
  dplyr::group_by(condition) %>%
  dplyr::filter(AIC == min(AIC, na.rm = TRUE)) %>%
  dplyr::ungroup()

write.csv(best_model, "AIC_best_model_each_condition.csv", row.names = FALSE)

print(all_aic)
print(best_model)
