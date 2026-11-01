######
# Experiment 2 Step 2: Judgment Task — Model Fitting (Continuous Group Only)
#
# Analysis pipeline:
#   1. Setup
#   2. Plotting theme
#   3. Load and prepare data
#   4. Calculate "sad" response proportions (continuous group only)
#   5. Define model fitting functions (logistic / linear / step; AIC selection)
#   6. Group-level model fitting and visualization (continuous group only)
#   7. Individual-level model fitting
#   8. Individual judgment curve plots
#   9. Paper figures (group curve + model comparison)
#  10. Save results
#
# Key design notes:
#   - Only the continuous group completed the judgment task.
#     The control group has no judgment data and is not involved in this step.
#   - Coordinate transform: emotion_scaled = (morph% - 50) / 10
#   - Model selection by AIC; boundary & slope always extracted from logistic fit
#   - group_fitting_results.csv   → read by Step 4 (XAB boundary classification)
#   - individual_fitting_results.csv → read by Step 3 (QC & descriptives)
######


#===== 1. Setup =====

rm(list = ls())

library(tidyverse)
library(minpack.lm)
library(broom)
library(scales)
library(Rmisc)
library(conflicted)

conflicted::conflicts_prefer(dplyr::summarise)
conflicted::conflicts_prefer(dplyr::rename)
conflicted::conflicts_prefer(dplyr::mutate)
conflicted::conflicts_prefer(dplyr::filter)
conflicted::conflicts_prefer(dplyr::select)

set.seed(123)
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
projectDir <- normalizePath(file.path(script_dir, "..", "..", ".."))
dataDir     <- file.path(projectDir, "data", "processed", "exp02")
resultsDir  <- file.path(projectDir, "results", "exp02", "judgment")
paperFigDir <- file.path(resultsDir, "paper_figures")

dir.create(file.path(resultsDir, "individual_plots", "continuous"), showWarnings = FALSE, recursive = TRUE)
dir.create(paperFigDir, showWarnings = FALSE, recursive = TRUE)

source(file.path(script_dir, "package_preference_config.R"))
setwd(resultsDir)
cat("Working directory:", getwd(), "\n\n")


#===== 2. Colors =====

color_continuous <- "#7FB3FF"
color_logistic   <- "#FF7F0E"
color_linear     <- "#1F77B4"
color_step       <- "#2CA02C"


#===== 3. Load and Prepare Data =====

combined_data <- read_csv(file.path(dataDir, "exp02_trial_data.csv"), show_col_types = FALSE)
cat(sprintf("Loaded %d rows, %d subjects\n\n", nrow(combined_data), n_distinct(combined_data$subject_id)))

id_data <- combined_data %>%
  filter(!is.na(judgment_rt), judgment_condition == "continuous") %>%
  select(subject_id, judgment_condition, emotion_level, face_identity,
         face_gender, block_id, trial_id, judgment_response, judgment_rt)

cat(sprintf("Judgment trials (continuous group): %d trials, %d subjects\n\n",
            nrow(id_data), n_distinct(id_data$subject_id)))


#===== 4. Calculate "Sad" Response Proportions =====

# Subject x morph level summary (judgment_response is a continuous rating)
id_summary <- id_data %>%
  group_by(subject_id, judgment_condition, emotion_level) %>%
  summarise(
    n_trials     = n(),
    sad_response = mean(judgment_response, na.rm = TRUE),
    mean_rt      = mean(judgment_rt,   na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    emotion_numeric = as.numeric(as.character(emotion_level)),
    emotion_scaled  = (emotion_numeric - 50) / 10
  )

cat("Sad response proportions computed (continuous rating, scaled).\n")
cat(sprintf("Sample (first 11 rows):\n"))
print(head(id_summary, 11))
cat("\n")


#===== 5. Model Fitting Functions =====

my_logistic_fun   <- function(x, p1, p2) 1 / (1 + exp(-(p1 + p2 * x)))
my_linear_fun     <- function(x, p1, p2) p1 * x + p2
my_step_fun       <- function(x, p1, p2) ifelse(x < 0, p1, p2)
rss_fun           <- function(p, x, y, fun) sum((y - fun(x, p[1], p[2]))^2)
calc_AIC_from_RSS <- function(rss, n, k) n * log(rss / n) + 2 * k


# --- 5.1 Group-level: fit all three models, return best by AIC ---
fit_three_models_group <- function(data, x_var = "emotion_scaled", y_var = "mean_sad") {
  
  x <- data[[x_var]]; y <- data[[y_var]]; n <- length(y); k <- 2
  
  safe_nls <- function(formula, data, start) {
    tryCatch(nlsLM(formula, data = data, start = start,
                   control = nls.lm.control(maxiter = 200)), error = function(e) NULL)
  }
  
  logi_fit <- safe_nls(y ~ my_logistic_fun(x, p1, p2), data.frame(x = x, y = y), list(p1 = 0, p2 = 1))
  logistic <- if (!is.null(logi_fit)) {
    rss <- sum(residuals(logi_fit)^2)
    list(converged = TRUE, p1 = as.numeric(coef(logi_fit)["p1"]), p2 = as.numeric(coef(logi_fit)["p2"]),
         aic = calc_AIC_from_RSS(rss, n, k), rss = rss,
         r_squared = 1 - rss / sum((y - mean(y))^2), model_type = "logistic", fit_obj = logi_fit)
  } else list(converged = FALSE, aic = Inf, rss = Inf, model_type = "logistic")
  
  lin_fit <- safe_nls(y ~ my_linear_fun(x, p1, p2), data.frame(x = x, y = y), list(p1 = 0.1, p2 = 0.5))
  linear <- if (!is.null(lin_fit)) {
    rss <- sum(residuals(lin_fit)^2)
    list(converged = TRUE, p1 = as.numeric(coef(lin_fit)["p1"]), p2 = as.numeric(coef(lin_fit)["p2"]),
         aic = calc_AIC_from_RSS(rss, n, k), rss = rss,
         r_squared = 1 - rss / sum((y - mean(y))^2), model_type = "linear", fit_obj = lin_fit)
  } else list(converged = FALSE, aic = Inf, rss = Inf, model_type = "linear")
  
  step_opt <- optim(par = c(0, 1), fn = rss_fun, x = x, y = y, fun = my_step_fun)
  rss_s    <- step_opt$value
  step <- list(converged = TRUE, p1 = step_opt$par[1], p2 = step_opt$par[2],
               aic = calc_AIC_from_RSS(rss_s, n, k), rss = rss_s,
               r_squared = 1 - rss_s / sum((y - mean(y))^2), model_type = "step")
  
  all  <- list(logistic = logistic, linear = linear, step = step)
  best <- all[[which.min(sapply(all, `[[`, "aic"))]]
  list(logistic = logistic, linear = linear, step = step, best = best)
}


# --- 5.2 Individual-level: fit models, boundary/slope always from logistic ---
fit_individual_models <- function(data) {
  
  result <- list(best_model_type = NA, best_aic = NA,
                 logistic_converged = FALSE, linear_converged = FALSE, step_converged = FALSE,
                 boundary = NA, slope = NA, r_squared = NA, p1 = NA, p2 = NA)
  
  x <- data$emotion_scaled; y <- data$sad_response; n <- length(y); k <- 2
  
  logi_res <- tryCatch({
    m   <- nlsLM(y ~ 1 / (1 + exp(-(p1 + p2 * x))), start = list(p1 = 0, p2 = 1),
                 control = nls.lm.control(maxiter = 200))
    rss <- sum(residuals(m)^2)
    list(converged = TRUE, p1 = as.numeric(coef(m)["p1"]), p2 = as.numeric(coef(m)["p2"]),
         aic = calc_AIC_from_RSS(rss, n, k), r_squared = 1 - rss / sum((y - mean(y))^2))
  }, error = function(e) list(converged = FALSE, aic = Inf))
  
  lin_res <- tryCatch({
    m   <- nlsLM(y ~ p1 * x + p2, start = list(p1 = 0.1, p2 = 0.5),
                 control = nls.lm.control(maxiter = 200))
    rss <- sum(residuals(m)^2)
    list(converged = TRUE, aic = calc_AIC_from_RSS(rss, n, k))
  }, error = function(e) list(converged = FALSE, aic = Inf))
  
  step_opt <- optim(par = c(0, 1), fn = rss_fun, x = x, y = y, fun = my_step_fun)
  step_res <- list(converged = TRUE, aic = calc_AIC_from_RSS(step_opt$value, n, k))
  
  all_aic <- c(logistic = logi_res$aic, linear = lin_res$aic, step = step_res$aic)
  best_nm <- names(which.min(all_aic))
  
  if (logi_res$converged) {
    result$boundary  <- (-logi_res$p1 / logi_res$p2) * 10 + 50
    result$slope     <- logi_res$p2
    result$r_squared <- logi_res$r_squared
    result$p1        <- logi_res$p1
    result$p2        <- logi_res$p2
  }
  result$best_model_type    <- best_nm
  result$best_aic           <- all_aic[best_nm]
  result$logistic_converged <- logi_res$converged
  result$linear_converged   <- lin_res$converged
  result$step_converged     <- step_res$converged
  result
}


#===== 6. Group-Level Model Fitting =====

cat("\n=== GROUP-LEVEL MODEL FITTING ===\n\n")

group_data <- id_summary %>%
  group_by(judgment_condition, emotion_level) %>%
  summarise(
    n_subj          = n_distinct(subject_id),
    mean_sad        = mean(sad_response, na.rm = TRUE),
    sd_sad          = sd(sad_response,   na.rm = TRUE),
    se_sad          = sd_sad / sqrt(n_subj),
    emotion_numeric = first(emotion_numeric),
    emotion_scaled  = first(emotion_scaled),
    .groups = "drop"
  )

cat("Fitting continuous group...\n")
fit_result    <- fit_three_models_group(group_data)
group_results <- list(continuous = list(data = group_data, three_models = fit_result))

cat(sprintf("  AIC -- Logistic: %.2f  Linear: %.2f  Step: %.2f  -> Best: %s\n",
            fit_result$logistic$aic, fit_result$linear$aic,
            fit_result$step$aic, fit_result$best$model_type))

if (fit_result$best$model_type == "logistic" && fit_result$best$converged) {
  boundary_grp <- (-fit_result$best$p1 / fit_result$best$p2) * 10 + 50
  cat(sprintf("  Boundary: %.2f%%  Slope: %.3f\n", boundary_grp, fit_result$best$p2))
}
cat("\n")


#===== 7. Individual-Level Model Fitting =====

cat("=== INDIVIDUAL-LEVEL MODEL FITTING ===\n\n")

unique_subjects <- unique(id_summary$subject_id)
fitting_summary <- data.frame()

for (subj_id in unique_subjects) {
  subj_data <- id_summary %>% filter(subject_id == subj_id)
  fit       <- fit_individual_models(subj_data)
  fitting_summary <- bind_rows(fitting_summary, data.frame(
    subject_id         = subj_id,
    judgment_condition      = "continuous",
    best_model_type    = fit$best_model_type,
    best_aic           = fit$best_aic,
    logistic_converged = fit$logistic_converged,
    linear_converged   = fit$linear_converged,
    step_converged     = fit$step_converged,
    boundary           = fit$boundary,
    slope              = fit$slope,
    r_squared          = fit$r_squared,
    p1                 = fit$p1,
    p2                 = fit$p2,
    stringsAsFactors   = FALSE
  ))
}

cat(sprintf("Fitted %d subjects (%d with valid boundary)\n\n",
            nrow(fitting_summary), sum(!is.na(fitting_summary$boundary))))

model_dist <- fitting_summary %>% filter(!is.na(best_model_type)) %>% count(best_model_type)
cat("Best model distribution:\n"); print(model_dist); cat("\n")


#===== 8. Individual Logistic Curve Plots =====

cat("=== INDIVIDUAL PLOTS ===\n\n")

simple_logistic <- function(x, boundary, slope_scaled) {
  x_sc <- (x - 50) / 10
  p1   <- -slope_scaled * (boundary - 50) / 10
  1 / (1 + exp(-(p1 + slope_scaled * x_sc)))
}

for (subj_id in unique_subjects) {
  
  subj_result <- fitting_summary %>% filter(subject_id == subj_id)
  if (nrow(subj_result) == 0 || is.na(subj_result$boundary)) next
  
  boundary  <- subj_result$boundary
  slope     <- subj_result$slope
  r2        <- subj_result$r_squared
  best_type <- subj_result$best_model_type
  
  subj_points <- id_summary %>%
    filter(subject_id == subj_id) %>%
    select(x = emotion_scaled, y = sad_response)
  
  pred_data <- data.frame(x = seq(-5, 5, length.out = 200)) %>%
    mutate(y = simple_logistic(x * 10 + 50, boundary, slope))
  
  p <- ggplot() +
    geom_point(data = subj_points, aes(x = x, y = y),
               color = "#7FB3FF", alpha = 0.8, size = 3) +
    geom_line(data = pred_data, aes(x = x, y = y),
              color = "#F28E8E", linewidth = 1.5) +
    scale_x_continuous(
      name   = "Morph: % Sad",
      breaks = c(-5, -3, -1, 1, 3, 5),
      labels = c("0%", "20%", "40%", "60%", "80%", "100%"),
      expand = expansion(mult = c(0, 0.05))
    ) +
    scale_y_continuous(
      name   = "Response",
      breaks = c(0, 0.5, 1),
      limits = c(0, 1.1),
      expand = expansion(mult = c(0, 0))
    ) +
    annotate("text", x = -5.8, y = 0, label = "Happy", angle = 90,
             hjust = 0.5, vjust = -2.5, size = 5, fontface = "italic") +
    annotate("text", x = -5.8, y = 1, label = "Sad",   angle = 90,
             hjust = 0.5, vjust = -2.5, size = 5, fontface = "italic") +
    coord_cartesian(xlim = c(-5, 5), clip = "off") +
    theme_minimal(base_size = 18) +
    theme(
      plot.margin  = margin(t = 10, r = 10, b = 10, l = 10),
      axis.line.x  = element_line(color = "black", linewidth = 1),
      axis.line.y  = element_line(color = "black", linewidth = 1),
      axis.ticks   = element_line(color = "black"),
      axis.title.y = element_text(color = "black", size = 22, face = "bold", margin = margin(r = 15)),
      axis.title.x = element_text(color = "black", size = 22, face = "bold"),
      axis.text.y  = element_text(color = "black", margin = margin(r = 5)),
      axis.text.x  = element_text(color = "black", margin = margin(t = 10)),
      panel.grid   = element_blank(),
      plot.title   = element_text(hjust = 0.5, face = "bold")
    )
  
  ggsave(filename = file.path(resultsDir, "individual_plots", "continuous",
                              paste0(subj_id, "_judgment_curve.jpg")),
         plot = p, width = 4.6, height = 4, dpi = 150)
}

cat(sprintf("Saved %d individual plots\n\n", sum(!is.na(fitting_summary$boundary))))


#===== 9. Paper Figures =====

cat("=== PAPER FIGURES ===\n\n")

# --- Fig 6a: Group-level judgment curve (mean +/- SE) ---
p_group_curve <- ggplot() +
  geom_line(
    data = group_data,
    aes(x = emotion_scaled, y = mean_sad),
    color = color_continuous, linewidth = 1.25
  ) +
  geom_errorbar(
    data = group_data,
    aes(x = emotion_scaled, ymin = mean_sad - se_sad, ymax = mean_sad + se_sad),
    width = 0.40, linewidth = 0.5, color = "black"
  ) +
  geom_point(
    data = group_data,
    aes(x = emotion_scaled, y = mean_sad),
    color = color_continuous, size = 0.05, shape = 19
  ) +
  scale_x_continuous(
    name   = "Morph: % Sad",
    breaks = c(-5, -3, -1, 1, 3, 5),
    labels = c("0%", "20%", "40%", "60%", "80%", "100%"),
    limits = c(-5, 5.2),
    expand = expansion(mult = c(0, 0.05))
  ) +
  scale_y_continuous(name = "Response", breaks = c(0, 0.5, 1), limits = c(0, 1.05), expand = expansion(mult = 0)) +
  annotate("text", x = -2.5, y = 0, label = "Happy", angle = 90,
           hjust = 0.5, vjust = -9, size = 7, fontface = "italic", color = "black") +
  annotate("text", x = -2.5, y = 1, label = "Sad",   angle = 90,
           hjust = 0.5, vjust = -9, size = 7, fontface = "italic", color = "black") +
  coord_cartesian(clip = "off", xlim = c(-5, 5)) +
  theme_minimal(base_size = 22) +
  theme(
    aspect.ratio    = 0.8,
    legend.position = "none",
    legend.title    = element_blank(),
    axis.line       = element_line(color = "black", linewidth = 1.1),
    axis.ticks      = element_line(color = "black", linewidth = 1.1),
    panel.grid      = element_blank(),
    plot.margin     = margin(t = 30, r = 10, b = 10, l = 10),
    axis.title      = element_text(size = 20, face = "bold"),
    axis.title.y    = element_text(size = 26, face = "bold", margin = margin(r = 18), hjust = 0.5),
    axis.title.x    = element_text(size = 26, face = "bold", margin = margin(t = 15)),
    axis.text.y     = element_text(color = "black", margin = margin(r = 5)),
    axis.text.x     = element_text(color = "black", margin = margin(t = 13))
  )

ggsave(file.path(paperFigDir, "Fig6a_group_judgment_curve.jpg"),
       p_group_curve, width = 6, height = 5, dpi = 300)
cat("  Saved: Fig6a_group_judgment_curve.jpg\n")


# --- Fig 6b: Model comparison plot ---
make_model_comparison_plot <- function(group_type, fig_name) {
  
  subset <- group_results[[group_type]]$data
  fit    <- group_results[[group_type]]$three_models
  x_seq  <- seq(min(subset$emotion_scaled), max(subset$emotion_scaled), length.out = 200)
  
  logi_pred <- if (fit$logistic$converged) {
    eta <- fit$logistic$p1 + fit$logistic$p2 * x_seq
    1 / (1 + exp(-pmax(pmin(eta, 35), -35)))
  } else rep(NA_real_, length(x_seq))
  
  lin_pred  <- if (fit$linear$converged) {
    pmin(pmax(fit$linear$p1 * x_seq + fit$linear$p2, 0), 1)
  } else rep(NA_real_, length(x_seq))
  
  step_pred <- pmin(pmax(my_step_fun(x_seq, fit$step$p1, fit$step$p2), 0), 1)
  
  df_preds <- bind_rows(
    data.frame(emotion_scaled = x_seq, pred = logi_pred, Model = "Logistic"),
    data.frame(emotion_scaled = x_seq, pred = lin_pred,  Model = "Linear"),
    data.frame(emotion_scaled = x_seq, pred = step_pred, Model = "Step")
  ) %>% filter(is.finite(pred))
  
  p <- ggplot(subset, aes(x = emotion_scaled, y = mean_sad)) +
    geom_point(color = "gray40", size = 2, alpha = 0.85) +
    geom_line(data = df_preds, aes(y = pred, color = Model), linewidth = 1.2) +
    scale_x_continuous(
      name   = "Morph: % Sad",
      breaks = c(-5, -3, -1, 1, 3, 5),
      labels = c("0%", "20%", "40%", "60%", "80%", "100%"),
      limits = c(-5, 5.2),
      expand = expansion(mult = c(0, 0.05))
    ) +
    scale_y_continuous(name = "Response", breaks = c(0, 0.5, 1), limits = c(0, 1.1), expand = expansion(mult = 0)) +
    scale_color_manual(values = c("Logistic" = color_logistic, "Linear" = color_linear, "Step" = color_step)) +
    annotate("text", x = -2.5, y = 0, label = "Happy", angle = 90,
             hjust = 0.5, vjust = -8.5, size = 6, fontface = "italic", color = "black") +
    annotate("text", x = -2.5, y = 1, label = "Sad",   angle = 90,
             hjust = 0.5, vjust = -8.5, size = 6, fontface = "italic", color = "black") +
    coord_cartesian(clip = "off") +
    theme_minimal(base_size = 18) +
    theme(
      legend.position    = "top",
      legend.box.spacing = unit(-10, "pt"),
      legend.title       = element_blank(),
      axis.line.x        = element_line(color = "black", linewidth = 1),
      axis.line.y        = element_line(color = "black", linewidth = 1),
      axis.ticks         = element_line(color = "black"),
      panel.grid         = element_blank(),
      plot.margin        = margin(t = 10, r = 10, b = 10, l = 10),
      axis.title         = element_text(size = 20, face = "bold"),
      axis.title.y       = element_text(size = 22, face = "bold", margin = margin(r = 15)),
      axis.title.x       = element_text(size = 22, face = "bold"),
      axis.text.x        = element_text(color = "black", margin = margin(t = 10)),
      axis.text.y        = element_text(color = "black", margin = margin(r = 5))
    )
  
  ggsave(file.path(paperFigDir, fig_name), p, width = 4.6, height = 4.0, dpi = 150)
  cat(sprintf("  Saved: %s\n", fig_name))
}

make_model_comparison_plot("continuous", "Fig6b_model_comparison_continuous.jpg")
cat("\nNote: Fig 6c (boundary) and Fig 6d (slope) are generated in Step 3.\n\n")


#===== 10. Save Results =====

cat("=== SAVING RESULTS ===\n\n")

best <- fit_result$best

group_fitting_results <- data.frame(
  judgment_condition      = "continuous",
  best_model_type    = best$model_type,
  best_aic           = best$aic,
  logistic_aic       = fit_result$logistic$aic,
  linear_aic         = fit_result$linear$aic,
  step_aic           = fit_result$step$aic,
  logistic_converged = fit_result$logistic$converged,
  linear_converged   = fit_result$linear$converged,
  step_converged     = fit_result$step$converged,
  boundary           = if (best$converged && best$model_type == "logistic")
    (-best$p1 / best$p2) * 10 + 50 else NA_real_,
  slope              = if (best$converged && best$model_type == "logistic")
    best$p2 else NA_real_,
  r_squared          = best$r_squared,
  p1                 = best$p1,
  p2                 = best$p2,
  stringsAsFactors   = FALSE
)

cat("Group-level fitting results:\n"); print(group_fitting_results); cat("\n")

write_csv(fitting_summary,       file.path(resultsDir, "individual_fitting_results.csv"))
write_csv(group_fitting_results, file.path(resultsDir, "group_fitting_results.csv"))

cat("  Saved: individual_fitting_results.csv  <- read by Step 3\n")
cat("  Saved: group_fitting_results.csv       <- read by Step 4\n\n")
cat("=== STEP 2 COMPLETE ===\n")
