######
# Step 2: Judgment Task — Model Fitting
#
# Analysis pipeline:
#   1. Setup
#   2. Plotting theme
#   3. Load and prepare data
#   4. Calculate "sad" response proportions
#   5. Define model fitting functions (logistic / linear / step; AIC selection)
#   6. Group-level model fitting and visualization
#   7. Individual-level model fitting
#   8. Individual judgment curve plots
#   9. Paper figures (group curve + model comparison)
#  10. Save results
#
# Key design notes:
#   - Coordinate transform: emotion_scaled = (morph% - 50) / 10
#   - Categorical group: GLM (binomial logit); boundary & slope always from logistic
#   - Continuous group:  nlsLM for logistic/linear, optim for step; AIC selects best model,
#                        but boundary & slope always extracted from logistic fit
#   - group_fitting_results.csv → read by Step 4 (XAB boundary classification)
#   - individual_fitting_results.csv → read by Step 3 (QC & descriptives)
#
# Last updated: 2026/2/7


# Step 2: Judgment Task — Model Fitting
# Last updated: 2026/2/7


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
conflicted::conflicts_prefer(dplyr::count)

set.seed(123)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
projectDir <- normalizePath(file.path(script_dir, "..", "..", ".."))
dataDir     <- file.path(projectDir, "data", "processed", "exp01")
resultsDir  <- file.path(projectDir, "results", "exp01", "judgment")
paperFigDir <- file.path(resultsDir, "paper_figures")

dir.create(paperFigDir, showWarnings = FALSE, recursive = TRUE)
setwd(resultsDir)
cat("Working directory:", getwd(), "\n\n")


#===== 2. Colors =====

color_categorical <- "#F28E8E"
color_continuous  <- "#7FB3FF"
color_logistic    <- "#FF7F0E"
color_linear      <- "#1F77B4"
color_step        <- "#2CA02C"


#===== 3. Load and Prepare Data =====

combined_data <- read_csv(file.path(dataDir, "exp01_trial_data.csv"), show_col_types = FALSE)
cat(sprintf("Loaded %d rows, %d subjects\n\n", nrow(combined_data), n_distinct(combined_data$subject_id)))

id_data <- combined_data %>%
  filter(!is.na(judgment_rt)) %>%
  select(subject_id, judgment_condition, emotion_level, face_identity,
         face_gender, block_id, trial_id, judgment_response, judgment_rt)

cat(sprintf("Judgment trials: %d total (%d categorical, %d continuous)\n\n",
            nrow(id_data),
            sum(id_data$judgment_condition == "categorical"),
            sum(id_data$judgment_condition == "continuous")))


#===== 4. Calculate "Sad" Response Proportions =====

# Subject x morph level summaries
# Categorical: judgment_response == 0 -> "sad"; Continuous: resp is continuous rating
id_summary <- id_data %>%
  group_by(subject_id, judgment_condition, emotion_level) %>%
  summarise(
    n_trials     = n(),
    sad_response = ifelse(first(judgment_condition) == "categorical",
                          mean(judgment_response == 0, na.rm = TRUE),
                          mean(judgment_response,      na.rm = TRUE)),
    mean_rt      = mean(judgment_rt, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    emotion_numeric = as.numeric(as.character(emotion_level)),
    emotion_scaled  = (emotion_numeric - 50) / 10
  )

# Trial-level version with scaling (used for categorical GLM fitting)
id_data_with_scaled <- id_data %>%
  mutate(
    emotion_numeric = as.numeric(as.character(emotion_level)),
    emotion_scaled  = (emotion_numeric - 50) / 10
  )


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
fit_individual_models <- function(data, judgment_condition) {
  
  result <- list(best_model_type = NA, best_aic = NA,
                 logistic_converged = FALSE, linear_converged = FALSE, step_converged = FALSE,
                 boundary = NA, slope = NA, r_squared = NA, p1 = NA, p2 = NA)
  
  if (judgment_condition == "categorical") {
    tryCatch({
      data$y_binary <- as.numeric(data$judgment_response == 0)
      m  <- glm(y_binary ~ emotion_scaled, data = data, family = binomial(link = "logit"))
      m0 <- glm(y_binary ~ 1,             data = data, family = binomial)
      p1 <- as.numeric(coef(m)[1]); p2 <- as.numeric(coef(m)[2])
      result$logistic_converged <- TRUE
      result$best_model_type    <- "logistic"
      result$best_aic           <- AIC(m)
      result$boundary           <- (-p1 / p2) * 10 + 50
      result$slope              <- p2
      result$r_squared          <- as.numeric(1 - logLik(m)[1] / logLik(m0)[1])
      result$p1 <- p1; result$p2 <- p2
    }, error = function(e) message("GLM fitting failed: ", conditionMessage(e)))
    
  } else {
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
  }
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

group_results <- list()

for (grp in c("categorical", "continuous")) {
  cat(sprintf("Fitting %s group...\n", grp))
  subset <- group_data %>% filter(judgment_condition == grp)
  fit    <- fit_three_models_group(subset)
  group_results[[grp]] <- list(data = subset, three_models = fit)
  cat(sprintf("  AIC -- Logistic: %.2f  Linear: %.2f  Step: %.2f  -> Best: %s\n",
              fit$logistic$aic, fit$linear$aic, fit$step$aic, fit$best$model_type))
  if (fit$best$model_type == "logistic" && fit$best$converged) {
    boundary <- (-fit$best$p1 / fit$best$p2) * 10 + 50
    cat(sprintf("  Boundary: %.2f%%  Slope: %.3f\n", boundary, fit$best$p2))
  }
  cat("\n")
}


#===== 7. Individual-Level Model Fitting =====

cat("=== INDIVIDUAL-LEVEL MODEL FITTING ===\n\n")

unique_subjects <- unique(id_summary$subject_id)
fitting_summary <- data.frame()

for (subj_id in unique_subjects) {
  
  judgment_condition <- id_summary %>% filter(subject_id == subj_id) %>% pull(judgment_condition) %>% first()
  
  subj_data <- if (judgment_condition == "categorical") {
    id_data_with_scaled %>% filter(subject_id == subj_id)
  } else {
    id_summary %>% filter(subject_id == subj_id)
  }
  
  fit <- fit_individual_models(subj_data, judgment_condition)
  
  fitting_summary <- bind_rows(fitting_summary, data.frame(
    subject_id         = subj_id,
    judgment_condition      = judgment_condition,
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

model_dist <- fitting_summary %>% filter(!is.na(best_model_type)) %>% count(judgment_condition, best_model_type)
cat("Best model distribution:\n"); print(model_dist); cat("\n")


#===== 8. Individual Judgment Curve Plots =====

cat("=== INDIVIDUAL PLOTS ===\n\n")

dir.create(file.path(resultsDir, "individual_plots", "categorical"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(resultsDir, "individual_plots", "continuous"),  showWarnings = FALSE, recursive = TRUE)

simple_logistic <- function(x, p1, p2) {
  x_sc <- (x - 50) / 10
  1 / (1 + exp(-(p1 + p2 * x_sc)))
}

for (subj_id in unique_subjects) {
  
  subj_result <- fitting_summary %>% filter(subject_id == subj_id)
  if (nrow(subj_result) == 0 || is.na(subj_result$boundary)) next
  
  judgment_condition <- subj_result$judgment_condition
  p1            <- subj_result$p1
  p2            <- subj_result$p2
  r2            <- subj_result$r_squared
  best_type     <- subj_result$best_model_type
  
  subj_points <- id_summary %>%
    filter(subject_id == subj_id) %>%
    select(x = emotion_scaled, y = sad_response)
  
  pred_data <- data.frame(x = seq(-5, 5, length.out = 200)) %>%
    mutate(y = simple_logistic(x * 10 + 50, p1, p2))
  
  # ↓ 以下内容移入循环体内
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
  
  ggsave(filename = file.path(resultsDir, "individual_plots", judgment_condition,
                              paste0(subj_id, "_judgment_curve.jpg")),
         plot = p, width = 4.6, height = 4, dpi = 150)
}  # ← for 循环在此正确关闭

cat(sprintf("Saved %d individual plots\n\n", sum(!is.na(fitting_summary$boundary))))

#===== 9. Paper Figures =====

cat("=== PAPER FIGURES ===\n\n")

# --- Fig 4a: Group-level judgment curves (mean +/- SE) ---
p_group_curve <- ggplot() +
  geom_line(
    data = group_data,
    aes(x = emotion_scaled, y = mean_sad, color = judgment_condition, group = judgment_condition),
    linewidth = 1.25, position = position_dodge(width = 0.4)
  ) +
  geom_errorbar(
    data = group_data,
    aes(x = emotion_scaled, ymin = mean_sad - se_sad, ymax = mean_sad + se_sad, group = judgment_condition),
    position = position_dodge(width = 0.35), width = 0.40, linewidth = 0.5, color = "black"
  ) +
  geom_point(
    data = group_data,
    aes(x = emotion_scaled, y = mean_sad, fill = judgment_condition, color = judgment_condition),
    size = 0.05, shape = 19, position = position_dodge(width = 0.4)
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
  scale_color_manual(values = c("categorical" = color_categorical, "continuous" = color_continuous),
                     labels = c("categorical" = "Categorical", "continuous" = "Continuous")) +
  scale_fill_manual(values  = c("categorical" = color_categorical, "continuous" = color_continuous),
                    labels  = c("categorical" = "Categorical", "continuous" = "Continuous")) +
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

ggsave(file.path(paperFigDir, "Fig4a_group_judgment_curve.jpg"),
       p_group_curve, width = 6, height = 5, dpi = 300)
cat("  Saved: Fig4a_group_judgment_curve.jpg\n")


# --- Fig 4b/4c: Model comparison plots (one per group) ---
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

make_model_comparison_plot("categorical", "Fig4b_model_comparison_categorical.jpg")
make_model_comparison_plot("continuous",  "Fig4c_model_comparison_continuous.jpg")
cat("\nNote: Fig 4d (boundary) and Fig 4e (slope) are generated in Step 3.\n\n")


#===== 10. Save Results =====

cat("=== SAVING RESULTS ===\n\n")

group_fitting_results <- map_dfr(c("categorical", "continuous"), function(grp) {
  fit  <- group_results[[grp]]$three_models
  best <- fit$best
  data.frame(
    judgment_condition      = grp,
    best_model_type    = best$model_type,
    best_aic           = best$aic,
    logistic_aic       = fit$logistic$aic,
    linear_aic         = fit$linear$aic,
    step_aic           = fit$step$aic,
    logistic_converged = fit$logistic$converged,
    linear_converged   = fit$linear$converged,
    step_converged     = fit$step$converged,
    boundary           = if (best$converged && best$model_type == "logistic")
      (-best$p1 / best$p2) * 10 + 50 else NA_real_,
    slope              = if (best$converged && best$model_type == "logistic")
      best$p2 else NA_real_,
    r_squared          = best$r_squared,
    p1                 = best$p1,
    p2                 = best$p2,
    stringsAsFactors   = FALSE
  )
})

cat("Group-level fitting results:\n"); print(group_fitting_results); cat("\n")

write_csv(fitting_summary,       file.path(resultsDir, "individual_fitting_results.csv"))
write_csv(group_fitting_results, file.path(resultsDir, "group_fitting_results.csv"))

cat("  Saved: individual_fitting_results.csv  <- read by Step 3\n")
cat("  Saved: group_fitting_results.csv       <- read by Step 4\n\n")
cat("=== STEP 2 COMPLETE ===\n")

