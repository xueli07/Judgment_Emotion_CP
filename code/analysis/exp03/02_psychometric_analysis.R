rm(list = ls())

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(getwd())
package_root <- normalizePath(file.path(script_dir, "..", ".."))
dataDir <- file.path(package_root, "data", "exp03")
analysisDir <- file.path(package_root, "results", "exp03")
dir.create(analysisDir, recursive = TRUE, showWarnings = FALSE)
setwd(analysisDir)

library(readxl)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(dplyr)
library(RColorBrewer)
scriptDir <- dirname(rstudioapi::getSourceEditorContext()$path)
source(paste(scriptDir, "helper_functions.R",sep = "\\"))

plot_color = c( "#F28E8E", "#7FB3FF")
##################################
# Load data

data_final <- read.csv(file.path(dataDir, "exp03_trial_data.csv"))
# Center emotion_level variable (and divide by 10 to make units more meaningful)
data_final$emotion_level <-(data_final$emotion_level-50)/10 
data_final$participant = as.factor(data_final$participant)
# Code and center condition variable (categorical = -.5, continuous = .5)
data_final$block_name = ifelse(data_final$block_name == "categorical", -.5, .5)

data_final <- data_final %>%
  mutate(face_id = sub(".*/(m_[^_/]+)_.*", "\\1", face_stim))

# --- Aggregate across participants
temp1 <- aggregate(resp ~ participant + block_name + emotion_level, data = data_final, FUN = "mean")
temp2 <- aggregate(RT ~ participant + block_name + emotion_level, data = data_final, FUN = "mean")

data_final_subj <- data.frame(temp1, temp2$RT)
colnames(data_final_subj)[ncol(data_final_subj)] <- c("rt")

data_final_subj <- data_final_subj[order(data_final_subj$participant, 
                                         data_final_subj$block_name, 
                                         data_final_subj$emotion_level),]


# --- Aggregate across stimuli
temp1 <- aggregate(resp ~ face_stim + block_name + emotion_level, data = data_final, FUN = "mean")
temp2 <- aggregate(RT ~ face_stim + block_name + emotion_level, data = data_final, FUN = "mean")

data_final_stim <- data.frame(temp1, temp2$RT)
colnames(data_final_stim)[ncol(data_final_stim)] <- c("rt")

data_final_stim <- data_final_stim[order(data_final_stim$face_stim, 
                                         data_final_stim$block_name, 
                                         data_final_stim$emotion_level),]

##################################
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(purrr)

# Extract unique participants
participants <- unique(data_final$participant)

# Initialize an empty dataframe to store results
results <- data.frame(participant = character(),
                      condition = character(),
                      PSE = numeric(),
                      slope = numeric(),
                      stringsAsFactors = FALSE)

# Define plotting function
plot_psychometric <- function(data, model, condition_type, participant_id) {
  # Generate prediction data
  pred_data <- data.frame(
    emotion_level = seq(min(data$emotion_level), 
                        max(data$emotion_level), 
                        length.out = 100)
  )
  
  # Get predicted values
  if (condition_type == "categorical") {
    pred_data$pred <- predict(model, newdata = pred_data, type = "response")
  } else {
    pred_data$pred <- predict(model, newdata = pred_data)
  }
  
  # Create base plot
  ggplot() +
    # Raw data points (with horizontal jitter)
    geom_point(
      data = data,
      aes(x = emotion_level, y = resp),
      position = position_jitter(width = 0.3, height = 0.02),
      color = plot_color[2],
      alpha = 0.4,
      size = 2
    ) +
    # Model fitted curve
    geom_line(
      data = pred_data,
      aes(x = emotion_level, y = pred),
      color = plot_color[1],
      linewidth = 1.5
    ) +
    # Axis settings
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
    # Text annotations
    annotate("text", x = -4, y = 0, label = "Calm",
             angle = 90, hjust = 0.8, vjust = -7,
             size = 5, fontface = "italic",color = "black") +
    annotate("text", x = -4, y = 1, label = "Fear",
             angle = 90, hjust = 0.2, vjust = -7,
             size = 5, fontface = "italic",color = "black") +
    # Theme settings
    theme_minimal(base_size = 18) +
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
  
  ggsave(paste0("fig5_onebyone\\fig5_emotion_categorization_", condition_type, "_",
                participant_id,".jpg"), width = 4.6, height = 4, dpi = 150)
  
}



# Loop over each participant
for (p in participants) {
  # Subset data for the current participant
  df <- data_final %>% filter(participant == p)
  
  # Split data by condition
  categorical_data <- df %>% filter(block_name == -0.5)
  continuous_data <- df %>% filter(block_name == 0.5)
  
  ### Logistic regression for categorical condition ###
  logit_model <- glm(resp ~ emotion_level, data = categorical_data, 
                     family = binomial(link = "logit"))
  logit_params <- coef(logit_model)
  logit_slope <- logit_params["emotion_level"]
  logit_PSE <- -logit_params["(Intercept)"] / logit_slope
  
  # Store categorical condition results
  results <- rbind(results, data.frame(participant = p,
                                       condition = "categorical",
                                       PSE = logit_PSE,
                                       slope = logit_slope))
  
  ### Nonlinear regression for continuous condition ###
  nls_model <- nls(resp ~ 1 / (1 + exp(-(b0 + b1 * emotion_level))),
                   data = continuous_data, start = list(b0 = 0, b1 = 1))
  
  if (!is.null(nls_model)) {
    nls_params <- coef(nls_model)
    nls_PSE <- -nls_params["b0"] / nls_params["b1"]
    nls_slope <- nls_params["b1"]
  } else {
    nls_PSE <- NA
    nls_slope <- NA
  }
  
  plot_psychometric(categorical_data, logit_model, 
                    "categorical", participant_id=p)
  plot_psychometric(continuous_data, nls_model, 
                    "continuous", participant_id=p)
  
  # Store continuous condition results
  results <- rbind(results, data.frame(participant = p,
                                       condition = "continuous",
                                       PSE = nls_PSE,
                                       slope = nls_slope))
  
  # Debugging print statements
  print(paste("Processed participant:", p, 
              "Categorical PSE:", logit_PSE, "Slope:", logit_slope,
              "Continuous PSE:", nls_PSE, "Slope:", nls_slope))
}


## Next: Comparison of PSE and slope

# Remove NA values before comparisons
results_clean <- results %>% filter(!is.na(PSE) & !is.na(slope))

# Compare PSE and slope between conditions using paired t-tests
library(bruceR)
df_wide <- results_clean  %>%
  pivot_wider(
    id_cols = participant,     
    names_from = condition,     
    values_from = c(PSE, slope) 
  )


pse_comparison <- TTEST(df_wide ,y=c("PSE_continuous","PSE_categorical"),paired =TRUE)
slope_comparison <- TTEST(df_wide ,y=c("slope_continuous","slope_categorical"),paired =TRUE)
#pse_comparison <- t.test(PSE ~ condition, data = results_clean, paired = TRUE)
#pse_comparison <- t.test(PSE ~ condition, data = results_clean)
#slope_comparison <- t.test(slope ~ condition, data = results_clean, paired = TRUE)
#slope_comparison <- t.test(slope ~ condition, data = results_clean)
# Print results
print(pse_comparison)
print(slope_comparison)

# library(lsr)
# cohensD(PSE ~ condition, method = "paired", data = results_clean)
# # Calculate Cohen's d for PSE comparison
# cohens_d_pse <- pse_comparison$statistic / sqrt(pse_comparison$parameter + 1)
# cat("Cohen's d (PSE):", cohens_d_pse, "\n")
# 
# # Calculate Cohen's d for slope comparison
# cohens_d_slope <- slope_comparison$statistic / sqrt(slope_comparison$parameter + 1)
# cat("Cohen's d (slope):", cohens_d_slope, "\n")
# 
# 
datac_PSE <- summarySE(data=results_clean, measurevar="PSE", groupvars=c("condition"), na.rm=FALSE,
                        conf.interval=.95, .drop=TRUE)

ggplot(datac_PSE, aes(x = "", y = PSE, fill = condition)) +
  # 1. Decrease geom_col width (narrow the bars)
  # 2. Decrease position_dodge width (keep narrowed bars close together)
  geom_col(position = position_dodge(width = 1.0), width = 0.4) +
  
  # 3. Synchronize position_dodge for error bars to align exactly with bar centers!
  geom_errorbar(aes(ymin = PSE - se, ymax = PSE + se),
                position = position_dodge(width = 1.0), 
                width = 0.2, # Slightly narrow the error bar width to match thinner bars
                linewidth = 1) +
  geom_hline(yintercept = 0, color = "black", size = 1.2) +
  scale_fill_manual(
    values = plot_color,
    labels = c("Categorical", "Continuous")
  ) +
  theme_classic(base_size = 24) +
  labs(y = "PSE", x = "") + # Explicitly set x-axis label to empty
  scale_y_continuous(
    breaks = seq(-1, 0, by = 0.5),
    limits = c(-1.35,0.2)
  ) +
  #coord_cartesian(ylim = c(-1, 0)) +
  theme(
    aspect.ratio = 0.9,
    legend.position = "none",
    legend.box.spacing = unit(-5, "pt"),
    axis.line.x = element_blank(),
    #axis.line.x = element_line(color = "black", linewidth = 1.2),
    axis.line.y = element_line(color = "black", linewidth = 1.2),
    axis.ticks.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    legend.title = element_blank(),
    legend.box.margin = margin(t = 0, b = 0, l = 0, r = 0), # Adjust legend margins
    plot.margin = margin(t = 30, b = 1, l = 10, r = 10),
    axis.title = element_text(size = 26, face = "bold"),
    axis.title.y = element_text(size = 26, face = "bold", margin = margin(r = 15)),
    #axis.title.x = element_text(size = 20, face = "bold")
    # Adjust overall plot 
  )


ggsave(paste0("fig6_emotion_categorization_psychometric_PSE_nolabel.jpg"), width = 6, height = 6, dpi = 300)


datac_slope <- summarySE(data=results_clean, measurevar="slope", groupvars=c("condition"), na.rm=FALSE,
                         conf.interval=.95, .drop=TRUE)

ggplot(datac_slope, aes(x = "", y = slope, fill = condition)) +
  geom_col(position = position_dodge(width = 1.0), width = 0.4) +
  geom_errorbar(aes(ymin = slope - se, ymax = slope + se),
                position = position_dodge(width = 1.0), 
                width = 0.2,
                linewidth = 1) +
  #geom_hline(yintercept = 0, color = "black", size = 1.2) +
  scale_fill_manual(
    values = plot_color,
    labels = c("Categorical", "Continuous")
  ) +
  theme_classic(base_size = 24) +
  labs(y = "Slope", x = "") + 
  scale_y_continuous(breaks = 0:3, expand = expansion(mult = 0),minor_breaks = seq(0.5, 2.5, by = 0.5),limits = c(0, 3.5))+# Explicitly set x-axis label to empty
  #coord_cartesian(ylim = c(0, 2.5)) +
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
    legend.box.margin = margin(t = 0, b = 0, l = 0, r = 0), # Adjust legend margins
    plot.margin = margin(t = 30, b = 1, l = 10, r = 10),
    axis.title = element_text(size = 20, face = "bold"),
    axis.title.y = element_text(size = 26, face = "bold", margin = margin(r = 15)),
    axis.title.x = element_text(size = 26, face = "bold")
    # Adjust overall plot 
  )

ggsave(paste0("fig6_emotion_categorization_psychometric_slope_nolabel.jpg"), width = 6, height = 6, dpi = 300)
