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

str(data_final)
data_final$participant = as.factor(data_final$participant)
data_final$block_name = as.factor(data_final$block_name)
data_final$emotion_level2 <-(data_final$emotion_level-50)/10 
data_final$emotion_level = as.factor(data_final$emotion_level)



# --- Aggregate across participants
temp1 <- aggregate(resp ~ participant + block_name + emotion_level, data = data_final, FUN = "mean")
temp2 <- aggregate(RT ~ participant + block_name + emotion_level, data = data_final, FUN = "mean")

data_final_subj <- data.frame(temp1, temp2$RT)
colnames(data_final_subj)[ncol(data_final_subj)] <- c("rt")

data_final_subj <- data_final_subj[order(data_final_subj$participant, 
                                         data_final_subj$block_name, 
                                         data_final_subj$emotion_level),]


# --- Aggregate across stimuli
temp1 <- aggregate(resp ~ face_stim + block_name + emotion_level2, data = data_final, FUN = "mean")
temp2 <- aggregate(RT ~ face_stim + block_name + emotion_level2, data = data_final, FUN = "mean")

data_final_stim <- data.frame(temp1, temp2$RT)
colnames(data_final_stim)[ncol(data_final_stim)] <- c("rt")

data_final_stim <- data_final_stim[order(data_final_stim$face_stim, 
                                         data_final_stim$block_name, 
                                         data_final_stim$emotion_level2),]

##################################
# Do a descriptive plot

# Requires ID variable for summarySEwithin - CG
datac <- summarySEwithin(data_final_subj, measurevar="resp", withinvars=c("emotion_level","block_name"), idvar="participant") 
datac
str(data_final_subj)

# Data Transformation ---------------------------------------------------
# Transform raw data
data_final_subj <- data_final_subj %>%
  mutate(
    # Convert factors to numeric values (1-5)
    emotion_numeric = as.numeric(emotion_level),
    # Linear transformation to a -2 to 2 scale
    emotion_scaled = emotion_numeric - 3
  )

# Transform summarized data
datac <- datac %>%
  mutate(
    emotion_numeric = as.numeric(emotion_level),
    emotion_scaled = emotion_numeric - 3
  )

##################################
# Do a descriptive plot (Fig2) -- Make errorbars more prominent + use solid lines only

p_fig2 <- ggplot() +
  
  # 1) Line: All solid lines (remove linetype mapping)
  geom_line(
    data = datac,
    aes(x = emotion_scaled, y = resp,
        color = block_name, group = block_name),
    linewidth = 1.25,
    position = position_dodge(width = 0.4)
  ) + 
  
  # 2) Errorbars: Thicker, darker, and brought to the front
  geom_errorbar(
    data = datac,
    aes(x = emotion_scaled, y = resp,
        ymin = resp - se, ymax = resp + se,
        group = block_name),
    position = position_dodge(width = 0.35),
    width = 0.20,            # Wider, more prominent
    linewidth = 0.5,         # Thicker
    color = "black",         # Darker color
    alpha = 1
  ) + 
  
  # 3) Mean points (aggregated data)
  geom_point(
    data = datac,
    aes(x = emotion_scaled, y = resp,
        fill = block_name, color = block_name),
    size = 0.05,
    shape = 19,
    position = position_dodge(width = 0.4)
  ) +
  
  # 4) Subject dots (keep unchanged - currently commented out)
  # geom_point(
  #   data = data_final_subj,
  #   aes(x = emotion_scaled, y = resp, color = block_name),
  #   position = position_jitterdodge(
  #     jitter.width = 0.3,
  #     jitter.height = 0.02,
  #     dodge.width = 0.4
  #   ),
  #   alpha = 0.4,
  #   size = 2,
  #   shape = 19
  # ) +
  
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
  
  # Annotations (keep unchanged)
  annotate("text",
           x = -2.5, y = 0, label = "Calm",
           angle = 90, hjust = 0.5, vjust = -3.8,
           size = 6, fontface = "italic", color = "black") +
  annotate("text",
           x = -2.5, y = 1, label = "Fear",
           angle = 90, hjust = 0.5, vjust = -3.8,
           size = 6, fontface = "italic", color = "black") +
  
  # Colors (keep unchanged)
  scale_color_manual(values = plot_color,labels = c("categorical" = "Categorical", "continuous" = "Continuous")) +
  scale_fill_manual(values = plot_color, labels = c("categorical" = "Categorical", "continuous" = "Continuous")) +
  
  theme_minimal(base_size = 22) +
  theme(
    aspect.ratio = 0.8,
    legend.position = "none",
    legend.box.spacing = unit(-20, "pt"),
    legend.title = element_blank(),
    axis.line = element_line(color = "black", linewidth = 1.1),
    axis.ticks = element_line(color = "black", linewidth = 1.1),
    panel.grid = element_blank(),
    plot.margin = margin(t = 30, r = 10, b = 10, l = 10),
    axis.title = element_text(size = 20, face = "bold"),
    axis.title.y = element_text(size = 22, face = "bold", margin = margin(r = 15),hjust = 0.5),
    axis.title.x = element_text(size = 22, face = "bold",margin = margin(t = 15)),
    axis.text = element_text(color = "black")
  ) +
  coord_cartesian(clip = "off", xlim = c(-2, 2))

print(p_fig2)

ggsave("fig2_emotion_categorization_allsubject_errorbar.jpg",
       p_fig2, width = 6, height = 5, dpi = 300)

##################################
# Define models
# Model comparison
library(dplyr)
library(quickpsy)

data_final_stim$emotion_level = as.numeric(data_final_stim$emotion_level2)

categorical_data <- subset(data_final_stim, block_name == "categorical")
continuous_data <- subset(data_final_stim, block_name == "continuous")

# Define your custom functions
my_logistic_fun <- function(x, p1, p2) {
  1 / (1 + exp(-(p1 + p2 * x)))
}
# A logistic function

my_linear_fun <- function(x, p1, p2) {
  p1 * x + p2
}
# A simple linear function, y = p1*x + p2

my_step_fun <- function(x, p1, p2) {
  ifelse(x < 0, p1, p2)
}
# A step function that returns p1 when x < 50% and p2 otherwise.

##################################
# Run model comparison for categorical data
# 1. Fit the logistic model using nls
logistic_fit <- nls(resp ~ my_logistic_fun(emotion_level, p1, p2),
                    data = categorical_data,
                    start = list(p1 = 0, p2 = 1))

# 2. Fit the linear model using nls
linear_fit <- nls(resp ~ my_linear_fun(emotion_level, p1, p2),
                  data = categorical_data,
                  start = list(p1 = 1, p2 = 0))

# 3. Fit the step model.
# Because the step function is not differentiable, nls might have trouble.
# Instead, we use 'optim' to minimize the sum of squared errors (SSE).

# The step function, as defined, has a sudden jump at the threshold 
# (in your case, at x=3). This jump creates a point of discontinuity where 
# the function isn't smooth, so its derivative (slope) is undefined at that point.

# Many optimization functions, such as nls (non-linear least squares), 
# rely on calculating derivatives (or gradients) to find the best-fitting parameters. 
# Since the step function doesn't have a derivative everywhere, these methods may 
# struggle or fail.

# To work around this, we use optim, which is a more general optimization tool. 
# optim doesn't require the function to be smooth or differentiable everywhere. 
# It can optimize the parameters by minimizing the sum of squared errors (SSE) 
# even when the function has discontinuities.

# nls is designed for smooth functions where derivatives can be computed.
# optim is more flexible and can handle non-differentiable functions like your step function.

# Define a residual sum-of-squares (RSS) function
rss_fun <- function(p, x, y, fun) {
  sum((y - fun(x, p[1], p[2]))^2)
}

# Use optim for the step function (threshold fixed at 3)
step_optim <- optim(par = c(0, 1),
                    fn = rss_fun,
                    x = categorical_data$emotion_level,
                    y = categorical_data$resp,
                    fun = my_step_fun)

# Extract the fitted parameters for the step function
step_params <- step_optim$par

# 4. Compare models using AIC.
# Since nls doesn't directly provide AIC, we compute it using:
#    AIC = n * log(RSS/n) + 2*k
# where n is the number of observations and k is the number of parameters.

n <- nrow(categorical_data)
k <- 2  # all models have two parameters

# Get RSS for the logistic and linear models
rss_logistic <- sum(resid(logistic_fit)^2)
rss_linear   <- sum(resid(linear_fit)^2)
rss_step     <- rss_fun(step_params, categorical_data$emotion_level, categorical_data$resp, my_step_fun)

AIC_logistic <- n * log(rss_logistic/n) + 2 * k
AIC_linear   <- n * log(rss_linear/n) + 2 * k
AIC_step     <- n * log(rss_step/n) + 2 * k

# Print the AICs
aic_table <- data.frame(
  Model = c("Logistic", "Linear", "Step"),
  AIC   = c(AIC_logistic, AIC_linear, AIC_step)
)
print(aic_table)

# 5. Visualize the fits
# Create a sequence of x values for predictions
x_seq <- seq(min(categorical_data$emotion_level), max(categorical_data$emotion_level), length.out = 100)

# Predictions from the logistic model
logistic_pred <- predict(logistic_fit, newdata = data.frame(emotion_level = x_seq))

# Predictions from the linear model
linear_pred <- predict(linear_fit, newdata = data.frame(emotion_level = x_seq))

# Predictions from the step model using our fitted parameters
step_pred <- my_step_fun(x_seq, step_params[1], step_params[2])

library(ggplot2)

# Combine the predictions into a single data frame
df_logistic <- data.frame(emotion_level = x_seq, pred = logistic_pred, Model = "Logistic")
df_linear   <- data.frame(emotion_level = x_seq, pred = linear_pred, Model = "Linear")
df_step     <- data.frame(emotion_level = x_seq, pred = step_pred, Model = "Step")

df_preds <- rbind(df_logistic, df_linear, df_step)

# Final adjusted plot
p <- ggplot(categorical_data, aes(x = emotion_level, y = resp)) +
  geom_point(color = "gray40", size = 2, alpha = 0.8) +
  geom_line(data = df_preds, aes(x = emotion_level, y = pred, color = Model), size = 1.2) +
  scale_x_continuous(name = "Morph: % Fear", 
                     breaks = c(-4, -2, 0, 2, 4),
                     labels = c("10%", "30%", "50%", "70%", "90%"),
                     limits = c(-5, 4.2),
                     expand = expansion(mult = c(0, 0.05))) + # No x-axis expansion
  scale_y_continuous(name = "Response", 
                     breaks = c(0, 0.5, 1),
                     limits = c(0, 1.1),
                     expand = expansion(mult = 0)) +
  scale_color_manual(values = c("#1F77B4", "#FF7F0E", "#2CA02C")) +
  theme_minimal(base_size = 18) +
  theme(
    legend.position = "top",
    legend.box.spacing = unit(-10, "pt"),
    legend.title = element_blank(),
    
    # Axis lines: added linewidth to make them more prominent
    axis.line.x = element_line(color = "black", linewidth = 1),
    axis.line.y = element_line(color = "black", linewidth = 1),
    axis.line = element_line(color = "black"),
    
    # Axis ticks: added linewidth to match axis lines
    axis.ticks = element_line(color = "black"),
    panel.grid = element_blank(),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10), # Increased left margin
    axis.title = element_text(size = 20, face = "bold"),
    axis.title.y = element_text(size = 22, face = "bold", margin = margin(r = 15)),
    axis.title.x = element_text(size = 22, face = "bold"),
    
    # Axis text (breaks): kept original settings
    axis.text = element_text(color = "black")
  ) +
  coord_cartesian(clip = "off") # Crucial for outside annotations

# Add annotations to left of y-axis
p <- p + 
  annotate("text", x = -4, y = 0, label = "Calm",  # Position left of x-axis start
           angle = 90, hjust = 0.8, vjust = -7, size = 5, fontface = "italic", color = "black") +
  annotate("text", x = -4, y = 1, label = "Fear",
           angle = 90, hjust = 0.3, vjust = -7, size = 5, fontface = "italic", color = "black")

print(p)

ggsave(paste0("fig3_emotion_cagegorization_model_comparison_categorical.jpg"), width = 4.6, height = 4, dpi = 150)

##################################
# Run model comparison for continuous data
# 1. Fit the logistic model using nls
logistic_fit <- nls(resp ~ my_logistic_fun(emotion_level, p1, p2),
                    data = continuous_data,
                    start = list(p1 = 0, p2 = 1))

# 2. Fit the linear model using nls
linear_fit <- nls(resp ~ my_linear_fun(emotion_level, p1, p2),
                  data = continuous_data,
                  start = list(p1 = 1, p2 = 0))

# 3. Fit the step model.

# Define a residual sum-of-squares (RSS) function
rss_fun <- function(p, x, y, fun) {
  sum((y - fun(x, p[1], p[2]))^2)
}

# Use optim for the step function (threshold fixed at 3)
step_optim <- optim(par = c(0, 1),
                    fn = rss_fun,
                    x = continuous_data$emotion_level,
                    y = continuous_data$resp,
                    fun = my_step_fun)

# Extract the fitted parameters for the step function
step_params <- step_optim$par

# 4. Compare models using AIC.
# Since nls doesn't directly provide AIC, we compute it using:
#    AIC = n * log(RSS/n) + 2*k
# where n is the number of observations and k is the number of parameters.

n <- nrow(continuous_data)
k <- 2  # all models have two parameters

# Get RSS for the logistic and linear models
rss_logistic <- sum(resid(logistic_fit)^2)
rss_linear   <- sum(resid(linear_fit)^2)
rss_step     <- rss_fun(step_params, continuous_data$emotion_level, continuous_data$resp, my_step_fun)

AIC_logistic <- n * log(rss_logistic/n) + 2 * k
AIC_linear   <- n * log(rss_linear/n) + 2 * k
AIC_step     <- n * log(rss_step/n) + 2 * k

# Print the AICs
aic_table <- data.frame(
  Model = c("Logistic", "Linear", "Step"),
  AIC   = c(AIC_logistic, AIC_linear, AIC_step)
)
print(aic_table)

# 5. Visualize the fits
# Create a sequence of x values for predictions
x_seq <- seq(min(continuous_data$emotion_level), max(continuous_data$emotion_level), length.out = 100)

# Predictions from the logistic model
logistic_pred <- predict(logistic_fit, newdata = data.frame(emotion_level = x_seq))

# Predictions from the linear model
linear_pred <- predict(linear_fit, newdata = data.frame(emotion_level = x_seq))

# Predictions from the step model using our fitted parameters
step_pred <- my_step_fun(x_seq, step_params[1], step_params[2])

library(ggplot2)

# Combine the predictions into a single data frame
df_logistic <- data.frame(emotion_level = x_seq, pred = logistic_pred, Model = "Logistic")
df_linear   <- data.frame(emotion_level = x_seq, pred = linear_pred, Model = "Linear")
df_step     <- data.frame(emotion_level = x_seq, pred = step_pred, Model = "Step")

df_preds <- rbind(df_logistic, df_linear, df_step)

# Final adjusted plot
p <- ggplot(continuous_data, aes(x = emotion_level, y = resp)) +
  geom_point(color = "gray40", size = 2, alpha = 0.8) +
  geom_line(data = df_preds, aes(x = emotion_level, y = pred, color = Model), size = 1.2) +
  scale_x_continuous(name = "Morph: % Fear", 
                     breaks = c(-4, -2, 0, 2, 4),
                     labels = c("10%", "30%", "50%", "70%", "90%"),
                     limits = c(-5, 4.2),
                     expand = expansion(mult = c(0, 0.05))) + 
  scale_y_continuous(name = "Response", 
                     breaks = c(0, 0.5, 1),
                     limits = c(0.0, 1.1),
                     expand = expansion(mult = c(0, 0))) +
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
    panel.grid = element_blank(),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10), # Increased left margin
    axis.title = element_text(size = 20, face = "bold"),
    axis.title.y = element_text(size = 22, face = "bold", margin = margin(r = 15)),
    axis.title.x = element_text(size = 22, face = "bold"),
    axis.text = element_text(color = "black")
  ) +
  coord_cartesian(clip = "off") # Crucial for outside annotations

# Add annotations to left of y-axis
p <- p + 
  annotate("text", x = -4, y = 0, label = "Calm",  # Position left of x-axis start
           angle = 90, hjust = 0.8, vjust = -8, size = 5, fontface = "italic",color = "black") +
  annotate("text", x = -4, y = 1, label = "Fear",
           angle = 90, hjust = 0.3, vjust = -8, size = 5, fontface = "italic",color = "black")

print(p)

ggsave(paste0("fig4_emotion_categorization_model_comparison_continuous.jpg"), width = 4.6, height = 4, dpi = 150)
