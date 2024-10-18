# Load necessary libraries
library(lcmm)
library(dplyr)
library(ggplot2)
library(readxl)
options(repos = c(CRAN = "https://cloud.r-project.org")) # Set CRAN mirror
library(tidyr)

# Set working directory (adjust the path accordingly)
#setwd("path/to/your/directory")

# 1. Data Preparation
# -------------------

# Load your data from the Excel file
# Replace "BD_Rute.xlsx" with your actual file name
data <- read_excel("BD_Rute.xlsx", sheet = "R")

# Filter data to include only rows where specified scores are not missing
data <- data %>%
  filter(!is.na(Score_Depressao_T0) & !is.na(Score_Depressao_T1) & !is.na(Score_Depressao_T3))

# Reshape the data to long format
data_long <- data %>%
  select(ID, Score_Depressao_T0, Score_Depressao_T1, Score_Depressao_T3) %>%
  pivot_longer(cols = starts_with("Score_Depressao"), 
               names_to = "Time", 
               values_to = "DepressionScore")

# Map time points to numerical values
data_long$TimeNumeric <- case_when(
  data_long$Time == "Score_Depressao_T0" ~ 0,
  data_long$Time == "Score_Depressao_T1" ~ 1,
  data_long$Time == "Score_Depressao_T3" ~ 3
)

# Merge predictors: exclude only the depression scores but keep the ID column
predictors <- data %>%
  select(ID, everything(), -c(Score_Depressao_T0, Score_Depressao_T1, Score_Depressao_T3))

# Handle missing values in predictors (optional: you can choose a different method)
# Here, we'll remove predictors with more than 50% missing values and impute the rest
threshold <- 0.5
predictor_na <- sapply(predictors, function(x) mean(is.na(x)))
predictors <- predictors[, predictor_na < threshold]

# Define a custom mode function to find the most frequent value
calculate_mode <- function(x) {
  uniq_vals <- unique(na.omit(x))
  uniq_vals[which.max(tabulate(match(x, uniq_vals)))]
}

# Handle missing values in predictors
# Impute remaining missing values with median for numeric and mode for categorical
impute_function <- function(x) {
  if (is.numeric(x)) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
  } else {
    x[is.na(x)] <- calculate_mode(x)  # Use custom mode function for categorical data
  }
  return(x)
}
predictors_imputed <- as.data.frame(lapply(predictors, impute_function))

# Combine the predictors with the long data, keeping ID for the join
data_long <- data_long %>%
  left_join(predictors_imputed, by = "ID")

# Convert categorical variables to factors
categorical_vars <- sapply(data_long, is.character)
data_long[categorical_vars] <- lapply(data_long[categorical_vars], as.factor)

# 2. Model Specification and Fitting
# ----------------------------------

# Define the fixed effects formula (including time and predictors)
# To simplify, we'll perform variable selection using univariate analysis
# and include only significant predictors in the model

# Univariate analysis to select significant predictors
predictor_names <- names(predictors_imputed)
significant_predictors <- c()

for (var in predictor_names) {
  # Check if the predictor has variation (not constant)
  if (length(unique(data_long[[var]])) > 1) {
    formula <- as.formula(paste("DepressionScore ~", var))
    
    # Use try() to handle any errors gracefully during model fitting
    model <- try(lm(formula, data = data_long), silent = TRUE)
    
    # Only proceed if model fitting was successful
    if (!inherits(model, "try-error")) {
      # Check if the model has a valid second coefficient (to avoid subscript issues)
      if (nrow(summary(model)$coefficients) > 1) {
        p_value <- summary(model)$coefficients[2, 4]
        if (p_value < 0.05) {
          significant_predictors <- c(significant_predictors, var)
        }
      }
    }
  }
}

# Limit to top 20 predictors based on p-values
if (length(significant_predictors) > 20) {
  p_values <- sapply(significant_predictors, function(var) {
    formula <- as.formula(paste("DepressionScore ~", var))
    model <- lm(formula, data = data_long)
    summary(model)$coefficients[2, 4]
  })
  significant_predictors <- significant_predictors[order(p_values)][1:20]
}

# Create the fixed effects formula
fixed_effects <- paste("DepressionScore ~ TimeNumeric +", paste(significant_predictors, collapse = " + "))

# Convert formula to object
fixed_effects_formula <- as.formula(fixed_effects)

# 3. Model Selection
# ------------------

# Determine the optimal number of trajectory groups (ng)

# Model Selection: Try models with 1 to 5 trajectory groups and compare BIC
bic_values <- c()
models <- list()

# Step 1: Fit the model with ng = 1
set.seed(123)
model_1 <- hlme(
  fixed = fixed_effects_formula,
  random = ~ TimeNumeric,
  subject = "ID",
  ng = 1,
  data = data_long,
  verbose = FALSE,
  nwg = FALSE,   # nwg must be FALSE for ng = 1
  na.action = 1  # Remove rows with missing data
)

# Store the model with ng = 1
models[[1]] <- model_1
bic_values[1] <- model_1$BIC

# Step 2: Fit models with ng > 1 using initial values from model_1
for (k in 2:5) {
  set.seed(123)
  
  # Set nwg = TRUE only when ng > 1
  nwg_value <- TRUE
  
  # Fit the model with initial values from model_1
  model_k <- hlme(
    fixed = fixed_effects_formula,
    random = ~ TimeNumeric,
    mixture = ~ TimeNumeric,    # Mixture for ng > 1
    subject = "ID",
    ng = k,
    data = data_long,
    verbose = FALSE,
    nwg = nwg_value,            # Conditional on ng
    B = model_1,                # Use initial values from model_1
    na.action = 1               # Remove rows with missing data
  )
  
  # Store BIC values and models
  bic_values[k] <- model_k$BIC
  models[[k]] <- model_k
}

# Plot BIC values and determine the optimal number of groups
plot(1:5, bic_values, type = "b", xlab = "Number of Trajectory Groups", ylab = "BIC", main = "BIC for Different Number of Groups")
optimal_ng <- which.min(bic_values)
cat("Optimal number of trajectory groups based on BIC:", optimal_ng, "\n")

# 4. Fit the Final Model
# ----------------------

# Fit the final model based on optimal number of groups
final_model <- models[[optimal_ng]]

# 5. Results Interpretation
# -------------------------

# Display the summary of the final model
summary(final_model)

# Extract the coefficients
coefficients <- summarytable(final_model)
write.csv(coefficients, file = "gbtm/output/R/coefficients.csv", row.names = FALSE)

# Simplify the coefficients output by focusing on significant predictors
# We'll extract coefficients with p-value < 0.05
significant_coefs <- coefficients %>%
  filter(`p-values` < 0.05)

write.csv(significant_coefs, file = "gbtm/output/R/significant_coefficients.csv", row.names = FALSE)

cat("Significant coefficients saved to 'gbtm/output/R/significant_coefficients.csv'\n")

# 6. Visualization
# ----------------

# Plot the estimated trajectories
plot(final_model, which = "fit", shade = TRUE)

# Save the plot
png(filename = "gbtm/output/R/trajectory_plot.png", width = 800, height = 600)
plot(final_model, which = "fit", shade = TRUE)
dev.off()

cat("Trajectory plot saved to 'gbtm/output/R/trajectory_plot.png'\n")

# 7. Class Membership Probabilities
# ---------------------------------

# Extract posterior probabilities of class membership
class_membership <- posterior(final_model)

# Add class membership to the data
data_long$class <- class_membership$class

# Save class membership
write.csv(class_membership, file = "gbtm/output/R/class_membership.csv", row.names = FALSE)

cat("Class membership probabilities saved to 'gbtm/output/R/class_membership.csv'\n")

# 8. Visualize Predictors by Class
# --------------------------------

# For each significant predictor, plot its distribution by trajectory class
for (var in significant_predictors) {
  p <- ggplot(data_long, aes_string(x = var, fill = as.factor(data_long$class))) +
    geom_histogram(position = "dodge", bins = 30) +
    labs(title = paste("Distribution of", var, "by Trajectory Class"), x = var, fill = "Class") +
    theme_minimal()
  
  # Save the plot
  ggsave(filename = paste0("gbtm/output/R/", var, "_by_class.png"), plot = p, width = 8, height = 6)
}

cat("Predictor distribution plots saved to 'gbtm/output/R/'\n")

# 9. Conclusion
# -------------

cat("GBTM analysis completed. Check the 'gbtm/output/R/' directory for results.\n")