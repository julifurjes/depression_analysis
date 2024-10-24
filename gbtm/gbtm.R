# Load necessary libraries
library(lcmm)
library(dplyr)
library(ggplot2)
library(readxl)
library(tidyr)
library(glmnet)      # For LASSO variable selection
library(car)         # For calculating VIF
library(corrplot)    # For correlation plots
library(caret)
library(modeest)     # For mode estimation in missing data imputation

# Set CRAN mirror (adjust if necessary)
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Start capturing console output
sink("gbtm/output/R/console_output.txt", split = TRUE)

# 1. Data Preparation
# -------------------

# Load your data from the Excel file
data <- read_excel("BD_Rute.xlsx", sheet = "R")

# Remove columns containing 'desc' in their names (case-insensitive)
desc_cols <- grep("desc", names(data), ignore.case = TRUE)
if (length(desc_cols) > 0) {
  data <- data[, -desc_cols]
}

# Handle 'outra' and 'outros' columns
outra_cols <- grep("outra", names(data), ignore.case = TRUE)
for (col in outra_cols) {
  data[[col]] <- ifelse(is.na(data[[col]]), 0, 1)
}

outros_cols <- grep("outros", names(data), ignore.case = TRUE)
for (col in outros_cols) {
  data[[col]] <- ifelse(is.na(data[[col]]), 0, 1)
}

# Adjust 'score_eq5d' variables
if ("score_eq5d" %in% names(data)) {
  data$score_eq5d <- data$score_eq5d + 1
}

if ("score_eq5d_T2" %in% names(data)) {
  data$score_eq5d_T2 <- data$score_eq5d_T2 + 1
}

# Handle mixed-type columns
mixed_columns <- c("Outras_Pub_desc", "Outra_Doenca_Cardiaca", 
                   "haq_desc_outros", "Outra_Doenca_Aparelho_Digestivo", 
                   "Outras_Priv_desc", "Desc_Outro_Prest_Assist_Dom", 
                   "Outros_Meds", "Descricao_Outro_Local", 
                   "Outra_Doenca_Pulmonar", "Desc", 
                   "Outra_Doenca_Mental", "Outra_Doenca_Neurologica", 
                   "Outro_desc", "Outro_Exerc_desc", "Outra_desc")

for (col in mixed_columns) {
  if (col %in% names(data)) {
    data[[col]][is.na(data[[col]])] <- ""
    data[[col]] <- as.character(data[[col]])
  }
}

# Fill missing IDs and convert to factor
data$ID[is.na(data$ID)] <- "Missing"
data$ID <- as.factor(data$ID)

# Filter data to include only rows where specified scores are not missing
data <- data %>%
  filter(!is.na(Score_Depressao_T0) &
         !is.na(Score_Depressao_T1) &
         !is.na(Score_Depressao_T3))

# Reshape the depression scores to long format
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

# Identify repeated measures variables
repeated_vars <- grep("_T[0-9]+$", names(data), value = TRUE)
# Exclude the depression scores (already handled)
repeated_vars <- setdiff(
  repeated_vars, 
  c("Score_Depressao_T0", "Score_Depressao_T1", "Score_Depressao_T3")
)

# Pivot repeated measures variables to long format
data_repeated_long <- data %>%
  select(ID, all_of(repeated_vars)) %>%
  pivot_longer(
    cols = -ID,
    names_to = c(".value", "Time"),
    names_pattern = "(.+)_T([0-9]+)"
  )

# Ensure Time is numeric
data_repeated_long$Time <- as.numeric(data_repeated_long$Time)

# Merge with the main long dataset
data_long <- data_long %>%
  left_join(data_repeated_long, by = c("ID", "TimeNumeric" = "Time"))

# Convert character columns to factors
categorical_vars <- sapply(data_long, is.character)
data_long[categorical_vars] <- lapply(data_long[categorical_vars], as.factor)

# Handle missing values
impute_function <- function(x) {
  if (is.numeric(x)) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
  } else if (is.factor(x)) {
    x[is.na(x)] <- mfv(x, na_rm = TRUE)[1]
  }
  return(x)
}

data_long <- data_long %>%
  mutate(across(everything(), impute_function))

# 2. Variable Selection Using LASSO
# ---------------------------------

# Prepare data for LASSO
exclude_cols <- c("ID", "Time", "DepressionScore", "TimeNumeric")
predictor_names <- setdiff(names(data_long), exclude_cols)

# Ensure predictors are numeric (convert factors to numeric codes)
data_lasso <- data_long %>%
  mutate(across(all_of(predictor_names), ~ as.numeric(as.factor(.))))

# Remove near-zero variance predictors
nzv <- nearZeroVar(data_lasso[, predictor_names])
if (length(nzv) > 0) {
  predictor_names <- predictor_names[-nzv]
}

# Create matrix of predictors and response
x_matrix <- as.matrix(data_lasso[, predictor_names])
y_vector <- data_lasso$DepressionScore

# Standardize predictors
x_scaled <- scale(x_matrix)

# Perform LASSO regression with 10-fold cross-validation
set.seed(123)
lasso_cv <- cv.glmnet(x_scaled, y_vector, alpha = 1, nfolds = 10)

# Get the best lambda (more regularized model)
best_lambda <- lasso_cv$lambda.1se

# Extract coefficients at best lambda
lasso_coef <- coef(lasso_cv, s = best_lambda)
lasso_coef <- as.matrix(lasso_coef)

# Select predictors with non-zero coefficients
selected_predictors <- rownames(lasso_coef)[which(lasso_coef != 0)]
selected_predictors <- selected_predictors[
  !selected_predictors %in% "(Intercept)"
]

cat("Predictors selected by LASSO:\n")
print(selected_predictors)

# 3. Multicollinearity Check
# --------------------------

# Create data frame with selected predictors
data_selected <- data_lasso[, c("DepressionScore", selected_predictors)]

# Calculate correlation matrix
cor_matrix <- cor(data_selected[, selected_predictors])
corrplot(cor_matrix, method = "color", tl.cex = 0.7)

# Calculate VIF
vif_values <- vif(lm(DepressionScore ~ ., data = data_selected))
print("VIF values:")
print(vif_values)

# Remove predictors with VIF > 5
high_vif <- names(vif_values[vif_values > 5])
if (length(high_vif) > 0) {
  selected_predictors <- setdiff(selected_predictors, high_vif)
  data_selected <- data_selected[, c("DepressionScore", selected_predictors)]
  cat("Predictors after removing high VIF (>5):\n")
  print(selected_predictors)
}

# 4. Model Specification and Fitting
# ----------------------------------

# Create the fixed effects formula
fixed_effects <- paste("DepressionScore ~ TimeNumeric +",
                       paste(selected_predictors, collapse = " + "))

# Convert formula to object
fixed_effects_formula <- as.formula(fixed_effects)

# Include covariates in the class-membership model
class_membership_formula <- paste("~",
                                  paste(selected_predictors, collapse = " + "))

# Model Selection: Try models with 1 to 5 trajectory groups and compare BIC
bic_values <- c()
models <- list()

# Step 1: Fit the model with ng = 1 (single group)
set.seed(123)

# Ensure ID is numeric for hlme function
data_long$ID <- as.numeric(as.factor(data_long$ID))

# Fit the model with ng = 1 (no mixture needed)
model_1 <- hlme(
  fixed = fixed_effects_formula,
  random = ~ TimeNumeric,
  subject = "ID",
  ng = 1,
  data = data_long,
  verbose = FALSE,
  nwg = FALSE,
  na.action = 1
)

# Store the BIC for the single-group model
bic_values[1] <- model_1$BIC
models[[1]] <- model_1

# Step 2: Fit models with ng > 1 using gridsearch for better initial values
for (k in 2:5) {
  set.seed(123)

  # Use gridsearch to find suitable initial values
  gs <- gridsearch(
    rep = 5,  # Number of random sets of initial values (adjust as needed)
    maxiter = 50,  # Maximum number of iterations for each model
    minit = model_1,
    hlme(
      fixed = fixed_effects_formula,
      random = ~ TimeNumeric,
      mixture = ~ TimeNumeric,
      classmb = as.formula(class_membership_formula),
      subject = "ID",
      ng = k,
      data = data_long,
      verbose = FALSE,
      nwg = FALSE,
      na.action = 1
    )
  )

  model_k <- gs

  # Store the BIC value for the model
  bic_values[k] <- model_k$BIC
  models[[k]] <- model_k
}

# Step 3: Plot the BIC values to determine the optimal number of groups
plot(1:5, bic_values, type = "b",
     xlab = "Number of Trajectory Groups",
     ylab = "BIC",
     main = "BIC for Different Number of Groups")
optimal_ng <- which.min(bic_values)
cat("Optimal number of trajectory groups based on BIC:", optimal_ng, "\n")

# 5. Fit the Final Model
# ----------------------

# Fit the final model based on optimal number of groups
final_model <- models[[optimal_ng]]

# 6. Results Interpretation
# -------------------------

# Display the summary of the final model
summary(final_model)

# Extract the coefficients
coefficients <- summarytable(final_model)
write.csv(coefficients,
          file = "gbtm/output/R/coefficients.csv", row.names = FALSE)

# Simplify the coefficients output by focusing on significant predictors
significant_coefs <- coefficients %>%
  filter(`p-values` < 0.05)

write.csv(significant_coefs,
          file = "gbtm/output/R/significant_coefficients.csv", row.names = FALSE)

cat("Significant coefficients saved to 'gbtm/output/R/significant_coefficients.csv'\n")

# 7. Visualization
# ----------------

# Plot the estimated trajectories
plot(final_model, which = "fit", shade = TRUE)

# Save the plot
png(filename = "gbtm/output/R/trajectory_plot.png", width = 800, height = 600)
plot(final_model, which = "fit", shade = TRUE)
dev.off()

cat("Trajectory plot saved to 'gbtm/output/R/trajectory_plot.png'\n")

# 8. Class Membership Probabilities
# ---------------------------------

# Extract posterior probabilities of class membership
class_membership <- posterior(final_model)

# Add class membership to the data
data_selected$class <- class_membership$class

# Save class membership
write.csv(class_membership,
          file = "gbtm/output/R/class_membership.csv", row.names = FALSE)

cat("Class membership probabilities saved to 'gbtm/output/R/class_membership.csv'\n")

# 9. Visualize Predictors by Class
# --------------------------------

# For each significant predictor, plot its distribution by trajectory class
for (var in selected_predictors) {
  if (is.numeric(data_selected[[var]])) {
    p <- ggplot(data_selected, aes_string(x = var, fill = as.factor(data_selected$class))) +
      geom_histogram(position = "dodge", bins = 30) +
      labs(title = paste("Distribution of", var, "by Trajectory Class"), x = var, fill = "Class") +
      theme_minimal()
  } else {
    p <- ggplot(data_selected, aes_string(x = var, fill = as.factor(data_selected$class))) +
      geom_bar(position = "dodge") +
      labs(title = paste("Distribution of", var, "by Trajectory Class"), x = var, fill = "Class") +
      theme_minimal()
  }
  
  # Save the plot
  ggsave(filename = paste0("gbtm/output/R/", var, "_by_class.png"), plot = p, width = 8, height = 6)
}

cat("Predictor distribution plots saved to 'gbtm/output/R/'\n")

# 10. Conclusion
# -------------

cat("GBTM analysis completed. Check the 'gbtm/output/R/' directory for results.\n")

# Stop capturing console output
sink()

# End of script