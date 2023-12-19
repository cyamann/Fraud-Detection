# Install and load necessary libraries
library(randomForest)
library(data.table)
library(caret)
library(doParallel)

# Load the dataset
dt <- fread("new_csv.csv")

# Drop unnecessary columns
dt <- dt[, c("step", "nameOrig", "nameDest") := NULL]

# Extract features and target variable
X <- dt[, !"isFraud", with = FALSE]
y <- dt$isFraud

# Identify categorical columns
categorical_columns <- c("type")

# One-hot encode categorical columns using data.table
X_encoded <- as.data.table(model.matrix(~ . - 1, data = X[, c(categorical_columns, "isFlaggedFraud"), with = FALSE]))

# Combine one-hot encoded features with non-categorical features
X_encoded <- cbind(X[, !"isFlaggedFraud", with = FALSE], X_encoded)

# Convert to data.table for efficient operations
X_encoded <- as.data.table(X_encoded)

# Split the dataset into training and testing sets
set.seed(42)  # For reproducibility
index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_encoded[index, ]
X_test <- X_encoded[-index, ]
y_train <- y[index]
y_test <- y[-index]

# Hyperparameter tuning
tuneGrid <- expand.grid(
  ntree = c(50, 100, 150),
  mtry = floor(sqrt(ncol(X_train)))
)

# Sequentially train Random Forest models and store their performance
rf_models <- list()

for (i in 1:nrow(tuneGrid)) {
  model <- randomForest(
    x = X_train, 
    y = as.factor(y_train), 
    ntree = tuneGrid$ntree[i], 
    mtry = tuneGrid$mtry[i], 
    classwt = c("0" = 1, "1" = 1), 
    importance = TRUE, 
    seed = 42
  )
  
  # Make predictions on the test set
  y_pred <- predict(model, newdata = X_test)
  
  # Make sure factor levels are aligned
  y_test_factor <- as.factor(y_test)
  y_pred_factor <- as.factor(y_pred)
  levels(y_test_factor) <- levels(y_pred_factor)
  
  # Store model and performance
  kappa_value <- confusionMatrix(y_pred_factor, y_test_factor)$overall["Kappa"]
  rf_models[[i]] <- list(model = model, performance = kappa_value)
}

# Extract the Kappa values from the rf_models list
kappa_values <- sapply(rf_models, function(model) model$performance)

# Identify the best model based on performance
best_model_index <- which.max(kappa_values)
best_model <- rf_models[[best_model_index]]$model

# Make predictions on the test set using the best model
y_pred <- predict(best_model, newdata = X_test)

# Evaluate the model
y_test_factor <- as.factor(y_test)
y_pred_factor <- as.factor(y_pred)
levels(y_test_factor) <- levels(y_pred_factor)

conf_matrix <- confusionMatrix(y_pred_factor, y_test_factor)

# Display confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Display other metrics
print("Precision, Recall, and F1 Score:")
print(conf_matrix$byClass)