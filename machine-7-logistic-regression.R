# Install and load necessary libraries
library(data.table)
library(caret)
library(glmnet)

# Load the dataset
dt <- fread("deneme_csv.csv")

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

# Convert the target variable to a factor
y_train <- as.factor(y_train)

# Convert the target variable to a factor
y_train <- as.factor(y_train)

# Train Logistic Regression model
logistic_model <- glm(
  formula = y_train ~ .,
  family = binomial,
  data = X_train
)

# Make predictions on the test set
y_pred_prob <- predict(logistic_model, newdata = X_test, type = "response")
y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)

# Evaluate the model
y_test_factor <- as.factor(y_test)
y_pred_factor <- as.factor(y_pred)

conf_matrix <- confusionMatrix(y_pred_factor, y_test_factor)

# Display confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Display other metrics
print("Precision, Recall, and F1 Score:")
print(conf_matrix$byClass)
