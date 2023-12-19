# Install and load necessary libraries
library(data.table)
library(caret)
library(e1071)
library(caret)

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

# Drop the "isFlaggedFraud" variable
X_train <- X_train[, !"isFlaggedFraud", with = FALSE]
X_test <- X_test[, !"isFlaggedFraud", with = FALSE]

# Train SVM model
svm_model <- svm(y_train ~ ., data = X_train, kernel = "radial")

# Make predictions on the test set
y_pred <- predict(svm_model, newdata = X_test)

# Evaluate the model
y_test_factor <- as.factor(y_test)
y_pred_factor <- as.factor(y_pred)

# Convert y_pred_factor levels to "0" and "1"
y_pred_factor <- as.factor(ifelse(as.numeric(levels(y_pred_factor)) > 0.5, 1, 0))

# Check and display factor levels again
print("Levels in y_test_factor:")
print(levels(y_test_factor))
print("Levels in y_pred_factor:")
print(levels(y_pred_factor))

# Ensure the length of y_pred_factor matches y_test_factor
min_length <- min(length(y_pred_factor), length(y_test_factor))
y_pred_factor <- y_pred_factor[1:min_length]
y_test_factor <- y_test_factor[1:min_length]

# Create a confusion matrix with common levels
conf_matrix <- confusionMatrix(y_pred_factor, y_test_factor)

# Display confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Display other metrics
print("Precision, Recall, and F1 Score:")
print(conf_matrix$byClass)
