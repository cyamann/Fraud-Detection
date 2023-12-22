library(data.table)
library(randomForest)
library(caret)
library(doParallel)
library(e1071)

perform_data_preprocessing <- function(file_path, target_column) {
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
  
  return(list(X_encoded = X_encoded, y = y))
}
naive_bayes <- function(){
  
  # Set up k-fold cross-validation
  ctrl <- trainControl(method = "cv", number = k)
  
  # Train Naive Bayes model with k-fold cross-validation
  naive_bayes_model <- train(
    x = X_train,
    y = y_train,
    method = "naive_bayes",
    trControl = ctrl
  )
  
  # Print cross-validation results
  print("Cross-Validation Results:")
  print(summary(naive_bayes_model))
  
  # Make predictions on the test set with probabilities
  y_pred_probs <- predict(naive_bayes_model, newdata = X_test, type = "prob")
  
  # Check the structure of y_pred_probs
  str(y_pred_probs)
  
  # Assuming y_pred_probs is a data frame or matrix
  # Extract probabilities for the positive class (adjust column name as needed)
  y_pred_probs_class_1 <- y_pred_probs[, "1"]
  
  # Convert predicted probabilities to binary predictions (0 or 1)
  y_pred <- ifelse(y_pred_probs_class_1 > 0.5, 1, 0)
  
  
  # Evaluate the model
  y_test_factor <- as.factor(y_test)
  y_pred_factor <- as.factor(y_pred)
  
  # Create a confusion matrix with common levels
  conf_matrix <- confusionMatrix(y_pred_factor, y_test_factor)
  
  # Display confusion matrix
  print("Confusion Matrix:")
  print(conf_matrix)
  
  # Display other metrics
  print("Precision, Recall, and F1 Score:")
  print(conf_matrix$byClass)
}
decision_tree <- function(X_train, y_train, k = 10) {
  library(rpart)
  
  # Train decision tree model
  tree_model <- rpart(y_train ~ ., data = X_train, method = "class")
  
  # Combine features and target variable for cross-validation
  df <- cbind(as.data.table(X_train), isFraud = y_train)
  col <- ncol(df)
  row <- nrow(df)
  k_list <- as.list(0)
  k_index <- as.list(0)
  
  # Generate random indexes and split the indexes into k lists
  partition <- floor(row / k)
  tot <- partition * k
  set.seed(1)
  index <- sample(1:tot, tot)
  x <- seq_along(index)
  k_index <- split(index, ceiling(x / partition))
  
  # Use index list to divide df into k lists
  for (i in 1:k) {
    k_list[[i]] <- df[index[k_index[[i]]], ]
  }
  
  # Append leftover rows to the last list
  if (partition * k != row) {
    tot1 <- (partition * k) + 1
    k_list[[k]] <- rbind(k_list[[k]], df[tot1:row, ])
  }
  
  # Create training data using k-1 folds and predict the kth fold using a predictive model
  out <- data.frame()
  for (j in 1:k) {
    train_temp <- do.call(rbind, k_list[-j])
    test_temp <- as.data.frame(k_list[j])
    
    # Call to prediction models and binding the output of each fold
    pred_val <- predict(tree_model, newdata = test_temp, type = "class")
    
    out <- rbind(out, data.frame(predicted = pred_val))
  }
  
  # Evaluate the model
  y_pred_cv <- as.factor(out$predicted)
  y_test_factor <- as.factor(y_train)
  conf_matrix_cv <- confusionMatrix(y_pred_cv, y_test_factor)
  
  # Display confusion matrix
  print("Confusion Matrix (Cross-Validation):")
  print(conf_matrix_cv)
  
  # Display other metrics
  print("Accuracy, Precision, Recall, and F1 Score (Cross-Validation):")
  print(conf_matrix_cv$overall)
}

logistic_regression<-function(X, y, folds = 5){
  folds_list <- createFolds(y, k = folds, list = TRUE)
  
  # Initialize variables to store results
  precision <- recall <- f1_score <- numeric()
  
  # Perform cross-validation
  for (fold_num in seq_along(folds_list)) {
    fold <- folds_list[[fold_num]]
    
    # Split the data into training and validation sets
    X_train <- X[-fold, ]
    X_valid <- X[fold, ]
    y_train <- y[-fold]
    y_valid <- y[fold]
    
    # Convert the target variable to a factor
    y_train <- as.factor(y_train)
    
    # Train Logistic Regression model
    logistic_model <- glm(
      formula = y_train ~ .,
      family = binomial,
      data = X_train
    )
    
    # Make predictions on the validation set
    y_pred_prob <- predict(logistic_model, newdata = X_valid, type = "response")
    y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)
    
    # Evaluate the model
    conf_matrix <- confusionMatrix(as.factor(y_pred), y_valid)
    
    # Store metrics
    precision <- c(precision, conf_matrix$byClass["Precision"])
    recall <- c(recall, conf_matrix$byClass["Recall"])
    f1_score <- c(f1_score, conf_matrix$byClass["F1"])
    
    # Print metrics for each fold
    cat("Fold", fold_num, "Precision:", conf_matrix$byClass["Precision"], "\n")
    cat("Fold", fold_num, "Recall:", conf_matrix$byClass["Recall"], "\n")
    cat("Fold", fold_num, "F1 Score:", conf_matrix$byClass["F1"], "\n")
  }
  
  # Display average metrics across folds
  cat("\nAverage Precision:", mean(precision), "\n")
  cat("Average Recall:", mean(recall), "\n")
  cat("Average F1 Score:", mean(f1_score), "\n")
}

random_forest <- function(X, y) {
  # Set up parallel processing
  cl <- makeCluster(detectCores())
  registerDoParallel(cl)
  
  # Create a cross-validation object
  cv <- createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)
  
  # Initialize a list to store models and performance
  rf_models <- list()
  
  # Perform cross-validation
  for (i in seq_along(cv)) {
    # Split data into training and testing sets
    train_indices <- unlist(cv[-i])
    test_indices <- cv[[i]]
    
    X_train <- X[train_indices, ]
    X_test <- X[test_indices, ]
    y_train <- y[train_indices]
    y_test <- y[test_indices]
    
    # Hyperparameter tuning
    tuneGrid <- expand.grid(
      ntree = c(50, 100, 150),
      mtry = floor(sqrt(ncol(X_train)))
    )
    
    # Train Random Forest models and store their performance
    models <- list()
    
    for (j in 1:nrow(tuneGrid)) {
      model <- randomForest(
        x = X_train, 
        y = as.factor(y_train), 
        ntree = tuneGrid$ntree[j], 
        mtry = tuneGrid$mtry[j], 
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
      models[[j]] <- list(model = model, performance = kappa_value)
    }
    
    # Extract the Kappa values from the models list
    kappa_values <- sapply(models, function(model) model$performance)
    
    # Identify the best model based on performance
    best_model_index <- which.max(kappa_values)
    best_model <- models[[best_model_index]]$model
    
    # Store the best model and its performance
    rf_models[[i]] <- list(model = best_model, performance = kappa_values[best_model_index])
  }
  
  # Stop the parallel processing
  stopCluster(cl)
  
  # Extract the Kappa values from the rf_models list
  kappa_values <- sapply(rf_models, function(model) model$performance)
  
  # Identify the average performance across folds
  average_performance <- mean(kappa_values)
  
  # Identify the best model based on average performance
  best_model_index <- which.max(average_performance)
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
  
  # Print cross-validation results
  for (i in seq_along(rf_models)) {
    cat(sprintf("Fold %d - Kappa: %.4f\n", i, rf_models[[i]]$performance))
  }
  
  # Print average performance
  cat(sprintf("\nAverage Performance Across Folds: %.4f\n", average_performance))
}
#####MAIN CODE#####


# Example usage
file_path <- "new_csv.csv"
target_column <- "isFraud"

result <- perform_data_preprocessing(file_path, target_column)#preprocessing
X_encoded <- result$X_encoded
y <- result$y
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
y <- as.factor(y)

# Convert y_train to factor with two levels
y_train <- as.factor(y_train)
#n<-naive_bayes()#naive bayes
#d<-decision_tree(X_train, y_train)
#lr<-logistic_regression(X_encoded, y)
#r<-random_forest(X_encoded, y)
