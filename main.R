library(data.table)
library(randomForest)
library(caret)
library(doParallel)
library(e1071)
library(kernlab)
library(ggplot2)
library(rpart)
library(pROC)

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
naive_bayes <- function(X_train, y_train, X_test, y_test, laplace = 1) {
  
  # Train Naive Bayes model
  naive_bayes_model <- naiveBayes(x = X_train, y = y_train, laplace = laplace)
  
  # Make predictions on the test set with probabilities
  y_pred_probs <- predict(naive_bayes_model, newdata = X_test, type = "raw")
  
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
  print("########################################################")
  print("Naive Bayes Model")
  
  # Display confusion matrix
  print("Confusion Matrix:")
  print(conf_matrix)
  
  # Display other metrics
  print("Precision, Recall, and F1 Score:")
  print(conf_matrix$byClass)
  
  # Plot performance metrics
  plot_naive_bayes(conf_matrix,y_test, y_pred_probs_class_1)
  
  print("########################################################")
}
plot_naive_bayes <- function(conf_matrix, y_test, y_pred_probs_class_1) {
  # Extracting performance metrics from the confusion matrix
  precision <- conf_matrix$byClass["Pos Pred Value"]
  recall <- conf_matrix$byClass["Sensitivity"]
  f1_score <- conf_matrix$byClass["F1"]
  
  # Calculate AUC
  auc <- roc(y_test, y_pred_probs_class_1)$auc
  
  # Prepare data for plotting
  data <- data.frame(
    Metric = c("Precision", "Recall", "F1 Score", "AUC"),
    Value = c(precision, recall, f1_score, auc)
  )
  
  # Plotting
  p <- ggplot(data, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Naive Bayes Performance Metrics", x = "Metric", y = "Value") +
    scale_fill_manual(values = c("Precision" = "blue", "Recall" = "green", "F1 Score" = "red", "AUC" = "purple"))
  print(p)
}


decision_tree <- function(X_train, y_train, k = 10, cp_values = seq(0.01, 0.1, 0.01)) {
  
  # Create a grid of hyperparameters
  hyperparameters <- expand.grid(cp = cp_values)
  
  # Set up k-fold cross-validation
  ctrl <- trainControl(method = "cv", number = k)
  
  # Train decision tree model with parameter tuning and k-fold cross-validation
  tree_model <- train(
    x = X_train,
    y = y_train,
    method = "rpart",
    trControl = ctrl,
    tuneGrid = hyperparameters
  )
  
  # Extract the best complexity parameter
  best_cp <- tree_model$bestTune$cp
  print(paste("Best Complexity Parameter:", best_cp))
  
  # Combine features and target variable for cross-validation
  df <- cbind(as.data.table(X_train), isFraud = y_train)
  
  # Generate random indexes and split the indexes into k lists
  partition <- floor(nrow(df) / k)
  tot <- partition * k
  set.seed(1)
  index <- sample(1:tot, tot)
  x <- seq_along(index)
  k_index <- split(index, ceiling(x / partition))
  
  # Use index list to divide df into k lists
  k_list <- lapply(1:k, function(i) df[index[k_index[[i]]], ])
  
  # Append leftover rows to the last list
  if (partition * k != nrow(df)) {
    tot1 <- (partition * k) + 1
    k_list[[k]] <- rbind(k_list[[k]], df[tot1:nrow(df), ])
  }
  
  # Create training data using k-1 folds and predict the kth fold using a predictive model
  out <- data.frame()
  for (j in 1:k) {
    train_temp <- do.call(rbind, k_list[-j])
    test_temp <- as.data.frame(k_list[j])
    
    # Train the decision tree model on the training data
    tree_model_fold <- rpart(isFraud ~ ., data = train_temp, method = "class", cp = best_cp)
    
    # Make predictions on the test data
    pred_val <- predict(tree_model_fold, newdata = test_temp, type = "class")
    
    out <- rbind(out, data.frame(predicted = pred_val))
  }
  
  
  # Evaluate the model
  y_pred_cv <- as.factor(out$predicted)
  y_test_factor <- as.factor(y_train)
  conf_matrix_cv <- confusionMatrix(y_pred_cv, y_test_factor)
  print("########################################################")
  print("Decision Tree Model")
  
  # Display confusion matrix
  print("Confusion Matrix (Cross-Validation):")
  print(conf_matrix_cv)
  
  # Display other metrics
  print("Accuracy, Precision, Recall, and F1 Score (Cross-Validation):")
  print(conf_matrix_cv$overall)
  
  # Plot decision tree performance metrics
  plot_decision_tree(conf_matrix_cv)
  
  print("########################################################")
}

plot_decision_tree <- function(conf_matrix_cv) {
  # Extracting performance metrics from the confusion matrix
  TP <- conf_matrix_cv[["byClass"]][["Pos Pred Value"]]
  recall <- conf_matrix_cv[["byClass"]][["Sensitivity"]]
  f1_score <- conf_matrix_cv[["byClass"]][["F1"]]
  
  # Convert to numeric to avoid errors
  TP <- as.numeric(TP)
  recall <- as.numeric(recall)
  f1_score <- as.numeric(f1_score)
  
  # Calculate AUC
  roc_curve <- roc(conf_matrix_cv$byClass$Sensitivity, 1 - conf_matrix_cv$byClass$Specificity)
  auc <- auc(roc_curve)
  
  # Prepare data for plotting
  data <- data.frame(
    Metric = c("Precision", "Recall", "F1 Score", "AUC"),
    Value = c(TP, recall, f1_score, auc)
  )
  
  # Create the ggplot object
  p <- ggplot(data, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Decision Tree Performance Metrics", x = "Metric", y = "Value") +
    scale_fill_manual(values = c("Precision" = "blue", "Recall" = "green", "F1 Score" = "red", "AUC" = "purple"))
  
  # Print the plot
  print(p)
}


logistic_regression <- function(X, y, folds = 5, lambda_values = seq(0.01, 1, 0.1)) {
  folds_list <- createFolds(y, k = folds, list = TRUE)
  
  # Initialize variables to store results
  precision <- recall <- f1_score <- numeric()
  
  # Set up the grid of hyperparameters for tuning
  hyperparameters <- data.frame(parameter = lambda_values)
  
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
    
    # Set up logistic regression model with parameter tuning
    logistic_model <- train(
      x = X_train,
      y = y_train,
      method = "glm",
      family = binomial,
      trControl = trainControl(method = "cv", number = 5),
      tuneGrid = hyperparameters
    )
    
    # Extract the best model from the training results
    best_model <- logistic_model$finalModel
    
    # Make predictions on the validation set
    y_pred_prob <- predict(best_model, newdata = X_valid, type = "response")
    y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)
    
    # Evaluate the model
    conf_matrix <- confusionMatrix(as.factor(y_pred), y_valid)
    
    # Store metrics
    precision <- c(precision, conf_matrix$byClass["Precision"])
    recall <- c(recall, conf_matrix$byClass["Recall"])
    f1_score <- c(f1_score, conf_matrix$byClass["F1"])
    
    # Print metrics for each fold
    cat("########################################################\n")
    cat("Logistic Regression Model (Fold", fold_num, ")\n")
    cat("Precision:", conf_matrix$byClass["Precision"], "\n")
    cat("Recall:", conf_matrix$byClass["Recall"], "\n")
    cat("F1 Score:", conf_matrix$byClass["F1"], "\n")
  }
  
  # Display average metrics across folds
  cat("\nAverage Precision:", mean(precision), "\n")
  cat("Average Recall:", mean(recall), "\n")
  cat("Average F1 Score:", mean(f1_score), "\n")
  
  # Plot logistic regression performance metrics
  plot_logistic_regression(precision, recall, f1_score,y_valid,y_pred_prob)
  
  cat("########################################################\n")
}

plot_logistic_regression <- function(precision, recall, f1_score, y_valid, y_pred_prob) {
  data <- data.frame(Metric = c("Precision", "Recall", "F1 Score", "AUC"),
                     Value = c(mean(precision), mean(recall), mean(f1_score), roc(y_valid, y_pred_prob)$auc))
  p <- ggplot(data, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    labs(title = "Logistic Regression Performance Metrics", x = "Metric", y = "Value") +
    scale_fill_brewer(palette = "Pastel1")
  
  print(p)
}


random_forest <- function(X, y) {
  # Set up parallel processing
  cl <- makeCluster(detectCores())
  registerDoParallel(cl)
  
  # Create a cross-validation object
  cv <- createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)
  
  # Initialize a list to store models and performance
  rf_models <- list()
  
  # Hyperparameter tuning grid
  tuneGrid <- expand.grid(
    ntree = c(50, 100, 150),
    mtry = floor(sqrt(ncol(X)))
  )
  
  # Perform cross-validation
  for (i in seq_along(cv)) {
    # Split data into training and testing sets
    train_indices <- unlist(cv[-i])
    test_indices <- cv[[i]]
    
    X_train <- X[train_indices, ]
    X_test <- X[test_indices, ]
    y_train <- y[train_indices]
    y_test <- y[test_indices]
    
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
  print("########################################################")
  print("Random Forest Model")
  
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
  plot_random_forest(rf_models)
  
  print("########################################################")
  
}
plot_random_forest <- function(rf_models) {
  # Assuming 'rf_models' is a list of lists containing model and performance for each fold
  kappa_values <- sapply(rf_models, function(model_list) model_list$performance)
  
  # Prepare data for plotting
  data <- data.frame(Fold = seq_along(kappa_values), Kappa = kappa_values)
  
  # Create the ggplot object
  p <- ggplot(data, aes(x = Fold, y = Kappa)) +
    geom_line() +
    geom_point() +
    theme_minimal() +
    labs(title = "Random Forest Kappa Metrics Over Folds", x = "Fold", y = "Kappa")
  
  # Print the plot
  print(p)
}

svm <- function() {
  # Extract features and target variable
  X <- dt[, !"isFraud", with = FALSE]
  y <- dt$isFraud
  
  # Convert outcome variable to a factor
  y <- factor(y)
  
  # Identify categorical columns
  categorical_columns <- "type"
  
  # One-hot encode categorical columns using model.matrix
  X_encoded <- model.matrix(~ . - 1, data = X[, c(categorical_columns, "isFlaggedFraud"), with = FALSE])
  
  # Combine one-hot encoded features with non-categorical features
  X_encoded <- cbind(X[, !"isFlaggedFraud", with = FALSE], X_encoded)
  
  # Convert to data.table for efficient operations
  X_encoded <- as.data.table(X_encoded)
  
  # Define the number of folds for cross-validation
  num_folds <- 5
  
  # Create an empty list to store results
  svm_results <- list()
  svm_performance_metrics <- vector("list", length = num_folds)
  
  for (fold in 1:num_folds) {
    set.seed(42 + fold)  # For reproducibility with different random seeds
    
    # Create train and test indices for this fold
    indices <- createDataPartition(y, p = 0.8, list = FALSE)
    X_train <- X_encoded[indices]
    X_test <- X_encoded[-indices]
    y_train <- y[indices]
    y_test <- y[-indices]
    
    # SVM Parameter Tuning
    svm_grid <- expand.grid(C = c(0.1, 1, 10),
                            sigma = c(0.01, 0.1, 1))
    
    # Train SVM model for this fold using kernlab::ksvm
    svm_model <- ksvm(y_train ~ ., data = X_train, kernel = "rbfdot",
                      C = svm_grid$C[1], kpar = list(sigma = svm_grid$sigma[1]))
    
    # Make predictions on the test set
    y_pred <- predict(svm_model, X_test)
    
    # Evaluate the model for this fold
    confusion_matrix <- table(Actual = y_test, Predicted = y_pred)
    
    # Store the results for this fold
    svm_results[[fold]] <- confusion_matrix
    svm_performance_metrics[[fold]] <- confusion_matrix
  }
  
  print("########################################################")
  print("SVM Model")
  
  # Display results for each fold
  for (fold in 1:num_folds) {
    cat("Fold", fold, "Confusion Matrix:\n")
    print(svm_results[[fold]])
    
    # Calculate and print precision, recall, and F1 score
    precision <- svm_results[[fold]][2, 2] / sum(svm_results[[fold]][, 2])
    recall <- svm_results[[fold]][2, 2] / sum(svm_results[[fold]][2, ])
    f1_score <- 2 * precision * recall / (precision + recall)
    
    cat("Precision:", precision, "\n")
    cat("Recall:", recall, "\n")
    cat("F1 Score:", f1_score, "\n\n")
  }
  
  # Calculate and display average performance metrics across folds
  avg_performance <- colMeans(do.call(rbind, svm_performance_metrics))
  cat("Average Performance Across Folds:\n")
  print(avg_performance)
  
  # Call the plot function with the correct 'results' list
  plot_svm(svm_results)
  print("########################################################")
}

plot_svm <- function(results) {
  # Assuming 'results' is a list of confusion matrices from each fold
  precision <- sapply(results, function(cm) (cm[2, 2] / sum(cm[, 2])))
  recall <- sapply(results, function(cm) (cm[2, 2] / sum(cm[2, ])))
  f1_score <- sapply(results, function(cm) (2 * precision(cm) * recall(cm)) / (precision(cm) + recall(cm)))
  
  # Prepare data for plotting
  data <- data.frame(
    Metric = rep(c("Precision", "Recall", "F1 Score"), times = length(results)),
    Fold = factor(rep(seq_along(results), each = 3)),
    Value = c(precision, recall, f1_score)
  )
  
  # Create the ggplot object
  p <- ggplot(data, aes(x = Fold, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    labs(title = "SVM Performance Metrics", x = "Fold", y = "Value") +
    scale_fill_brewer(palette = "Pastel1")
  
  # Print the plot
  print(p)
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
# Example usage with Laplace smoothing
#naive_bayes(X_train, y_train, X_test, y_test, laplace = 0.1)
#decision_tree(X_train, y_train, k = 10, cp_values = seq(0.01, 0.1, 0.01))
logistic_regression(X, y, folds = 5, lambda_values = seq(0.01, 1, 0.1))
#r<-random_forest(X_encoded, y)
#s<-svm()