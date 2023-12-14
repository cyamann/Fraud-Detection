data <- read.csv("new-csv.csv")
library(dplyr)
library(caTools)
library(rpart)

# Split the data into training and test sets
split_values <- sample.split(data$isFraud, SplitRatio = 0.65)
train_set <- subset(data, split_values == TRUE)
test_set <- subset(data, split_values == FALSE)

# Define the formula for the rpart model
formula <- as.formula("isFraud ~ step + type + amount + nameOrig + oldbalanceOrg + newbalanceOrig + nameDest + oldbalanceDest + newbalanceDest + isFlaggedFraud")

# Train the rpart model
mod_class <- rpart(formula, data = train_set, method = "class")

# Convert factor levels in the test set
test_set$nameOrig <- factor(test_set$nameOrig, levels = levels(train_set$nameOrig))
test_set$nameDest <- factor(test_set$nameDest, levels = levels(train_set$nameDest))

# Make predictions on the test set
result_class <- predict(mod_class, test_set, type = "class")

# Display the confusion matrix
table(test_set$isFraud, result_class)
#regression
library(ggplot2)
#building lm model
mod_regres<-lm(isFraud~., data=train_set)
result_regres<-predict(mod_regres,test_set)
Final_Data<-cbind(Actual=test_set$isFraud,Predicted=result_regres)
Final_Data<-as.data.frame(Final_Data)
View(Final_Data)