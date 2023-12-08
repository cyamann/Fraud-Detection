library(ranger)
data <- read.csv("C:/project/fraud_subset.csv")
library(data.table)
library(caret)

# Assuming "isFraud" is the target variable
target_variable <- data$isFraud

# Summary of amount
summary(data$amount)
sd(data$amount)
IQR(data$amount)
var(data$amount)

# Manipulation
data$amount <- scale(data$amount)

# Create data2 without the first column and scale amount
data2 <- data[, -1]
data2$amount <- scale(data2$amount)

set.seed(12)
install.packages('caTools')
library('caTools')
library(reprex)

# Convert "isFraud" to a factor in the original dataset
data$isFraud <- as.factor(data$isFraud)

# Use the entire data2 dataframe in sample.split
sample_data <- sample.split(data$isFraud, SplitRatio = 4/5)
train_data <- subset(data, sample_data == TRUE)
test_data <- subset(data, sample_data == FALSE)
dim(train_data)
dim(test_data)
#logistik model
# Assuming your data frame is named "your_data"
# Replace "your_data" with the actual name of your data frame

# Load necessary library for logistic regression
library(glmnet)
data$type <- as.factor(data$type)
data$nameDest <- as.factor(data$nameDest)

Logistic_Model <- glm(isFraud ~ step + type + amount + nameOrig + oldbalanceOrg + newbalanceOrig +
                        nameDest + oldbalanceDest + newbalanceDest + isFlaggedFraud,
                      data = data, family = binomial())



summary(Logistic_Model)
