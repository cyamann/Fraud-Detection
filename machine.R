library(ranger)
data <- read.csv("fraud.csv")
library(data.table)
library(caret)
summary(data)
names(data)
#summary of amount
summary(data$amount)
sd(data$amount)
IQR(data$amount)
var(data$amount)
#manipulation
data$amount <- scale(data$amount)
set.seed(12)
library(caTools)
data$Class <- as.numeric(data$Class)
sample_data <- sample.split(data$Class, SplitRatio = 0.80)
train_data <- subset(data, sample_data=TRUE)
test_data <- subset(data, sample_data=FALSE)


