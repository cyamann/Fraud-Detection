library(data.table)
library(caret)
library(biglm)

# Veri setini okuma
data <- fread("C:/Users/ozany/OneDrive/Belgeler/GitHub/Fraud-Detection/new_csv.csv")

# Veri setini faktörleştirmek ve ölçeklendirmek
data$isFraud <- as.factor(data$isFraud)
data$type <- as.factor(data$type)
data$nameDest <- as.factor(data$nameDest)
data[, amount_scaled := scale(amount)]

# Veri setini bölme
set.seed(12)
sample_data <- sample.split(data$isFraud, SplitRatio = 4/5)
train_data <- data[sample_data == TRUE, ]
test_data <- data[sample_data == FALSE, ]

# Veri setini küçük parçalara ayırma
chunk_size <- 100000 # Örneğin, her bir parça 100,000 satır
num_chunks <- ceiling(nrow(train_data) / chunk_size)

# Her bir parça üzerinde model oluşturma
fit <- NULL
for (i in 1:num_chunks) {
  chunk <- train_data[((i-1) * chunk_size + 1):(min(i * chunk_size, nrow(train_data))), ]
  if (is.null(fit)) {
    fit <- biglm(isFraud ~ step + type + amount_scaled + nameOrig + oldbalanceOrg + newbalanceOrig +
                   nameDest + oldbalanceDest + newbalanceDest + isFlaggedFraud, data = chunk)
  } else {
    fit <- update(fit, data = chunk)
  }
}

# Model özeti
summary(fit)
