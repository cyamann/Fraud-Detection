library(data.table)
library(caret)
library(biglm)

# Veri setini okuma ve ön işleme
data <- fread("C:/Users/ozany/OneDrive/Belgeler/GitHub/Fraud-Detection/new_csv.csv")
data$isFraud <- as.factor(data$isFraud)
data$type <- as.factor(data$type)
data$nameDest <- as.factor(data$nameDest)
data[, amount_scaled := scale(amount)]

# Veri setini bölme (10,000 satır)
set.seed(12)
sample_data <- sample.split(data$isFraud, SplitRatio = 0.5) # Veri setinin yarısını eğitim için kullanıyoruz
train_data <- data[sample_data == TRUE, ]

# Model oluşturma (chunk_size 5,000)
chunk_size <- 5000
num_chunks <- ceiling(nrow(train_data) / chunk_size)

fit <- NULL
for (i in 1:num_chunks) {
  chunk <- train_data[((i-1) * chunk_size + 1):(min(i * chunk_size, nrow(train_data))), ]
  if (is.null(fit)) {
    fit <- biglm(isFraud ~ step + type + amount_scaled + nameOrig + oldbalanceOrg + newbalanceOrig +
                   nameDest + oldbalanceDest + newbalanceDest + isFlaggedFraud, data = chunk)
  } else {
    fit <- update(fit, chunk)
  }
}

# Model özeti
summary(fit)
