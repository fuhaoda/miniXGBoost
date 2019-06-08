rm(list=ls())
set.seed(1234)

# requirs RStudio
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) 

library("xgboost")

training.SampleSize <- 500
testing.SampleSize <- 10000;
nCol <- 20
sgma <-  0.5

beta <-  rnorm(nCol)

# generate training data
train.fMatrix <- matrix(runif(training.SampleSize*nCol), nrow = training.SampleSize)
train.err <- rnorm(training.SampleSize, mean=0, sd=sgma)
train.y <- train.fMatrix%*%beta + train.err


test.fMatrix <- matrix(runif(testing.SampleSize*nCol), nrow = testing.SampleSize)
test.err <- rnorm(testing.SampleSize, mean=0, sd=sgma)
test.y <- test.fMatrix%*%beta + test.err

xgboost.fit <- xgboost(data = train.fMatrix, label = train.y, max_depth = 3, eta = .1, nthread = 1, nrounds = 800, objective = "reg:linear", verbose = 0)
yhat.xgboost <- predict(xgboost.fit, test.fMatrix)

lmfit <- lm(train.y ~ train.fMatrix-1)
yhat.lm <- test.fMatrix%*%lmfit$coefficients


mse.xgboost <- mean((test.y-yhat.xgboost)^2)/2
mse.lm <- mean((test.y-yhat.lm)^2)/2

print(paste("MSE from XGBoost = ", mse.xgboost))

print(paste("MSE from Linear Model = ", mse.lm))

write.table(cbind(train.y,train.fMatrix), file = "SimTrain.csv",row.names = F, col.names = FALSE,sep=",")

write.table(cbind(test.y,test.fMatrix), file = "SimTest.csv",row.names = F, col.names = FALSE,sep=",")