#################################################################
## File:            SentimentModels.R
##
## Description:     Merges stock data with sentiment scores,
##                  prepares data for modeling, and then performs
##                  various modeling algorithms.

require(randomForest)
require(xgboost)
require(caret)
require(miscTools)
require(ggplot2)
require(doParallel)

setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis")

########################################
###### Combine Price & Sentiment #######
########################################

## Get hourly data and extract date and hour
AAPL.stock <- read.csv("Data/Hourly Stock Data/getHistory_AAPL.csv", stringsAsFactors = FALSE)
AAPL.stock$timestamp <- ymd_hms(strptime(AAPL.stock$timestamp,"%FT%H:%M:%S", tz = "EST"), tz = "EST")
AAPL.stock$date <- date(AAPL.stock$timestamp)
AAPL.stock$hour <- hour(AAPL.stock$timestamp)

## Merge sentiment data with stock data for model development
AAPL.train <- merge(AAPL, AAPL.stock[, c("date", "hour", "high")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
#length.complete <- length(AAPL.train <- AAPL.train[AAPL.train$date > "2018-02-26"])
AAPL.train <- AAPL.train[AAPL.train$date > "2018-02-26" & AAPL.train$date < "2018-03-14",] #Incomplete sentiment data before 2-23-18
AAPL.train[is.na(AAPL.train)] <- 0 #Fill any NA sentiment scores with 0
colnames(AAPL.train) <- gsub(x = colnames(AAPL.train), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames

head(AAPL.train[,c(1,2,3,201)],21)
tail(AAPL.train[,c(1,2,3,201)],21)

glimpse(AAPL.train)

#############################################
###### Set aside a Day for Prediction #######
#############################################

## Merge sentiment data with stock data for model prediction
AAPL.pred <- merge(AAPL, AAPL.stock[, c("date", "hour", "high")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
AAPL.pred <- AAPL.pred[AAPL.pred$date > "2018-03-13",] #Incomplete sentiment data before 2-23-18
AAPL.pred[is.na(AAPL.pred)] <- 0 #Fill any NA sentiment scores with 0
colnames(AAPL.pred) <- gsub(x = colnames(AAPL.pred), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames

head(AAPL.pred[,c(1,2,3,201)],21)
tail(AAPL.pred[,c(1,2,3,201)],21)

glimpse(AAPL.pred)


###########################################
###### Generate Random Forest Model #######
###########################################

## Following inspired by http://blog.yhat.com/posts/comparing-random-forests-in-python-and-r.html
cols <- colnames(AAPL.train)
cols <- cols[!cols %in% "date"]

set.seed(123)
ts.control <- trainControl(method="timeslice", initialWindow = 35, horizon = 14, fixedWindow = FALSE, allowParallel = TRUE, search = "grid") #35 day cv training, 14 day cv testing
tuneGridRF <- expand.grid(.mtry=c(1:72))
metric <- "Rsquared"
rf <- train(high ~ ., data = AAPL.train[,cols], method = "rf", metric = metric, trControl = ts.control, tuneGrid = tuneGridRF, importance=TRUE)
#tuneGridRF <- expand.grid(.mtry=68)
#rf <- train(high ~ ., data = AAPL.train[,cols], method = "rf", metric = metric, trControl = ts.control, tuneGrid = tuneGridRF, importance=TRUE) #AAPL best mtry = 68
print(rf)
plot(rf)


(r2.train <- rSquared(AAPL.train$high, AAPL.train$high - predict(rf, AAPL.train[,cols])))
(r2.pred <- rSquared(AAPL.pred$high, AAPL.pred$high - predict(rf, AAPL.pred[,cols])))
(mse.train <- mean((AAPL.train$high - predict(rf, AAPL.train[,cols]))^2))
(mse.pred <- mean((AAPL.pred$high - predict(rf, AAPL.pred[,cols]))^2))

p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=AAPL.train$high, pred=predict(rf, AAPL.train[,cols])))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("RandomForest Regression in R r^2=", r2.train, sep=""))

p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=AAPL.pred$high, pred=predict(rf, AAPL.pred[,cols])))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("RandomForest Regression in R r^2=", r2.pred, sep=""))

# plot(as.numeric(AAPL.train$high), type = "l", lty = 1, xaxt = "n")
# axis(1, at=1:length(AAPL.train$high), labels=FALSE)
# text(1:length(AAPL.train$high), par("usr")[3] - 0.2, labels = paste(AAPL.train$date, AAPL.train$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.8)
# lines(predict(rf, AAPL.train[,cols]), type = "l", lty = 2, lwd = 2, col = "red")

plot(as.numeric(c(AAPL.train$high, AAPL.pred$high)), type = "l", lty = 1, xaxt = "n")
axis(1, at=1:(sum(length(AAPL.train$high), length(AAPL.pred$high))), labels=FALSE)
text(1:(sum(length(AAPL.train$high), length(AAPL.pred$high))), par("usr")[3] - 0.2, labels = c(paste(AAPL.train$date, AAPL.train$hour), paste(AAPL.pred$date, AAPL.pred$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(c(predict(rf, AAPL.train[,cols]),predict(rf, AAPL.pred[,cols])), type = "l", lty = 2, lwd = 2, col = "red")

varImpPlot(rf)
feat.imp <- varImp(rf)
head(importance(rf))

###########################################
######### Generate XG Boost Model #########
###########################################
cols <- cols[!cols %in% "high"]
X.train <- data.matrix(AAPL.train[,cols])
X.test <- data.matrix(AAPL.pred[,cols])
Y <- AAPL.train$high
# xgData <- xgb.DMatrix(data = X, label = Y)
# xgb <- xgboost(data = xgData, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "reg:linear")
# 
# plot(as.numeric(AAPL.train$high), type = "l", lty = 1)
# lines(predict(xgb, X), type = "l", lty = 2, lwd = 2, col = "red")

ts.control <- trainControl(method="timeslice", initialWindow = 35, horizon = 14, fixedWindow = FALSE, allowParallel = TRUE, search = "grid") #35 day cv training, 14 day cv testing
metric <- "Rsquared"
tuneGridXGB <- expand.grid( #See parameter descriptions at http://xgboost.readthedocs.io/en/latest/parameter.html
    nrounds=c(350),
    eta = c(0, 0.05, 0.1, 0.3),
    gamma = c(0.01, 1, 5),
    max_depth = c(0, 3, 6),
    colsample_bytree = 0.5,
    subsample = c(0.50, 1),
    min_child_weight = c(0, 1))

start <- Sys.time() #Start timer
xgbmod <- train(
    x = X.train,
    y = Y,
    method = 'xgbTree',
    metric = metric,
    trControl = ts.control,
    #tuneGrid = tuneGridXGB,
    importance=TRUE)
print(Sys.time() - start)

xgbmod

# nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
# 16      50         1 0.3     0              0.8                1         1

# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were nrounds = 350, max_depth = 6, eta = 0.05, gamma =
#     0.01, colsample_bytree = 0.5, min_child_weight = 0 and subsample = 0.5.

(r2.train <- rSquared(AAPL.train$high, AAPL.train$high - predict(xgbmod, X.train)))
(r2.pred <- rSquared(AAPL.pred$high, AAPL.pred$high - predict(xgbmod, X.test)))
(mse.train <- mean((AAPL.train$high - predict(rf, AAPL.train[,cols]))^2))
(mse.pred <- mean((AAPL.pred$high - predict(rf, AAPL.pred[,cols]))^2))

p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=AAPL.train$high, pred=predict(xgbmod, X.train)))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("RandomForest Regression in R r^2=", r2.train, sep=""))

p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=AAPL.pred$high, pred=predict(rf, AAPL.pred[,cols])))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("RandomForest Regression in R r^2=", r2.pred, sep=""))


plot(as.numeric(c(AAPL.train$high, AAPL.pred$high)), type = "l", lty = 1, xaxt = "n")
axis(1, at=1:(sum(length(AAPL.train$high), length(AAPL.pred$high))), labels=FALSE)
text(1:(sum(length(AAPL.train$high), length(AAPL.pred$high))), par("usr")[3] - 0.2, labels = c(paste(AAPL.train$date, AAPL.train$hour), paste(AAPL.pred$date, AAPL.pred$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(c(predict(xgbmod, X.train),c(predict(xgbmod, X.test))), type = "l", lty = 2, lwd = 2, col = "red")

plot(as.numeric(AAPL.pred$high), type = "l", lty = 1)
lines(predict(xgbmod, X.test), type = "l", lty = 2, lwd = 2, col = "red")

feat.imp <- varImp(xgbmod)
