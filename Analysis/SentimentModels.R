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

setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis")
# dailyHI <- read.csv("Data/DailyHI.csv", stringsAsFactors = FALSE)
# dailyHI$DATE <- as.Date(dailyHI$DATE, "%m/%d/%Y")
# 
# AAPL <- merge(AAPL, dailyHI[, c("DATE", "AAPL")], by.x = "timestamp", by.y = "DATE", all.x = TRUE)

########################################
###### Combine Price & Sentiment #######
########################################

## Get hourly data and extract date and hour
AAPL.stock <- read.csv("Data/Hourly Stock Data/getHistory_AAPL.csv", stringsAsFactors = FALSE)
AAPL.stock$timestamp <- ymd_hms(strptime(AAPL.stock$timestamp,"%FT%H:%M:%S", tz = "EST"), tz = "EST")
AAPL.stock$date <- date(AAPL.stock$timestamp)
AAPL.stock$hour <- hour(AAPL.stock$timestamp)

## Merge sentiment data with stock data for model development
AAPL.complete <- merge(AAPL, AAPL.stock[, c("date", "hour", "high")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
AAPL.complete <- AAPL.complete[AAPL.complete$date > "2018-02-26" & AAPL.complete$date < "2018-03-14",] #Incomplete sentiment data before 2-23-18
AAPL.complete[is.na(AAPL.complete)] <- 0 #Fill any NA sentiment scores with 0
colnames(AAPL.complete) <- gsub(x = colnames(AAPL.complete), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames

head(AAPL.complete[,c(1,2,3,201)],21)
tail(AAPL.complete[,c(1,2,3,201)],21)

glimpse(AAPL.complete)

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

glimpse(AAPL.complete)


###########################################
###### Generate Random Forest Model #######
###########################################

## Following inspired by http://blog.yhat.com/posts/comparing-random-forests-in-python-and-r.html
cols <- colnames(AAPL.complete)
cols <- cols[!cols %in% "date"]
rf <- randomForest(high ~ ., data = AAPL.complete[,cols], ntree = 100)

(r2 <- rSquared(AAPL.pred$high, AAPL.pred$high - predict(rf, AAPL.pred[,cols])))

(mse <- mean((AAPL.pred$high - predict(rf, AAPL.pred[,cols]))^2))


p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=AAPL.pred$high, pred=predict(rf, AAPL.pred[,cols])))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("RandomForest Regression in R r^2=", r2, sep=""))

# plot(as.numeric(AAPL.complete$high), type = "l", lty = 1, xaxt = "n")
# axis(1, at=1:length(AAPL.complete$high), labels=FALSE)
# text(1:length(AAPL.complete$high), par("usr")[3] - 0.2, labels = paste(AAPL.complete$date, AAPL.complete$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.8)
# lines(predict(rf, AAPL.complete[,cols]), type = "l", lty = 2, lwd = 2, col = "red")

plot(as.numeric(c(AAPL.complete$high, AAPL.pred$high)), type = "l", lty = 1, xaxt = "n")
axis(1, at=1:(sum(length(AAPL.complete$high), length(AAPL.pred$high))), labels=FALSE)
text(1:(sum(length(AAPL.complete$high), length(AAPL.pred$high))), par("usr")[3] - 0.2, labels = c(paste(AAPL.complete$date, AAPL.complete$hour), paste(AAPL.pred$date, AAPL.pred$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(c(predict(rf, AAPL.complete[,cols]),predict(rf, AAPL.pred[,cols])), type = "l", lty = 2, lwd = 2, col = "red")

varImpPlot(rf)
head(importance(rf))

###########################################
######### Generate XG Boost Model #########
###########################################
cols <- cols[!cols %in% "high"]
X.train <- data.matrix(AAPL.complete[,cols])
X.test <- data.matrix(AAPL.pred[,cols])
Y <- AAPL.complete$high
# xgData <- xgb.DMatrix(data = X, label = Y)
# xgb <- xgboost(data = xgData, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "reg:linear")
# 
# plot(as.numeric(AAPL.complete$high), type = "l", lty = 1)
# lines(predict(xgb, X), type = "l", lty = 2, lwd = 2, col = "red")

tuneGridXGB <- expand.grid( #See parameter descriptions at http://xgboost.readthedocs.io/en/latest/parameter.html
    nrounds=c(350, 900),
    eta = c(0, 0.05, 0.1, 1),
    gamma = c(0.01, 1, 5),
    max_depth = c(3, 6),
    colsample_bytree = c(0.5, 0.75),
    subsample = c(0.50, 0.75),
    min_child_weight = c(0, 1))

start <- Sys.time() #Start timer
xgbmod <- train(
    x = X.train,
    y = Y,
    method = 'xgbTree',
    metric = 'RMSE',
    tuneGrid = tuneGridXGB)
print(Sys.time() - start)

# fitControl <- trainControl(search = "random")
# 
# start <- Sys.time() #Start timer
# set.seed(1)
# xgbmod <- train(
#     x = X.train,
#     y = Y,
#     method = 'xgbTree',
#     metric = 'RMSE',
#     #tuneGrid = tuneGridXGB)
#     tuneLength = 100,
#     trControl = fitControl)
# 
# print(Sys.time() - start)

plot(as.numeric(c(AAPL.complete$high, AAPL.pred$high)), type = "l", lty = 1, xaxt = "n")
axis(1, at=1:(sum(length(AAPL.complete$high), length(AAPL.pred$high))), labels=FALSE)
text(1:(sum(length(AAPL.complete$high), length(AAPL.pred$high))), par("usr")[3] - 0.2, labels = c(paste(AAPL.complete$date, AAPL.complete$hour), paste(AAPL.pred$date, AAPL.pred$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(c(predict(xgbmod, X.train),c(predict(xgbmod, X.test))), type = "l", lty = 2, lwd = 2, col = "red")

plot(as.numeric(AAPL.pred$high), type = "l", lty = 1)
lines(predict(xgbmod, X.test), type = "l", lty = 2, lwd = 2, col = "red")


