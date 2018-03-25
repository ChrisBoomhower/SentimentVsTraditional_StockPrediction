#################################################################
## File:            SentimentModels.R
##
## Description:     Merges stock data with sentiment scores,
##                  prepares data for modeling, and then performs
##                  various modeling algorithms.

## Housekeeping tasks
rm(list = ls())
gc()

require(randomForest)
require(xgboost)
require(caret)
require(miscTools)
require(ggplot2)
require(doParallel)
registerDoParallel(cores=4)
library(lattice)

setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis")

########################################
###### Combine Price & Sentiment #######
###### Set Aside Data for Prediction ###
###### Check for Data Stationarity #####
########################################

getTrainTest <- function(tick){
    ########################################
    ###### Combine Price & Sentiment #######
    ########################################
    sink(paste0("getTrainTest_", tick, ".txt"))
    
    ## Get hourly data and extract date and hour
    stock <- read.csv(paste0("Data/Hourly Stock Data/getHistory_", tick, ".csv"), stringsAsFactors = FALSE)
    stock$timestamp <- ymd_hms(strptime(stock$timestamp,"%FT%H:%M:%S", tz = "EST"), tz = "EST")
    stock$date <- date(stock$timestamp)
    stock$hour <- hour(stock$timestamp)
    stock$high.diff <- c(NA,diff(stock$high))
    
    ## Merge sentiment data with stock data for model development
    tickSent <- readRDS(paste0('Data/TickerRDS/', tick, '.RDS'))
    tickSent.train <- merge(tickSent, stock[, c("date", "hour", "high")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
    tickSent.train <- tickSent.train[tickSent.train$date > "2018-02-26" & tickSent.train$date < "2018-03-14",] #Incomplete sentiment data before 2-23-18
    tickSent.train[is.na(tickSent.train)] <- 0 #Fill any NA sentiment scores with 0
    colnames(tickSent.train) <- gsub(x = colnames(tickSent.train), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames
    
    cat("\nTrain Head\n")
    print(head(tickSent.train[,c(1,2,3,201)],21))
    cat("\nTrain Tail\n")
    print(tail(tickSent.train[,c(1,2,3,201)],21))
    
    cat("\nglimpse Train\n")
    print(glimpse(tickSent.train))
    
    #############################################
    ###### Set Aside Data for Prediction ########
    #############################################
    
    ## Merge sentiment data with stock data for model prediction
    tickSent.pred <- merge(tickSent, stock[, c("date", "hour", "high")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
    tickSent.pred <- tickSent.pred[tickSent.pred$date > "2018-03-13",] #Incomplete sentiment data before 2-23-18
    tickSent.pred[is.na(tickSent.pred)] <- 0 #Fill any NA sentiment scores with 0
    colnames(tickSent.pred) <- gsub(x = colnames(tickSent.pred), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames
    
    cat("\nTest Head\n")
    print(head(tickSent.pred[,c(1,2,3,201)],21))
    cat("\nTest Tail\n")
    print(tail(tickSent.pred[,c(1,2,3,201)],21))
    
    print(cat("\nglimpse Test\n"))
    glimpse(tickSent.pred)
    
    ##########################################
    ###### Check for Data Stationarity #######
    ##########################################
    pdf(paste0(tick,'_Stationarity.pdf'))
    ## Check stationarity of variables in model
    temp <- apply(rbind(tickSent.train[2:length(tickSent.train)],tickSent.pred[2:length(tickSent.pred)]), 2, adf.test, alternative = "stationary", k=0)
    plot(sapply(temp, function(x) x$p.value), xlab = "Column Location", ylab = "Dickey-Fuller p.value", main = "Dickey-Fuller p.values") #Note that while sentiment values are stationary, stock price is not
    
    temp <- apply(rbind(tickSent.train[2:length(tickSent.train)],tickSent.pred[2:length(tickSent.pred)]), 2, adf.test, alternative = "stationary")
    plot(sapply(temp, function(x) x$p.value), xlab = "Column Location", ylab = "Augmented Dickey-Fuller p.value", main = "Augmented Dickey-Fuller p.values") #Note that while sentiment values are stationary, stock price is not
    
    ## Create high.diff training set
    tickSent.diff.train <- merge(tickSent, stock[, c("date", "hour", "high.diff")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
    tickSent.diff.train <- tickSent.diff.train[tickSent.diff.train$date > "2018-02-26" & tickSent.diff.train$date < "2018-03-14",] #Incomplete sentiment data before 2-23-18
    tickSent.diff.train[is.na(tickSent.diff.train)] <- 0 #Fill any NA sentiment scores with 0
    colnames(tickSent.diff.train) <- gsub(x = colnames(tickSent.diff.train), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames
    
    cat("\nSanity check for matching Train df size\n")
    print(nrow(tickSent.train) == nrow(tickSent.diff.train)) #Sanity check on size
    
    ## Create high.diff test set
    tickSent.diff.pred <- merge(tickSent, stock[, c("date", "hour", "high")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
    tickSent.diff.pred <- tickSent.diff.pred[tickSent.diff.pred$date > "2018-03-13",] #Incomplete sentiment data before 2-23-18
    tickSent.diff.pred[is.na(tickSent.diff.pred)] <- 0 #Fill any NA sentiment scores with 0
    colnames(tickSent.diff.pred) <- gsub(x = colnames(tickSent.diff.pred), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames
    
    cat("\nSanity check for matching Test df size\n")
    print(nrow(tickSent.pred) == nrow(tickSent.diff.pred)) #Sanity check on size
    
    #sum(!(cumsum(c(AAPL.stock$high[1], diff(AAPL.stock$high))) == AAPL.stock$high)) #Checking to see how to compare models when using differenced price
    dev.off()
    
    sink() #Stop writing to text file for later review
    
    return(list(tickSent.train, tickSent.pred, tickSent.diff.train, tickSent.diff.pred))
}

AAPL <- getTrainTest("AAPL")
AMZN <- getTrainTest("AMZN")
BA   <- getTrainTest("BA")
DWDP <- getTrainTest("DWDP")
JNJ  <- getTrainTest("JNJ")
JPM  <- getTrainTest("JPM")
NEE  <- getTrainTest("NEE")
PG   <- getTrainTest("PG")
SPG  <- getTrainTest("SPG")
VZ   <- getTrainTest("VZ")
XOM  <- getTrainTest("XOM")

###########################################
###### Generate Random Forest Model #######
###########################################

## Following inspired by http://blog.yhat.com/posts/comparing-random-forests-in-python-and-r.html
cols <- colnames(AAPL.train)
cols <- cols[!cols %in% "date"]

## Create Random Forest Seeds
set.seed(123)
seeds <- vector(mode = "list", length = 30) #Length based on number of resamples + 1 for final model iteration
for(i in 1:29) seeds[[i]] <- sample.int(1000, 72) #sample.int second argument value based on expand.grid length
seeds[[30]] <- sample.int(1000, 1)

## Setup training parameters
ts.control <- trainControl(method="timeslice", initialWindow = 35, horizon = 14, fixedWindow = FALSE, allowParallel = TRUE, seeds = seeds, search = "grid") #35 day cv training, 14 day cv testing
tuneGridRF <- expand.grid(.mtry=c(1:72))
metric <- "Rsquared"

## Perform training
start <- Sys.time() #Start timer
rf <- train(high ~ ., data = AAPL.train[,cols], method = "rf", metric = metric, trControl = ts.control, tuneGrid = tuneGridRF, importance=TRUE)
print(Sys.time() - start)
#tuneGridRF <- expand.grid(.mtry=22)
#rf <- train(high ~ ., data = AAPL.train[,cols], method = "rf", metric = metric, trControl = ts.control, tuneGrid = tuneGridRF, importance=TRUE) #AAPL best mtry = 22
print(rf)
plot(rf)

## Evaluate metrics
(r2.train <- rSquared(AAPL.train$high, AAPL.train$high - predict(rf, AAPL.train[,cols])))
(r2.pred <- rSquared(AAPL.pred$high, AAPL.pred$high - predict(rf, AAPL.pred[,cols])))
(mse.train <- mean((AAPL.train$high - predict(rf, AAPL.train[,cols]))^2))
(mse.pred <- mean((AAPL.pred$high - predict(rf, AAPL.pred[,cols]))^2))

## Plot Rsquared Evaluation
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

## Plot trained and predicted performance
plot(as.numeric(c(AAPL.train$high, AAPL.pred$high)), type = "l", lty = 1, xaxt = "n")
axis(1, at=1:(sum(length(AAPL.train$high), length(AAPL.pred$high))), labels=FALSE)
text(1:(sum(length(AAPL.train$high), length(AAPL.pred$high))), par("usr")[3] - 0.2, labels = c(paste(AAPL.train$date, AAPL.train$hour), paste(AAPL.pred$date, AAPL.pred$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(c(predict(rf, AAPL.train[,cols]),predict(rf, AAPL.pred[,cols])), type = "l", lty = 2, lwd = 2, col = "red")

## Plot just predicted performance
plot(as.numeric(AAPL.pred$high), type = "l", lty = 1, xaxt = "n")
axis(1, at=1:(length(AAPL.pred$high)), labels=FALSE)
text(1:(length(AAPL.pred$high)), par("usr")[3] - 0.2, labels = paste(AAPL.pred$date, AAPL.pred$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(predict(rf, AAPL.pred[,cols]), type = "l", lty = 2, lwd = 2, col = "red")

## Get feature importance
feat.imp <- varImp(rf)
plot(feat.imp)

###########################################
######### Generate XG Boost Model #########
###########################################
## Setup data
cols <- cols[!cols %in% "high"]
X.train <- data.matrix(AAPL.train[,cols])
X.test <- data.matrix(AAPL.pred[,cols])
Y <- AAPL.train$high

## Create Random Forest seeds
set.seed(123)
seeds <- vector(mode = "list", length = 30) #Length based on number of resamples + 1 for final model iteration
for(i in 1:29) seeds[[i]] <- sample.int(1000, 144) #sample.int second argument value based on expand.grid nrows
seeds[[30]] <- sample.int(1000, 1)

## Setup training parameters
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

## Perform training
start <- Sys.time() #Start timer
xgbmod <- train(
    x = X.train,
    y = Y,
    method = 'xgbTree',
    metric = metric,
    trControl = ts.control,
    tuneGrid = tuneGridXGB,
    importance=TRUE)
print(Sys.time() - start)
print(xgbmod)
plot(xgbmod)

# nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
# 16      50         1 0.3     0              0.8                1         1

# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were nrounds = 350, max_depth = 6, eta = 0.05, gamma =
#     0.01, colsample_bytree = 0.5, min_child_weight = 0 and subsample = 0.5.

## Evaluate metrics
(r2.train <- rSquared(AAPL.train$high, AAPL.train$high - predict(xgbmod, X.train)))
(r2.pred <- rSquared(AAPL.pred$high, AAPL.pred$high - predict(xgbmod, X.test)))
(mse.train <- mean((AAPL.train$high - predict(xgbmod, X.train))^2))
(mse.pred <- mean((AAPL.pred$high - predict(xgbmod, X.test))^2))

## Plot Rsquared Evaluation
p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=AAPL.train$high, pred=predict(xgbmod, X.train)))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("XGBoost Regression in R r^2=", r2.train, sep=""))

p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=AAPL.pred$high, pred=predict(rf, AAPL.pred[,cols])))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("XGBoost Regression in R r^2=", r2.pred, sep=""))

## Plot trained and predicted performance
plot(as.numeric(c(AAPL.train$high, AAPL.pred$high)), type = "l", lty = 1, xaxt = "n")
axis(1, at=1:(sum(length(AAPL.train$high), length(AAPL.pred$high))), labels=FALSE)
text(1:(sum(length(AAPL.train$high), length(AAPL.pred$high))), par("usr")[3] - 0.2, labels = c(paste(AAPL.train$date, AAPL.train$hour), paste(AAPL.pred$date, AAPL.pred$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(c(predict(xgbmod, X.train),c(predict(xgbmod, X.test))), type = "l", lty = 2, lwd = 2, col = "red")

## Plot just predicted performance
plot(as.numeric(AAPL.pred$high), type = "l", lty = 1, xaxt = "n")
axis(1, at=1:(length(AAPL.pred$high)), labels=FALSE)
text(1:(length(AAPL.pred$high)), par("usr")[3] - 0.2, labels = paste(AAPL.pred$date, AAPL.pred$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(predict(xgbmod, X.test), type = "l", lty = 2, lwd = 2, col = "red")

feat.imp <- varImp(xgbmod)


####################################################
######### Generate Linear Regression Model #########
####################################################
## Setup data
X.train <- AAPL.train[,cols]
X.test <- AAPL.pred[,cols]
Y.train <- AAPL.train$high
Y.test <- AAPL.pred$high

## Check for and remove collinearity between variables
lc <- findLinearCombos(X.train)
X.train <- X.train[,lc$remove]
X.test <- X.test[,lc$remove]

## Cehck for and remove collinearity between variables again
lc <- findLinearCombos(X.train)
X.train <- X.train[,lc$remove]
X.test <- X.test[,lc$remove]

## Create Random Forest seeds
set.seed(123)

## Setup training parameters
ts.control <- trainControl(method="timeslice", initialWindow = 35, horizon = 14, fixedWindow = FALSE, allowParallel = TRUE) #35 day cv training, 14 day cv testing
tuneLength.num <- 20

## Perform training
start <- Sys.time() #Start timer
lm.mod <- train(high ~ ., data = X.train,
                method = "lm",
                trControl = ts.control,
                tuneLength=tuneLength.num)
print(Sys.time() - start)
lm.mod

## Evaluate metrics
(r2.train <- rSquared(Y.train, Y.train - predict(lm.mod, X.train)))
(r2.pred <- rSquared(Y.test, Y.test - predict(lm.mod, X.test)))
(mse.train <- mean((Y.train - predict(lm.mod, X.train))^2))
(mse.pred <- mean((Y.test - predict(lm.mod, X.test))^2))

## Plot Rsquared Evaluation
p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=Y.train, pred=predict(lm.mod, X.train)))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("RandomForest Regression in R r^2=", r2.train, sep=""))

p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=AAPL.pred$high, pred=predict(lm.mod, AAPL.pred[,cols])))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("RandomForest Regression in R r^2=", r2.pred, sep=""))

## Plot trained and predicted performance
plot(as.numeric(c(Y.train, Y.test)), type = "l", lty = 1, xaxt = "n")
axis(1, at=1:(sum(length(Y.train), length(Y.test))), labels=FALSE)
text(1:(sum(length(Y.train), length(Y.test))), par("usr")[3] - 0.2, labels = c(paste(AAPL.train$date, AAPL.train$hour), paste(AAPL.pred$date, AAPL.pred$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(c(predict(lm.mod, X.train),c(predict(lm.mod, X.test))), type = "l", lty = 2, lwd = 2, col = "red")

## Plot just predicted performance
plot(as.numeric(AAPL.pred$high), type = "l", lty = 1, ylim = c(min(predict(lm.mod, X.test))-5, max(predict(lm.mod, X.test))))
axis(1, at=1:(length(AAPL.pred$high)), labels=FALSE)
text(1:(length(AAPL.pred$high)), par("usr")[3] - 0.2, labels = paste(AAPL.pred$date, AAPL.pred$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.8, offset = -0.1)
lines(predict(lm.mod, X.test), type = "l", lty = 2, lwd = 2, col = "red")

## Get coefficients and confidence intervals
coef(lm.mod$finalModel)
confint(lm.mod$finalModel)



###########################################
######### Compare Sentiment Models ########
###########################################

resamps <- resamples(list(RandomForest = rf, XGBoost = xgbmod, LinearRegression = lm.mod))
summary(resamps)
trellis.par.set(caretTheme())
dotplot(resamps, metric = "Rsquared", main = "Sentiment Model Review - R^2")
dotplot(resamps, metric = "RMSE", main = "Sentiment Model Review - RMSE")
dotplot(resamps, metric = "MAE", main = "Sentiment Model Review - MAE")

