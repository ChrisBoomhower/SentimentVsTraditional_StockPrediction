#################################################################
## File:            SentimentModels.R
##
## Description:     Merges stock data with sentiment scores,
##                  prepares data for modeling, and then performs
##                  various modeling algorithms.

require(randomForest)
require(caret)
require(miscTools)
require(ggplot2)

setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis")
AAPL.complete <- AAPL

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

## Merge sentiment data with stock data
AAPL.complete <- merge(AAPL.complete, AAPL.stock[, c("date", "hour", "high")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
AAPL.complete <- AAPL.complete[AAPL.complete$date > "2018-02-22",] #Incomplete sentiment data before 2-23-18
AAPL.complete[is.na(AAPL.complete)] <- 0 #Fill any NA sentiment scores with 0
head(AAPL.complete[,c(1,2,3,222)],50)

glimpse(AAPL.complete)

###########################################
###### Generate Random Forest Model #######
###########################################

## Following inspired by http://blog.yhat.com/posts/comparing-random-forests-in-python-and-r.html
colnames(AAPL.complete) <- gsub(x = colnames(AAPL.complete), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames
cols <- colnames(AAPL.complete)
cols <- cols[!cols %in% "date"]
rf <- randomForest(high ~ ., data = AAPL.complete[,cols], ntree = 100)

(r2 <- rSquared(AAPL.complete$high, AAPL.complete$high - predict(rf, AAPL.complete[,cols])))

(mse <- mean((AAPL.complete$high - predict(rf, AAPL.complete[,cols]))^2))


p <- ggplot(aes(x=actual, y=pred),
            data=data.frame(actual=AAPL.complete$high, pred=predict(rf, AAPL.complete[,cols])))
p + geom_point() +
    geom_abline(color="red") +
    ggtitle(paste("RandomForest Regression in R r^2=", r2, sep=""))

plot(as.numeric(AAPL.complete$high), type = "l", lty = 1)
lines(predict(rf, AAPL.complete[,cols]), type = "l", lty = 2, lwd = 2, col = "red")


varImpPlot(rf)
head(importance(rf))
varImp(rf, type = 2)

