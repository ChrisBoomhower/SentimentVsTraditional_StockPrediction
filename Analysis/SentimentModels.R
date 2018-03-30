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
require(lattice)
require(TSPred)
require(lubridate)
require(dplyr)
require(tseries)

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
    tickSent.train <- tickSent.train[tickSent.train$date > "2018-02-26" & tickSent.train$date <= "2018-03-14",] #Incomplete sentiment data before 2-26-18
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
    tickSent.pred <- tickSent.pred[tickSent.pred$date > "2018-03-14",] #20% of data set aside for predictions
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
    pdf(paste0('Stationarity_', tick, '.pdf'))
    ## Check stationarity of variables in model
    temp <- apply(rbind(tickSent.train[2:length(tickSent.train)],tickSent.pred[2:length(tickSent.pred)]), 2, adf.test, alternative = "stationary", k=0)
    plot(sapply(temp, function(x) x$p.value), xlab = "Column Location", ylab = "Dickey-Fuller p.value", main = "Dickey-Fuller p.values") #Note that while sentiment values are stationary, stock price is not
    
    temp <- apply(rbind(tickSent.train[2:length(tickSent.train)],tickSent.pred[2:length(tickSent.pred)]), 2, adf.test, alternative = "stationary")
    plot(sapply(temp, function(x) x$p.value), xlab = "Column Location", ylab = "Augmented Dickey-Fuller p.value", main = "Augmented Dickey-Fuller p.values") #Note that while sentiment values are stationary, stock price is not
    
    ## Create high.diff training set
    tickSent.diff.train <- merge(tickSent, stock[, c("date", "hour", "high.diff")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
    tickSent.diff.train <- tickSent.diff.train[tickSent.diff.train$date > "2018-02-26" & tickSent.diff.train$date <= "2018-03-14",] #Incomplete sentiment data before 2-23-18
    tickSent.diff.train[is.na(tickSent.diff.train)] <- 0 #Fill any NA sentiment scores with 0
    colnames(tickSent.diff.train) <- gsub(x = colnames(tickSent.diff.train), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames
    colnames(tickSent.diff.train)[which(colnames(tickSent.diff.train) %in% "high.diff")] <- "high"
    
    cat("\nSanity check for matching Train df size\n")
    print(nrow(tickSent.train) == nrow(tickSent.diff.train)) #Sanity check on size
    
    ## Create high.diff test set
    tickSent.diff.pred <- merge(tickSent, stock[, c("date", "hour", "high.diff")], by = c("date", "hour"), all.y = TRUE) #Keep only rows containing stock price (since sentiment was lagged before this, we still have lagged sentiment effects)
    tickSent.diff.pred <- tickSent.diff.pred[tickSent.diff.pred$date > "2018-03-14",] #20% of data set aside for predictions
    tickSent.diff.pred[is.na(tickSent.diff.pred)] <- 0 #Fill any NA sentiment scores with 0
    colnames(tickSent.diff.pred) <- gsub(x = colnames(tickSent.diff.pred), pattern = "-", replacement = "N") #randomForest function doesn't like '-' in colnames
    colnames(tickSent.diff.pred)[which(colnames(tickSent.diff.pred) %in% "high.diff")] <- "high"
    
    cat("\nSanity check for matching Test df size\n")
    print(nrow(tickSent.pred) == nrow(tickSent.diff.pred)) #Sanity check on size
    
    #sum(!(cumsum(c(AAPL.stock$high[1], diff(AAPL.stock$high))) == AAPL.stock$high)) #Checking to see how to compare models when using differenced price
    
    on.exit(dev.off())
    on.exit(sink(), add = TRUE) #Stop writing to text file for later review
    
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
doRF <- function(train.df, pred.df, tick, metric, xform = "", orig.train.df = train.df, orig.pred.df = pred.df){
    tryCatch({
        ## Write outputs to external files for later review
        sink(paste0("doRF_", tick, xform, "_", metric, ".txt"))
        pdf(paste0('RFplots_',tick, xform, '_', metric, '.pdf'))
        
        ## Flow and plots inspired by and modified from http://blog.yhat.com/posts/comparing-random-forests-in-python-and-r.html
        ## Setup data
        cols <- colnames(train.df)
        cols <- cols[!cols %in% "date"]
        
        ## Create Random Forest Seeds
        # Seeding and timeslice methodology inspired by https://rpubs.com/crossxwill/time-series-cv
        set.seed(123)
        seeds <- vector(mode = "list", length = 44) #Length based on number of resamples + 1 for final model iteration
        for(i in 1:43) seeds[[i]] <- sample.int(1000, 72) #sample.int second argument value based on expand.grid length
        seeds[[44]] <- sample.int(1000, 1)
        
        ## Setup training parameters
        ts.control <- trainControl(method="timeslice", initialWindow = 35, horizon = 14, fixedWindow = FALSE, allowParallel = TRUE, seeds = seeds, search = "grid") #35 day cv training, 14 day cv testing
        tuneGridRF <- expand.grid(.mtry=c(1:72))
        #metric <- "Rsquared"
        
        ## Perform training
        start <- Sys.time() #Start timer
        rf <- train(high ~ ., data = train.df[,cols], method = "rf", metric = metric, trControl = ts.control, tuneGrid = tuneGridRF, importance=TRUE)
        print(Sys.time() - start)
        #tuneGridRF <- expand.grid(.mtry=22)
        #rf <- train(high ~ ., data = train.df[,cols], method = "rf", metric = metric, trControl = ts.control, tuneGrid = tuneGridRF, importance=TRUE) #AAPL best mtry = 22
        cat("\nRF Output\n")
        print(rf)
        print(plot(rf))
        
        ## Evaluate metrics
        r2.train <- rSquared(train.df$high, train.df$high - predict(rf, train.df[,cols]))
        r2.pred <- rSquared(pred.df$high, pred.df$high - predict(rf, pred.df[,cols]))
        mse.train <- mean((train.df$high - predict(rf, train.df[,cols]))^2)
        mse.pred <- mean((pred.df$high - predict(rf, pred.df[,cols]))^2)
        rmse.train <- sqrt(mse.train)
        rmse.pred <- sqrt(mse.pred)
        mae.train <- mean(abs(train.df$high - predict(rf, train.df[,cols])))
        mae.pred <- mean(abs(pred.df$high - predict(rf, pred.df[,cols])))
        mape.train <- MAPE(train.df$high, predict(rf, train.df[,cols]))
        mape.pred <- MAPE(pred.df$high, predict(rf, pred.df[,cols]))
        
        ## Plot Rsquared Evaluation
        p <- ggplot(aes(x=actual, y=pred),
                    data=data.frame(actual=train.df$high, pred=predict(rf, train.df[,cols])))
        print(p + geom_point() +
            geom_abline(color="red") +
            ggtitle(paste(tick, "RandomForest Regression: Training r^2 =", r2.train)))
        
        p <- ggplot(aes(x=actual, y=pred),
                    data=data.frame(actual=pred.df$high, pred=predict(rf, pred.df[,cols])))
        print(p + geom_point() +
            geom_abline(color="red") +
            ggtitle(paste(tick, "RandomForest Regression: Prediction r^2 =", r2.pred)))
        
        if(xform == "diff"){
            ## Plot trained and predicted performance
            plot(as.numeric(c(cumsum(c(orig.train.df$high[1], train.df$high)), cumsum(c(orig.pred.df$high[1], pred.df$high)))), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Cumulative Sum of Price[1] and DIff = Price", xaxt = "n", main = paste(tick, "RF Performance (Diff): Training + Prediction"))
            axis(1, at=1:(sum(length(train.df$high), length(pred.df$high))), labels=FALSE)
            text(1:(sum(length(train.df$high), length(pred.df$high))), par("usr")[3] - 0.2, labels = c(paste(train.df$date, train.df$hour), paste(pred.df$date, pred.df$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.5, offset = -0.1)
            lines(c(cumsum(c(orig.train.df$high[1], predict(rf, train.df[,cols]))), cumsum(c(orig.pred.df$high[1], predict(rf, pred.df[,cols])))), type = "l", lty = 2, lwd = 2, col = "red")
            abline(v = length(train.df$high)+1, lty = 2, col = "blue")
            
            ## Plot just predicted performance
            plot(as.numeric(cumsum(c(orig.pred.df$high[1], pred.df$high))), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Cumulative Sum of Price[1] and DIff = Price", xaxt = "n", ylim = c(min(c(cumsum(c(orig.pred.df$high[1], predict(rf, pred.df[,cols]))), cumsum(c(orig.pred.df$high[1], pred.df$high)))), max(c(cumsum(c(orig.pred.df$high[1], predict(rf, pred.df[,cols]))), cumsum(c(orig.pred.df$high[1], pred.df$high))))), main = paste(tick, "RF Performance (Diff): Prediction"))
            axis(1, at=1:(length(pred.df$high)), labels=FALSE)
            text(1:(length(pred.df$high)), par("usr")[3] - 0.2, labels = paste(pred.df$date, pred.df$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.6, offset = -0.1)
            lines(cumsum(c(orig.pred.df$high[1], predict(rf, pred.df[,cols]))), type = "l", lty = 2, lwd = 2, col = "red")
        }
        else{
            ## Plot trained and predicted performance
            plot(as.numeric(c(train.df$high, pred.df$high)), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Price", xaxt = "n", main = paste(tick, "RF Performance: Training + Prediction"))
            axis(1, at=1:(sum(length(train.df$high), length(pred.df$high))), labels=FALSE)
            text(1:(sum(length(train.df$high), length(pred.df$high))), par("usr")[3] - 0.2, labels = c(paste(train.df$date, train.df$hour), paste(pred.df$date, pred.df$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.5, offset = -0.1)
            lines(c(predict(rf, train.df[,cols]),predict(rf, pred.df[,cols])), type = "l", lty = 2, lwd = 2, col = "red")
            abline(v = length(train.df$high)+1, lty = 2, col = "blue")
            
            ## Plot just predicted performance
            plot(as.numeric(pred.df$high), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Price", xaxt = "n", ylim = c(min(c(predict(rf, pred.df[,cols]), pred.df$high)), max(c(predict(rf, pred.df[,cols]), pred.df$high))), main = paste(tick, "RF Performance: Prediction"))
            axis(1, at=1:(length(pred.df$high)), labels=FALSE)
            text(1:(length(pred.df$high)), par("usr")[3] - 0.2, labels = paste(pred.df$date, pred.df$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.6, offset = -0.1)
            lines(predict(rf, pred.df[,cols]), type = "l", lty = 2, lwd = 2, col = "red")
        }
        
        ## Get feature importance
        feat.imp <- varImp(rf)
        plot(feat.imp, main = paste(tick, "RF Feature Importance"))
        
        on.exit(dev.off())
        on.exit(sink(), add = TRUE)
    }, error = function(e){
        on.exit(dev.off())
        on.exit(sink(), add = TRUE)
        print(paste(tick, "RF failed"))
    })
    
    
    return(list(rf, list(r2.train, r2.pred), list(mse.train, mse.pred), list(rmse.train, rmse.pred), list(mae.train, mae.pred), list(mape.train, mape.pred), feat.imp))
}

startOverall <- Sys.time() #Start Overall timer

AAPL.rf.Rsquared <- doRF(AAPL[[1]], AAPL[[2]], "AAPL", "Rsquared")
AMZN.rf.Rsquared <- doRF(AMZN[[1]], AMZN[[2]], "AMZN", "Rsquared")
BA.rf.Rsquared   <- doRF(BA[[1]], BA[[2]], "BA", "Rsquared")
DWDP.rf.Rsquared <- doRF(DWDP[[1]], DWDP[[2]], "DWDP", "Rsquared")
JNJ.rf.Rsquared  <- doRF(JNJ[[1]], JNJ[[2]], "JNJ", "Rsquared")
JPM.rf.Rsquared  <- doRF(JPM[[1]], JPM[[2]], "JPM", "Rsquared")
NEE.rf.Rsquared  <- doRF(NEE[[1]], NEE[[2]], "NEE", "Rsquared")
PG.rf.Rsquared   <- doRF(PG[[1]], PG[[2]], "PG", "Rsquared")
SPG.rf.Rsquared  <- doRF(SPG[[1]], SPG[[2]], "SPG", "Rsquared")
VZ.rf.Rsquared   <- doRF(VZ[[1]], VZ[[2]], "VZ", "Rsquared")
XOM.rf.Rsquared  <- doRF(XOM[[1]], XOM[[2]], "XOM", "Rsquared")

AAPL.rf.RMSE <- doRF(AAPL[[1]], AAPL[[2]], "AAPL", "RMSE")
AMZN.rf.RMSE <- doRF(AMZN[[1]], AMZN[[2]], "AMZN", "RMSE")
BA.rf.RMSE   <- doRF(BA[[1]], BA[[2]], "BA", "RMSE")
DWDP.rf.RMSE <- doRF(DWDP[[1]], DWDP[[2]], "DWDP", "RMSE")
JNJ.rf.RMSE  <- doRF(JNJ[[1]], JNJ[[2]], "JNJ", "RMSE")
JPM.rf.RMSE  <- doRF(JPM[[1]], JPM[[2]], "JPM", "RMSE")
NEE.rf.RMSE  <- doRF(NEE[[1]], NEE[[2]], "NEE", "RMSE")
PG.rf.RMSE   <- doRF(PG[[1]], PG[[2]], "PG", "RMSE")
SPG.rf.RMSE  <- doRF(SPG[[1]], SPG[[2]], "SPG", "RMSE")
VZ.rf.RMSE   <- doRF(VZ[[1]], VZ[[2]], "VZ", "RMSE")
XOM.rf.RMSE  <- doRF(XOM[[1]], XOM[[2]], "XOM", "RMSE")

AAPLdiff.rf.Rsquared <- doRF(AAPL[[3]], AAPL[[4]], "AAPL", "Rsquared", "diff", AAPL[[1]], AAPL[[2]])
AMZNdiff.rf.Rsquared <- doRF(AMZN[[3]], AMZN[[4]], "AMZN", "Rsquared", "diff", AMZN[[1]], AMZN[[2]])
BAdiff.rf.Rsquared   <- doRF(BA[[3]], BA[[4]], "BA", "Rsquared", "diff", BA[[1]], BA[[2]])
DWDPdiff.rf.Rsquared <- doRF(DWDP[[3]], DWDP[[4]], "DWDP", "Rsquared", "diff", DWDP[[1]], DWDP[[2]])
JNJdiff.rf.Rsquared  <- doRF(JNJ[[3]], JNJ[[4]], "JNJ", "Rsquared", "diff", JNJ[[1]], JNJ[[2]])
JPMdiff.rf.Rsquared  <- doRF(JPM[[3]], JPM[[4]], "JPM", "Rsquared", "diff", JPM[[1]], JPM[[2]])
NEEdiff.rf.Rsquared  <- doRF(NEE[[3]], NEE[[4]], "NEE", "Rsquared", "diff", NEE[[1]], NEE[[2]])
PGdiff.rf.Rsquared   <- doRF(PG[[3]], PG[[4]], "PG", "Rsquared", "diff", PG[[1]], PG[[2]])
SPGdiff.rf.Rsquared  <- doRF(SPG[[3]], SPG[[4]], "SPG", "Rsquared", "diff", SPG[[1]], SPG[[2]])
VZdiff.rf.Rsquared   <- doRF(VZ[[3]], VZ[[4]], "VZ", "Rsquared", "diff", VZ[[1]], VZ[[2]])
XOMdiff.rf.Rsquared  <- doRF(XOM[[3]], XOM[[4]], "XOM", "Rsquared", "diff", XOM[[1]], XOM[[2]])

AAPLdiff.rf.RMSE <- doRF(AAPL[[3]], AAPL[[4]], "AAPL", "RMSE", "diff", AAPL[[1]], AAPL[[2]])
AMZNdiff.rf.RMSE <- doRF(AMZN[[3]], AMZN[[4]], "AMZN", "RMSE", "diff", AMZN[[1]], AMZN[[2]])
BAdiff.rf.RMSE   <- doRF(BA[[3]], BA[[4]], "BA", "RMSE", "diff", BA[[1]], BA[[2]])
DWDPdiff.rf.RMSE <- doRF(DWDP[[3]], DWDP[[4]], "DWDP", "RMSE", "diff", DWDP[[1]], DWDP[[2]])
JNJdiff.rf.RMSE  <- doRF(JNJ[[3]], JNJ[[4]], "JNJ", "RMSE", "diff", JNJ[[1]], JNJ[[2]])
JPMdiff.rf.RMSE  <- doRF(JPM[[3]], JPM[[4]], "JPM", "RMSE", "diff", JPM[[1]], JPM[[2]])
NEEdiff.rf.RMSE  <- doRF(NEE[[3]], NEE[[4]], "NEE", "RMSE", "diff", NEE[[1]], NEE[[2]])
PGdiff.rf.RMSE   <- doRF(PG[[3]], PG[[4]], "PG", "RMSE", "diff", PG[[1]], PG[[2]])
SPGdiff.rf.RMSE  <- doRF(SPG[[3]], SPG[[4]], "SPG", "RMSE", "diff", SPG[[1]], SPG[[2]])
VZdiff.rf.RMSE   <- doRF(VZ[[3]], VZ[[4]], "VZ", "RMSE", "diff", VZ[[1]], VZ[[2]])
XOMdiff.rf.RMSE  <- doRF(XOM[[3]], XOM[[4]], "XOM", "RMSE", "diff", XOM[[1]], XOM[[2]])

print(Sys.time() - startOverall)

###########################################
######### Generate XG Boost Model #########
###########################################
doXGB <- function(train.df, pred.df, tick, metric, xform = "", orig.train.df = train.df, orig.pred.df = pred.df){
    tryCatch({
        ## Write outputs to external files for later review
        sink(paste0("doXGB_", tick, xform, "_", metric, ".txt"))
        pdf(paste0('XGBplots_',tick, xform, '_', metric, '.pdf'))
        
        ## Flow and plots inspired by and modified from http://blog.yhat.com/posts/comparing-random-forests-in-python-and-r.html
        ## Setup data
        ## Setup data
        cols <- colnames(train.df)
        cols <- cols[!cols %in% c("date", "high")]
        X.train <- data.matrix(train.df[,cols])
        X.test <- data.matrix(pred.df[,cols])
        Y.train <- train.df$high
        Y.test <- pred.df$high
        
        ## Create seeds
        set.seed(123)
        seeds <- vector(mode = "list", length = 44) #Length based on number of resamples + 1 for final model iteration
        for(i in 1:43) seeds[[i]] <- sample.int(1000, 12) #sample.int second argument value based on expand.grid nrows
        seeds[[44]] <- sample.int(1000, 1)
        
        ## Setup training parameters
        ts.control <- trainControl(method="timeslice", initialWindow = 35, horizon = 14, fixedWindow = FALSE, allowParallel = TRUE, search = "grid") #35 day cv training, 14 day cv testing
        metric <- "RMSE"
        tuneGridXGB <- expand.grid( #See parameter descriptions at http://xgboost.readthedocs.io/en/latest/parameter.html
            nrounds=350,
            eta = c(0.3, 0.5),
            gamma = c(0, 1, 5),
            max_depth = 6,
            colsample_bytree = c(0.5, 1),
            subsample = 0.5,
            min_child_weight = 1)
        
        ## Perform training
        start <- Sys.time() #Start timer
        xgbmod <- train(
            x = X.train,
            y = Y.train,
            method = 'xgbTree',
            metric = metric,
            trControl = ts.control,
            tuneGrid = tuneGridXGB,
            importance=TRUE)
        print(Sys.time() - start)
        cat("\nXGB Output\n")
        print(xgbmod)
        print(plot(xgbmod))
        
        ## Evaluate metrics
        r2.train <- rSquared(Y.train, Y.train - predict(xgbmod, X.train))
        r2.pred <- rSquared(Y.test, Y.test - predict(xgbmod, X.test))
        mse.train <- mean((Y.train - predict(xgbmod, X.train))^2)
        mse.pred <- mean((Y.test - predict(xgbmod, X.test))^2)
        rmse.train <- sqrt(mse.train)
        rmse.pred <- sqrt(mse.pred)
        mae.train <- mean(abs(Y.train - predict(xgbmod, X.train)))
        mae.pred <- mean(abs(Y.test - predict(xgbmod, X.test)))
        mape.train <- MAPE(Y.train, predict(xgbmod, X.train))
        mape.pred <- MAPE(Y.test, predict(xgbmod, X.test))
        
        ## Plot Rsquared Evaluation
        p <- ggplot(aes(x=actual, y=pred),
                    data=data.frame(actual=Y.train, pred=predict(xgbmod, X.train)))
        print(p + geom_point() +
            geom_abline(color="red") +
            ggtitle(paste("XGBoost Regression in R r^2=", r2.train, sep="")))
        
        p <- ggplot(aes(x=actual, y=pred),
                    data=data.frame(actual=Y.test, pred=predict(xgbmod, X.test)))
        print(p + geom_point() +
            geom_abline(color="red") +
            ggtitle(paste("XGBoost Regression in R r^2=", r2.pred, sep="")))
        
        if(xform == "diff"){
            ## Plot trained and predicted performance
            plot(as.numeric(c(cumsum(c(orig.train.df$high[1], Y.train)), cumsum(c(orig.pred.df$high[1], Y.test)))), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Cumulative Sum of Price[1] and DIff = Price", xaxt = "n", main = paste(tick, "XGBoost Performance (Diff): Training + Prediction"))
            axis(1, at=1:(sum(length(Y.train), length(Y.test))), labels=FALSE)
            text(1:(sum(length(Y.train), length(Y.test))), par("usr")[3] - 0.2, labels = c(paste(train.df$date, train.df$hour), paste(pred.df$date, pred.df$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.5, offset = -0.1)
            lines(c(cumsum(c(orig.train.df$high[1], predict(xgbmod, X.train))), cumsum(c(orig.pred.df$high[1], predict(xgbmod, X.test)))), type = "l", lty = 2, lwd = 2, col = "red")
            abline(v = length(Y.train)+1, lty = 2, col = "blue")
            
            ## Plot just predicted performance
            plot(as.numeric(cumsum(c(orig.pred.df$high[1], Y.test))), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Cumulative Sum of Price[1] and DIff = Price", xaxt = "n", ylim = c(min(c(cumsum(c(orig.pred.df$high[1], predict(xgbmod, X.test))), cumsum(c(orig.pred.df$high[1], Y.test)))), max(c(cumsum(c(orig.pred.df$high[1], predict(xgbmod, X.test))), cumsum(c(orig.pred.df$high[1], Y.test))))), main = paste(tick, "XGBoost Performance (Diff): Prediction"))
            axis(1, at=1:(length(Y.test)), labels=FALSE)
            text(1:(length(Y.test)), par("usr")[3] - 0.2, labels = paste(pred.df$date, pred.df$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.6, offset = -0.1)
            lines(cumsum(c(orig.pred.df$high[1], predict(xgbmod, X.test))), type = "l", lty = 2, lwd = 2, col = "red")
        }
        else{
            ## Plot trained and predicted performance
            plot(as.numeric(c(Y.train, Y.test)), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Price", xaxt = "n", main = paste(tick, "XGBoost Performance: Training + Prediction"))
            axis(1, at=1:(sum(length(Y.train), length(Y.test))), labels=FALSE)
            text(1:(sum(length(Y.train), length(Y.test))), par("usr")[3] - 0.2, labels = c(paste(train.df$date, train.df$hour), paste(pred.df$date, pred.df$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.5, offset = -0.1)
            lines(c(predict(xgbmod, train.df[,cols]),predict(xgbmod, X.test)), type = "l", lty = 2, lwd = 2, col = "red")
            abline(v = length(Y.train)+1, lty = 2, col = "blue")
            
            ## Plot just predicted performance
            plot(as.numeric(Y.test), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Price", xaxt = "n", ylim = c(min(c(predict(xgbmod, X.test), Y.test)), max(c(predict(xgbmod, X.test), Y.test))), main = paste(tick, "XGBoost Performance: Prediction"))
            axis(1, at=1:(length(Y.test)), labels=FALSE)
            text(1:(length(Y.test)), par("usr")[3] - 0.2, labels = paste(pred.df$date, pred.df$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.6, offset = -0.1)
            lines(predict(xgbmod, X.test), type = "l", lty = 2, lwd = 2, col = "red")
        }
        tryCatch({
            feat.imp <- varImp(xgbmod)
            print(plot(feat.imp, main = paste(tick, "RF Feature Importance")))
            
            on.exit(dev.off())
            on.exit(sink(), add = TRUE)
            return(list(xgbmod, list(r2.train, r2.pred), list(mse.train, mse.pred), list(rmse.train, rmse.pred), list(mae.train, mae.pred), list(mape.train, mape.pred), feat.imp))
        }, error = function(e){
            on.exit(dev.off())
            on.exit(sink(), add = TRUE)
            return(list(xgbmod, list(r2.train, r2.pred), list(mse.train, mse.pred), list(rmse.train, rmse.pred), list(mae.train, mae.pred), list(mape.train, mape.pred), NA))
        })
    }, error = function(e){
        on.exit(dev.off())
        on.exit(sink(), add = TRUE)
        print(paste(tick, "XGB failed"))
    })
    
}

startOverall <- Sys.time() #Start Overall timer

AAPL.xgb.Rsquared <- doXGB(AAPL[[1]], AAPL[[2]], "AAPL", "Rsquared")
AMZN.xgb.Rsquared <- doXGB(AMZN[[1]], AMZN[[2]], "AMZN", "Rsquared")
BA.xgb.Rsquared   <- doXGB(BA[[1]], BA[[2]], "BA", "Rsquared")
DWDP.xgb.Rsquared <- doXGB(DWDP[[1]], DWDP[[2]], "DWDP", "Rsquared")
JNJ.xgb.Rsquared  <- doXGB(JNJ[[1]], JNJ[[2]], "JNJ", "Rsquared")
JPM.xgb.Rsquared  <- doXGB(JPM[[1]], JPM[[2]], "JPM", "Rsquared")
NEE.xgb.Rsquared  <- doXGB(NEE[[1]], NEE[[2]], "NEE", "Rsquared")
PG.xgb.Rsquared   <- doXGB(PG[[1]], PG[[2]], "PG", "Rsquared")
SPG.xgb.Rsquared  <- doXGB(SPG[[1]], SPG[[2]], "SPG", "Rsquared")
VZ.xgb.Rsquared   <- doXGB(VZ[[1]], VZ[[2]], "VZ", "Rsquared")
XOM.xgb.Rsquared  <- doXGB(XOM[[1]], XOM[[2]], "XOM", "Rsquared")

AAPL.xgb.RMSE <- doXGB(AAPL[[1]], AAPL[[2]], "AAPL", "RMSE")
AMZN.xgb.RMSE <- doXGB(AMZN[[1]], AMZN[[2]], "AMZN", "RMSE")
BA.xgb.RMSE   <- doXGB(BA[[1]], BA[[2]], "BA", "RMSE")
DWDP.xgb.RMSE <- doXGB(DWDP[[1]], DWDP[[2]], "DWDP", "RMSE")
JNJ.xgb.RMSE  <- doXGB(JNJ[[1]], JNJ[[2]], "JNJ", "RMSE")
JPM.xgb.RMSE  <- doXGB(JPM[[1]], JPM[[2]], "JPM", "RMSE")
NEE.xgb.RMSE  <- doXGB(NEE[[1]], NEE[[2]], "NEE", "RMSE")
PG.xgb.RMSE   <- doXGB(PG[[1]], PG[[2]], "PG", "RMSE")
SPG.xgb.RMSE  <- doXGB(SPG[[1]], SPG[[2]], "SPG", "RMSE")
VZ.xgb.RMSE   <- doXGB(VZ[[1]], VZ[[2]], "VZ", "RMSE")
XOM.xgb.RMSE  <- doXGB(XOM[[1]], XOM[[2]], "XOM", "RMSE")

AAPLdiff.xgb.Rsquared <- doXGB(AAPL[[3]], AAPL[[4]], "AAPL", "Rsquared", "diff", AAPL[[1]], AAPL[[2]])
AMZNdiff.xgb.Rsquared <- doXGB(AMZN[[3]], AMZN[[4]], "AMZN", "Rsquared", "diff", AMZN[[1]], AMZN[[2]])
BAdiff.xgb.Rsquared   <- doXGB(BA[[3]], BA[[4]], "BA", "Rsquared", "diff", BA[[1]], BA[[2]])
DWDPdiff.xgb.Rsquared <- doXGB(DWDP[[3]], DWDP[[4]], "DWDP", "Rsquared", "diff", DWDP[[1]], DWDP[[2]])
JNJdiff.xgb.Rsquared  <- doXGB(JNJ[[3]], JNJ[[4]], "JNJ", "Rsquared", "diff", JNJ[[1]], JNJ[[2]])
JPMdiff.xgb.Rsquared  <- doXGB(JPM[[3]], JPM[[4]], "JPM", "Rsquared", "diff", JPM[[1]], JPM[[2]])
NEEdiff.xgb.Rsquared  <- doXGB(NEE[[3]], NEE[[4]], "NEE", "Rsquared", "diff", NEE[[1]], NEE[[2]])
PGdiff.xgb.Rsquared   <- doXGB(PG[[3]], PG[[4]], "PG", "Rsquared", "diff", PG[[1]], PG[[2]])
SPGdiff.xgb.Rsquared  <- doXGB(SPG[[3]], SPG[[4]], "SPG", "Rsquared", "diff", SPG[[1]], SPG[[2]])
VZdiff.xgb.Rsquared   <- doXGB(VZ[[3]], VZ[[4]], "VZ", "Rsquared", "diff", VZ[[1]], VZ[[2]])
XOMdiff.xgb.Rsquared  <- doXGB(XOM[[3]], XOM[[4]], "XOM", "Rsquared", "diff", XOM[[1]], XOM[[2]])

AAPLdiff.xgb.RMSE <- doXGB(AAPL[[3]], AAPL[[4]], "AAPL", "RMSE", "diff", AAPL[[1]], AAPL[[2]])
AMZNdiff.xgb.RMSE <- doXGB(AMZN[[3]], AMZN[[4]], "AMZN", "RMSE", "diff", AMZN[[1]], AMZN[[2]])
BAdiff.xgb.RMSE   <- doXGB(BA[[3]], BA[[4]], "BA", "RMSE", "diff", BA[[1]], BA[[2]])
DWDPdiff.xgb.RMSE <- doXGB(DWDP[[3]], DWDP[[4]], "DWDP", "RMSE", "diff", DWDP[[1]], DWDP[[2]])
JNJdiff.xgb.RMSE  <- doXGB(JNJ[[3]], JNJ[[4]], "JNJ", "RMSE", "diff", JNJ[[1]], JNJ[[2]])
JPMdiff.xgb.RMSE  <- doXGB(JPM[[3]], JPM[[4]], "JPM", "RMSE", "diff", JPM[[1]], JPM[[2]])
NEEdiff.xgb.RMSE  <- doXGB(NEE[[3]], NEE[[4]], "NEE", "RMSE", "diff", NEE[[1]], NEE[[2]])
PGdiff.xgb.RMSE   <- doXGB(PG[[3]], PG[[4]], "PG", "RMSE", "diff", PG[[1]], PG[[2]])
SPGdiff.xgb.RMSE  <- doXGB(SPG[[3]], SPG[[4]], "SPG", "RMSE", "diff", SPG[[1]], SPG[[2]])
VZdiff.xgb.RMSE   <- doXGB(VZ[[3]], VZ[[4]], "VZ", "RMSE", "diff", VZ[[1]], VZ[[2]])
XOMdiff.xgb.RMSE  <- doXGB(XOM[[3]], XOM[[4]], "XOM", "RMSE", "diff", XOM[[1]], XOM[[2]])

print(Sys.time() - startOverall)


####################################################
######### Generate Linear Regression Model #########
####################################################
doLM <- function(train.df, pred.df, tick, metric, xform = "", orig.train.df = train.df, orig.pred.df = pred.df){
    tryCatch({    
        ## Write outputs to external files for later review
        sink(paste0("doLM_", tick, xform, "_", metric, ".txt"))
        pdf(paste0('LMplots_',tick, xform, "_", metric, '.pdf'))
        
        ## Flow and plots inspired by and modified from http://blog.yhat.com/posts/comparing-random-forests-in-python-and-r.html
        ## Setup data
        cols <- colnames(train.df)
        cols <- cols[!cols %in% "date"]
        
        X.train <- train.df[,cols]
        X.test <- pred.df[,cols]
        Y.train <- train.df$high
        Y.test <- pred.df$high
        
        ## Check for and remove collinearity between variables
        repeat{
            lc <- findLinearCombos(X.train[,!(colnames(X.train) %in% "high")])
            if(!is.null(lc$remove)){
                X.train <- X.train[,-lc$remove]
                X.test <- X.test[,-lc$remove]
            }
            else break
        }
        
        ## Create Random Forest seeds
        set.seed(123)
        
        ## Setup training parameters
        ts.control <- trainControl(method="timeslice", initialWindow = 35, horizon = 14, fixedWindow = FALSE, allowParallel = TRUE) #35 day cv training, 14 day cv testing
        tuneLength.num <- 2
        #metric <- "RMSE"
        
        ## Perform training
        start <- Sys.time() #Start timer
        lm.mod <- train(high ~ ., data = X.train,
                        method = "lm",
                        metric = metric,
                        trControl = ts.control,
                        tuneLength=tuneLength.num)
        print(Sys.time() - start)
        cat("\nRF Output\n")
        print(lm.mod)
        
        ## Evaluate metrics
        r2.train <- rSquared(train.df$high, train.df$high - predict(lm.mod, train.df[,cols]))
        r2.pred <- rSquared(pred.df$high, pred.df$high - predict(lm.mod, pred.df[,cols]))
        mse.train <- mean((train.df$high - predict(lm.mod, train.df[,cols]))^2)
        mse.pred <- mean((pred.df$high - predict(lm.mod, pred.df[,cols]))^2)
        rmse.train <- sqrt(mse.train)
        rmse.pred <- sqrt(mse.pred)
        mae.train <- mean(abs(train.df$high - predict(lm.mod, train.df[,cols])))
        mae.pred <- mean(abs(pred.df$high - predict(lm.mod, pred.df[,cols])))
        mape.train <- MAPE(train.df$high, predict(lm.mod, train.df[,cols]))
        mape.pred <- MAPE(pred.df$high, predict(lm.mod, pred.df[,cols]))
        
        
        ## Plot Rsquared Evaluation
        p <- ggplot(aes(x=actual, y=pred),
                    data=data.frame(actual=Y.train, pred=predict(lm.mod, X.train)))
        print(p + geom_point() +
            geom_abline(color="red") +
            ggtitle(paste(tick, "Linear Regression: Training r^2 =", r2.train)))
        
        p <- ggplot(aes(x=actual, y=pred),
                    data=data.frame(actual=pred.df$high, pred=predict(lm.mod, pred.df[,cols])))
        print(p + geom_point() +
            geom_abline(color="red") +
            ggtitle(paste(tick, "Linear Regression: Prediction r^2 =", r2.pred)))
        
        if(xform == "diff"){
            ## Plot trained and predicted performance
            plot(as.numeric(c(cumsum(c(orig.train.df$high[1], Y.train)), cumsum(c(orig.pred.df$high[1], Y.test)))), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Cumulative Sum of Price[1] and DIff = Price", xaxt = "n", main = paste(tick, "LM Performance (Diff): Training + Prediction"))
            axis(1, at=1:(sum(length(Y.train), length(Y.test))), labels=FALSE)
            text(1:(sum(length(Y.train), length(Y.test))), par("usr")[3] - 0.2, labels = c(paste(train.df$date, train.df$hour), paste(pred.df$date, pred.df$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.5, offset = -0.1)
            lines(c(cumsum(c(orig.train.df$high[1], predict(lm.mod, X.train))), cumsum(c(orig.pred.df$high[1], predict(lm.mod, X.test)))), type = "l", lty = 2, lwd = 2, col = "red")
            abline(v = length(Y.train)+1, lty = 2, col = "blue")
    
            ## Plot just predicted performance
            plot(as.numeric(cumsum(c(orig.pred.df$high[1], Y.test))), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Cumulative Sum of Price[1] and DIff = Price", xaxt = "n", ylim = c(min(c(cumsum(c(orig.pred.df$high[1], predict(lm.mod, X.test))), cumsum(c(orig.pred.df$high[1], Y.test)))), max(c(cumsum(c(orig.pred.df$high[1], predict(lm.mod, X.test))), cumsum(c(orig.pred.df$high[1], Y.test))))), main = paste(tick, "LM Performance (Diff): Prediction"))
            axis(1, at=1:(length(Y.test)), labels=FALSE)
            text(1:(length(Y.test)), par("usr")[3] - 0.2, labels = paste(pred.df$date, pred.df$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.6, offset = -0.1)
            lines(cumsum(c(orig.pred.df$high[1], predict(lm.mod, X.test))), type = "l", lty = 2, lwd = 2, col = "red")
        }
        else{
            ## Plot trained and predicted performance
            plot(as.numeric(c(Y.train, Y.test)), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Price", xaxt = "n", main = paste(tick, "LM Performance: Training + Prediction"))
            axis(1, at=1:(sum(length(Y.train), length(Y.test))), labels=FALSE)
            text(1:(sum(length(Y.train), length(Y.test))), par("usr")[3] - 0.2, labels = c(paste(train.df$date, train.df$hour), paste(pred.df$date, pred.df$hour)), srt = 90, pos = 2, xpd = TRUE, cex = 0.5, offset = -0.1)
            lines(c(predict(lm.mod, train.df[,cols]),predict(lm.mod, pred.df[,cols])), type = "l", lty = 2, lwd = 2, col = "red")
            abline(v = length(Y.train)+1, lty = 2, col = "blue")
    
            ## Plot just predicted performance
            plot(as.numeric(Y.test), type = "l", lty = 1, xlab = "Date & Hour", ylab = "Price", xaxt = "n", ylim = c(min(c(predict(lm.mod, pred.df[,cols]), Y.test)), max(c(predict(lm.mod, pred.df[,cols]), Y.test))), main = paste(tick, "LM Performance: Prediction"))
            axis(1, at=1:(length(Y.test)), labels=FALSE)
            text(1:(length(Y.test)), par("usr")[3] - 0.2, labels = paste(pred.df$date, pred.df$hour), srt = 90, pos = 2, xpd = TRUE, cex = 0.6, offset = -0.1)
            lines(predict(lm.mod, pred.df[,cols]), type = "l", lty = 2, lwd = 2, col = "red")
        }
    
        ## Get coefficients and confidence intervals
        coeffs <- coef(lm.mod$finalModel)
        confints <- confint(lm.mod$finalModel)
        cat("/nCoefficients/n")
        print(coeffs)
        cat("/nConfidence Intervals/n")
        print(confints)
        
        on.exit(dev.off())
        on.exit(sink(), add = TRUE)
    }, error = function(e){
        on.exit(dev.off())
        on.exit(sink(), add = TRUE)
        print(paste(tick, "LM failed"))
    })

    return(list(lm.mod, list(r2.train, r2.pred), list(mse.train, mse.pred), list(rmse.train, rmse.pred), list(mae.train, mae.pred), list(mape.train, mape.pred), coeffs, confints))
}

startOverall <- Sys.time() #Start Overall timer

AAPL.lm.Rsquared <- doLM(AAPL[[1]], AAPL[[2]], "AAPL", "Rsquared")
AMZN.lm.Rsquared <- doLM(AMZN[[1]], AMZN[[2]], "AMZN", "Rsquared")
BA.lm.Rsquared   <- doLM(BA[[1]], BA[[2]], "BA", "Rsquared")
DWDP.lm.Rsquared <- doLM(DWDP[[1]], DWDP[[2]], "DWDP", "Rsquared")
JNJ.lm.Rsquared  <- doLM(JNJ[[1]], JNJ[[2]], "JNJ", "Rsquared")
JPM.lm.Rsquared  <- doLM(JPM[[1]], JPM[[2]], "JPM", "Rsquared")
NEE.lm.Rsquared  <- doLM(NEE[[1]], NEE[[2]], "NEE", "Rsquared")
PG.lm.Rsquared   <- doLM(PG[[1]], PG[[2]], "PG", "Rsquared")
SPG.lm.Rsquared  <- doLM(SPG[[1]], SPG[[2]], "SPG", "Rsquared")
VZ.lm.Rsquared   <- doLM(VZ[[1]], VZ[[2]], "VZ", "Rsquared")
XOM.lm.Rsquared  <- doLM(XOM[[1]], XOM[[2]], "XOM", "Rsquared")

AAPL.lm.RMSE <- doLM(AAPL[[1]], AAPL[[2]], "AAPL", "RMSE")
AMZN.lm.RMSE <- doLM(AMZN[[1]], AMZN[[2]], "AMZN", "RMSE")
BA.lm.RMSE   <- doLM(BA[[1]], BA[[2]], "BA", "RMSE")
DWDP.lm.RMSE <- doLM(DWDP[[1]], DWDP[[2]], "DWDP", "RMSE")
JNJ.lm.RMSE  <- doLM(JNJ[[1]], JNJ[[2]], "JNJ", "RMSE")
JPM.lm.RMSE  <- doLM(JPM[[1]], JPM[[2]], "JPM", "RMSE")
NEE.lm.RMSE  <- doLM(NEE[[1]], NEE[[2]], "NEE", "RMSE")
PG.lm.RMSE   <- doLM(PG[[1]], PG[[2]], "PG", "RMSE")
SPG.lm.RMSE  <- doLM(SPG[[1]], SPG[[2]], "SPG", "RMSE")
VZ.lm.RMSE   <- doLM(VZ[[1]], VZ[[2]], "VZ", "RMSE")
XOM.lm.RMSE  <- doLM(XOM[[1]], XOM[[2]], "XOM", "RMSE")

AAPLdiff.lm.Rsquared <- doLM(AAPL[[3]], AAPL[[4]], "AAPL", "Rsquared", "diff", AAPL[[1]], AAPL[[2]])
AMZNdiff.lm.Rsquared <- doLM(AMZN[[3]], AMZN[[4]], "AMZN", "Rsquared", "diff", AMZN[[1]], AMZN[[2]])
BAdiff.lm.Rsquared   <- doLM(BA[[3]], BA[[4]], "BA", "Rsquared", "diff", BA[[1]], BA[[2]])
DWDPdiff.lm.Rsquared <- doLM(DWDP[[3]], DWDP[[4]], "DWDP", "Rsquared", "diff", DWDP[[1]], DWDP[[2]])
JNJdiff.lm.Rsquared  <- doLM(JNJ[[3]], JNJ[[4]], "JNJ", "Rsquared", "diff", JNJ[[1]], JNJ[[2]])
JPMdiff.lm.Rsquared  <- doLM(JPM[[3]], JPM[[4]], "JPM", "Rsquared", "diff", JPM[[1]], JPM[[2]])
NEEdiff.lm.Rsquared  <- doLM(NEE[[3]], NEE[[4]], "NEE", "Rsquared", "diff", NEE[[1]], NEE[[2]])
PGdiff.lm.Rsquared   <- doLM(PG[[3]], PG[[4]], "PG", "Rsquared", "diff", PG[[1]], PG[[2]])
SPGdiff.lm.Rsquared  <- doLM(SPG[[3]], SPG[[4]], "SPG", "Rsquared", "diff", SPG[[1]], SPG[[2]])
VZdiff.lm.Rsquared   <- doLM(VZ[[3]], VZ[[4]], "VZ", "Rsquared", "diff", VZ[[1]], VZ[[2]])
XOMdiff.lm.Rsquared  <- doLM(XOM[[3]], XOM[[4]], "XOM", "Rsquared", "diff", XOM[[1]], XOM[[2]])

AAPLdiff.lm.RMSE <- doLM(AAPL[[3]], AAPL[[4]], "AAPL", "RMSE", "diff", AAPL[[1]], AAPL[[2]])
AMZNdiff.lm.RMSE <- doLM(AMZN[[3]], AMZN[[4]], "AMZN", "RMSE", "diff", AMZN[[1]], AMZN[[2]])
BAdiff.lm.RMSE   <- doLM(BA[[3]], BA[[4]], "BA", "RMSE", "diff", BA[[1]], BA[[2]])
DWDPdiff.lm.RMSE <- doLM(DWDP[[3]], DWDP[[4]], "DWDP", "RMSE", "diff", DWDP[[1]], DWDP[[2]])
JNJdiff.lm.RMSE  <- doLM(JNJ[[3]], JNJ[[4]], "JNJ", "RMSE", "diff", JNJ[[1]], JNJ[[2]])
JPMdiff.lm.RMSE  <- doLM(JPM[[3]], JPM[[4]], "JPM", "RMSE", "diff", JPM[[1]], JPM[[2]])
NEEdiff.lm.RMSE  <- doLM(NEE[[3]], NEE[[4]], "NEE", "RMSE", "diff", NEE[[1]], NEE[[2]])
PGdiff.lm.RMSE   <- doLM(PG[[3]], PG[[4]], "PG", "RMSE", "diff", PG[[1]], PG[[2]])
SPGdiff.lm.RMSE  <- doLM(SPG[[3]], SPG[[4]], "SPG", "RMSE", "diff", SPG[[1]], SPG[[2]])
VZdiff.lm.RMSE   <- doLM(VZ[[3]], VZ[[4]], "VZ", "RMSE", "diff", VZ[[1]], VZ[[2]])
XOMdiff.lm.RMSE  <- doLM(XOM[[3]], XOM[[4]], "XOM", "RMSE", "diff", XOM[[1]], XOM[[2]])

print(Sys.time() - startOverall)

###########################################
######### Compare Sentiment Models ########
###########################################

## Compare models for each ticker graphically
compareMods <- function(pat){
    pdf(paste0('Comparison_', pat, '.pdf'))
    
    ## Get resample results for each model
    modList <- ls(pattern = paste0(pat,"."), envir=.GlobalEnv)
    if(pat == 'PG') modList <- modList[!grepl('^SPG', modList)]
    comp <- lapply(modList, function(x) get(x)[[1]])
    names(comp) <- modList
    comp <- resamples(comp)
    
    ## Compare metrics of resample results
    trellis.par.set(caretTheme())
    print(dotplot(comp, metric = "Rsquared", main = "Sentiment Model Review - R^2"))
    print(dotplot(comp, metric = "RMSE", main = "Sentiment Model Review - RMSE"))
    print(dotplot(comp, metric = "MAE", main = "Sentiment Model Review - MAE"))
    
    ## Compare and rank overall metrics
    modInfo <- data.frame(model.name = character(), ticker = character(), r2.train = numeric(), r2.pred = numeric(), mse.train = numeric(), mse.pred = numeric(), rmse.train = numeric(), rmse.pred = numeric(), mae.train = numeric(), mae.pred = numeric(), mape.train = numeric(), mape.pred = numeric())
    for(m in modList){
        modInfo  <-  rbind(modInfo, data.frame(model.name = m, ticker = strsplit(m, '\\.|d')[[1]][1],
                                                 r2.train = get(m)[[2]][[1]], r2.pred = get(m)[[2]][[2]],
                                                 mse.train = get(m)[[3]][[1]], mse.pred = get(m)[[3]][[2]],
                                                 rmse.train = get(m)[[4]][[1]], rmse.pred = get(m)[[4]][[2]],
                                                 mae.train = get(m)[[5]][[1]], mae.pred = get(m)[[5]][[2]],
                                                 mape.train = get(m)[[6]][[1]], mape.pred = get(m)[[6]][[2]]))
    }
    
    model.name <- modInfo$model.name
    ticker <- modInfo$ticker
    r2.train <- rank(-modInfo$r2.train, ties.method = "min")
    r2.pred <- NA #NA chosen since prediction rSquared values are wonky
    mse.train <- rank(modInfo$mse.train, ties.method = "min")
    mse.pred <- rank(modInfo$mse.pred, ties.method = "min")
    rmse.train <- rank(modInfo$rmse.train, ties.method = "min")
    rmse.pred <- rank(modInfo$rmse.pred, ties.method = "min")
    mae.train <- rank(modInfo$mae.train, ties.method = "min")
    mae.pred <- rank(modInfo$mae.pred, ties.method = "min")
    mape.train <- rank(modInfo$mape.train, ties.method = "min")
    mape.pred <- rank(modInfo$mape.pred, ties.method = "min")
    
    ## Make final rankings
    modRanks <- data.frame(model.name, ticker, r2.train, r2.pred, mse.train, mse.pred, rmse.train, rmse.pred, mae.train, mae.pred, mape.train, mape.pred)
    modRanks$rank.overall <- rank(rowSums(modRanks[,3:length(modRanks)], na.rm = TRUE), ties.method = "min")
    modRanks$rank.train <- rank(rowSums(modRanks[,c(3,5,7,9,11)], na.rm = TRUE), ties.method = "min")
    modRanks$rank.pred <- rank(rowSums(modRanks[,c(4,6,8,10,12)], na.rm = TRUE), ties.method = "min")
    
    
    ## Write to global variables
    assign(paste0('Comparison.', pat), modInfo, envir=.GlobalEnv)
    assign(paste0('Ranks.', pat), modRanks, envir=.GlobalEnv)
    
    on.exit(dev.off())
    
    return(comp)
}

## Compare models for each ticker
compareMods('AAPL')
compareMods('AMZN')
compareMods('BA')
compareMods('DWDP')
compareMods('JNJ')
compareMods('JPM')
compareMods('NEE')
compareMods('PG')
compareMods('SPG')
compareMods('VZ')
compareMods('XOM')

###########################################
######### Output data to Tableau ##########
###########################################

## Output data for visualization in Tableau
allTickDF <- rbind(AAPL[[1]], AAPL[[2]]) #First, output non-diffed price and scores
allTickDF$highDiff <- c(AAPL[[3]][[ncol(AAPL[[3]])]], AAPL[[4]][[ncol(AAPL[[4]])]])
allTickDF$ticker <- 'AAPL'
ticks10 <- c('AMZN', 'BA', 'DWDP', 'JNJ', 'JPM', 'NEE', 'PG', 'SPG', 'VZ', 'XOM')
for(t in ticks10){
    temp <- rbind(get(t)[[1]], get(t)[[2]])
    temp$highDiff <- c(get(t)[[3]][[ncol(get(t)[[3]])]], get(t)[[4]][[ncol(get(t)[[4]])]])
    temp$ticker <- t
    allTickDF <- rbind(allTickDF, temp)
}
rm(t, temp, ticks10)
write.csv(allTickDF, 'Data/Tableau/allTickDF.csv', row.names = FALSE)

## Output feature importance where available
featureRank <- data.frame(model = character(), ticker = character(), feature = character(), importance = numeric())
decTrees <- ls(pattern = '.xgb.|.rf.')
for (m in decTrees){
    if(!is.na(get(m)[length(get(m))])){
        temp <- data.frame(model = m, ticker = strsplit(m, '\\.')[[1]][1],
                           feature = row.names(get(m)[[length(get(m))]][[1]]),
                           importance = get(m)[[length(get(m))]][[1]])
        featureRank <- rbind(featureRank, temp)
    }
}
rm(m, temp, decTrees)
write.csv(featureRank, 'Data/Tableau/featureRank.csv', row.names = FALSE)

## Output coefficients and confidence intervals where available
coeffConfs <- data.frame(model = character(), ticker = character(), variable = character(),
                          coeff = numeric(), confInt.Lower = numeric(), confInt.Upper = numeric())
linMods <- ls(pattern = '.lm.')
for (m in linMods){
    if(!is.na(get(m)[length(get(m))])){
        temp <- data.frame(model = m, ticker = strsplit(m, '\\.')[[1]][1],
                           variable = names(get(m)[[length(get(m))-1]]),
                           coeff = get(m)[[length(get(m))-1]],
                           confInt.Lower = get(m)[[length(get(m))]][,1],
                           confInt.Upper = get(m)[[length(get(m))]][,2])
        coeffConfs <- rbind(coeffConfs, temp)
    }
}
rm(m, temp, linMods)
write.csv(coeffConfs, 'Data/Tableau/coeffConfs.csv', row.names = FALSE)

## Output metric and ranking tables
comps <- ls(pattern = '^Comparison\\.')
modelMetrics <- get(comps[1])
for(c in comps[-1]){
    modelMetrics <- rbind(modelMetrics, get(c))
}
rm(comps, c)
write.csv(modelMetrics, 'Data/Tableau/modelMetrics.csv', row.names = FALSE)

rankings <- ls(pattern = '^Ranks\\.')
modelRanks <- get(rankings[1])
for(r in rankings[-1]){
    modelRanks <- rbind(modelRanks, get(r))
}
rm(rankings, r)
write.csv(modelRanks, 'Data/Tableau/modelRanks.csv', row.names = FALSE)