#################################################################
## File:            getStockTwits.R
##
## Description:     This file is a scheduled script designed to
##                  gather message data from StockTwits for our
##                  portfolio ticker symbols. The data are stored
##                  in external RDS files for later processing.
##
## Frequency:       Scheduled to run hourly

library(jsonlite)
require(plyr)

setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis/Data/StockTwits")

## Get system time for script run
date<-Sys.time()
date<-as.character(date)
date <- gsub(":","_", date)

## Function to obtain and write twit data to dataframe class
getTwit <- function(symb){
    ## Get twits
    recentTwits <- stream_in(url(paste0("https://api.stocktwits.com/api/2/streams/symbol/", symb, ".json")))
    
    ## Save twits
    setwd(symb)
    name<-paste0(symb,"_",date,".RDS")
    saveRDS(as.data.frame(recentTwits$messages), file = name)
    setwd("..")
    
    return(paste(symb, "data saved"))
}

## Get twit data for portfolio tickers based on folder listings
for(s in list.files(pattern = '[^getStockTwits.R]')){
    getTwit(s)
}
