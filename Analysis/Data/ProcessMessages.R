#################################################################
## File:            ProcessMessages.R
##
## Description:     Consolidate StockTwits data pulled previously
##                  and generate sentiment scores.

require(plyr)
require(jsonlite)

## Consolidate StockTwits ticker symbol data into single dataframes with duplicates dropped
setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis/Data/StockTwits")
ST.tickers <- list.files(pattern = '[^getStockTwits.R]')
for(s in ST.tickers){
    setwd(s)

    files <- list.files(pattern = '\\.RDS$')
    dat_list <- lapply(files, readRDS)
    dat_list <- lapply(dat_list, flatten) #Need to flatten dataframes since dfs from JSON contain nested dfs
    df <- ldply(dat_list, data.frame) #Combine list dataframes into single dataframe
    assign(paste0(s,".ST"), unique(df)) #Drop duplicates and assign to variable named after ticker symbol

    setwd('..')
}
rm(df)

## Consolidate Twitter ticker symbol data into single dataframes with duplicates dropped
setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis/Data/TwitterData")
T.tickers <- list.files(pattern = '[^getTweets.R]')
for(s in T.tickers){
    setwd(s)
    
    files <- list.files(pattern = '\\.RDS$')
    dat_list <- lapply(files, readRDS)
    df <- ldply(dat_list, data.frame)
    assign(paste0(s,".T"), unique(df)) #Drop duplicates and assign to variable named after ticker symbol
    
    setwd('..')
}
rm(df)

