#################################################################
## File:            getNews.R
##
## Description:     This file is a scheduled script designed to
##                  gather news article descriptions from YahooFinance
##                  for our portfolio ticker symbols. The data are
##                  stored in external RDS files for later processing.
##
## Frequency:       Scheduled to run every 1 hour

setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis/Data/YahooFinance")

library(tm.plugin.webmining)
library(tm)

## Get system time for script run
date<-Sys.time()
date<-as.character(date)
date <- gsub(":","_", date)
date

## Function to extract meta elements to vector
enum <- function(data.list, data.vec){
    en = 1
    for(i in data.list){
        data.vec[en] <- i
        en = en + 1
    }
    
    return(data.vec)
}

## Function to get YahooFinance news articles
getStockNews <- function(symb){
    ## Get news
    yahoofinance <- WebCorpus(YahooFinanceSource(symb))
    
    ## Extract timestamps
    dt.list <- lapply(yahoofinance, meta, "datetimestamp")
    dt.vec <- .POSIXct(character(length(dt.list)))
    dt.vec <- enum(dt.list, dt.vec)
    attributes(dt.vec)$tzone <- "America/New_York"
    
    ## Extract descriptions
    des.list <- lapply(yahoofinance, meta, "description")
    des.vec <- character(length(des.list))
    des.vec <- enum(des.list, des.vec)
    
    ## Extract origin
    orig.list <- lapply(yahoofinance, meta, "origin")
    orig.vec <- character(length(orig.list))
    orig.vec <- enum(orig.list, orig.vec)
    
    ## Extract id
    id.list <- lapply(yahoofinance, meta, "origin")
    id.vec <- character(length(id.list))
    id.vec <- enum(id.list, id.vec)
    
    ## combine article data into dataframe
    news <- data.frame(timestamp = dt.vec, description = des.vec, url = orig.vec, articleID = id.vec, stringsAsFactors = FALSE)
    
    ## Save news dataframe
    setwd(symb)
    name<-paste0(symb,"_",date,".RDS")
    saveRDS(news, file =name)
    setwd("..")
    
    return(paste(symb, "data saved"))
}

## Get YahooFinance data for portfolio tickers based on folder listings
for(s in list.files(pattern = '[^getNews.R|getNews.bat|getNews.Rout]')){
    try(print(getStockNews(s)))
}