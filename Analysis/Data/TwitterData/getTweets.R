#################################################################
## File:            getTweets.R
##
## Description:     This file is a scheduled script designed to
##                  gather message data from Twitter for our
##                  portfolio ticker symbols. The data are stored
##                  in external RDS files for later processing.
##
## Frequency:       Scheduled to run every 30 minutes

setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis/Data/TwitterData")

library(twitteR)
library(httr)
library(taskscheduleR)

## Get system time for script run
date<-Sys.time()
date<-as.character(date)
date <- gsub(":","_", date)
date

## Get API OAuth credentials and authorize account
api_key <- readLines("C:/Users/Owner/OneDrive for Business/Semester_6/Special Topics in Data Science/Project 1/OAuth/API_KEY.txt")
api_secret <- readLines("C:/Users/Owner/OneDrive for Business/Semester_6/Special Topics in Data Science/Project 1/OAuth/API_SECRET.txt")
access_token <- readLines("C:/Users/Owner/OneDrive for Business/Semester_6/Special Topics in Data Science/Project 1/OAuth/ACCESS_TOKEN.txt")
access_token_secret <- readLines("C:/Users/Owner/OneDrive for Business/Semester_6/Special Topics in Data Science/Project 1/OAuth/ACCESS_TOKEN_SECRET.txt")

setup_twitter_oauth(api_key, api_secret, access_token=access_token, access_secret=access_token_secret)

## Function to obtain and write twit data to dataframe class
getTweet <- function(symb){
    ## Get tweets
    recentTweets <- searchTwitter(paste0("#",symb), n=1000, lang = "en", )
    recentTweets.df <- twListToDF(recentTweets)
    
    ## Save twits
    setwd(symb)
    name<-paste0(symb,"_",date,".RDS")
    saveRDS(recentTweets.df, file =name)
    setwd("..")
    
    return(paste(symb, "data saved"))
}

## Get tweet data for portfolio tickers based on folder listings
for(s in list.files(pattern = '[^getTweets.R|getTweets.bat|getTweets.Rout]')){
    try(print(getTweet(s)))
}
