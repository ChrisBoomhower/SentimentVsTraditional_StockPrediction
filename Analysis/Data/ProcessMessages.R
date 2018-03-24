#################################################################
## File:            ProcessMessages.R
##
## Description:     Consolidate StockTwits data pulled previously
##                  and generate sentiment scores.

require(plyr)
require(jsonlite)
require(stringr)
require(dplyr)
require(DataCombine)
require(lubridate)

Tableau = FALSE #Set conditional term for processing word dataframes

#################################
###### DATA CONSOLIDATION #######
#################################

# ## Consolidate StockTwits ticker symbol data into single dataframes with duplicates dropped
setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis/Data/StockTwits")
ST.tickers <- list.files(pattern = '[^getStockTwits.R|getStockTwits.bat|getStockTwits.Rout]')

cols.u <- c("id", "body", "created_at", "symbols", "links", "user.id", "likes.total")

start <- Sys.time() #Start timer
for(s in ST.tickers){
    setwd(s)

    files <- list.files(pattern = '\\.RDS$')
    dat_list <- lapply(files, readRDS)
    dat_list <- lapply(dat_list, flatten) #Need to flatten dataframes since dfs from JSON contain nested dfs
    df <- ldply(dat_list, data.frame) #Combine list dataframes into single dataframe

    ## Remove duplicates; of duplicates, keep only record with highest likes.total
    temp <- df[!duplicated(df[,cols.u]),]
    temp$likes.total <- ifelse(is.na(temp$likes.total), 0, temp$likes.total) #Fill NAs with 0
    df <- temp %>% group_by(id, body, created_at, user.id) %>% top_n(1, likes.total) #Inspired by https://stackoverflow.com/questions/24558328/how-to-select-the-row-with-the-maximum-value-in-each-group

    ## Convert creation time to date format and convert UTC time to EST
    df$created_at <- as.POSIXct(strptime(df$created_at,"%FT%H:%M:%SZ", tz = "UTC"))
    df$created_at <- format(df$created_at, tz="EST")
    
    assign(paste0(s,".ST"), df)

    setwd('..')
}
rm(df)
rm(temp)
rm(dat_list)
print(Sys.time() - start)

## Consolidate Twitter ticker symbol data into single dataframes with duplicates dropped
setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis/Data/TwitterData")
T.tickers <- list.files(pattern = '[^getTweets.R|getTweets.bat|getTweets.Rout]')

cols.u <- c("id", "text", "created", "screenName", "favoriteCount")

start <- Sys.time() #Start timer
for(s in T.tickers){
    setwd(s)

    files <- list.files(pattern = '\\.RDS$')
    dat_list <- lapply(files, readRDS)
    df <- ldply(dat_list, data.frame)

    ## Remove duplicates; of duplicates, keep only record with highest favoriteCount
    temp <- df[!duplicated(df[,cols.u]),]
    df <- temp %>% group_by(id, text, created, screenName) %>% top_n(1, favoriteCount) #Inspired by https://stackoverflow.com/questions/24558328/how-to-select-the-row-with-the-maximum-value-in-each-group

    ## Convert UTC time to EST
    df$created <- format(df$created, tz="EST")
    
    assign(paste0(s,".T"), df)

    setwd('..')
}
rm(df)
rm(temp)
rm(dat_list)
print(Sys.time() - start)

## Consolidate YahooFinance ticker symbol data into single dataframes with duplicates dropped
setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis/Data/YahooFinance")
YF.tickers <- list.files(pattern = '[^getNews.R|getNews.bat|getNews.Rout]')

start <- Sys.time() #Start timer
for(s in YF.tickers){
    setwd(s)

    files <- list.files(pattern = '\\.RDS$')
    dat_list <- lapply(files, readRDS)
    df <- ldply(dat_list, data.frame)

    df <- df[!duplicated(df$description),] #Drop duplicates
    
    ## Ensure EST timestamps (EST appears to be default but just want to make sure)
    df$created <- format(df$timestamp, tz="EST")
    
    assign(paste0(s,".YF"), df) #Assign to variable named after ticker symbol

    setwd('..')
}
rm(df)
rm(dat_list)
print(Sys.time() - start)

#################################
####### SCORING FUNCTIONS #######
#################################

## Get good and bad words
## See Minqing Hu and Bing Liu papers titled,
## "Mining and Summarizing Customer Reviews" and
## "Opinion Observer: Analyzing and Comparing Opinions on the Web,"
setwd("..")
positive = scan('positive-words.txt', what='character', comment.char=';') #Produce good word vector
negative = scan('negative-words.txt', what='character', comment.char=';') #Produce negative word vector

positive = c(positive, 'wtf', 'epicfail', 'bearish', 'bear')
negative = c(negative, 'upgrade', ':)', 'bullish', 'bull')

## Following function modified from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/blob/master/R/sentiment.R
make.score <- function(sentence, pos.words, neg.words) {
    
    ## clean up sentence contents:
    sentence = gsub('[[:punct:]]', '', sentence)
    sentence = gsub('[[:cntrl:]]', '', sentence)
    sentence = gsub('\\d+', '', sentence)
    
    ## convert to lower case:
    sentence = try(tolower(sentence))
    
    ## split into words and reduce list of single list object to just list
    word.list = str_split(sentence, '\\s+')
    words = unlist(word.list)
    
    ## compare message words against good and bad word dictionaries
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    
    ## get indices of matched terms
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    ## calculate score based on summations
    score = sum(pos.matches) - sum(neg.matches)
    
    #if(cloud) return(list(words, pos.matches, neg.matches)) #Use when generating df for Tableau wordcloud
    #else return(score) #Use when getting scores
    return(score)
}

score.sentiment = function(sentences, pos.words, neg.words, feed, addition = '', .progress='none'){
    if(feed == 'ST'){
        sentences <- ifelse(addition == '', sentences, paste(sentences, addition)) #Tag on additional column content when needed
    }
    sentences <- try(iconv(sentences, 'UTF-8', 'latin1')) #Inspired by https://stackoverflow.com/questions/9637278/r-tm-package-invalid-input-in-utf8towcs

    # we got a vector of sentences. plyr will handle a list or a vector as an "l" for us
    # we want a simple array of scores back, so we use "l" + "a" + "ply" = laply:
    scores = laply(sentences, make.score, pos.words, neg.words, .progress=.progress )
    
    scores.df = data.frame(score=scores, text=sentences)
    return(scores.df)
}

#################################
######## GENERATE SCORES ########
#################################

## Generate StockTwits scores
ST <- ls(pattern = '.\\.ST$') #Get list of StockTwits objects
for(s in ST){
    temp <- score.sentiment(get(s)[["body"]], positive, negative, feed = 'ST',
                            addition = get(s)[["entities.sentiment.basic"]],
                            .progress='text')

    ## Multiply score by likes.total and add score column to social dataframes
    assign(paste0(s,".scores"), temp)
    assign(s, `[[<-`(get(s), 'scoreLikes', #New variable assignment syntax inspired by https://stackoverflow.com/questions/15670193/how-to-use-assign-or-get-on-specific-named-column-of-a-dataframe
                     value =ifelse(get(s)[["likes.total"]] > 0,
                                   temp[,1] * get(s)[["likes.total"]],
                                   temp[,1])))

    ## Also add raw scores to social dataframes
    assign(s, `[[<-`(get(s), 'score', #New variable assignment syntax inspired by https://stackoverflow.com/questions/15670193/how-to-use-assign-or-get-on-specific-named-column-of-a-dataframe
                     value =temp[,1]))
}
rm(temp)

#FOLLOWING LINE FOR DEBUG ONLY
#temp <- score.sentiment(AAPL.ST[["body"]], positive, negative, addition = AAPL.ST[,"entities.sentiment.basic"], .progress='text')

## Generate Twitter scores
Tw <- ls(pattern = '.\\.T$') #Get list of Twitter objects
start <- Sys.time() #Start timer
for(s in Tw){
    temp <- score.sentiment(get(s)[["text"]], positive, negative,
                            feed = "Tw", addition = "",
                            .progress='text')

    ## Multiply score by likes.total and add score column to social dataframes
    assign(paste0(s,".scores"), temp)
    assign(s, `[[<-`(get(s), 'scoreLikes',
                     value =ifelse(get(s)[["favoriteCount"]] > 0,
                                   temp[,1] * get(s)[["favoriteCount"]],
                                   temp[,1])))

    ## Also add raw scores to social dataframes
    assign(s, `[[<-`(get(s), 'score',
                     value =temp[,1]))
}
rm(temp)
print(Sys.time() - start)

#FOLLOWING LINE FOR DEBUG ONLY
#temp <- score.sentiment(AAPL.T[["text"]], positive, negative, feed = "Tw", .progress='text')

## Generate Yahoo Finance scores
YF <- ls(pattern = '.\\.YF$') #Get list of Twitter objects
start <- Sys.time() #Start timer
for(s in YF){
    temp <- score.sentiment(get(s)[["description"]], positive, negative,
                            feed = "YF", addition = "",
                            .progress='text')
    
    ## Add raw scores to social dataframes
    assign(s, `[[<-`(get(s), 'score',
                     value =temp[,1]))
}
rm(temp)
print(Sys.time() - start)

#FOLLOWING LINE FOR DEBUG ONLY
#temp <- score.sentiment(AAPL.YF[["description"]], positive, negative, feed = "YF", addition = "", .progress='text')

######################################
####### Combine Ticker Sources #######
######################################

## Prep for hourly predictions
AAPL.ST.small <- as.data.frame(AAPL.ST[,c("created_at", "score")])
colnames(AAPL.ST.small) <- c("timestamp", "ST.score")
AAPL.ST.small$timestamp <- as.character(format(AAPL.ST.small$timestamp, tz="EST"))
AAPL.ST.small$timestamp <- ymd_hms(AAPL.ST.small$timestamp, tz="EST")
AAPL.ST.small$date <- date(AAPL.ST.small$timestamp) #date/hour extraction inspired by https://stackoverflow.com/questions/10705328/extract-hours-and-seconds-from-posixct-for-plotting-purposes-in-r
AAPL.ST.small$hour <- hour(AAPL.ST.small$timestamp)
AAPL.ST.small <- aggregate(ST.score~date+hour, AAPL.ST.small, sum)
AAPL.ST.small <- AAPL.ST.small[with(AAPL.ST.small, order(date, hour)),]
fill <- seq(ymd_h(paste(AAPL.ST.small[1, "date"], AAPL.ST.small[1, "hour"]), tz = "EST"),
            ymd_h(paste(AAPL.ST.small[nrow(AAPL.ST.small)-1, "date"], AAPL.ST.small[nrow(AAPL.ST.small)-1, "hour"]),tz = "EST"),
            by="hour")
AAPL.ST.small <- full_join(AAPL.ST.small, data.frame(date = date(fill), hour = hour(fill))) #This methodology inspired by https://stackoverflow.com/questions/16787038/insert-rows-for-missing-dates-times
for(i in seq(1,72)) AAPL.ST.small <- slide(AAPL.ST.small, Var = "ST.score", slideBy = -i)
AAPL.ST.small <- tail(AAPL.ST.small, -72) #Drop rows with lagged NA values
AAPL.ST.small <- AAPL.ST.small[,-c(3:9)] #Remove most recent 7 hours of sentiment data

AAPL.T.small <- as.data.frame(AAPL.T[,c("created", "score")])
colnames(AAPL.T.small) <- c("timestamp", "T.score")
AAPL.T.small$timestamp <- as.character(format(AAPL.T.small$timestamp, tz="EST"))
AAPL.T.small$timestamp <- ymd_hms(AAPL.T.small$timestamp, tz="EST")
AAPL.T.small$date <- date(AAPL.T.small$timestamp)
AAPL.T.small$hour <- hour(AAPL.T.small$timestamp)
AAPL.T.small <- aggregate(T.score~date+hour, AAPL.T.small, sum)
AAPL.T.small <- AAPL.T.small[with(AAPL.T.small, order(date, hour)),]
fill <- seq(ymd_h(paste(AAPL.T.small[1, "date"], AAPL.T.small[1, "hour"]), tz = "EST"),
            ymd_h(paste(AAPL.T.small[nrow(AAPL.T.small)-1, "date"], AAPL.T.small[nrow(AAPL.T.small)-1, "hour"]),tz = "EST"),
            by="hour")
AAPL.T.small <- full_join(AAPL.T.small, data.frame(date = date(fill), hour = hour(fill))) #This methodology inspired by https://stackoverflow.com/questions/16787038/insert-rows-for-missing-dates-times
for(i in seq(1,72)) AAPL.T.small <- slide(AAPL.T.small, Var = "T.score", slideBy = -i)
AAPL.T.small <- tail(AAPL.T.small, -72) #Drop rows with lagged NA values
AAPL.T.small <- AAPL.T.small[,-c(3:9)] #Remove most recent 7 hours of sentiment data

AAPL.YF.small <- as.data.frame(AAPL.YF[,c("timestamp", "score")])
colnames(AAPL.YF.small) <- c("timestamp", "YF.score")
AAPL.YF.small$timestamp <- as.character(format(AAPL.YF.small$timestamp, tz="EST"))
AAPL.YF.small$timestamp <- ymd_hms(AAPL.YF.small$timestamp, tz="EST")
AAPL.YF.small$date <- date(AAPL.YF.small$timestamp)
AAPL.YF.small$hour <- hour(AAPL.YF.small$timestamp)
AAPL.YF.small <- aggregate(YF.score~date+hour, AAPL.YF.small, sum)
AAPL.YF.small <- AAPL.YF.small[with(AAPL.YF.small, order(date, hour)),]
fill <- seq(ymd_h(paste(AAPL.YF.small[1, "date"], AAPL.YF.small[1, "hour"]), tz = "EST"),
            ymd_h(paste(AAPL.YF.small[nrow(AAPL.YF.small)-1, "date"], AAPL.YF.small[nrow(AAPL.YF.small)-1, "hour"]),tz = "EST"),
            by="hour")
AAPL.YF.small <- full_join(AAPL.YF.small, data.frame(date = date(fill), hour = hour(fill))) #This methodology inspired by https://stackoverflow.com/questions/16787038/insert-rows-for-missing-dates-times
for(i in seq(1,72)) AAPL.YF.small <- slide(AAPL.YF.small, Var = "YF.score", slideBy = -i)
AAPL.YF.small <- tail(AAPL.YF.small, -72) #Drop rows with lagged NA values
AAPL.YF.small <- AAPL.YF.small[,-c(3:9)] #Remove most recent 7 hours of sentiment data

AAPL <- merge(AAPL.ST.small, AAPL.T.small, by = c("date", "hour"))
AAPL <- merge(AAPL, AAPL.YF.small, by = c("date", "hour"))
AAPL <- AAPL[with(AAPL, order(date, hour)),]

saveRDS(AAPL, 'Data/TickerRDS/AAPL.RDS')

#fill <- seq(ymd_h(paste(AAPL[1, "date"], AAPL[1, "hour"]), tz = "EST"),
#    ymd_h(paste(AAPL[nrow(AAPL)-1, "date"], AAPL[nrow(AAPL)-1, "hour"]),tz = "EST"),
#    by="hour")
#AAPL <- full_join(AAPL, data.frame(date = date(fill), hour = hour(fill))) #This methodology inspired by https://stackoverflow.com/questions/16787038/insert-rows-for-missing-dates-times
#AAPL <- tail(AAPL, -72) #Drop rows with lagged NA values

## Prep for day-to-day predictions

# AAPL.ST.small <- as.data.frame(AAPL.ST[,c("created_at", "score")])
# colnames(AAPL.ST.small) <- c("timestamp", "ST.score")
# AAPL.ST.small$timestamp <- as.Date(format(AAPL.ST.small$timestamp, tz="EST"))
# AAPL.ST.small <- aggregate(ST.score~timestamp, AAPL.ST.small, sum)
# AAPL.ST.small <- slide(AAPL.ST.small, Var = "ST.score", slideBy = -1)
# AAPL.ST.small <- slide(AAPL.ST.small, Var = "ST.score", slideBy = -2)
# AAPL.ST.small <- slide(AAPL.ST.small, Var = "ST.score", slideBy = -3)
# 
# AAPL.T.small <- AAPL.T[,c("created", "score")]
# colnames(AAPL.T.small) <- c("timestamp", "T.score")
# AAPL.T.small$timestamp <- as.Date(format(AAPL.T.small$timestamp, tz="EST"))
# AAPL.T.small <- aggregate(T.score~timestamp, AAPL.T.small, sum)
# AAPL.T.small <- slide(AAPL.T.small, Var = "T.score", slideBy = -1)
# AAPL.T.small <- slide(AAPL.T.small, Var = "T.score", slideBy = -2)
# AAPL.T.small <- slide(AAPL.T.small, Var = "T.score", slideBy = -3)
# 
# AAPL.YF.small <- AAPL.YF[,c("timestamp", "score")]
# colnames(AAPL.YF.small) <- c("timestamp", "YF.score")
# AAPL.YF.small$timestamp <- as.Date(format(AAPL.YF.small$timestamp, tz="EST"))
# AAPL.YF.small <- aggregate(YF.score~timestamp, AAPL.YF.small, sum)
# AAPL.YF.small <- slide(AAPL.YF.small, Var = "YF.score", slideBy = -1)
# AAPL.YF.small <- slide(AAPL.YF.small, Var = "YF.score", slideBy = -2)
# AAPL.YF.small <- slide(AAPL.YF.small, Var = "YF.score", slideBy = -3)
# 
# AAPL <- merge(AAPL.ST.small, AAPL.T.small, by = "timestamp")
# AAPL <- merge(AAPL, AAPL.YF.small, by = "timestamp")



# AMZN.ST.small <- AMZN.ST[,c("created_at", "score")]
# colnames(AMZN.ST.small) <- c("timestamp", "score")
# AMZN.T.small <- AMZN.T[,c("created", "score")]
# colnames(AMZN.T.small) <- c("timestamp", "score")
# AMZN.YF.small <- AMZN.YF[,c("timestamp", "score")]
# AMZN <- as.data.frame(rbind(AMZN.ST.small,AMZN.T.small,AMZN.YF.small))
# 
# BA.ST.small <- BA.ST[,c("created_at", "score")]
# colnames(BA.ST.small) <- c("timestamp", "score")
# Boeing.T.small <- Boeing.T[,c("created", "score")]
# colnames(Boeing.T.small) <- c("timestamp", "score")
# BA.YF.small <- BA.YF[,c("timestamp", "score")]
# BA <- as.data.frame(rbind(BA.ST.small,Boeing.T.small,BA.YF.small))
# 
# DWDP.ST.small <- DWDP.ST[,c("created_at", "score")]
# colnames(DWDP.ST.small) <- c("timestamp", "score")
# DowDuPont.T.small <- DowDuPont.T[,c("created", "score")]
# colnames(DowDuPont.T.small) <- c("timestamp", "score")
# DWDP.YF.small <- DWDP.YF[,c("timestamp", "score")]
# DWDP <- as.data.frame(rbind(DWDP.ST.small,DowDuPont.T.small,DWDP.YF.small))
# 
# JNJ.ST.small <- JNJ.ST[,c("created_at", "score")]
# colnames(JNJ.ST.small) <- c("timestamp", "score")
# JNJ.T.small <- JNJ.T[,c("created", "score")]
# colnames(JNJ.T.small) <- c("timestamp", "score")
# JNJ.YF.small <- JNJ.YF[,c("timestamp", "score")]
# JNJ <- as.data.frame(rbind(JNJ.ST.small,JNJ.T.small,JNJ.YF.small))
# 
# JPM.ST.small <- JPM.ST[,c("created_at", "score")]
# colnames(JPM.ST.small) <- c("timestamp", "score")
# JPM.T.small <- JPM.T[,c("created", "score")]
# colnames(JPM.T.small) <- c("timestamp", "score")
# JPM.YF.small <- JPM.YF[,c("timestamp", "score")]
# JPM <- as.data.frame(rbind(JPM.ST.small,JPM.T.small,JPM.YF.small))
# 
# NEE.ST.small <- NEE.ST[,c("created_at", "score")]
# colnames(NEE.ST.small) <- c("timestamp", "score")
# NEE.T.small <- NEE.T[,c("created", "score")]
# colnames(NEE.T.small) <- c("timestamp", "score")
# NEE.YF.small <- NEE.YF[,c("timestamp", "score")]
# NEE <- as.data.frame(rbind(NEE.ST.small,NEE.T.small,NEE.YF.small))
# 
# PG.ST.small <- PG.ST[,c("created_at", "score")]
# colnames(PG.ST.small) <- c("timestamp", "score")
# Proctor&Gamble.T.small <- Proctor&Gamble.T[,c("created", "score")]
# colnames(Proctor&Gamble.T.small) <- c("timestamp", "score")
# PG.YF.small <- PG.YF[,c("timestamp", "score")]
# PG <- as.data.frame(rbind(PG.ST.small,Proctor&Gamble.T.small,PG.YF.small))
# 
# SPG.ST.small <- SPG.ST[,c("created_at", "score")]
# colnames(SPG.ST.small) <- c("timestamp", "score")
# SPG.T.small <- SPG.T[,c("created", "score")]
# colnames(SPG.T.small) <- c("timestamp", "score")
# SPG.YF.small <- SPG.YF[,c("timestamp", "score")]
# SPG <- as.data.frame(rbind(SPG.ST.small,SPG.T.small,SPG.YF.small))
# 
# VZ.ST.small <- VZ.ST[,c("created_at", "score")]
# colnames(VZ.ST.small) <- c("timestamp", "score")
# Verizon.T.small <- Verizon.T[,c("created", "score")]
# colnames(Verizon.T.small) <- c("timestamp", "score")
# VZ.YF.small <- VZ.YF[,c("timestamp", "score")]
# VZ <- as.data.frame(rbind(VZ.ST.small,Verizon.T.small,VZ.YF.small))
# 
# XOM.ST.small <- XOM.ST[,c("created_at", "score")]
# colnames(XOM.ST.small) <- c("timestamp", "score")
# Exxon.T.small <- Exxon.T[,c("created", "score")]
# colnames(Exxon.T.small) <- c("timestamp", "score")
# XOM.YF.small <- XOM.YF[,c("timestamp", "score")]
# XOM <- as.data.frame(rbind(XOM.ST.small,Exxon.T.small,XOM.YF.small))

## Cleanup unneeded objects
rm(list=ls(pattern = '.\\.scores$'))
rm(cols.u, files, s, ST.tickers, start, T.tickers, YF.tickers)
gc()

######################################
####### GET WORDS FOR TABLEAUE #######
######################################

if(Tableau){ #Takes ~24hrs to run, so only run when needed
    word.freq <- function(sentence, s) {
        
        ## clean up sentence contents:
        sentence = gsub('[[:punct:]]', '', sentence)
        sentence = gsub('[[:cntrl:]]', '', sentence)
        sentence = gsub('\\d+', '', sentence)
        
        ## convert to lower case:
        sentence = try(tolower(sentence))
        
        ## split into words and reduce list of single list object to just list
        word.list = str_split(sentence, '\\s+')
        words = unlist(word.list)
        
        words.add <- table(words)
        ## Combine with ticker by feed words
        sim <- intersect(names(get(paste0("words.", s))), names(words.add)) #Process for combining tables inspired by https://stackoverflow.com/questions/12897220/how-to-merge-tables-in-r
        assign(paste0("words.", s), c(get(paste0("words.", s))[!(names(get(paste0("words.", s))) %in% sim)], #Update global variable
                        words.add[!(names(words.add) %in% sim)],
                        get(paste0("words.", s))[sim] + words.add[sim]),
               envir = .GlobalEnv)
        
        # ## Combine with all words table (contains words for all feed types)
        # sim <- intersect(names(words.all), names(words.add)) #Process for combining tables inspired by https://stackoverflow.com/questions/12897220/how-to-merge-tables-in-r
        # words.all <<- c(words.all[!(names(words.all) %in% sim)], #Update global variable
        #          words.add[!(names(words.add) %in% sim)],
        #          words.all[sim] + words.add[sim])
        
        return(NA) #Use when getting scores
    }
    ## Word generation wrapper function
    genWords <- function(feed, message){
        start <- Sys.time() #Start timer
        
        for(s in feed){
            ## Initialize ticker table (purposely use numeric type since numbers removed from message texts)
            assign(paste0("words.", s), table(1), envir = .GlobalEnv) #Assign to global variable named after ticker symbol
            
            ## Apply word.freq function
            sentences <- try(iconv(get(s)[[message]], 'UTF-8', 'latin1'))
            lapply(sentences, word.freq, s)
            
            ## Coerce words and word sentiment to dataframe
            df.w <- as.data.frame(get(paste0("words.", s)), row.names = names(get(paste0("words.", s))))
            df.w$Sentiment <- ifelse(rownames(df.w) %in% positive, "Good",
                                             ifelse(rownames(df.w) %in% negative, "Bad", "Neutral"))
            names(df.w) <- c("Count", "Sentiment")
            df.w <- df.w[!is.na(df.w$Count),]
            df.w <- df.w[rownames(df.w) != 1,] #Remove the "1" initializer row
            
            ## Add source tag for easy filtering and aggregation
            df.w$Media_Source <- unlist(strsplit(s,"\\."))[2]
            
            ## Add source tag for easy filtering and aggregation
            df.w$Ticker <- unlist(strsplit(s,"\\."))[1]
            
            ## Write ticker word frequency df to global environment
            assign(paste0("words.", s), df.w, envir = .GlobalEnv) #Assign to variable named after ticker symbol
            
        }
            
        print(Sys.time() - start)
        return(paste(feed, "word extraction for Tablaue complete"))
    }
    
    ## Initialize words.all table (purposely use numeric type since numbers removed from message texts)
    # words.all <- table(1)
    
    ## Extract words for Tableau visualizations
    genWords(ST, "body")
    genWords(YF, "description")
    genWords(Tw, "text")
    
    # ## Coerce words and word sentiment to dataframe
    # words.all.df <- as.data.frame(words.all, row.names = names(words.all))
    # words.all.df$Sentiment <- ifelse(rownames(words.all.df) %in% positive, "Good",
    #                     ifelse(rownames(words.all.df) %in% negative, "Bad", "Neutral"))
    # names(words.all.df) <- c("Count", "Sentiment")
    # words.all.df <- words.all.df[!is.na(words.all.df$Count),]
    # words.all.df <- words.all.df[rownames(words.all.df) != 1,] #Remove the "1" initializer row
    # 
    # ## Export words.all for Tableau
    # write.csv(words.all.df, "Tableau/wordsCombined.csv")
    
    ## Combine and export words by ticker/source for Tableau
    wo <- ls(pattern = '^words') #Get list of word dataframe objects
    wo <- wo[!unlist(lapply(wo, grepl, '[words.all|words.all.df]'))] #Remove overall words lists
    words.by.source <- do.call(rbind, lapply(wo, get))
    words.by.source$Word <- gsub('[[:digit:]]+', '', rownames(words.by.source))
    write.csv(words.by.source, "Tableau/wordsBySource.csv", row.names = FALSE)
}
