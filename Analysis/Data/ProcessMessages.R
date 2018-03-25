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

## Function to combine sources and pad missing timestamps when no messages were posted
combineSent <- function(ST.t, Tw.t, YF.t){
    ## Prep for hourly predictions
    ST.small <- as.data.frame(get(ST.t)[,c("created_at", "score")])
    colnames(ST.small) <- c("timestamp", "ST.score")
    ST.small$timestamp <- as.character(format(ST.small$timestamp, tz="EST"))
    ST.small$timestamp <- ymd_hms(ST.small$timestamp, tz="EST")
    ST.small$date <- date(ST.small$timestamp) #date/hour extraction inspired by https://stackoverflow.com/questions/10705328/extract-hours-and-seconds-from-posixct-for-plotting-purposes-in-r
    ST.small$hour <- hour(ST.small$timestamp)
    ST.small <- aggregate(ST.score~date+hour, ST.small, sum)
    ST.small <- ST.small[with(ST.small, order(date, hour)),]
    fill <- seq(ymd_h(paste(ST.small[1, "date"], ST.small[1, "hour"]), tz = "EST"),
                ymd_h(paste(ST.small[nrow(ST.small)-1, "date"], ST.small[nrow(ST.small)-1, "hour"]),tz = "EST"),
                by="hour")
    ST.small <- full_join(ST.small, data.frame(date = date(fill), hour = hour(fill))) #This methodology inspired by https://stackoverflow.com/questions/16787038/insert-rows-for-missing-dates-times
    for(i in seq(1,72)) ST.small <- slide(ST.small, Var = "ST.score", slideBy = -i)
    ST.small <- tail(ST.small, -72) #Drop rows with lagged NA values
    ST.small <- ST.small[,-c(3:9)] #Remove most recent 7 hours of sentiment data
    
    T.small <- as.data.frame(get(Tw.t)[,c("created", "score")])
    colnames(T.small) <- c("timestamp", "T.score")
    T.small$timestamp <- as.character(format(T.small$timestamp, tz="EST"))
    T.small$timestamp <- ymd_hms(T.small$timestamp, tz="EST")
    T.small$date <- date(T.small$timestamp)
    T.small$hour <- hour(T.small$timestamp)
    T.small <- aggregate(T.score~date+hour, T.small, sum)
    T.small <- T.small[with(T.small, order(date, hour)),]
    fill <- seq(ymd_h(paste(T.small[1, "date"], T.small[1, "hour"]), tz = "EST"),
                ymd_h(paste(T.small[nrow(T.small)-1, "date"], T.small[nrow(T.small)-1, "hour"]),tz = "EST"),
                by="hour")
    T.small <- full_join(T.small, data.frame(date = date(fill), hour = hour(fill))) #This methodology inspired by https://stackoverflow.com/questions/16787038/insert-rows-for-missing-dates-times
    for(i in seq(1,72)) T.small <- slide(T.small, Var = "T.score", slideBy = -i)
    T.small <- tail(T.small, -72) #Drop rows with lagged NA values
    T.small <- T.small[,-c(3:9)] #Remove most recent 7 hours of sentiment data
    
    YF.small <- as.data.frame(get(YF.t)[,c("timestamp", "score")])
    colnames(YF.small) <- c("timestamp", "YF.score")
    YF.small$timestamp <- as.character(format(YF.small$timestamp, tz="EST"))
    YF.small$timestamp <- ymd_hms(YF.small$timestamp, tz="EST")
    YF.small$date <- date(YF.small$timestamp)
    YF.small$hour <- hour(YF.small$timestamp)
    YF.small <- aggregate(YF.score~date+hour, YF.small, sum)
    YF.small <- YF.small[with(YF.small, order(date, hour)),]
    fill <- seq(ymd_h(paste(YF.small[1, "date"], YF.small[1, "hour"]), tz = "EST"),
                ymd_h(paste(YF.small[nrow(YF.small)-1, "date"], YF.small[nrow(YF.small)-1, "hour"]),tz = "EST"),
                by="hour")
    YF.small <- full_join(YF.small, data.frame(date = date(fill), hour = hour(fill))) #This methodology inspired by https://stackoverflow.com/questions/16787038/insert-rows-for-missing-dates-times
    for(i in seq(1,72)) YF.small <- slide(YF.small, Var = "YF.score", slideBy = -i)
    YF.small <- tail(YF.small, -72) #Drop rows with lagged NA values
    YF.small <- YF.small[,-c(3:9)] #Remove most recent 7 hours of sentiment data
    
    sent.df <- merge(ST.small, T.small, by = c("date", "hour"))
    sent.df <- merge(sent.df, YF.small, by = c("date", "hour"))
    sent.df <- sent.df[with(sent.df, order(date, hour)),]
    
    return(sent.df)
}

## Create list of sources for each ticker
listCombo <- list()
for(i in 1:length(ST)){
    listCombo[[i]] <- c(ST[i], Tw[i], YF[i])
}

## Save combined data for each ticker to RDS file
for(i in 1:length(listCombo)){
    saveRDS(combineSent(listCombo[[i]][1], listCombo[[i]][2], listCombo[[i]][3]), paste0('Data/TickerRDS/', strsplit(ST[i], "\\.")[[1]][1],'.RDS'))
}

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
