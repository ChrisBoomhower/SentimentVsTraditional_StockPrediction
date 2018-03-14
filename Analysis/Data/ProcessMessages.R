#################################################################
## File:            ProcessMessages.R
##
## Description:     Consolidate StockTwits data pulled previously
##                  and generate sentiment scores.

require(plyr)
require(jsonlite)
require(stringr)
require(dplyr)

## Consolidate StockTwits ticker symbol data into single dataframes with duplicates dropped
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
for(s in T.tickers[]){
    setwd(s)

    files <- list.files(pattern = '\\.RDS$')
    dat_list <- lapply(files, readRDS)
    df <- ldply(dat_list, data.frame)
    
    ## Remove duplicates; of duplicates, keep only record with highest favoriteCount
    temp <- df[!duplicated(df[,cols.u]),]
    df <- temp %>% group_by(id, text, created, screenName) %>% top_n(1, favoriteCount) #Inspired by https://stackoverflow.com/questions/24558328/how-to-select-the-row-with-the-maximum-value-in-each-group
    
    assign(paste0(s,".T"), df)

    setwd('..')
}
rm(df)
rm(temp)
rm(dat_list)
print(Sys.time() - start)
# 
# ## Consolidate YahooFinance ticker symbol data into single dataframes with duplicates dropped
# setwd("C:/Users/Owner/Documents/GitHub/MSDS_8390/SentimentVsTraditional_StockPrediction/Analysis/Data/YahooFinance")
# YF.tickers <- list.files(pattern = '[^getNews.R|getNews.bat|getNews.Rout]')
# for(s in YF.tickers){
#     setwd(s)
#     
#     files <- list.files(pattern = '\\.RDS$')
#     dat_list <- lapply(files, readRDS)
#     df <- ldply(dat_list, data.frame)
#     assign(paste0(s,".YF"), unique(df)) #Drop duplicates and assign to variable named after ticker symbol
#     
#     setwd('..')
# }
# rm(df)

## Generate sentiment scores
setwd("..")
positive = scan('positive-words.txt', what='character', comment.char=';') #Produce good word vector
negative = scan('negative-words.txt', what='character', comment.char=';') #Produce negative word vector

positive = c(positive, 'wtf', 'epicfail', 'bearish', 'bear')
negative = c(negative, 'upgrade', ':)', 'bullish', 'bull')

## Following function modified from https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/blob/master/R/sentiment.R
score.sentiment = function(sentences, pos.words, neg.words, feed, addition = '', .progress='none'){
    if(feed == 'ST'){
        sentences <- ifelse(addition == '', sentences, paste(sentences, addition)) #Tag on additional column content when needed
    }
    sentences <- try(iconv(sentences, 'UTF-8', 'latin1')) #Inspired by https://stackoverflow.com/questions/9637278/r-tm-package-invalid-input-in-utf8towcs
    #print(addition)
    #print(sentences)
    # we got a vector of sentences. plyr will handle a list or a vector as an "l" for us
    # we want a simple array of scores back, so we use "l" + "a" + "ply" = laply:
    scores = laply(sentences, function(sentence, pos.words, neg.words) {
        
        # clean up sentences with R's regex-driven global substitute, gsub():
        sentence = gsub('[[:punct:]]', '', sentence)
        sentence = gsub('[[:cntrl:]]', '', sentence)
        sentence = gsub('\\d+', '', sentence)
        # and convert to lower case:
        sentence = try(tolower(sentence))
        
        # split into words. str_split is in the stringr package
        word.list = str_split(sentence, '\\s+')
        # sometimes a list() is one level of hierarchy too much
        words = unlist(word.list)
        
        # compare our words to the dictionaries of positive & negative terms
        pos.matches = match(words, pos.words)
        neg.matches = match(words, neg.words)
        
        # match() returns the position of the matched term or NA
        # we just want a TRUE/FALSE:
        pos.matches = !is.na(pos.matches)
        neg.matches = !is.na(neg.matches)
        
        # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
        score = sum(pos.matches) - sum(neg.matches)
        
        return(score)
    }, pos.words, neg.words, .progress=.progress )
    
    scores.df = data.frame(score=scores, text=sentences)
    return(scores.df)
}

## Generate StockTwits scores
ST <- ls(pattern = '.\\.ST$') #Get list of StockTwits objects
for(s in ST){
    temp <- score.sentiment(get(s)[["body"]], positive, negative, feed = 'ST',
                            addition = get(s)[["entities.sentiment.basic"]],
                            .progress='text')
    
    ## Multiply score by likes.total and add score column to social dataframes
    assign(paste0(s,".scores"), temp)
    assign(s, `[[<-`(get(s), 'score', #New variable assignment syntax inspired by https://stackoverflow.com/questions/15670193/how-to-use-assign-or-get-on-specific-named-column-of-a-dataframe
                     value =ifelse(get(s)[["likes.total"]] > 0,
                                   temp[,1] * get(s)[["likes.total"]],
                                   temp[,1])))
}
rm(temp)

# genScores(ST, "body", "entities.sentiment.basic", "likes.total")
# 
# genScores <- function(obj, main, adder, multi){
#     for(s in obj){
#         temp <- score.sentiment(get(s)[[main]], positive, negative,
#                                 addition = get(s)[[adder]],
#                                 .progress='text')
#         
#         ## Multiply score by likes.total and add score column to social dataframes
#         assign(paste0(s,".scores"), temp)
#         assign(s, `[[<-`(get(s), 'score', #New variable assignment syntax inspired by https://stackoverflow.com/questions/15670193/how-to-use-assign-or-get-on-specific-named-column-of-a-dataframe
#                          value =ifelse(get(s)[[multi]] > 0,
#                                        temp[,1] * get(s)[["multi"]],
#                                        temp[,1])))
#     }
#     rm(temp)
# }

#FOLLOWING LINE FOR DEBUG ONLY
#temp <- score.sentiment(AAPL.ST[["body"]], positive, negative, addition = AAPL.ST[,"entities.sentiment.basic"], .progress='text')

## Generate Twitter scores
T <- ls(pattern = '.\\.T$') #Get list of Twitter objects
start <- Sys.time() #Start timer
for(s in T){
    temp <- score.sentiment(get(s)[["text"]], positive, negative,
                            feed = "T", addition = "",
                            .progress='text')
    
    ## Multiply score by likes.total and add score column to social dataframes
    assign(paste0(s,".scores"), temp)
    assign(s, `[[<-`(get(s), 'score', #New variable assignment syntax inspired by https://stackoverflow.com/questions/15670193/how-to-use-assign-or-get-on-specific-named-column-of-a-dataframe
                     value =ifelse(get(s)[["favoriteCount"]] > 0,
                                   temp[,1] * get(s)[["favoriteCount"]],
                                   temp[,1])))
}
rm(temp)
print(Sys.time() - start)

#FOLLOWING LINE FOR DEBUG ONLY
#temp <- score.sentiment(AAPL.T[["text"]], positive, negative, feed = "T", .progress='text')
