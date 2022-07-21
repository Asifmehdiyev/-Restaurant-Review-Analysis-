# **********  Import necessary libraries *******
library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(inspectdf)

# ***** Import 'nlpdata' and get familiarized with it. ******
df <- read.csv('nlpdata.csv')
View(df)

df %>% glimpse()
df %>% inspect_na()
df %>% dim()
df %>% colnames()


# ************* Data preprocessing ************
df$Liked<-df$Liked %>% as.character()   

df<-df[!duplicated(df$Review), ]
df %>% dim()

df<-select(df,Liked,  everything());

for (i in 1:nrow(df))
{
  string<-df$Review[i]
  temp <- tolower(string)
  temp <- stringr::str_replace_all(temp,"[^a-zA-Z\\s]", " ") 
  temp <- stringr::str_replace_all(temp,"[\\s]+", " ")
  temp <- stringr::str_replace_all(temp,"subject ", "")  
  indexes <- which(temp == "")
  if(length(indexes) > 0){
    temp <- temp[-indexes]
  }
  temp <- gsub("[^\x01-\x7F]", "", temp)
  df$Review[i] <-temp
}


df %>% class()

# ************ Split data ************
set.seed(123)
split <- df$Liked %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)

it_train <- train$Review %>% 
  itoken(preprocesser = toLower,
        tokenizer = word_tokenizer,
        ids = train$...1,
        progressbar = F)



# ********** Prune some words ***********
stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")


vocab <- it_train %>% create_vocabulary(stopwords = stop_words)

vocab %>% tail()

vectorizer <- vocab %>% vocab_vectorizer() 
dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()
identical(rownames(dtm_train) %>% as.numeric(), train$...1)

# ********************* Modeling ************
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)# again lower number of iterations for faster training
glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

it_test <- test$Review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$...1,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)

# *****************************************
# AUC score of test data is higher than the train data. 
# Our model has learned well the data points existing in the test data.