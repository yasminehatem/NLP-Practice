
# read in the libraries we're going to use
library(tidyverse) # general utility & workflow functions
library(tidytext) # tidy implimentation of NLP methods
library(topicmodels) # for LDA topic modelling 
library(tm) # general text mining functions
library(SnowballC) # for stemming

# read in our data
texts <- read_csv("../input/fake.csv")

# subsample the dataset (so we can calculate LDA quicker)
set.seed(1234) #setting this so we'll always get the same subset
row_indexes <- sample(1:nrow(texts), 1600, replace = F) # randomly generate 2000 row indexes
texts_subsample <-slice(texts, row_indexes) # get rows at those indexes
head(texts_subsample)
summary(texts_subsample)


# function to get & plot the most informative terms by a specificed number
# of topics, using LDA
top_terms_by_topic_LDA <- function(input_text, # should be a columm from a dataframe
                                   plot = T, # return a plot? TRUE by defult
                                   number_of_topics = 4) # number of topics (4 by default)
{    
  # create a corpus (type of object expected by tm) and document term matrix
  Corpus <- Corpus(VectorSource(input_text)) # make a corpus object
  DTM <- DocumentTermMatrix(Corpus) # get the count of words/document
  
  # remove any empty rows in our document term matrix (if there are any 
  # we'll get an error when we try to run our LDA)
  unique_indexes <- unique(DTM$i) # get the index of each unique value
  DTM <- DTM[unique_indexes,] # get a subset of only those indexes
  
  # preform LDA & get the words/topic in a tidy text format
  lda <- LDA(DTM, k = number_of_topics, control = list(seed = 1234))
  topics <- tidy(lda, matrix = "beta") # convert the LDA output to a tidy
  
  # get the top ten terms for each topic
  top_terms <- topics  %>% # take the topics data frame and..
    group_by(topic) %>% # treat each topic as a different group
    top_n(10, beta) %>% # get the top 10 most informative words
    ungroup() %>% # ungroup
    arrange(topic, -beta) # arrange words in descending informativeness
  
  # if the user asks for a plot (TRUE by default)
  if(plot == T){
    # plot the top ten terms for each topic in order
    top_terms %>% # take the top terms
      mutate(term = reorder(term, beta)) %>% # sort terms by beta value 
      ggplot(aes(term, beta, fill = factor(topic))) + # plot beta by theme
      geom_col(show.legend = FALSE) + # as a bar plot
      facet_wrap(~ topic, scales = "free") + # which each topic in a seperate plot
      labs(x = NULL, y = "Beta") + # no x label, change y label 
      coord_flip() # turn bars sideways
  }else{ 
    # if the user does not request a plot
    # return a list of sorted terms instead
    return(top_terms)
  }
}


top_terms_by_topic_LDA(texts_subsample$text, number_of_topics = 4)




# create a document term matrix to clean
myCorpus <- Corpus(VectorSource(texts_subsample$text)) 
textsDTM <- DocumentTermMatrix(myCorpus)
# convert the document term matrix to a tidytext corpus
textsDTM_tidy <- tidy(textsDTM)

# remove stopwords
textsDTM_tidy_cleaned <- textsDTM_tidy %>% # take our tidy dtm and...
  anti_join(stop_words, by = c("term" = "word"))   # remove English stopwords and...
cleaned_documents <- textsDTM_tidy_cleaned %>%
  group_by(document) %>%
  mutate(terms = toString(rep(term, count))) %>%
  select(document, terms) %>%
  unique()
head(textsDTM_tidy_cleaned)
head(cleaned_documents)


# take a look at the new most informative terms
top_terms_by_topic_LDA(cleaned_documents$terms, number_of_topics = 4)


# stem the words (e.g. convert each word to its stem, where applicable)
textsDTM_tidy_cleaned <- textsDTM_tidy_cleaned %>% 
  mutate(stem = wordStem(term))
cleaned_documents <- textsDTM_tidy_cleaned %>%
  group_by(document) %>% 
  mutate(terms = toString(rep(stem, count))) %>%
  select(document, terms) %>%
  unique()


# now let's look at the new most informative terms
top_terms_by_topic_LDA(cleaned_documents$terms, number_of_topics = 4)

 
# Since the texts in this dataset are labeled, let's explore them using TF-IDF. For this we can use the same function that I wrote for the tutorial.

# %% [code]
# function that takes in a dataframe and the name of the columns
# with the document texts and the topic labels. If plot is set to
# false it will return the tf-idf output rather than a plot.
top_terms_by_topic_tfidf <- function(text_df, text_column, group_column, plot = T){
  # name for the column we're going to unnest_tokens_ to
  # (you only need to worry about enquo stuff if you're
  # writing a function using using tidyverse packages)
  group_column <- enquo(group_column)
  text_column <- enquo(text_column)
  
  # get the count of each word in each review
  words <- text_df %>%
    unnest_tokens(word, !!text_column) %>%
    count(!!group_column, word) %>% 
    ungroup()
  
  # get the number of words per text
  total_words <- words %>% 
    group_by(!!group_column) %>% 
    summarize(total = sum(n))
  
  # combine the two dataframes we just made
  words <- left_join(words, total_words)
  
  # get the tf_idf & order the words by degree of relevence
  tf_idf <- words %>%
    bind_tf_idf(word, !!group_column, n) %>%
    select(-total) %>%
    arrange(desc(tf_idf)) %>%
    mutate(word = factor(word, levels = rev(unique(word))))
  
  if(plot == T){
    # convert "group" into a quote of a name
    # (this is due to funkiness with calling ggplot2
    # in functions)
    group_name <- quo_name(group_column)
    
    # plot the 10 most informative terms per topic
    tf_idf %>% 
      group_by(!!group_column) %>% 
      top_n(10) %>% 
      ungroup %>%
      ggplot(aes(word, tf_idf, fill = as.factor(group_name))) +
      geom_col(show.legend = FALSE) +
      labs(x = NULL, y = "tf-idf") +
      facet_wrap(reformulate(group_name), scales = "free") +
      coord_flip()
  }else{
    # return the entire tf_idf dataframe
    return(tf_idf)
  }
}



# Plot the top terms for each label in the "type" column.
top_terms_by_topic_tfidf(text_df = texts, # dataframe
                         text_column = text, # column with text
                         group_column = type, # column with topic label
                         plot = T) 


# plot the top terms by language, using the labels in the "language" column
top_terms_by_topic_tfidf(text_df = texts, # dataframe
                         text_column = text, # column with text
                         group_column = language, # column with topic label
                         plot = T)

