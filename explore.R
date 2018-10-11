###
# Kaggle Competition #
                    ##

# Group C2-11 for SYS 6018: Data Mining

##########
# Step 1 #
##########
# Setup prior to running

# Install these packages IF NEEDED
install.packages("tm", "tidyverse", "snow", "SnowballC","e1071","glmnet","text2vec", "glmnet", "doParallel", "Matrix", "dummies", "parallel")

# General use package
library(tidyverse)

# Some language processing libraries
library(tm)
library(openNLP)
library(stringr)
library(plyr)


# Set this to the parent directory of where all your files are stored
wd = "F:/2018 Fall/SYS 6018, Data Mining/assignments/kaggle/03_Age/sys6018-competition-blogger-characteristics"
setwd(wd)

# Start reading in the data.
train = read.csv("Original Files/train.csv")
test = read.csv("Original Files/test.csv")

##########
# Step 2 #
##########
# Cleaning

# Check for missing values
missing.train = sapply(train, function(x) length(which(is.na(x))))
data.frame(missing.train[missing.train>0])
# Nothing is missing. Good!

# Check the names of the columns
colnames(train)
# [1] "post.id" "user.id" "gender"  "topic"   "sign"    "date"    "text"    "age"   

# Check the summary of the data
summary(train)

#    post.id          user.id         gender                        topic       
# Min.   :     1   Min.   :    1   female:227840   indUnk              :159419  
# 1st Qu.:168419   1st Qu.: 4760   male  :215121   Student             :102746  
# Median :341752   Median : 9630                   Technology          : 28346  
# Mean   :340076   Mean   : 9655                   Arts                : 24313  
# 3rd Qu.:510601   3rd Qu.:14278                   Education           : 18563  
# Max.   :681161   Max.   :19319                   Communications-Media: 13195  
#                                                  (Other)             : 96379
#
#      sign                    date                              text       
# Taurus : 42051   02,August,2004:  9407            urlLink        :   316  
# Libra  : 41088   01,August,2004:  8521                           :   310  
# Virgo  : 40923   03,August,2004:  8110           urlLink         :   230  
# Scorpio: 38548   10,August,2004:  5886                           :   219  
# Cancer : 38104   02,July,2004  :  5605                           :   203  
# Leo    : 36809   09,August,2004:  5400                           :   176  
# (Other):205438   (Other)       :400032   (Other)                 :441507  
#
#      age       
# Min.   :13.00  
# 1st Qu.:17.00  
# Median :24.00  
# Mean   :23.54  
# 3rd Qu.:26.00  
# Max.   :48.00  

# Some things that immediately jump out are (1) text, it looks odd but this is
# probably because there are too many different strings with odd spaces, and 
# (2) that the maximum post.id is greater than the size of the training set.

# Plot post.id by and index number
plot(train$post.id, ylab="Post ID", main="Post ID Sampling")
# Add a line for y=x to compare what a 1:1 ratio would look like
abline(0,1)
# It seems that there was random sampling between the training and test
# sets. Maybe the max number is the sum of train and test?

# Check the dimensionality of the datasets
dim(train)[1]+dim(test)[1]
# [1] 681284
max(train$post.id)
# [1] 681161
max(test$post.id)
# [1] 681284
# Yes, this is exactly what happened

# Another thing to check with date data is to see if it's correctly formatted as
# a date instead of a string
unlist(lapply(train, class))
unlist(lapply(test, class))
#   post.id   user.id    gender     topic      sign      date      text       age 
# "integer" "integer"  "factor"  "factor"  "factor"  "factor"  "factor" "integer" 

# Check the factor levels for date
table(train$date)

# Check missing dates
dim(train[grepl(",,", train$date),])
# 24 missing dates

# We have some months in different languages
# English
eng = c("january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december")
# Additional languages attempted but not worth the effort
# # Spanish
# esp = c("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
#         "septiembre", "octubre", "noviembre", "diciembre")
# # French
# fre = c("janvier","fevrier","mars","avril","mai","juin","juillet",
#         "aout","septembre","octobre","novembre","decembre")
# # Portugese
# por = c("janeiro","fevereiro","marco","abril","maio","junho","julho",
#         "agosto","septembro","outubro","novembro","dezembro")
# # German
# ger = c("januar","februar","marz","april","mai","juni","juli",
#         "august","september","oktober","november","dezember")
# # Italian
# ita = c("gennaio","febbraio","marzo","aprile","maggio","giugno","luglio",
#         "agosto","settembre","ottobre","novembre","dicembre")

# Standardize to lower-case first
train$date = tolower(train$date)

# Index which includes only english dates entries
index = grepl(paste(eng,collapse="|"), train$date, ignore.case=TRUE)
train.en=train[index,]
train.other=train[!index,]

# Train has 442 961 observations
# Index has 438 749 observations
sum(index)/dim(train)[1]
# This is a loss of less than 1%

# Check to see their age distributions
par(mfrow=c(1,2))
hist(train.en$age)
hist(train.other$age)
# Notice that non-english dates have a higher proportion of sampling
# with the first two age groups. This may be important?

# Notice that the test has all english dates!
index2 = grepl(paste(eng,collapse="|"), test$date, ignore.case=TRUE)
length(index2)
test.en=test[index,]

# Make into date format
train.en$date = as.Date(train.en$date, format="%d,%B,%Y")
# Same for test
test$date = as.Date(test$date, format="%d,%B,%Y")

# For all other dates use median interpolation
# First find the median date
med = median(train.en$date[!is.na(train.en$date)])
# All missing dates
train.en$date[is.na(train.en$date)] = med
# All non-english dates
train.other$date = med
# Same for test
test$date[is.na(test$date)] = med

# Bring it all together
train.new = rbind(train.en, train.other)
dim(train) == dim(train.new)

# Error check
sum(is.na(train.new$date))

##########
# Step 3 #
##########
# Basic text cleaning
# (Sorry if there's redundancy here and there, the code got a bit complex)

# First thing, change text into a character variable!
unlist(lapply(train.new, class))
train.new$text = as.character(train.new$text)
class(train.new$text)

# To reduce the size of the data, aggregate by id number
names(train.new)

# Needed a mode function
# https://stackoverflow.com/questions/2547402/is-there-a-built-in-function-for-finding-the-mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
names(train.new)
# "post.id" "user.id" "gender"  "topic"   "sign"    "date"    "text"    "age" 

gen = aggregate(train.new$gender, list(train.new$user.id), Mode)[,2]
top = aggregate(train.new$topic, list(train.new$user.id), Mode)[,2]
sign = aggregate(train.new$sign, list(train.new$user.id), Mode)[,2]
date = aggregate(train.new$date, list(train.new$user.id), median)[,2]
fulltxt = aggregate(train.new$text, list(train.new$user.id), toString)[,2]
age = aggregate(train.new$age, list(train.new$user.id), Mode)[,2]
user = aggregate(train.new$user.id, list(train.new$user.id), Mode)[,2]

# Now clean the text
# Setup a new dataframe as to not override the old one
train.new = data.frame(user.id.=user,gender.=gen,topic.=top,sign.=sign,date.=date,text=fulltxt,age.=age)
train.clean = train.new

# Let's do some filtering
# Trim whitespace
train.clean$text = trimws(train.clean$text)
# Convert ascii characters (will remove most)
train.clean$text = iconv(train.clean$text, to='ASCII', sub='')
# Count the number of total characters
train.clean$char.1 = nchar(train.clean$text)
# Remove stopwords
train.clean$text = removeWords(train.clean$text, stopwords("english"))
# Remove extra spaces
train.clean$text = str_squish(train.clean$text)
# Count the number of 'relevant' characters, removed stopwords and whitespace
# Maybe younger people use more stopwords?
train.clean$char.2 = nchar(train.clean$text)
# Count the number of digits
train.clean$num = nchar(gsub("[^0-9]+", "", train.clean$text))
# Remove numbers
train.clean$text = gsub("[0-9]+","",train.clean$text)
# Count the number of upper case words
train.clean$upper = ldply(str_match_all(train.clean$text,"[A-Z]"),length)[,1]
# Change all to lower case
train.clean$text = tolower(train.clean$text)
# Count the number of punctuation marks
train.clean$punct = str_count(train.clean$text,"[:punct:]")
# Remove punctuation marks
train.clean$text = removePunctuation(train.clean$text)
# Stem words
train.clean$text = stemDocument(train.clean$text)

# Repeat to test
test.new=test
gen = aggregate(test.new$gender, list(test.new$user.id), Mode)[,2]
top = aggregate(test.new$topic, list(test.new$user.id), Mode)[,2]
sign = aggregate(test.new$sign, list(test.new$user.id), Mode)[,2]
date = aggregate(test.new$date, list(test.new$user.id), median)[,2]
fulltxt = aggregate(test.new$text, list(test.new$user.id), toString)[,2]
user = aggregate(test.new$user.id, list(test.new$user.id), Mode)[,2]
test.new = data.frame(user.id.=user,gender.=gen,topic.=top,sign.=sign,date.=date,text=fulltxt)

test.clean = test.new
# Trim whitespace
test.clean$text = trimws(test.clean$text)
# Convert ascii characters (will remove most)
test.clean$text = iconv(test.clean$text, to='ASCII', sub='')
# Count the number of total characters
test.clean$char.1 = nchar(test.clean$text)
# Remove stopwords
test.clean$text = removeWords(test.clean$text, stopwords("english"))
# Remove extra spaces
test.clean$text = str_squish(test.clean$text)
# Count the number of 'relevant' characters, removed stopwords and whitespace
# Maybe younger people use more stopwords?
test.clean$char.2 = nchar(test.clean$text)
# Count the number of digits
test.clean$num = nchar(gsub("[^0-9]+", "", test.clean$text))
# Remove numbers
test.clean$text = gsub("[0-9]+","",test.clean$text)
# Count the number of upper case words
test.clean$upper = ldply(str_match_all(test.clean$text,"[A-Z]"),length)[,1]
# Change all to lower case
test.clean$text = tolower(test.clean$text)
# Count the number of punctuation marks
test.clean$punct = str_count(test.clean$text,"[:punct:]")
# Remove punctuation marks
test.clean$text = removePunctuation(test.clean$text)
# Stem words
test.clean$text = stemDocument(test.clean$text)

# Write to csv
# Optional, does take lots of time so it's commented out here
# write.csv(train.clean, "train_clean.csv",row.names=FALSE)
# write.csv(test.clean, "test_clean.csv",row.names=FALSE)

##########
# Step 4 #
##########
# Identifying trends (non-text data)

# Age

# Recall that there were three distinct age groups
par(mfrow=c(1,1))
hist(train.clean$age, breaks=40)
table(train.clean$age)
# It looks like 13-17 is one group, 23-27 another group, and 33-48 the last group.
# Try to fix the skew a bit?
library(e1071)
skewness(train.clean$age)
skewness(log(train.clean$age))
hist(log(train.clean$age), breaks=40)
train.clean$age = log(train.clean$age)

# Gender

# Let's look at gender overall.
table(train.clean$gender.)
sum(train.clean$gender.=="female")/length(train.clean$gender.)
# 0.5143 Pretty even

# <=17
table(train.clean$gender.[exp(train.clean$age)<=17])

# >=23 and <=27
table(train.clean$gender.[exp(train.clean$age)>=23 & exp(train.clean$age)<=27])

# >=33 and <=48
table(train.clean$gender.[exp(train.clean$age)>=33 & exp(train.clean$age)<=48])

# All of them are pretty even which will make it very hard to get useful
# information for age~gender alone. Maybe consider interaction with the text?

# Topic

# Let's use barplots

par(mfrow=c(2,2))

# All
ct = table(train.clean$topic.)
ct = ct/sum(ct)
barplot(ct, main="All")

# <= 17 year olds
ct.2 = table(train.clean$topic.[exp(train.clean$age)<=17])
ct.2 = ct.2/sum(ct.2)
barplot(ct.2, main="<= 17 Year Olds")

# >=23 and <=27
ct.3 = table(train.clean$topic.[exp(train.clean$age)>=23 & exp(train.clean$age)<=27])
ct.3 = ct.3/sum(ct.3)
barplot(ct.3, main="23-27 Year Olds")

# >=33 and <=48
ct.4 = table(train.clean$topic.[exp(train.clean$age)>=33 & exp(train.clean$age)<=48])
ct.4 = ct.4/sum(ct.4)
barplot(ct.4, main="33-48 Year Olds")

# Conclusions
# <= 17 year olds have much more STUDENT posts
# 23-27 year olds have more ARTS and TECHNOLOGY posts
# 33-48 year olds have slightly more indUnk and INTERNET posts

# Let's make a difference plot of 3 groups
par(mfrow=c(3,3))

barplot(ct.2, main="Group 1")
barplot(ct.2-ct.3, main="Group 1 vs. Group 2")
barplot(ct.2-ct.4, main="Group 1 vs. Group 3")

barplot(ct.3-ct.2, main="Group 2 vs. Group 1")
barplot(ct.3, main="Group 2")
barplot(ct.3-ct.4, main="Group 2 vs. Group 3")

barplot(ct.4-ct.2, main="Group 3 vs. Group 1")
barplot(ct.4-ct.3, main="Group 3 vs. Group 2")
barplot(ct.4, main="Group 3")

# Date

# Is there any relationship between posting date. and age?
par(mfrow=c(1,1))
plot(train.clean$age,train.clean$date.)
# Maybe within each of these groups you can see that within the age group
# <=17 younger individuals there is an inverse relationship between age and date.
# 23-27 there is no discernable pattern
# 33-48 there is a relationship between age and date.

# Any patterns by year?
gp = aggregate(train.clean$age, list(format(train.clean$date., "%Y")), median)
plot(gp$Group.1, gp$x, xlab="Year", ylab="Age")
# Not really, maybe a downwards trend for year~age?

# Any patterns by month?
gp.2 = aggregate(train.clean$age, list(format(train.clean$date., "%m")), mean)
plot(gp.2$Group.1, gp.2$x, xlab="Month", ylab="Age")
# No.

# Can we really tell anything? Not really.

# user.id

plot(train.clean$user.id., train.clean$age)
# Pretty evenly distributed between the different age groups.

# Sign

plot(train.clean$sign., train.clean$age)
# Pretty evenly distributed. Can't tell much!

# char.1 - total number of characters pre-processing

plot(train.clean$char.1, train.clean$age)
# Can't tell too much

# char.2 - total number of characters after stopwords removed
# This is used as a count because of the hypothesis that younger
# participants use more stopwords as a proportion of their sentences

plot(train.clean$char.2, train.clean$age)
# Can't tell too much

# char.2/char.1 - Proprortion remaining after cleaning

plot(train.clean$char.2/(train.clean$char.1+1), train.clean$age)
# Now we're seeing something.
gp.3 = aggregate(train.clean$char.2/(train.clean$char.1+1), list(train.clean$age), median)
plot(gp.3$Group.1, gp.3$x, xlab="Age", ylab="char2/char1")

# num prop

plot(train.clean$num/(train.clean$char.1+1), train.clean$age)
gp.4 = aggregate(train.clean$num/(train.clean$char.1+1), list(train.clean$age), median)
plot(gp.4$Group.1, gp.4$x, xlab="Age", ylab="Median Propotional Use of Numbers in Post")
# Maybe a trend in the mean number of numbers used per post?

# upper prop

plot(train.clean$upper/(train.clean$char.1+1), train.clean$age)
gp.5 = aggregate(train.clean$upper/(train.clean$char.1+1), list(train.clean$age), median)
plot(gp.5$Group.1, gp.5$x, xlab="Age", ylab="Median Proportional Use of Upper-Cased Letters in Post")

# punct prop

plot(train.clean$punct/(train.clean$char.1+1), train.clean$age)
gp.6 = aggregate(train.clean$punct/(train.clean$char.1+1), list(train.clean$age), median)
plot(gp.6$Group.1, gp.6$x, xlab="Age", ylab="Median Proportional Use of Punctuation in Post")
# Oh, there definitely is something here (I think)

##########
# Step 5 #
##########
# Simple Modeling

# Temp df
df = train.clean
names(df)
# [1] user.id" "gender"  "topic"   "sign"    "date"    "text"    "age"     "char.1"  "char.2"  "num"     "upper"   "punct" 

# Create a number of variables based on the columns we currently have.
# These are all PROPORTIONS of ____ based on sentence length.
# x/(y+1) is used because if y=0 we get an error for x/y.
df$ch = df$char.2/(df$char.1+1)
df$n = df$num/(df$char.1+1)
df$u = df$upper/(df$char.1+1)
df$p = df$punct/(df$char.1+1)
df$text = NULL
names(df) = c("user.id.","gender.","topic.", "sign.","date.",
              "age.","char.1.","char.2.","num.","upper.","punct.","ch.",
              "n.","u.","p.")

# Repeat to test
tst = test.clean
tst$ch = tst$char.2/(tst$char.1+1)
tst$n = tst$num/(tst$char.1+1)
tst$u = tst$upper/(tst$char.1+1)
tst$p = tst$punct/(tst$char.1+1)
tst$text = NULL
names(tst) = c("user.id.","gender.","topic.", "sign.","date.",
                 "char.1.","char.2.","num.","upper.","punct.","ch.",
                 "n.","u.","p.")

library(dummies)
library(Matrix)
# One hot encoding for categorical variables
train_dum_1=dummy(df[,c("gender.")])
train_dum_2=dummy(df[,c("sign.")])
train_dum_3=dummy(df[,c("topic.")])
train_dum=cbind(train_dum_1,train_dum_2,train_dum_3)
colnames(train_dum)=c(levels(df$gender.),levels(df$sign.),levels(df$topic.))
train_dum_sparse = Matrix(train_dum, sparse=TRUE)

test_dum_1=dummy(tst[,c("gender.")])
test_dum_2=dummy(tst[,c("sign.")])
test_dum_3=dummy(tst[,c("topic.")])
test_dum=cbind(test_dum_1,test_dum_2,test_dum_3)
colnames(test_dum)=c(levels(tst$gender.),levels(tst$sign.),levels(tst$topic.))
test_dum_sparse = Matrix(test_dum, sparse=TRUE)

# IF interested,
# Check models in exploreOLD.R that were attempted.
# They should fit in here. They were not very good models.
#

# We can't stop here, keep going!
# We will have to use more information. This is going nowhere!

##########
# Step 6 #
##########
# In-depth Text Analsys + Modeling

# Here more advanced methods are used in extracting as much information
# as possible from the corpus. The first method below is the one
# provided by the Professor from the lecture.

# Make a temporary dataframe in case for transformations.
df = train.clean
dim(df)

# Only need id and strings
df.str = df[,c("user.id.","text")]

# The tm package requires specific names
names(df.str) = c("doc_id", "text")

# Turn into a corpus object
blogs = VCorpus(DataframeSource(df.str))

# Sub-corpus: Get blogs 1 and 2
inspect(blogs[1:2])

# Access documents: Get blog 1 and its contents
blogs[[1]]
blogs[[1]]$content

# compute TF-IDF matrix and inspect sparsity
blogs.tfidf = DocumentTermMatrix(blogs,control=list(weighting=weightTfIdf))
blogs.tfidf

# <<DocumentTermMatrix (documents: 12880, terms: 737685)>>
# Non-/sparse entries: 12611133/9488771667
# Sparsity           : 100%
# Maximal term length: 4480
# Weighting          : term frequency - inverse document frequency (normalized) (tf-idf)

# Still way too sparse!

# Inspect blogs
as.matrix(blogs.tfidf[1:2,1:200])
# aaaaaahhhh
gc()

tfidf.95 = removeSparseTerms(blogs.tfidf, 0.95)
tfidf.95

# <<DocumentTermMatrix (documents: 12880, terms: 3651)>>
# Non-/sparse entries: 8342419/38682461
# Sparsity           : 82%
# Maximal term length: 12
# Weighting          : term frequency - inverse document frequency (normalized) (tf-idf)
as.matrix(tfidf.95[1:5, 1:5])

# Other values like 0.70 were tried but 0.95 seemed like a good fit
# because the table can be filtered down from here. Additionally,
# since there were 3 groups a value near 1-1/3=0.66 wasn't preferred.

# Append to df to make a super table
train.idf.95 = df
train.idf.95$post.id = NULL
train.idf.95$text = NULL
train.idf.95$ch = train.idf.95$char.2/(train.idf.95$char.1+1)
train.idf.95$n = train.idf.95$num/(train.idf.95$char.1+1)
train.idf.95$u = train.idf.95$upper/(train.idf.95$char.1+1)
train.idf.95$p = train.idf.95$punct/(train.idf.95$char.1+1)
# Change names because of collisions later.
train.idf.95$age = NULL
names(train.idf.95)
names(train.idf.95) = c("user.id.","gender.","topic.", "sign.","date.",
                        "age.","char.1.","char.2.","num.","upper.","punct.","ch.",
                        "n.","u.","p.")
df.tfidf.95 = as.data.frame(as.matrix(tfidf.95))
df.95 = cbind(train.idf.95,df.tfidf.95)
dim(df.95)
names(df.95)

# Chose important words (but first cleanup any old files to save on space)
gc()

# Separate into three distinct age groups to see which words were
# the most important in each of the groups
group.1 = df.95[df$age<3,]
group.2 = df.95[df$age>3 & df$age<3.4,]
group.3 = df.95[df$age>3.4,]

ind = dim(group.1)[2]

# Group 1 (young)
plot(colSums(group.1[1:2,16:ind]))
n.1=names(group.1)[16:ind][colSums(group.1[1:2,16:ind])>0.01]
n.1
# [1] "found"   "learn"   "now"     "urllink" "wait"

#  [1] "bean"       "california" "car"        "chris"      "dash"       "director"   "drove"     
# [8] "evil"       "fat"        "frank"      "guy"        "hello"      "lol"        "matt"      
# [15] "mcdonald"   "michell"    "nail"       "pound"      "robot"      "sexi"       "trash"     
# [22] "urllink"    "went"       "william"    "wind"       "yell" 

# Group 2 (middle-age)

plot(colSums(group.2[1:2,16:ind]))
n.2=names(group.2)[16:ind][colSums(group.2[1:2,16:ind])>0.01]
n.2
# [1] "almost" "eat"    "live"   "need"   "never"  "well"   "world" 

#  [1] "anyway"   "band"     "boost"    "carpet"   "celebr"   "coach"    "competit" "corner"  
# [9] "crown"    "dash"     "dave"     "divid"    "draw"     "drum"     "eight"    "elimin"  
# [17] "envelop"  "flood"    "fought"   "germani"  "goal"     "hockey"   "host"     "itali"   
# [25] "leagu"    "low"      "match"    "minut"    "moor"     "paul"     "pie"      "pitch"   
# [33] "play"     "pressur"  "prior"    "score"    "shock"    "shot"     "soccer"   "spanish" 
# [41] "struck"   "toast"    "world"  

# Group 3 (not so middle-age)
plot(colSums(group.3[1:2,16:ind]))
n.3=names(group.3)[16:ind][colSums(group.3[1:2,16:ind])>0.01]
n.3
# [1] "can"     "cool"    "enjoy"   "hour"    "mean"    "money"   "need"    "now"     "real"   
# [10] "show"    "thank"   "urllink" "without"

#  [1] "chip"       "christ"     "church"     "comp"       "game"       "god"        "jesus"     
# [8] "michael"    "play"       "player"     "saint"      "smith"      "tournament"

# All of the key words
key.words = unique(c(n.1,n.2,n.3))

# Since these are the most important words for distinguishing
# between each of the groups, maybe they can be used in linear
# regression? As long as they are distinct from one another
# it should be useful.

# Let's try another method using the text2vec package
#
#
# This is the core of the program
#
#

# https://cran.r-project.org/web/packages/text2vec/vignettes/text-vectorization.html
library(text2vec)

# Create a word tokenizer
it_train=itoken(train.clean$text, tokenizer=word_tokenizer,ids=train.clean$post.id)

# Create the vocabulary - up to 2 ngrams
vocab = create_vocabulary(it_train, ngram = c(1L, 2L))

# Prune vocabulary
pruned_vocab = prune_vocabulary(vocab, term_count_min=50, doc_proportion_max=0.40, doc_proportion_min=0.005)

# Vectorize vocabulary
vectorizer = vocab_vectorizer(pruned_vocab)

# Apply to the training data
dtm_train = create_dtm(it_train, vectorizer)

# Create a framework for TF-IDF
tfidf = TfIdf$new()

# Fit training data to this TF-IDF framework
dtm_train_tfidf = fit_transform(dtm_train, tfidf)

# Make to matrix
dim(dtm_train_tfidf) = c(dim(train.clean)[1],length(dtm_train_tfidf)/dim(train.clean)[1])

# Set names of column
colnames(dtm_train_tfidf) = pruned_vocab$term

# Same to test
it_test=itoken(test.clean$text, tokenizer=word_tokenizer,ids=test.clean$post.id)
dtm_test = create_dtm(it_test, vectorizer)
dtm_test_tfidf= fit_transform(dtm_test, tfidf)
dim(dtm_test_tfidf) = c(dim(test.clean)[1],length(dtm_test_tfidf)/dim(test.clean)[1])
colnames(dtm_test_tfidf) = pruned_vocab$term

# Sentiment analysis
library(syuzhet)
senti_train = get_nrc_sentiment(as.character(train.new$text))
senti_train_prop = senti_train/rowSums(senti_train)
senti_train_prop_sparse = Matrix(as.matrix(senti_train_prop))

senti_test = get_nrc_sentiment(as.character(test.new$text))
senti_test_prop = senti_test/rowSums(senti_test)
senti_test_prop_sparse = Matrix(as.matrix(senti_test_prop))

###
# Let's keep going #
###

# https://m-clark.github.io/text-analysis-with-R/part-of-speech-tagging.html
# https://datascience.stackexchange.com/questions/5316/general-approach-to-extract-key-text-from-sentence-nlp
# Part of speech tagging

# Initialize tokenizer  
sent_token_annotator = Maxent_Sent_Token_Annotator()
word_token_annotator = Maxent_Word_Token_Annotator()
pos_tag_annotator = Maxent_POS_Tag_Annotator()
pos = c("CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP",
        "NNPS","PDT","POR","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH",
        "VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB")

# Annotation function
annot = function(string) {
  a2 = annotate(string, list(sent_token_annotator, word_token_annotator))
  a3 = annotate(string, pos_tag_annotator, a2)
  a3w = subset(a3, type=="word")
  sapply(a3w$features, "[[", "POS")[1]
}

# Get the column names
length(colnames(dtm_train))
col_pos = unlist(lapply(colnames(dtm_train), annot))
length(col_pos)
# Replace all "." with "NA" and "$" with "D"
col_pos = replace(col_pos, col_pos==".", "NA")
col_pos = replace(col_pos, col_pos=="PRP$", "PRPD")
col_pos = replace(col_pos, col_pos=="WP$", "WPD")
table(col_pos)
length(col_pos)

# Get a dtm table with new column names
dtm_pos_train = dtm_train
colnames(dtm_pos_train) = col_pos

# Get unique column names
uni = unique(col_pos)

# Initialize matrix
outmat = Matrix(nrow=dim(dtm_pos_train)[1], ncol=0)

# Calculate rowsums for same column names
for (pos in uni) {
  print(pos)
  if (pos != "TO" & pos!="WPD") {
    index = colnames(dtm_pos_train)==pos
    somecol = Matrix(rowSums(dtm_pos_train[,index]))
    outmat = cbind(outmat, somecol)
  } else if (pos=="TO") {
    outmat = cbind(outmat, dtm_pos_train[,"TO"])
  } else if (pos=="WPD") {
    outmat = cbind(outmat, dtm_pos_train[,"WPD"])
  }
}
# Set column names
colnames(outmat) = uni

# Repeat to test
length(colnames(dtm_test))
# Could this be redundant?
col_pos = unlist(lapply(colnames(dtm_test), annot))
length(col_pos)
col_pos = replace(col_pos, col_pos==".", "NA")
col_pos = replace(col_pos, col_pos=="PRP$", "PRPD")
col_pos = replace(col_pos, col_pos=="WP$", "WPD")
table(col_pos)
length(col_pos)
dtm_pos_test = dtm_test
colnames(dtm_pos_test) = col_pos
uni.test = unique(col_pos)
outmat.test = Matrix(nrow=dim(dtm_pos_test)[1], ncol=0)
for (pos in uni) {
  print(pos)
  if (pos != "TO" & pos!="WPD") {
    index = colnames(dtm_pos_test)==pos
    somecol = Matrix(rowSums(dtm_pos_test[,index]))
    outmat.test = cbind(outmat.test, somecol)
  } else if (pos=="TO") {
    outmat.test = cbind(outmat.test, dtm_pos_test[,"TO"])
  } else if (pos=="WPD") {
    outmat.test = cbind(outmat.test, dtm_pos_test[,"WPD"])
  }
}
colnames(outmat.test) = uni

# Get proportions
outmat = outmat/rowSums(outmat)
outmat.test = outmat.test/rowSums(outmat.test)

# All done
dim(outmat)
dim(outmat.test)

# Bring all the predictors together
train_X = cbind(dtm_train_tfidf,train_dum_sparse,senti_train_prop_sparse,outmat)
test_X = cbind(dtm_test_tfidf,test_dum_sparse,senti_test_prop_sparse,outmat.test)

# Check the dimension
dim(train_X)
dim(test_X)

# Parallel Processing

######################################
# WARNING: ThIS WILL EAT UP YOUR CPU #
# YOU MAY WANT TO ONLY USE 1 CORE IF #
# YOU DON'T WANT YOUR CPU USAGE TO   #
# BE 100%. CAREFUL ON RUNNING ON     #
# WEAKER SYSTEMS!!!!                 #
######################################

# Here is where you can modify age.
skewness(df$age.)
lambda = -.67
y.mod = (df$age.**lambda-1)/lambda
skewness(y.mod)
par(mfrow=c(1,1))
hist(y.mod, breaks=100)

library(glmnet)
library(doParallel)
library(parallel)

registerDoParallel(detectCores())
gc()
# Alpha == 1 indicates LASSO. We will not use this and instead use Alpha==0
# for ridge regression.
glm_model = cv.glmnet(train_X,y.mod,alpha=0,type.measure="mae",parallel=TRUE)
plot(glm_model)
min(glm_model$cvm)

# Alpha LogMAE   MAE box-cox
#     0  0.156 3.889   0.019
#     1  0.160 4.058

# Calcualted lambda value. This value determines which variables have their
# weights set to 0. Smaller lambda = use more variables.
lam=glm_model$lambda.min
lam

# Coefficients of model given optimal lambda
co=coef(glm_model, s="lambda.min")
print(glm_model$glmnet.fit)

# Make predictions on the data
pred=predict(glm_model, test_X, type="response", s=lam)
hist(pred, breaks=100)

# Inverse transformation used on y or y.mod

# This is for no transform
# pred = pred

# This is for log
# pred=exp(pred)

# This is for boxcox
pred=(pred*lambda+1)**(1/lambda)

hist(pred, breaks=100)
qqnorm(pred)
qqline(pred)

# Return the data frame in the format kaggle likes
out = data.frame(user.id=tst$user.id, age=pred)
names(out) = c("user.id", "age")

# Some naive transformations
out$age[out$age<13] = 13
out$age[out$age>48] = 48
hist(out$age, breaks=100)

# Naive transformation. Simply the age doesnt fit the orignial categories,
# push them into the closest age group.
out$age[out$age>17 & out$age<20] = 17
out$age[out$age>20 & out$age<23] = 23
out$age[out$age>27 & out$age<30] = 27
out$age[out$age>30 & out$age<33] = 33
hist(out$age, breaks=100)

# # Not-so-naive transformation. Set the age to the median of the most likely group
# table(train$age)
# #   13    14    15    16    17    23    24    25    26    27    33    34    35    36    37    38    39    40    41    42    43    44    45    46    47    48 
# # 9597 19615 27595 49818 53315 47195 54102 43920 35244 31448 10713  7896 10398  8935  6484  4298  3505  4194  2801  1262  2049  1597  3065  1157  1568  1190 
# median(train$age[train$age<=17])
# # [1] 16
# median(train$age[train$age>=23 & train$age<=27])
# # [1] 25
# median(train$age[train$age>=33])
# # [1] 36
# 
# # Find the cutoffs between these ranges
# c1=mean(train$age[train$age>=16 & train$age<=25])
# # [1] 20.879
# c2=mean(train$age[train$age>=25 & train$age<=36])
# # [1] 28.077
# 
# # Make the transformations
# out$age[out$age>17 & out$age<c1] = 16
# out$age[out$age>c1 & out$age<23] = 25
# out$age[out$age>27 & out$age<c2] = 25
# out$age[out$age>c2 & out$age<33] = 36
# 
# # This ends up being a more centered distribution
# hist(out$age, breaks=100)

# Write csv.
write.csv(out, "final_1.csv", row.names = FALSE)