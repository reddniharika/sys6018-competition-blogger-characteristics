########### Exploratory Data Analysis

##########
# Step 1 #
##########
# Working directory and reading in raw data, general setup
install.packages("tm", "tidyverse", "parallel", "snow", "SnowballC","e1071","glmnet")
library(parallel)
library(snow)
detectCores()
cls = snow::makeCluster(detectCores(), "SOCK")

library(tidyverse)

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

missing.test = sapply(test, function(x) length(which(is.na(x))))
data.frame(missing.test[missing.test>0])
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

dim(train)[1]+dim(test)[1]
# [1] 681284
max(train$post.id)
# [1] 681161
max(test$post.id)
# [1] 681284

# Yes, this is exactly what happened.

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

# Index includes only english dates entries
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
# with the first two age groups! This could be important

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

# Text

# First thing, change text into a character variable!
unlist(lapply(train.new, class))
train.new$text = as.character(train.new$text)
class(train.new$text)

# Now clean the text
library(tm)
library(stringr)
library(plyr)
library(SnowballC)

train.clean = train.new

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

# Write to csv
write.csv(train.clean, "train_clean.csv",row.names=FALSE)

# Now repeat to test dataset
test.clean = test

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
write.csv(test.clean, "test_clean.csv",row.names=FALSE)

##########
# Step 3 #
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
table(train.clean$gender)
sum(train.clean$gender=="female")/length(train.clean$gender)
# 0.5143 Pretty even

# <=17
table(train.clean$gender[exp(train.clean$age)<=17])

# >=23 and <=27
table(train.clean$gender[exp(train.clean$age)>=23 & exp(train.clean$age)<=27])

# >=33 and <=48
table(train.clean$gender[exp(train.clean$age)>=33 & exp(train.clean$age)<=48])

# All of them are pretty even which will make it very hard to get useful
# information for age~gender alone. Maybe consider interaction with the text?

# Topic

# Let's use barplots

par(mfrow=c(2,2))

# All
ct = table(train.clean$topic)
ct = ct/sum(ct)
barplot(ct, main="All")

# <= 17 year olds
ct.2 = table(train.clean$topic[exp(train.clean$age)<=17])
ct.2 = ct.2/sum(ct.2)
barplot(ct.2, main="<= 17 Year Olds")

# >=23 and <=27
ct.3 = table(train.clean$topic[exp(train.clean$age)>=23 & exp(train.clean$age)<=27])
ct.3 = ct.3/sum(ct.3)
barplot(ct.3, main="23-27 Year Olds")

# >=33 and <=48
ct.4 = table(train.clean$topic[exp(train.clean$age)>=33 & exp(train.clean$age)<=48])
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

# Is there any relationship between posting date and age?
par(mfrow=c(1,1))
plot(train.clean$age,train.clean$date)
# Maybe within each of these groups you can see that within the age group
# <=17 younger individuals there is an inverse relationship between age and date
# 23-27 there is no discernable pattern
# 33-48 there is a relationship between age and date

# Any patterns by year?
gp = aggregate(train.clean$age, list(format(train.clean$date, "%Y")), median)
plot(gp$Group.1, gp$x, xlab="Year", ylab="Age")
# Not really, maybe a downwards trend for year~age?

# Any patterns by month?
gp.2 = aggregate(train.clean$age, list(format(train.clean$date, "%m")), mean)
plot(gp.2$Group.1, gp.2$x, xlab="Month", ylab="Age")
# No.

# Can we really tell anything? Not really.

# user.id

plot(train.clean$user.id, train.clean$age)
# Pretty evenly distributed between the different age groups.

# Sign

plot(train.clean$sign, train.clean$age)
# Pretty evenly distributed. Can't tell much!

# char.1

plot(train.clean$char.1, train.clean$age)
# Can't tell too much

# char.2

plot(train.clean$char.2, train.clean$age)
# Can't tell too much

# char.2/char.1

plot(train.clean$char.2/(train.clean$char.1+1), train.clean$age)
# Now we're seeing something.
gp.3 = aggregate(train.clean$char.2/(train.clean$char.1+1), list(train.clean$age), median)
plot(gp.3$Group.1, gp.3$x, xlab="Age", ylab="char2/char1")

# num/char.1

plot(train.clean$num/(train.clean$char.1+1), train.clean$age)
gp.4 = aggregate(train.clean$num/(train.clean$char.1+1), list(train.clean$age), median)
plot(gp.4$Group.1, gp.4$x, xlab="Age", ylab="Median Propotional Use of Numbers in Post")
# Maybe a trend in the mean number of numbers used per post?

# upper

plot(train.clean$upper/(train.clean$char.1+1), train.clean$age)
gp.5 = aggregate(train.clean$upper/(train.clean$char.1+1), list(train.clean$age), median)
plot(gp.5$Group.1, gp.5$x, xlab="Age", ylab="Median Proportional Use of Upper-Cased Letters in Post")

# punct
plot(train.clean$punct/(train.clean$char.1+1), train.clean$age)
gp.6 = aggregate(train.clean$punct/(train.clean$char.1+1), list(train.clean$age), median)
plot(gp.6$Group.1, gp.6$x, xlab="Age", ylab="Median Proportional Use of Punctuation in Post")
# Oh, there definitely is something here

#######
# Preliminary linear model!
#######

# Temp df
df = train.clean
names(df)
# [1] "post.id" "user.id" "gender"  "topic"   "sign"    "date"    "text"    "age"     "char.1"  "char.2"  "num"     "upper"   "punct" 

df$ch = df$char.2/(df$char.1+1)
df$n = df$num/(df$char.1+1)
df$u = df$upper/(df$char.1+1)
df$p = df$punct/(df$char.1+1)
df$text = NULL

# Repeat to test
tst = test.clean
tst$ch = tst$char.2/(tst$char.1+1)
tst$n = tst$num/(tst$char.1+1)
tst$u = tst$upper/(tst$char.1+1)
tst$p = tst$punct/(tst$char.1+1)
tst$text = NULL

# Model 1

model=lm(age~topic+sign+date, data=df)
anova(model)

pred = predict(model, tst)
hist(pred)

out = data.frame(user.id=test.clean$user.id, age=pred)
means = ddply(out, ~user.id, summarise, age=mean(age))
means$age = round(means$age)
hist(means$age, breaks=100)

# >=13 and <=17
means$age[means$age<13] = 13
means$age[means$age>17 & means$age<20] = 17
# >=23 and <=27
means$age[means$age>20 & means$age<23] = 23
means$age[means$age>27 & means$age<30] = 27
# >=33 and <=48
means$age[means$age>30 & means$age<33] = 33
means$age[means$age>48] = 48

hist(means$age, breaks=100)

write.csv(means, "test_output.csv", row.names = FALSE)

# Model 2

model.2=lm(age~topic+sign+date+ch+n+u+p, data=df)
anova(model.2)

pred.2 = predict(model.2, tst)
hist(pred.2)

out = data.frame(user.id=test.clean$user.id, age=pred.2)
means.2 = ddply(out, ~user.id, summarise, age=mean(age))
means.2$age = round(means$age)
hist(means.2$age, breaks=100)

# >=13 and <=17
means.2$age[means.2$age<13] = 13
means.2$age[means.2$age>17 & means.2$age<20] = 17
# >=23 and <=27
means.2$age[means.2$age>20 & means.2$age<23] = 23
means.2$age[means.2$age>27 & means.2$age<30] = 27
# >=33 and <=48
means.2$age[means.2$age>30 & means.2$age<33] = 33
means.2$age[means.2$age>48] = 48

hist(means.2$age, breaks=100)

write.csv(means.2, "test_output_2.csv", row.names = FALSE)

# Model 3

# [1] "post.id" "user.id" "gender"  "topic"   "sign"    "date"    "text"    "age"     "char.1"  "char.2"  "num"     "upper"   "punct"  

# Here's a function for mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
# https://stackoverflow.com/questions/32684931/how-to-aggregate-data-in-r-with-mode-most-common-value-for-each-row

df.agg = data.frame(
  user.id = aggregate(df$user.id, list(df$user.id), Mode)[,2],
  age = aggregate(df$age, list(df$user.id), median)[,2],
  gender = aggregate(df$gender, list(df$user.id), Mode)[,2],
  topic = aggregate(df$topic, list(df$user.id), Mode)[,2],
  sign = aggregate(df$sign, list(df$user.id), Mode)[,2],
  text = as.character(aggregate(df$text, list(df$user.id), function(x) paste(x, sep="", collapse=" "))[,2]),
  date = aggregate(df$date, list(df$user.id), median)[,2],
  char.1 = aggregate(df$char.1, list(df$user.id), median)[,2],
  char.2 = aggregate(df$char.2, list(df$user.id), median)[,2],
  num = aggregate(df$num, list(df$user.id), median)[,2],
  upper = aggregate(df$upper, list(df$user.id), median)[,2],
  punct = aggregate(df$punct, list(df$user.id), median)[,2]
)

df.agg$text = as.character(df.agg$text)
df.agg$ch = df.agg$char.2/(df.agg$char.1+1)
df.agg$n = df.agg$num/(df.agg$char.1+1)
df.agg$u = df.agg$upper/(df.agg$char.1+1)
df.agg$p = df.agg$punct/(df.agg$char.1+1)

temp = head(df.agg)
View(temp)

names(df.agg)

model.3=lm(age~topic+sign+date+num+upper+punct+char.1+char.2+ch+n+u+p,data=df.agg)
anova(model.3)

# Model selection
library(randomForest)
fit = randomForest(age~topic+sign+date+num+upper+punct+char.1+char.2+ch+n+u, data=df.agg, importance=TRUE, ntree=500)
fit$importance[order(-fit$importance[,1]),]

#             %IncMSE IncNodePurity
# topic  0.0459995868     404.00401
# char.1 0.0173464808      97.46159
# char.2 0.0164612837      95.20468
# upper  0.0155151918      86.97802
# punct  0.0135988234     109.91225
# u      0.0129607156     138.29239
# num    0.0047914304      38.90674
# n      0.0038961605      58.55020
# ch     0.0026711346      96.62709
# date   0.0009411056      87.10879
# sign   0.0002106824     117.60989

# It looks like topic, char.1, char.2, upper, punct, and u are good variables

model.4 = lm(age~topic+char.1+char.2+upper+punct+u, data=df.agg)
pred.4 = predict(model.4, tst)
hist(exp(pred.4))

out = data.frame(user.id=tst$user.id, age=exp(pred.4))
means = ddply(out, ~user.id, summarise, age=mean(age))
means$age = round(means$age)
hist(means$age, breaks=100)

# >=13 and <=17
means$age[means$age<13] = 13
means$age[means$age>17 & means$age<20] = 17
# >=23 and <=27
means$age[means$age>20 & means$age<23] = 23
means$age[means$age>27 & means$age<30] = 37
# >=33 and <=48
means$age[means$age>30 & means$age<33] = 33
means$age[means$age>48] = 48

hist(means$age, breaks=100)

write.csv(means, "test_output_4.csv", row.names = FALSE)

# We will have to use more information. This is going nowhere!

##########
# Step 4 #
##########
# TF-IDF Matricies

# Make a temporary dataframe in case for transformations.
df = train.clean
dim(df)

### First try out for the entire dataset regardless of age group

# Only need id and strings
df.str = df[,c("post.id","text")]

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

# <<DocumentTermMatrix (documents: 442961, terms: 737685)>>
# Non-/sparse entries: 33252199/326732433086
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

# <<DocumentTermMatrix (documents: 442961, terms: 105)>>
# Non-/sparse entries: 7955168/38555737
# Sparsity           : 83%
# Maximal term length: 7
# Weighting          : term frequency - inverse document frequency (normalized) (tf-idf)

as.matrix(tfidf.95[1:5, 1:5])

# Append to df to make a super table
train.idf.95 = df
train.idf.95$post.id = NULL
train.idf.95$text = NULL
train.idf.95$ch = train.idf.95$char.2/(train.idf.95$char.1+1)
train.idf.95$n = train.idf.95$num/(train.idf.95$char.1+1)
train.idf.95$u = train.idf.95$upper/(train.idf.95$char.1+1)
train.idf.95$p = train.idf.95$punct/(train.idf.95$char.1+1)
# Change names because of collisions later.
names(train.idf.95) = c("user.id.","gender.","topic.", "sign.","date.",
                        "age.","char.1.","char.2.","num.","upper.","punct.","ch.",
                        "n.","u.","p.")
df.tfidf.95 = as.data.frame(as.matrix(tfidf.95))
df.95 = cbind(train.idf.95,df.tfidf.95)
dim(df.95)

names(df.95)

# Tokenize (choose important words)
gc()
names(df.95)

group.1 = df.95[df$age<3,]
group.2 = df.95[df$age>3 & df$age<3.4,]
group.3 = df.95[df$age>3.4,]

plot(colSums(group.1[1:2,16:283]))
n.1=names(group.1)[16:284][colSums(group.1[1:2,16:283])>0.05]
n.1
# [1] "found"   "learn"   "now"     "urllink" "wait"

plot(colSums(group.2[1:2,16:283]))
n.2=names(group.2)[16:284][colSums(group.2[1:2,16:283])>0.05]
n.2
# [1] "almost" "eat"    "live"   "need"   "never"  "well"   "world" 

plot(colSums(group.3[1:2,16:283]))
n.3=names(group.3)[16:284][colSums(group.3[1:2,16:283])>0.05]
n.3
# [1] "can"     "cool"    "enjoy"   "hour"    "mean"    "money"   "need"    "now"     "real"   
# [10] "show"    "thank"   "urllink" "without"

key.words = unique(c(n.1,n.2,n.3))

# https://cran.r-project.org/web/packages/text2vec/vignettes/text-vectorization.html
install.packages("text2vec")
library(text2vec)

it_train=itoken(train.clean$text, tokenizer=word_tokenizer,ids=train.clean$post.id)
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)
tfidf = TfIdf$new()
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
it_test=itoken(test.clean$text, tokenizer=word_tokenizer,ids=test.clean$post.id)
dtm_test_tfidf = create_dtm(it_test, vectorizer)
dtm_test_tfidf=transform(dtm_test_tfidf, tfidf)
glmnet_classifer = cv.glmnet(x=dtm_train_tfidf,y=train.clean$age,
                             alpha=1,
                             type.measure="mae")
plot(glmnet_classifer)

lam=glmnet_classifer$lambda.min
# [1] 0.0005819068
coef(glmnet_classifer, s="lambda.min")
pred=predict(glmnet_classifer,c(123123),s="lambda.min")
print(glmnet_classifer$glmnet.fit)
pred=predict(glmnet_classifer, dtm_test_tfidf, type="response", s=lam)
pred = exp(pred)
pred[pred<13] = 13
pred[pred>48] = 48
hist(pred)


out = data.frame(user.id=tst$user.id, age=pred)
names(out) = c("user.id", "age")
means = ddply(out, ~user.id, summarise, age=mean(age))
means$age = round(means$age)
hist(means$age, breaks=100)

write.csv(means, "test_output_6.csv", row.names = FALSE)
# Best score so far, I'll work on this later