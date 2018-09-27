########### Exploratory Data Analysis

##########
# Step 1 #
##########
# Working directory and reading in raw data

# Set this to the parent directory of where all your files are stored
wd = "F:/2018 Fall/SYS 6018, Data Mining/assignments/kaggle/03_Age/sys6018-competition-blogger-characteristics"
setwd(wd)

# Start reading in the data.
train = read.csv("Original Files/train.csv")
test = read.csv("Original Files/test.csv")

##########
# Step 2 #
##########
# Identify odd values

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
eng = c("january", "february", "march", "april", "may", "june", "july", "August",
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
438749/442961
# This is a loss of less than 1%

# Check to see their age distributions
par(mfrow=c(1,2))
hist(train.en$age)
hist(train.other$age)

# Notice that the test has all english dates!
index2 = grepl(paste(eng,collapse="|"), test$date, ignore.case=TRUE)
length(index2)
test.en=test[index,]

# Let's go ahead with just using english dates and non-empty dates
train.en$date = as.Date(train.en$date, format="%d,%B,%Y")

# Write to csv
write.csv(train.en, "train_en.csv",row.names=FALSE)