###       ###
# Old Files #
###       ###

# Tommy Jun (tbj2cu)
# for SYS 6018: Data Mining

# Here's a list of all the old things which didn't make it into the final
# either because it was too complicated, wrong, or not as accurate

# Note that these lines of code will not run as is.

# The four following are a set of four initial models tried
# 1. Multi-linear regression
# 2. Multi-linear regression with derived text features
# 3. Multi-linear regression with derived text features but
#    instead of taking each document as a row, all of the
#    documents belonging to each unique person was aggregated
# 4. Multi-linear regression similar to 3 but with features
#    derived using recursive variable selection via random forest
# All of these models didn't perform too well, so switching over
# to more `in-depth` models was required.

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
