# ECON 576 Machine Learning Competition Code, for Ian Bach and Matt Fiorini
library(tidyverse)
library(mice)

train = read.csv("https://raw.githubusercontent.com/MattFiorini/ECON576_MLcomp/main/data/train.csv", sep = ",")
test = read.csv("https://raw.githubusercontent.com/MattFiorini/ECON576_MLcomp/main/data/test.csv", sep = ",")
str(train)

realBlanks <- c("LotFrontage", "MasVnrType", "MasVnrArea", "Electrical"
                , "GarageYrBlt")

for (i in 1:length(realBlanks)){
  train[which(train[,realBlanks[i]]=='NA'),realBlanks[i]] <- NA 
  test[which(test[,realBlanks[i]]=='NA'),realBlanks[i]] <- NA 
}

s <- c("MSSubClass","MSZoning","Street","Alley","LotShape","LandContour","Utilities",
       "LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType",
       "HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType",
       "ExterQual","ExterCond","Foundation", "BsmtQual","BsmtCond",
       "BsmtExposure", "BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir",
       "Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish",
       "GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType",
       "SaleCondition")
for (i in 1:length(s)){
  train[,s[i]] <- as.factor(train[,s[i]])
  test[,s[i]] <- as.factor(test[,s[i]])
}

train = train %>% select(-Id, -Utilities, -Street)
rm(s, i)

sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))

None = c("Alley","MasVnrType","BsmtQual", "BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature")

for (col in None) {
  train[[col]] <- as.factor(ifelse(is.na(train[[col]]), "None", as.character(train[[col]])))
  test[[col]] <- as.factor(ifelse(is.na(test[[col]]), "None", as.character(test[[col]])))
}

train$GarageYrBlt <- ifelse(is.na(train$GarageYrBlt), train$YearBuilt, train$GarageYrBlt)
test$GarageYrBlt <- ifelse(is.na(test$GarageYrBlt), test$YearBuilt, test$GarageYrBlt)

imputedValues <- mice(data=train
                      , seed=2016     
                      , method="cart" 
                      , m=1           
                      , maxit = 5     
)

train <- mice::complete(imputedValues,1) 

imputedValuesTe <- mice(data=test
                        , seed=2016     
                        , method="cart" 
                        , m=1           
                        , maxit = 5    
)

test <- mice::complete(imputedValuesTe,1) 

sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))

d = train
names(d)

d = d[,c(78,1:77)]

names(d)[1] <- "y"
names(d)

rm(imputedValuesTe, imputedValues)

library(caret)
set.seed(1234)

# Using the dummyVars() function, create one-hot encoded variables for all
# the categorical variables. Do this for both the 'd' data.frame and 'te' set

names(d)
names(test)
str(test)

sapply(test, function(col) length(unique(col[!is.na(col)])) < 2)

#for d set
dummiesD = dummyVars(y ~ ., data=d)
ex = data.frame(predict(dummiesD, newdata = d))
names(ex) = gsub("\\.","",names(ex))
d = cbind(d$y, ex)

#for test set
str(test)
test = test %>% select(-Utilities,-Street)
dummiesTe = dummyVars(Id ~ ., data=test)
ex1 = data.frame(predict(dummiesTe, newdata = test))
names(ex1) = gsub("\\.","",names(ex1))
test = cbind(test$Id, ex1)
rm(dummiesD, ex, dummiesTe, ex1)

descCor = cor(d[,2:ncol(d)])
highlyCorDescr = findCorrelation(descCor, cutoff = 0.8)
filteredDesc = d[,2:ncol(d)][,-highlyCorDescr]
summary(descCor2[upper.tri(descCor2)])
names(d)[1] = "y"
d = cbind(d$y, filteredDesc)
names(d)[1] = "y"

rm(descCor, highlyCorDescr, filteredDesc)

y= d$y
d = cbind(rep(1, nrow(d)), d[2:ncol(d)])
names(d)[1] = "ones"
comboInfo = findLinearCombos(d)
d = d[, - comboInfo$remove]
d = d[,c(2:ncol(d))]
d = cbind(y , d)

rm(comboInfo)

nzv = nearZeroVar(d, saveMetrics = T)
head(nzv)
d = d[, c(T,!nzv$nzv[2:ncol(d)])]
rm(nzv)

set.seed(1234)

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)

m1 = train(y~.,data=d, method = "lm", trControl = ctrl, metric = "Rsquared")
m1
summary(m1)
