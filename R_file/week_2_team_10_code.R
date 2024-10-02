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

train$LotFrontage <- as.numeric(train$LotFrontage)
train$GarageYrBlt <- as.numeric(train$GarageYrBlt)
train$MasVnrArea <- as.numeric(train$MasVnrArea)

test$LotFrontage <- as.numeric(test$LotFrontage)
test$GarageYrBlt <- as.numeric(test$GarageYrBlt)
test$MasVnrArea <- as.numeric(test$MasVnrArea)

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


train = train %>% select(-Id, -all_of(s))
test = test %>% select(-all_of(s))

rm(s, i)

sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))

train$GarageYrBlt <- ifelse(is.na(train$GarageYrBlt), 0, train$GarageYrBlt)
test$GarageYrBlt <- ifelse(is.na(test$GarageYrBlt), 0, test$GarageYrBlt)

sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))

str(train)
str(test)

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

d = d[,c(36,1:35)]

names(d)[1] <- "y"
names(d)

rm(imputedValuesTe, imputedValues)

library(caret)
set.seed(1234)

inTrain = createDataPartition(y = d$y, p = 0.8, list = F)
tr = d[inTrain,]
te = d[-inTrain,]


##########################
#using non-cleaned data to understand the impact of a normal GBM and random forest model on the data set

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)

m1 = train(y ~ ., data = tr, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)
m2 = train(y ~ ., data = tr, method = "rf", trControl = ctrl, nodesize = 5, ntree = 250, importance = TRUE, metric = "RMSE")

#m1 train
defaultSummary(data=data.frame(obs=tr$y, pred=predict(m1, newdata=tr))
               , model=m1)
#m1 test
defaultSummary(data=data.frame(obs=te$y, pred=predict(m1, newdata=te))
               , model=m1)
#m2 train
defaultSummary(data=data.frame(obs=tr$y, pred=predict(m2, newdata=tr))
               , model=m2)
#m2 test
defaultSummary(data=data.frame(obs=te$y, pred=predict(m2, newdata=te))
               , model=m2)
#large difference between the model on the train and test data for m2 (Random Forest) indicate
#that the model is overfitting. Going with the GBM model based on consistency and ability to handle
# variances in the data better than the Random Forest. Will retrain the model on the complete dataset
# and leverage predictions on the test dataset

m_gbm = train(y ~ ., data = d, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)
summary(m_gbm)

defaultSummary(data=data.frame(obs=d$y, pred=predict(m_gbm, newdata=d))
               , model=m_gbm)

preds = predict(m_gbm, newdata = test)


results = data.frame(Id = test$Id, SalePrice = preds)
names(results) = c("Id","SalePrice")

getwd()
write.table(results, file="week_2_team_10.csv", quote = F, row.names = F, sep = ",")
