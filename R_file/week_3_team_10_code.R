# ECON 576 Machine Learning Competition Code, for Ian Bach and Matt Fiorini
library(tidyverse)
library(mice)
library(caret)
library(gbm)
library(xgboost)

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

s <- c("MSSubClass","MSZoning","Alley","LotShape","LandContour",
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


train = train %>% select(-Id, -Street, -Utilities)
test = test %>% select(-Street, -Utilities)

None = c("Alley","MasVnrType","BsmtQual", "BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature")

for (col in None) {
  train[[col]] <- as.factor(ifelse(is.na(train[[col]]), "None", as.character(train[[col]])))
  test[[col]] <- as.factor(ifelse(is.na(test[[col]]), "None", as.character(test[[col]])))
}


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

set.seed(1234)

#splitting dataset into non-vectored and vectored sets to determine success of each model
# set 1 will be XGBoost while set 2 will be additional data manipulation with GBM model
train_1 = train %>% select(-all_of(s))
test_1 = test %>% select(-all_of(s))

d1 = train_1
names(d1)

d1 = d1[,c(36,1:35)]

names(d1)[1] <- "y"
names(d1)

rm(imputedValuesTe, imputedValues)

#set 2 creation and manipulation

train_2 = train
test_2 = test

d2 = train_2
names(d2)

d2 = d2[,c(78,1:77)]

names(d2)[1] <- "y"
names(d2)

dummiesD = dummyVars(y ~ ., data=d2)
ex = data.frame(predict(dummiesD, newdata = d2))
names(ex) = gsub("\\.","",names(ex))
d2 = cbind(d2$y, ex)

#for test set
dummiesTe = dummyVars(Id ~ ., data=test_2)
ex1 = data.frame(predict(dummiesTe, newdata = test_2))
names(ex1) = gsub("\\.","",names(ex1))
test_2 = cbind(test_2$Id, ex1)
rm(dummiesD, ex, dummiesTe, ex1)

#normalize the values to a 0 to 1 range, leaving sale price as is
preProcValues <- preProcess(d2[2:ncol(d2)], method = c("center","scale"))
d2 <- predict(preProcValues, d2)

#for "test_2" data, except for the ID
preProcValues <- preProcess(test_2[2:ncol(test_2)], method = c("center","scale"))
test_2 <- predict(preProcValues, test_2)

#removing highly correlated values from d2 set
descCor = cor(d2[,2:ncol(d2)])
highlyCorDescr = findCorrelation(descCor, cutoff = 0.8)
filteredDesc = d2[,2:ncol(d2)][,-highlyCorDescr]
descCor2 = cor(filteredDesc)
summary(descCor2[upper.tri(descCor2)])
names(d2)[1] = "y"
d2 = cbind(d2$y, filteredDesc)
names(d2)[1] = "y"
#reducing collinearity
y= d2$y
d2 = cbind(rep(1, nrow(d2)), d2[2:ncol(d2)])
names(d2)[1] = "ones"
comboInfo = findLinearCombos(d2)
d2 = d2[, - comboInfo$remove]
d2 = d2[,c(2:ncol(d2))]
d2 = cbind(y , d2)

#data partition creation for each set

inTrain1 = createDataPartition(y = d1$y, p = 0.8, list = F)
tr1 = d1[inTrain1,]
te1 = d1[-inTrain1,]

inTrain2 = createDataPartition(y = d2$y, p = 0.8, list = F)
tr2 = d2[inTrain2,]
te2 = d2[-inTrain2,]

##########################
#using non-cleaned data to understand the impact of a normal GBM and random forest model on the data set

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)

#m1 is on the numeric only data, using XGBoost model
m1 = train(y ~ ., data = tr1, method = "xgbTree", trControl = ctrl, metric = "RMSE", tuneLength = 5)
#m2 is on the manipulated data using gradient boosting
m2 = train(y ~ ., data = tr2, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)
#m3 is considered our baseline model, is gradient booting on only numeric data
m3 = train(y ~ ., data = tr1, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)
#m4 is on the manipulated data, using XGBoost
m4 = train(y ~ ., data = tr2, method = "xgbTree", trControl = ctrl, metric = "RMSE", tuneLength = 5)

############

#m1 train
defaultSummary(data=data.frame(obs=tr1$y, pred=predict(m1, newdata=tr1))
               , model=m1)
#         RMSE     Rsquared          MAE 
# 1.192280e+04 9.743355e-01 9.184983e+03

#m1 test
defaultSummary(data=data.frame(obs=te1$y, pred=predict(m1, newdata=te1))
               , model=m1)
#         RMSE     Rsquared          MAE 
# 2.656784e+04 8.860896e-01 1.790126e+04

############

#m2 train
defaultSummary(data=data.frame(obs=tr2$y, pred=predict(m2, newdata=tr2))
               , model=m2)
#         RMSE     Rsquared          MAE 
# 1.815497e+04 9.386765e-01 1.229562e+04

#m2 test
defaultSummary(data=data.frame(obs=te2$y, pred=predict(m2, newdata=te2))
               , model=m2)
#         RMSE     Rsquared          MAE 
# 3.070959e+04 8.794477e-01 1.612029e+04

############

#m3 train
defaultSummary(data=data.frame(obs=tr1$y, pred=predict(m3, newdata=tr1))
               , model=m3)
#         RMSE     Rsquared          MAE 
# 2.104849e+04 9.200352e-01 1.353379e+04

#m3 test
defaultSummary(data=data.frame(obs=te1$y, pred=predict(m3, newdata=te1))
               , model=m3)
#         RMSE     Rsquared          MAE 
# 2.492193e+04 9.023918e-01 1.674809e+04

############

#m4 train
defaultSummary(data=data.frame(obs=tr2$y, pred=predict(m4, newdata=tr2))
               , model=m4)
#         RMSE     Rsquared          MAE 
# 8354.6695107    0.9870698 6297.8474663

#m4 test
defaultSummary(data=data.frame(obs=te2$y, pred=predict(m4, newdata=te2))
               , model=m4)
#         RMSE     Rsquared          MAE 
# 34050.534997     0.844401 17798.775598

############

#m3 is the best model (given the small difference between the train and test set), however the second
# best is m2, which is the GBM on the manipulated data

#prior to running model, we'll reclean the data to ensure that the column names match between the d2 set
# and test_2 set. Without doing this, we'll be unable to generate a model that can deliver predictions

test_id = test_2[,1] #removing the id from the list
vars = intersect(names(d2), names(test_2))
test_2 = test_2[, common_columns]
test_2 = cbind(test_id, test_2)
names(test_2)[1] = "Id"

y = d2[,1] #removing the y variable
d2 = d2[, common_columns]
d2 = cbind(y, d2)
names(d2)[1] = "y"

m_gbm = train(y ~ ., data = d2, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)
summary(m_gbm)

defaultSummary(data=data.frame(obs=d2$y, pred=predict(m_gbm, newdata=d2))
               , model=m_gbm)

preds = predict(m_gbm, newdata = test_2)


results = data.frame(Id = test_2$Id, SalePrice = preds)
names(results) = c("Id","SalePrice")

getwd()
write.table(results, file="week_3_team_10_model2.csv", quote = F, row.names = F, sep = ",")

#### testing d3 with manipulated d1 data

set.seed(1245)

d3 = d1
test_3 = test_1

preProcValues <- preProcess(d3[2:ncol(d3)], method = c("center","scale"))
d3 <- predict(preProcValues, d3)

#for "test_3" data, except for the ID
preProcValues <- preProcess(test_3[2:ncol(test_3)], method = c("center","scale"))
test_3 <- predict(preProcValues, test_3)

#removing highly correlated values from d2 set
y= d3$y

descCor = cor(d3[,2:ncol(d3)])
highlyCorDescr = findCorrelation(descCor, cutoff = 0.8)
filteredDesc = d3[,2:ncol(d3)][,-highlyCorDescr]
descCor2 = cor(filteredDesc)
summary(descCor2[upper.tri(descCor2)])
names(d3)[1] = "y"
d3 = cbind(y, filteredDesc)
names(d3)[1] = "y"

#reducing collinearity
y= d2$y
d3 = cbind(rep(1, nrow(d3)), d3[2:ncol(d3)])
names(d3)[1] = "ones"
comboInfo = findLinearCombos(d3)
d3 = d3[, - comboInfo$remove]
d3 = d3[,c(2:ncol(d3))]
d3 = cbind(y , d3)

inTrain3 = createDataPartition(y = d3$y, p = 0.8, list = F)
tr3 = d3[inTrain3,]
te3 = d3[-inTrain3,]

m5 = train(y ~ ., data = tr3, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

defaultSummary(data=data.frame(obs=tr3$y, pred=predict(m5, newdata=tr3))
               , model=m5)
#         RMSE     Rsquared          MAE 
# 2.532110e+04 8.899584e-01 1.681306e+04 

defaultSummary(data=data.frame(obs=te3$y, pred=predict(m5, newdata=te3))
               , model=m5)
#         RMSE     Rsquared          MAE 
# 2.151236e+04 9.121222e-01 1.609862e+04