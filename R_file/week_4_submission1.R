library(tidyverse)
library(mice)
library(caret)
library(gbm)
library(xgboost)
library(lightgbm)
library(corrplot)

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

Upper = train %>% filter(SalePrice > 214000)
Lower = train %>% filter(SalePrice < 129900)

variables = colnames(Upper)
#understanding the difference between the distribtuion of vairables in the lower and upper quartiles of sale price
for (var in variables) {
  cat("\n--------------------------\n")
  cat("Summary for variable:", var, "\n")
  
  cat("\nLower Quartile Summary:\n")
  print(summary(Lower[[var]]))
  
  cat("\nUpper Quartile Summary:\n")
  print(summary(Upper[[var]]))
  
  cat("--------------------------\n")
}
#basing decision off of if there is a significant difference between values or lower and upper quartiles
VarsKeep = c("MSSubClass","LotFrontage","LotArea","Neighborhood","HouseStyle","OverallQual",
             "YearBuilt","YearRemodAdd","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea",
             "ExterQual","Foundation","BsmtQual","BsmtFinType1","TotalBsmtSF","HeatingQC","X1stFlrSF",
             "X2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","FullBath","BedroomAbvGr","KitchenQual",
             "TotRmsAbvGrd","Fireplaces","GarageType","GarageFinish","GarageCars","GarageArea","SaleType",
             "SaleCondition","SalePrice")
#validating variables selected by observing the distribution of the selected variables in the test set
for (var in VarsKeep) {
  cat("\n--------------------------\n")
  cat("Summary for variable:", var, "\n")
  
  cat("\nLower Quartile Summary:\n")
  print(summary(test[[var]]))
}

d6 = train %>% select(all_of(VarsKeep))
testID = test[,1]
test_6 = test[,(colnames(d6[,1:34]))]
test_6 = cbind(testID,test_6)
names(test_6)[1] = "Id"
d6 = d6[,c(35,1:34)]
names(d6)[1] = "y"


dummiesD = dummyVars(y ~ ., data=d6)
ex = data.frame(predict(dummiesD, newdata = d6))
names(ex) = gsub("\\.","",names(ex))
d6 = cbind(d6$y, ex)
names(d6)[1] = "y"

#for test set
dummiesTe = dummyVars(Id ~ ., data=test_6)
ex1 = data.frame(predict(dummiesTe, newdata = test_6))
names(ex1) = gsub("\\.","",names(ex1))
test_6 = cbind(test_6$Id, ex1)
rm(dummiesD, ex, dummiesTe, ex1)
names(test_6)[1] = "Id"

#confirming what one-hot encoded variables were not present in both the test and train
#not including the y variable in the test set
test_add = setdiff(colnames(d6), colnames(test_6))[2:8]
#not including the ID variable in the train set
train_add = setdiff(colnames(test_6), colnames(d6))[2:4]
intersect(colnames(d6), colnames(test_6))

#adding missing columns and ensuring all the values are 0
d6 = d6 %>%
  mutate(!!!set_names(rep(0, length(train_add)), train_add))
test_6 = test_6 %>%
  mutate(!!!set_names(rep(0, length(test_add)), test_add))

set.seed(221133)
inTrain6 = createDataPartition(y = d6$y, p = 0.8, list = F)
tr6 = d6[inTrain6,]
te6 = d6[-inTrain6,]

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)

m6 = train(y ~ ., data = tr6, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

#m6 train
defaultSummary(data=data.frame(obs=tr6$y, pred=predict(m6, newdata=tr6))
               , model=m6)
#         RMSE     Rsquared          MAE 
# 2.294951e+04 9.113664e-01 1.424164e+04

#m6 test
defaultSummary(data=data.frame(obs=te6$y, pred=predict(m6, newdata=te6))
               , model=m6)
#         RMSE     Rsquared          MAE 
# 2.322942e+04 8.808396e-01 1.712016e+04

#re-running model to include the entire train dataset
m_gbm = train(y ~ ., data = d6, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

defaultSummary(data=data.frame(obs=d6$y, pred=predict(m_gbm, newdata=d6))
               , model=m_gbm)
#        RMSE    Rsquared         MAE 
# 22756.66961     0.90873 14547.90186

names(test_6)[1] = "Id"

preds = predict(m_gbm, newdata = test_6)

results = data.frame(Id = test_6$Id, SalePrice = preds)

names(results) = c("Id","SalePrice")

write.table(results, file="week_4_team10_results.csv", quote = F, row.names = F, sep = ",")
