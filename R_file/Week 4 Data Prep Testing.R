library(tidyverse)
library(mice)
library(caret)
library(gbm)
library(xgboost)
library(corrplot)

train = read.csv("https://raw.githubusercontent.com/MattFiorini/ECON576_MLcomp/main/data/train.csv", sep = ",")
test = read.csv("https://raw.githubusercontent.com/MattFiorini/ECON576_MLcomp/main/data/test.csv", sep = ",")

#train = read.csv("https://raw.githubusercontent.com/MattFiorini/Kaggle_MLHousingPrices/refs/heads/main/train.csv", sep = ",")
#test = read.csv("https://raw.githubusercontent.com/MattFiorini/Kaggle_MLHousingPrices/refs/heads/main/test.csv", sep = ",")

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

for (col in names(d1)) {
    # Create a histogram for the numeric column
    hist(d1[[col]], 
         main = paste("Histogram of", col),  # Title with column name
         xlab = col,                         # X-axis label with column name
         col = "lightblue",                  # Color of the bars
         border = "black")                   # Border color of the bars
    
    # Pause to allow viewing each plot before moving to the next
    readline(prompt = "Press [Enter] to see the next histogram...")
  }
}
#log to frontage, lotarea, yearbuilt, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, X1stFlrSF,
#X2ndFlrSF, LowQualFinSF, TotRmsAbvGrd, GarageYrBlt
str(d1)

log_update = c("LotFrontage","LotArea","YearBuilt", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "X1stFlrSF",
"X2ndFlrSF", "LowQualFinSF", "TotRmsAbvGrd", "GarageYrBlt")
d1[log_update] = lapply(d1[log_update], log)
train_1[log_update] = lapply(train_1[log_update], log)

# Create a correlation matrix for your dataframe 'd1'
correlation_matrix = cor(d1, use = "complete.obs")  # Excludes rows with missing values (if any)
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

set.seed(1997)
inTrain1 = createDataPartition(y = d1$y, p = 0.8, list = F)
tr1 = d1[inTrain1,]
te1 = d1[-inTrain1,]

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)


m1 = train(y ~ ., data = tr1, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

#m1 train
defaultSummary(data=data.frame(obs=tr1$y, pred=predict(m1, newdata=tr1))
               , model=m1)
#        RMSE     Rsquared          MAE 
#1.825321e+04 9.393083e-01 1.298512e+04

#m1 test
defaultSummary(data=data.frame(obs=te1$y, pred=predict(m1, newdata=te1))
               , model=m1)
#         RMSE     Rsquared          MAE 
# 3.843562e+04 7.770289e-01 1.868833e+04

#No log transformation applied
test_2 = test %>% select(-all_of(s))
d2 = train_1
names(d2)

d2 = d2[,c(36,1:35)]

names(d2)[1] <- "y"
names(d2)

correlation_matrix2 = cor(d2, use = "complete.obs")  # Excludes rows with missing values (if any)
corrplot(correlation_matrix2, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

set.seed(19990)
inTrain2 = createDataPartition(y = d2$y, p = 0.8, list = F)
tr2 = d2[inTrain2,]
te2 = d2[-inTrain2,]

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)

m2 = train(y ~ ., data = tr2, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

#m2 train
defaultSummary(data=data.frame(obs=tr2$y, pred=predict(m2, newdata=tr2))
               , model=m2)
# RMSE     Rsquared          MAE 
# 2.052585e+04 9.195765e-01 1.299551e+04

#m2 test
defaultSummary(data=data.frame(obs=te2$y, pred=predict(m2, newdata=te2))
               , model=m2)
#         RMSE     Rsquared          MAE 
#  2.776915e+04 9.036266e-01 1.756497e+04

##############
# log tranformation to response variable to better understand
d1$y = log(d1$y)

set.seed(2007)
inTrain3 = createDataPartition(y = d1$y, p = 0.8, list = F)
tr3 = d1[inTrain1,]
te3 = d1[-inTrain1,]

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)


m3 = train(y ~ ., data = tr3, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)


#m3 train
defaultSummary(data=data.frame(obs=tr3$y, pred=predict(m3, newdata=tr3))
               , model=m3)
#       RMSE   Rsquared        MAE 
# 0.09947962 0.93211462 0.07317921

#m3 test
defaultSummary(data=data.frame(obs=te3$y, pred=predict(m3, newdata=te3))
               , model=m3)
#       RMSE   Rsquared        MAE 
# 0.15370187 0.84726167 0.09423684

##########
#trying full dataset, onehot encode and keeping 0.5 cor or better values as d4
train_4 = train
test_4 = test

d4 = train_4
names(d4)

d4 = d4[,c(78,1:77)]

names(d4)[1] <- "y"
names(d4)

dummiesD = dummyVars(y ~ ., data=d4)
ex = data.frame(predict(dummiesD, newdata = d4))
names(ex) = gsub("\\.","",names(ex))
d4 = cbind(d4$y, ex)
names(d4)[1] = "y"

#for test set
dummiesTe = dummyVars(Id ~ ., data=test_4)
ex1 = data.frame(predict(dummiesTe, newdata = test_4))
names(ex1) = gsub("\\.","",names(ex1))
test_4 = cbind(test_4$Id, ex1)
rm(dummiesD, ex, dummiesTe, ex1)
names(test_4)[1] = "Id"

correlations = as.data.frame(cor(d4, d4$y, use = "complete.obs"))
correlations = correlations %>% filter(abs(V1)>0.3)
common_names = rownames(correlations)[-1]

y = d4$y
testID = test_4$Id

d4 = d4[, common_names]
test_4 = test_4[, common_names]
d4 = cbind(y, d4)
test_4 = cbind(testID, test_4)
names(test_4)[1] = "Id"
names(d4)

set.seed(56789)
inTrain4 = createDataPartition(y = d4$y, p = 0.8, list = F)
tr4 = d4[inTrain4,]
te4 = d4[-inTrain4,]
m4 = train(y ~ ., data = tr4, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)
#m4 train
defaultSummary(data=data.frame(obs=tr4$y, pred=predict(m4, newdata=tr4))
               , model=m4)
#         RMSE     Rsquared          MAE 
# 2.251793e+04 9.124153e-01 1.485138e+04
#m4 test
defaultSummary(data=data.frame(obs=te4$y, pred=predict(m4, newdata=te4))
               , model=m4)
#         RMSE     Rsquared          MAE 
# 2.480856e+04 8.827358e-01 1.729044e+04

##########
#Numeric Values where only 0.3 greater cor is kept for numeric values?
d5 = d2
test_5 = test_2

correlations = as.data.frame(cor(d5, d5$y, use = "complete.obs"))
correlations = correlations %>% filter(abs(V1)>0.3)
common_names2 = rownames(correlations)[-1]

d5 = d5[, common_names2]
test_5 = test_5[, common_names2]
d5 = cbind(y, d5)
test_5 = cbind(testID, test_5)
names(test_5)[1] = "Id"
names(d5)

set.seed(221133)
inTrain5 = createDataPartition(y = d5$y, p = 0.8, list = F)
tr5 = d5[inTrain5,]
te5 = d5[-inTrain5,]
m5 = train(y ~ ., data = tr5, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)
#m5 train
defaultSummary(data=data.frame(obs=tr5$y, pred=predict(m5, newdata=tr5))
               , model=m5)
#         RMSE     Rsquared          MAE 
# 2.637937e+04 8.828719e-01 1.742606e+04 

#m5 test
defaultSummary(data=data.frame(obs=te5$y, pred=predict(m5, newdata=te5))
               , model=m5)
#         RMSE     Rsquared          MAE 
# 2.443833e+04 8.680925e-01 1.779878e+04

##########
#Quartile Analysis
Upper = train %>% filter(SalePrice > 214000)
Lower = train %>% filter(SalePrice < 129900)

variables = colnames(Upper)

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

set.seed(221133)
inTrain6 = createDataPartition(y = d6$y, p = 0.8, list = F)
tr6 = d6[inTrain6,]
te6 = d6[-inTrain6,]

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
