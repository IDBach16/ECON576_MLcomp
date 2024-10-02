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

y= d2$y
d2 = cbind(rep(1, nrow(d2)), d2[2:ncol(d2)])
names(d2)[1] = "ones"
comboInfo = findLinearCombos(d2)
d2 = d2[, - comboInfo$remove]
d2 = d2[,c(2:ncol(d2))]
d2 = cbind(y , d2)

control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)

# Try a range of feature subset sizes, from 1 to 20 (or adjust based on your data)
sizes = c(20, 30, 40, 50, 60)
rfe_results <- rfe(x = d2[,2:ncol(d2)], y = d2$y, sizes = sizes, rfeControl = control)
plot(rfe_results, type = c("g", "o"))  # Plots performance across different subset sizes
#plot indicating that the lowest RMSE is at the 30 variable mark
selected_features <- predictors(rfe_results)
test_id = test_2[,1]
y = d2[,1]
d2 <- d2[, selected_features]
test_2 <- test_2[, selected_features]
d2 = cbind(y, d2)
test_2 = cbind(test_id, test_2)
names(test_2)[1] = "Id"

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
# 1.765363e+04 9.453311e-01 1.316698e+04

#m1 test
defaultSummary(data=data.frame(obs=te1$y, pred=predict(m1, newdata=te1))
               , model=m1)
#         RMSE     Rsquared          MAE 
# 4.597649e+04 7.213963e-01 2.171005e+04

############

#m2 train
defaultSummary(data=data.frame(obs=tr2$y, pred=predict(m2, newdata=tr2))
               , model=m2)
#         RMSE     Rsquared          MAE 
# 22224.340072     0.914372 14431.499515 

#m2 test
defaultSummary(data=data.frame(obs=te2$y, pred=predict(m2, newdata=te2))
               , model=m2)
#         RMSE     Rsquared          MAE 
# 2.401037e+04 8.922891e-01 1.684374e+04

############

#m3 train
defaultSummary(data=data.frame(obs=tr1$y, pred=predict(m3, newdata=tr1))
               , model=m3)
#         RMSE     Rsquared          MAE 
# 1.708218e+04 9.492032e-01 1.206797e+04

#m3 test
defaultSummary(data=data.frame(obs=te1$y, pred=predict(m3, newdata=te1))
               , model=m3)
#         RMSE     Rsquared          MAE 
# 3.891188e+04 7.675018e-01 1.900755e+04

############

#m4 train
defaultSummary(data=data.frame(obs=tr2$y, pred=predict(m4, newdata=tr2))
               , model=m4)
#         RMSE     Rsquared          MAE 
# 9133.5080098    0.9855313 6998.4248932

#m4 test
defaultSummary(data=data.frame(obs=te2$y, pred=predict(m4, newdata=te2))
               , model=m4)
#         RMSE     Rsquared          MAE 
# 2.407908e+04 8.940119e-01 1.651425e+04

############

#m4 performed incredibly well on the training set, and not as well on the training, m2 offered the most consistent performance
# between the 3 models

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

sizes = c(1:10, 20, 30)
rfe_results <- rfe(x = d3[,2:34], y = d3$y, sizes = sizes, rfeControl = control)
plot(rfe_results, type = c("g", "o"))  # Plots performance across different subset sizes
#plot indicating that the lowest RMSE is at the 30 variable mark
selected_features <- predictors(rfe_results)
test_id = test_3[,1]
y = d3[,1]
d3 <- d3[, selected_features]
test_3 <- test_3[, selected_features[-1]]
d3 = cbind(y, d3)
test_3 = cbind(test_id, test_3)

inTrain3 = createDataPartition(y = d3$y, p = 0.8, list = F)
tr3 = d3[inTrain3,]
te3 = d3[-inTrain3,]

m5 = train(y ~ ., data = tr3, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

defaultSummary(data=data.frame(obs=tr3$y, pred=predict(m5, newdata=tr3))
               , model=m5)
#         RMSE     Rsquared          MAE 
# 1.964605e+04 9.280733e-01 1.305043e+04 

defaultSummary(data=data.frame(obs=te3$y, pred=predict(m5, newdata=te3))
               , model=m5)
#         RMSE     Rsquared          MAE 
# 3.131331e+04 8.639456e-01 1.748170e+04