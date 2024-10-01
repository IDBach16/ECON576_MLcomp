# ECON 576 Machine Learning Competition Code, for Ian Bach and Matt Fiorini
library(tidyverse)
library(mice)
library(caret)
library(gbm)
library(xgboost)

#train = read.csv("https://raw.githubusercontent.com/MattFiorini/ECON576_MLcomp/main/data/train.csv", sep = ",")
#test = read.csv("https://raw.githubusercontent.com/MattFiorini/ECON576_MLcomp/main/data/test.csv", sep = ",")

train = read.csv("https://raw.githubusercontent.com/MattFiorini/Kaggle_MLHousingPrices/refs/heads/main/train.csv", sep = ",")
test = read.csv("https://raw.githubusercontent.com/MattFiorini/Kaggle_MLHousingPrices/refs/heads/main/test.csv", sep = ",")

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
d3 <- d2[, selected_features]
test_3 <- test_2[, selected_features]
d3 = cbind(y, d3)
test_3 = cbind(test_id, test_3)
names(test_3)[1] = "Id"


d4 = d1
test_4 = test_1

preProcValues <- preProcess(d4[2:ncol(d4)], method = c("center","scale"))
d4 <- predict(preProcValues, d4)

#for "test_3" data, except for the ID
preProcValues <- preProcess(test_4[2:ncol(test_4)], method = c("center","scale"))
test_4 <- predict(preProcValues, test_4)

#removing highly correlated values from d2 set
y= d4$y

descCor = cor(d4[,2:ncol(d4)])
highlyCorDescr = findCorrelation(descCor, cutoff = 0.8)
filteredDesc = d4[,2:ncol(d4)][,-highlyCorDescr]
descCor2 = cor(filteredDesc)
summary(descCor2[upper.tri(descCor2)])
names(d4)[1] = "y"
d4 = cbind(y, filteredDesc)
names(d4)[1] = "y"

sizes = c(1:10, 20, 30)
rfe_results <- rfe(x = d4[,2:33], y = d4$y, sizes = sizes, rfeControl = control)
plot(rfe_results, type = c("g", "o"))  # Plots performance across different subset sizes
#plot indicating that the lowest RMSE is at the 30 variable mark
selected_features <- predictors(rfe_results)
test_id = test_4[,1]
y = d4[,1]
d4 <- d4[, selected_features]
test_4 <- test_4[, selected_features[-1]]
d4 = cbind(y, d4)
test_4 = cbind(test_id, test_4)
#data partition creation for each set

set.seed(99999)
inTrain1 = createDataPartition(y = d1$y, p = 0.8, list = F)
tr1 = d1[inTrain1,]
te1 = d1[-inTrain1,]

inTrain2 = createDataPartition(y = d2$y, p = 0.8, list = F)
tr2 = d2[inTrain2,]
te2 = d2[-inTrain2,]

inTrain3 = createDataPartition(y = d3$y, p = 0.8, list = F)
tr3 = d3[inTrain3,]
te3 = d3[-inTrain3,]

inTrain4 = createDataPartition(y = d4$y, p = 0.8, list = F)
tr4 = d4[inTrain4,]
te4 = d4[-inTrain4,]
##########################
#using non-cleaned data to understand the impact of a normal GBM and random forest model on the data set

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)


m1 = train(y ~ ., data = tr1, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

m2 = train(y ~ ., data = tr2, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

m3 = train(y ~ ., data = tr3, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

m4 = train(y ~ ., data = tr4, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)

m1 = train(y ~ ., data = d1, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)
defaultSummary(data=data.frame(obs=d1$y, pred=predict(m1, newdata=d1))
               , model=m1)

############

#m1 train
defaultSummary(data=data.frame(obs=tr1$y, pred=predict(m1, newdata=tr1))
               , model=m1)


#m1 test
defaultSummary(data=data.frame(obs=te1$y, pred=predict(m1, newdata=te1))
               , model=m1)


############

#m2 train
defaultSummary(data=data.frame(obs=tr2$y, pred=predict(m2, newdata=tr2))
               , model=m2)
 

#m2 test
defaultSummary(data=data.frame(obs=te2$y, pred=predict(m2, newdata=te2))
               , model=m2)


############

#m3 train
defaultSummary(data=data.frame(obs=tr3$y, pred=predict(m3, newdata=tr3))
               , model=m3)


#m3 test
defaultSummary(data=data.frame(obs=te3$y, pred=predict(m3, newdata=te3))
               , model=m3)

############

#m4 train
defaultSummary(data=data.frame(obs=tr4$y, pred=predict(m4, newdata=tr4))
               , model=m4)
#m4 test
defaultSummary(data=data.frame(obs=te4$y, pred=predict(m4, newdata=te4))
               , model=m4)

############

library(h2o)
h2o.init(nthreads = 12, max_mem_size = "64g")

data1 = as.h2o(d1)
data2 = as.h2o(d2)
data3 = as.h2o(d3)
data4 = as.h2o(d4)

y = "y"
x1 = setdiff(names(data1),y)
parts1 = h2o.splitFrame(data1, 0.8, seed = 99)
train1 = parts1[[1]]
test1 = parts1[[2]]

x2 = setdiff(names(data2),y)
parts2 = h2o.splitFrame(data2, 0.8, seed = 99)
train2 = parts2[[1]]
test2 = parts2[[2]]

x3 = setdiff(names(data3),y)
parts3 = h2o.splitFrame(data3, 0.8, seed = 99)
train3 = parts3[[1]]
test3 = parts3[[2]]

x4 = setdiff(names(data4),y)
parts4 = h2o.splitFrame(data4, 0.8, seed = 99)
train4 = parts4[[1]]
test4 = parts4[[2]]

auto1 <- h2o.automl(x1, y, train1, max_runtime_secs=100)
#                                                 model_id     rmse       mse      mae     rmsle
#1  StackedEnsemble_BestOfFamily_5_AutoML_1_20240930_223246 27367.63 748987194 16858.37 0.1377021
#auto2 <- h2o.automl(x2, y, train2, max_runtime_secs=100)
auto3 <- h2o.automl(x3, y, train3, max_runtime_secs=100)
#                                                  model_id     rmse       mse      mae     rmsle
#1             GBM_grid_1_AutoML_2_20240930_224052_model_16 29287.64 857765808 17949.71 0.1437472
auto4 <- h2o.automl(x4, y, train4, max_runtime_secs=100)
#model_id     rmse       mse      mae     rmsle
#1  StackedEnsemble_BestOfFamily_4_AutoML_3_20240930_224812 29363.23 862199067 17425.59 0.1404390

auto_summary_train = h2o.performance(model = auto1, newdata = train1)
auto_summary_test = h2o.performance(model = auto1, newdata = test1)

test1 = as.h2o(test_1)
test3 = as.h2o(test_3)
test4 = as.h2o(test_4)

p1 <- h2o.predict(auto1, test1)
p1 <- as.data.frame(p1)
head(p1)
names(p1) <- "predict"

h2o_results1 <- data.frame(Id=test_1$Id, SalePrice=p1$predict)

p3 <- h2o.predict(auto3, test3)
p3 <- as.data.frame(p3)
head(p3)
names(p3) <- "predict"

h2o_results3 <- data.frame(Id=test_3$Id, SalePrice=p3$predict)

p4 <- h2o.predict(auto4, test4)
p4 <- as.data.frame(p4)
head(p4)
names(p4) <- "predict"

h2o_results4 <- data.frame(Id=test_4$Id, SalePrice=p4$predict)

write.table(x=h2o_results1, sep=",", file="h2o_results1.csv", row.names=F)
#0.14353 on Kaggle
write.table(x=h2o_results3, sep=",", file="h2o_results3.csv", row.names=F)
#0.15098 on Kaggle
write.table(x=h2o_results4, sep=",", file="h2o_results4.csv", row.names=F)
#error in creation of h2o 4

h2o.shutdown()
