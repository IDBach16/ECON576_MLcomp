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

train = train %>% select(-Utilities, -Street)
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

integer_columns <- sapply(train, function(col) is.integer(col))
(tr_integer_columns <- names(train)[integer_columns])
tr_integer_columns = append(tr_integer_columns, names(train)[3])

te_integer_columns <- sapply(test, function(col) is.integer(col))
(te_integer_columns <- names(test)[te_integer_columns])
te_integer_columns = append(te_integer_columns, names(test)[4])

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

d = d[,c(79,1:78)]

names(d)[1] <- "y"
names(d)

rm(imputedValuesTe, imputedValues)

library(caret)
set.seed(1234)

inTrain = createDataPartition(y = d$y, p = 0.8, list = F)
tr = d[inTrain,]
te = d[-inTrain,]

tr = tr %>% select(-Id)

#removing y from test set to confirm the results and see the difference in the end R^2
test_y = te$y
te = te[,2:79]


##########################
#using non-cleaned data to understand the impact of a normal GBM and random forest model on the data set

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)

m1 = train(y ~ ., data = tr, method = "gbm", trControl = ctrl, metric = "RMSE", verbose = F)
m2 = train(y ~ ., data = tr, method = "rf", trControl = ctrl, nodesize = 5, ntree = 250, importance = TRUE, metric = "RMSE")

preds1 = predict(m1, newdata = te)
preds2 = predict(m2, newdata = te)

model_comp = data.frame(test_y, preds1, preds2)

gbm_nontransformed = lm(preds1~test_y, data = model_comp)
rf_nontransformed = lm(preds2~test_y, data = model_comp)

summary(gbm_nontransformed)
summary(rf_nontransformed)