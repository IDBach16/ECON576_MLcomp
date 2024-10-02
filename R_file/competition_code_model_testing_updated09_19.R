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

#running this code to standardize the numeric values to only display within a 0 to 1 range
tr_numeric_cols <- intersect(names(d), tr_integer_columns)

#for "d" data
preProcValues <- preProcess(d[,tr_numeric_cols]
                            , method = c("center","scale"))
d <- predict(preProcValues, d)

#for "test" data
preProcValues <- preProcess(test[,tr_numeric_cols]
                            , method = c("center","scale"))
te <- predict(preProcValues, test)


set.seed(1234)
#splitting data_set into a 80/20 split
inTrain = createDataPartition(y = d$y, p = 0.8, list = F)
tr = d[inTrain,]
te = d[-inTrain,]

#removing y from test set to confirm the results and see the difference in the end R^2
test_y = te$y
te = te[,2:89]

ctrl = trainControl(method="cv", number = 10, classProbs = F, 
                    summaryFunction = defaultSummary, allowParallel = T)

#standard linear model that we had prior
m1 = train(y~.,data=tr, method = "lm", trControl = ctrl, metric = "Rsquared")

defaultSummary(data=data.frame(obs=tr$y, pred=predict(m1, newdata=tr))
                         , model=m1)
#basic GBM model
m2 = train(y ~ ., data = tr, method = "gbm", trControl = ctrl, metric = "Rsquared", verbose = F)

defaultSummary(data=data.frame(obs=tr$y, pred=predict(m1, newdata=tr))
               , model=m2)

#running predictions on the models to on the remaining 20% of data
preds1 = predict(m1, newdata = te)
preds2 = predict(m2, newdata = te)

#comparing the models and finding the R^2
model_comp = data.frame(preds1, preds2, test_y)
linear_rsquared = lm(preds1~test_y, data = model_comp)
gbm_rsquared = lm(preds2~test_y, data = model_comp)

summary(linear_rsquared)
summary(gbm_rsquared)
##########################

#### H2O CODE ######

library(h2o)
h2o.init(nthreads = 12, max_mem_size = "64g")

#importing in data to h2o, keeping same mock testing/training data set to ensure consistent comp with non h2o models
data = as.h2o(d)
tr_h2o = as.h2o(tr)
te_h2o = as.h2o(te)

y = "y"
x = setdiff(names(data),y)

#h2o base GBM model
m3 = h2o.gbm(y = y, training_frame = tr_h2o, seed = 99)
summary(m3)
preds3 = h2o.predict(m3,te_h2o)
preds3 = as.numeric(preds3)
model_comp$preds3 = preds3

#h2o automl function, will give the best results based on the cleaning/prepping that we've done
m_base = h2o.automl(x, y, tr_h2o, max_runtime_secs=300)
summary(m_base)
preds_best = h2o.predict(m_base, te_h2o)
preds_best = as.numeric(preds_best)
model_comp$preds_best = preds_best

#to stop h2o connection and free up computing space
h2o.shutdown()

head(model_comp)
model_comp[,c(5,1:2,4,3)]


linear_rsquared = lm(preds1~test_y, data = model_comp)
gbm_rsquared = lm(preds2~test_y, data = model_comp)
gbm_h2o = lm(preds3~test_y, data = model_comp)
best_model = lm(preds_best~test_y, data = model_comp)

summary(linear_rsquared)
#generated a 0.89 R Squared
summary(gbm_rsquared)
#generated a 0.87 R Squared
summary(gbm_h2o)
#generated a 0.85 R Squared
summary(best_model)
#generated a 0.90 R Squared

results = data.frame(Id = test$Id, SalePrice = preds)

names(results) = c("Id","SalePrice")

write.table(results, file="results.csv", quote = F, row.names = F, sep = ",")
