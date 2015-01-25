#Script del proyecto
library(caret)

# Loading and preprocessing the data

url1<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url1,"projtraining.csv") 
training<-read.csv("projtraining.csv", na.strings= c('NA','','#DIV/0!'))

url2<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url2,"projtesting.csv") 
testing<-read.csv("projtesting.csv",na.strings= c('NA','','#DIV/0!'))


# Data exploration
## 

# remove
# as.numeric(as.character(x))
sapply(training[1,],class)
# columnasfactor<-sapply(training[1,],is.factor)


# Lets see which columsn are near zero value, so we can eliminate those from the analysis
nsv<-nearZeroVar(training,saveMetrics=TRUE)
sum(nsv$nzv)
removezv<-which(nsv$nzv)

# The following columns are not considered as predictors since they are basically stamps and labels
stamps<-c(1,3,4,5)

# Which columns are mostly NA and do not produce much signal 
NAindex<-((sapply(1:ncol(training), function(x) {(sum(is.na(training[,x]))>950)})))
NAindex<-which(NAindex)

indexwNA<-union(removezv,stamps)
indexwoNA<-union(indexwNA,NAindex)

# View(training)
# 
# 
# View(training[,nsv$nzv])

# Standardize and Impute


set.seed(32323)
## Creating k folds, for cross validation, k = 10
foldstrain<-createFolds(y=training$classe,k=10, list=TRUE, returnTrain = TRUE)

sapply(foldstrain,length)

# To see if NA columns add any predicting value, we will start with
# We will do an initial approach with Classification and Regression trees model (rpart),
# In order to select the best configuration with a fast method, later on we can
# switch to Random Forest, once we have selected our final variables.
# using NA columns, with no preprocessing,
trainingbase<-training[foldstrain[[1]],]
testingbase<-training[-foldstrain[[1]],]
modFitBase<-train(classe ~ ., method='rpart', data=trainingbase[,-indexwNA])
print(modFitBase)

## We can see that the accuracy is very low. We should eliminate mostly NA variables to 
## to improve accuracy.

# predict(modFitBase, newdata= testingbase[,-indexwNA])
# 
# confusionMatrix(testingbase$classe,predict(modFitBase,testingbase[,-indexwNA]))


# Check and see if using NA columns reports any advantage
trainingnoNA<-training[foldstrain[[2]],]
testingnoNA<-training[-foldstrain[[2]],]
modFitnoNA<-train(classe ~ ., method='rpart', data=trainingnoNA[,-indexwoNA])
print(modFitnoNA)
predict(modFitnoNA, newdata= testingnoNA[,-indexwoNA])

confusionMatrix(testingnoNA$classe,predict(modFitnoNA,testingnoNA[,-indexwoNA]))

##Accuracy is up to 59% in sample, and 57% out of sample, but still pretty low. Lets try 
## centering and scaling the remaining variables.

traininscale<-training[foldstrain[[3]],]
testingscale<-training[-foldstrain[[3]],]
modFitscale<-train(classe ~ ., method='rpart', data=traininscale[,-indexwoNA],
                   preProcess=c('center','scale'))
print(modFitscale)
predict(modFitscale, newdata= testingscale[,-indexwoNA])

confusionMatrix(testingscale$classe,predict(modFitscale, newdata= testingscale[,-indexwoNA]))

# Check and see if PCA is better than scaling
trainingPCA<-training[foldstrain[[4]],]
testingsPCA<-training[-foldstrain[[4]],]
modFitPCA<-train(classe ~ ., method='rpart', data=trainingPCA[,-indexwoNA],
                   preProcess='pca')
print(modFitPCA)
predict(modFitPCA, newdata= testingsPCA[,-indexwoNA])

confusionMatrix(testingsPCA$classe,predict(modFitPCA, newdata= testingsPCA[,-indexwoNA]))

## We will stick to training with 55 variable, no scaling, centering or PCA.
## Lets see if a method such as Random Forests improves accuracy
trainingRF<-training[foldstrain[[5]],]
testingRF<-training[-foldstrain[[5]],]
modFitRF<-train(classe ~ ., method='rf', data=trainingRF[,-indexwoNA])
print(modFitRF)
predict(modFitRF, newdata= testingRF[,-indexwoNA])

##Estimation of out of sample error for this model can be obtained with the confusion Matrix
confusionMatrix(testingRF$classe,predict(modFitRF, newdata= testingRF[,-indexwoNA]))


##Finally, the predictions for the 20 cases to test are as follows
predictions<-predict(modFitRF, newdata= testing[,-indexwoNA])
confusionMatrix(testing$classe,predict(modFitRF, newdata= testing[,-indexwoNA]))


# Obtain average out of sample error with repeated cross validation (estimation of error)

## 

##Al parecer solo le das foldstrain como data y ya te aplica el modelo y te da los accuracies
# de todos los folds 