---
title: "Practical Machine Learning Course Project"
output: html_document
---

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Data

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

## Load Libraries
Firstly, we must load the appropriate libraries for the analysis.

```r
library(dplyr)
library(caret)
library(randomForest)
```

## Get and Load Data Into R
In this step, we will download the files directly from the source URL. 

```r
training.url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
testing.url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
download.file(training.url, destfile = './training_data.csv', method = 'curl')
download.file(testing.url, destfile = './testing_data.csv', method = 'curl')
training <- read.csv('training_data.csv', header = TRUE)
test.set <- read.csv('testing_data.csv', header = TRUE) # used for assignment
```

## Partition Data
We must split the data set into a training and testing partition for cross validation of our model. It is important to note that after performing this partition, nothing should be done to testing partition until the model is ready to make predictions. In this example, we split the data set into 75% training and 25% testing.

```r
set.seed(1234)
inTrain <- createDataPartition(training$classe, p = .75, list = FALSE)
training <- training[inTrain, ]
testing <- training[-inTrain, ]
```

## Pre-Procession
There are about 160 variables in this data set. In order to reduce the dimensions of our model, we should try remove variables that show little variance and variables that have mostly missing values.

Let's remove variables that show little variance with the 'nearZeroVar' function.

```r
n.zero.var <- nearZeroVar(training, saveMetrics = TRUE)
nzv.col.index <- n.zero.var$nzv

training <- training[, !nzv.col.index] 
```

Now let's remove the first 5 columns because they do not provide any information our model can use. 

```r
training <- training[, -(1:5)]
```

Lastly, let's take a look at each remaining variables to see the density of NA values. If the NA density is above 85% per variable, it will be removed.

```r
col.na <- sapply(training, function(x) {sum(is.na(x))})
training <- training[, which(col.na < nrow(training)*.85)]
```

We have reduced the number of variables to 54 from 160.

## Model Creation and Cross Validation
I will start with trying to fit a random forest model to the training set. Originally, I tried using the caret package but the algorithm took too long to run. After switching to the randomForest package, the model took less than 1 minute. 

```r
set.seed(321)
model.rf1 <- randomForest(classe ~ ., data = training)
rf.accuracy2 <- predict(model.rf1, testing)
cm <- confusionMatrix(rf.accuracy2, testing$classe)
cm$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      1.0000000      1.0000000      0.9990054      1.0000000      0.2821689 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

```r
cm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1046    0    0    0    0
##          B    0  711    0    0    0
##          C    0    0  642    0    0
##          D    0    0    0  615    0
##          E    0    0    0    0  693
```
The results indicate that this model very accurately predicts the correct class. The 95% confidence interval for accuracy is about 100%. Later, we will test this model against the test set for the assignment to further validate the accuracy of the model.

## Out of Sample Error and Expectations
This model is very accurate at predicting classe for this data set (nearly 100% accurate). I would expect the out of sample error for other similar data sets to be below this accuracy reported in the confusion matrix. I would closely monitor the accuracy on another data set and if accuracy is greately reduced, I would explore methods to reduce overfitting. 

## Test Predictions For Assignment
After submitting my predictions for the assignment, the model returned 100% correct. This model seems to work well. 


```r
test.set.answers <- predict(model.rf1, test.set)
test.set.answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

