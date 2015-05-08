library(psych)
library(e1071)
library(caret)
library(kernlab)
library(ROSE)
library(pnn)
library(lattice)
library(stats)
library(fpc)
library(cluster)
library(vegan)
library(mclust)
library(cvTools)
library(rpart)
library(randomForest)
library(ggplot2)
library(GGally)
library(forecast)
library(pso)

RawData <- read.table("E:/machine_learning_data.csv", header = TRUE, sep=",",
                      colClasses= c("numeric","factor","factor","factor", "numeric","numeric","numeric","numeric"))
names(RawData) <- c("day_of_month", "day_of_week","hour_of_day", "grid_index", "num_trips","avg_distance","avg_num_passenger", "std_num_passenger")
hist(RawData$avg_distance)

shortTrip <- RawData[RawData$avg_distance <= 3, ]
longTrip <- RawData[RawData$avg_distance > 3, ]


STData <- RawData
STData[STData$avg_distance <= 3, ]$avg_distance = rep(0, length(STData[STData$avg_distance <= 3, ]$avg_distance))
STData[STData$avg_distance > 3, ]$avg_distance = rep(1, length(STData[STData$avg_distance > 3, ]$avg_distance))

STData$avg_distance <- factor(STData$avg_distance,level = c(0,1))

trainingSTData_raw <- STData[STData$day_of_month <= 24, ]
testingSTData_raw <- STData[STData$day_of_month > 24, ]

trainingSTData_raw <- subset(trainingSTData_raw,select=-c(day_of_month))
testingSTData_raw <- subset(testingSTData_raw,select=-c(day_of_month))

training_sample <- trainingSTData_raw[sample(nrow(trainingSTData_raw),size=nrow(trainingSTData_raw)*0.01),]

#DataVIS <- subset(STData, select=-c(grid_index))
#names(DataVIS) <- c("day_of_month", "day_of_week","hour_of_day", "num_trips","avg_distance","avg_PSGR_pcar", "std_PSGR_pcar")

#ggpairs(DataVIS, diag=list(continuous="density", discrete="bar"), axisLabels="show", colour = "avg_distance",legends = TRUE)


# Trip Distance Category Classification
# Support Vector Machine

folds <- cvFolds(nrow(STData),K=5)

svm.radial<- train(avg_distance ~ ., data = training_sample, method = "svmRadial", preProc = c("center","scale"), tuneLength = 15)
pred.svm.radial <- predict(svm.radial , newdata = testingSTData_raw[,c("day_of_week","hour_of_day", "grid_index", "num_trips","avg_num_passenger", "std_num_passenger")], type="raw")
confusionMatrix(data = pred.svm.radial, reference = testingSTData_raw$avg_distance)


# Random Forest
RF <- train(avg_distance ~ ., data = training_sample, method = "rf", tuneLength = 15)
pred.RF <- predict(RF , newdata = testingSTData_raw[,c("day_of_week","hour_of_day", "grid_index", "num_trips","avg_num_passenger", "std_num_passenger")], type="raw")
confusionMatrix(data = pred.RF, reference = testingSTData_raw$avg_distance)

testingData_Output <- testingSTData_raw
testingData_Output$Pred <- pred.svm.radial

# PSO SVM
pso_accuracy<-c()
svmTrain <- function(parameter,trainingData, testingData, acc){
  svmParameter <- expand.grid(C = c(parameter[1]), sigma = c(parameter[2]))
  trainingData$avg_distance <- factor(trainingData$avg_distance)
  testingData$avg_distance <- factor(testingData$avg_distance)
  
  print(parameter)
  svm.radial_tune <- train(avg_distance ~ ., data = trainingData, method = "svmRadial",
                           preProc = c("center","scale"), tuneGrid = svmParameter)
  pred.svm.radial.bal_tune<- predict(svm.radial_tune, newdata = subset(testingData,select=-c(avg_distance)), type="raw")
  
  acc<- c(acc, confusionMatrix(data=pred.svm.radial.bal_tune, reference=testingData[["avg_distance"]])$overall[1])
  
  print(confusionMatrix(data=pred.svm.radial.bal_tune, reference=testingData[["avg_distance"]])$overall[1])
  acc<-c(acc, confusionMatrix(data=pred.svm.radial.bal_tune, reference=testingData[["avg_distance"]])$overall[1])
  return(-confusionMatrix(data=pred.svm.radial.bal_tune, reference=testingData[["avg_distance"]])$overall[1])
}


svmP <- psoptim(par=c(1,0.01),fn = svmTrain, lower=c(0.01,0.0001), upper=c(100000000,1000), trainingData = training_sample, testingData=testingSTData_raw, 
                acc= pso_accuracy, control=list(maxit.stagnate = 5,trace=TRUE, REPORT=1, maxit= 50))

PSO <- read.table("C:/Users/bingweiliu/Desktop/NYCTaxi/Book1.csv",  sep=",",
                  colClasses= c("numeric"))

write.csv(testingData_Output, file = "C:/Users/bingweiliu/Desktop/bingweiliu/NYCTaxi/nycTaxi_Predictions.csv")
