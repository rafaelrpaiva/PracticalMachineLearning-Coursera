## http://pkudinov.github.io/MachineLearningCoursera/
## https://rstudio-pubs-static.s3.amazonaws.com/29426_041c5ccb9a6a4bedb204e33144bb0ad4.html
## http://topepo.github.io/caret/training.html
## http://www.nicolomarchi.it/projects/practicalmachinelearning/exercise.html

## http://groupware.les.inf.puc-rio.br/har
### Step 1: Loading Libraries and Data
library(caret)
library(doParallel)
set.seed(171178)

trainingDS <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
testingDS <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))

### Step 2: Cleaning data 
# Using suggestion posted by Himanshu Rawat in
# https://class.coursera.org/predmachlearn-030/forum/thread?thread_id=61
## https://class.coursera.org/predmachlearn-030/forum/thread?thread_id=113
goodData <- colSums(is.na(trainingDS)) < nrow(trainingDS) * 0.5
trainingFilter <- trainingDS[, goodData]
trainingFilter <- trainingFilter[, -1]
testingFilter <- testingDS[, goodData]
testingFilter <- testingFilter[, -c(1, length(testingFilter))]

### Step 3: Creating data partition for cross-validation
inTrain  <- createDataPartition(trainingFilter$classe, p = 0.6, list = FALSE)
trainingCV <- trainingFilter[inTrain, ]
testingCV <- trainingFilter[-inTrain, ]

### Step 4: Running ML algorithms
# For more details, see http://topepo.github.io/caret/Random_Forest.html
# Relating to train control: http://topepo.github.io/caret/training.html#custom
registerDoParallel(cores = 3)
dateInit <- date()
modFit <- train(classe ~., method="rf", trControl=trainControl(method = "cv", number = 4), data=trainingCV)
dateEnd <- date()

varImp(modFit)

### Step xx: running the prediction into the testing dataset.

predictedTesting <- predict(modFit, newdata=testingCV)
confMatrix <- confusionMatrix(predictedTesting,testingCV$classe)
confMatrix

## Running Test

finalPredict <- predict(modFit, newdata=testingFilter)

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(finalPredict)