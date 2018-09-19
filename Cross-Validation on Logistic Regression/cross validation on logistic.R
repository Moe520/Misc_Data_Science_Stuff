
set.seed(1234567)


setwd("C:/Users/PC/Documents/GitHub/Moe_Antar/Cross-Validation on Logistic Regression")


cancer <- read.csv("BreastCancer.csv")[-1] # Don't include the ID column

attach(cancer)


# First let's remove incomplete cases as they can screw with our functions and generate NA's
cancer <- cancer[complete.cases(cancer),]

# Fit this data to a logistic regression model
fit.logistic <- glm(cancer$Class~., family="binomial",data=cancer)

# Now we need to assess how well the model performs

# Since the data is small enough , we can run a LOOCV cross-validation routine to assess the model's effectiveness



# set up empty vector to store results (if the prediction matched the target, outcome is one, otherwise outcome is 0)
result_matched <- NULL

## Leave One Out Cross Validation

for(i in 1:length(cancer$Class)){
  
  # At each iteration, we take out one row, and train the data on the rest
  # Then we use that model to predict that "Left-Out Row"
  training_set <- cancer[-i,] ; testing_set <- cancer[i,]
  
  #Generate a logistic model from the training set
  fit.logistic <- glm(cancer$Class~., family="binomial",data=cancer)
  
  #Use this generated model to predict thet "Left-Out" row
  prediction <- predict(fit.logistic, newdata = testing_set, type = "response")
  
  # Did it correctly predict?
  yhat_prediction <- ifelse(prediction > 0.5, 1, 0)
  
  target <- testing_set$Class
  
  #Store the result before moving  on to the next iteration
  result_matched[i] <- ifelse(yhat_prediction == target, 1, 0)
}

## Since we are dealing with a logistic model, RMSE is not a good measure because it assumes normality (which is impossible in a binomial distribution)


## So instead, we will use the rate of misclassification (i.e what percentage of the time was the model correct)


count_of_matches <-  length(which(result_matched == 1)) # How many times did the model correctly predict the target

count_of_matches/nrow(cancer) # what percentage of the attempts was successful in predicting the "Left-Out" row?


# K-fold CV for 10 iterations

scorecard <- NULL

#Cut the data into folds
folds <- cut(seq(1,nrow(cancer)),breaks=10,labels=FALSE)


for(i in 1:10){
  #Split up the data into folds 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- cancer[testIndexes, ]
  trainData <- cancer[-testIndexes, ]
  
  #####
  #Generate a logistic model from the training set
  fit.logistic <- glm(trainData$Class~., family="binomial",data=trainData)
  
  #Use generated model to predict the test data
  prediction <- predict(fit.logistic, newdata = testData, type = "response")
  
  # Did it correctly predict?
  temp_vector <- NULL
  for (i in 1:length(prediction)){
    yhat_prediction <- ifelse(prediction[i] > 0.5, 1, 0)
    target <- testData$Class[i]
    ifelse(yhat_prediction==target,temp_vector[i] <- 1,temp_vector[i] <- 0)
  }
  count_of_matches <-  length(which(temp_vector == 1))
  scorecard[i] <- count_of_matches/nrow(temp_vector)
}

mean(scorecard)


