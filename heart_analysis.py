# The following is an analysis of the risk factors of heart disease, done in R. 4 different statistical models are # used: two logistic regression models and two random forests.

# My code can be run with this dataset: 
# https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset


# Install Libraries

install.packages("ResourceSelection")
install.packages("pROC")
install.packages("rpart.plot")

# Prepare Dataset

heart_data <- read.csv(file="heart_disease.csv", header=TRUE, sep=",")

# Converting appropriate variables to factors  
heart_data <- within(heart_data, {
   target <- factor(target)
   sex <- factor(sex)
   cp <- factor(cp)
   fbs <- factor(fbs)
   restecg <- factor(restecg)
   exang <- factor(exang)
   slope <- factor(slope)
   ca <- factor(ca)
   thal <- factor(thal)
})

head(heart_data, 10)

print("Number of variables")
ncol(heart_data)

print("Number of rows")
nrow(heart_data)

# Model #1

# Create Logistic Multiple Regression Model for heart disease (target) 
#using variables age (age), resting blood pressure (trestbps), and maximum heart rate achieved (thalach) 
print("Logistic Multiple Regression Model")
logit <- glm(target ~ age + trestbps + thalach , data = heart_data, family = "binomial")

summary(logit)
library(ResourceSelection)

# Hosmer-Lemeshow GOF
print("Hosmer-Lemeshow Goodness of Fit Test")
hl = hoslem.test(logit$y, fitted(logit), g=50)
hl
# Wald Confidence Interval
print("Wald Confidence Interval")
conf_int <- confint.default(logit, level=0.95)
round(conf_int,4)

# Confusion Matrix!							
# Predict heart disease or no heart disease for the data set using the model
hd_model_data <- heart_data[c('trestbps', 'thalach', 'age')]
pred <- predict(logit, newdata=hd_model_data, type='response')

# If likelihood is > 50%, predict 1, else, predict 0
depvar_pred = as.factor(ifelse(pred >= 0.5, '1', '0'))

# This creates the confusion matrix
conf.matrix <- table(heart_data$target, depvar_pred)[c('0','1'),c('0','1')]
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ": target=")
colnames(conf.matrix) <- paste("Prediction", colnames(conf.matrix), sep = ": target=")

# Print nicely formatted confusion matrix
print("Confusion Matrix")
format(conf.matrix,justify="centre",digit=2)

# ROC curve
library(pROC)

labels <- heart_data$target
predictions <- logit$fitted.values

roc <- roc(labels ~ predictions)

print("Area Under the Curve (AUC)")
round(auc(roc),4)

print("ROC Curve")
# True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity)
plot(roc, legacy.axes = TRUE)

# Predictions!
# What is the probability of an individual who is 50 years old,
# has a resting blood pressure of 122, and has maximum heart rate of 140 having heart disease?
print("Prediction: patient has rbp of 122, max heart rate is 140, and age is 50")
newdata1 <- data.frame(trestbps=122, age=50, thalach=140)
pred1 <- predict(logit, newdata1, type='response')
round(pred1, 4)

# What is the probability of an individual who is 50 years old,
# has a resting blood pressure of 140, and has maximum heart rate of 170 having heart disease?
print("Prediction: patient has trestbps of 140, age of 50, and max heart rate of 170")
newdata2 <- data.frame(trestbps=140, age=50, thalach=170)
pred2 <- predict(logit, newdata2, type='response')
round(pred2, 4)

# Second Model!

# The multiple regression model for heart disease (target) using variables maximum heart rate achieved (thalach),
# age of the individual (age), sex of the individual (sex),  exercise-induced angina (exang), type of chest pain (cp),
# And the quadratic term for age and the interaction term between age and maximum heart rate achieved.
logit2 <- glm(target ~ age + trestbps + sex + exang + cp + thalach + I(age^2) + age:thalach, data = heart_data, family = "binomial")

summary(logit2)
library(ResourceSelection)

# Hosmer-Lemeshow GOF
print("Hosmer-Lemeshow Goodness of Fit Test")
hl2 = hoslem.test(logit2$y, fitted(logit2), g=50)
hl2

# Confusion Matrix!
# Predict heart disease or no heart disease for the data set using the model
hd_model_data2 <- heart_data[c('age', 'trestbps', 'sex', 'exang', 'cp', 'thalach')]
pred2 <- predict(logit2, newdata=hd_model_data2, type='response')

# If likelihood is > 50%, predict 1, else, predict 0
depvar_pred2 = as.factor(ifelse(pred2 >= 0.5, '1', '0'))

# This creates the confusion matrix
conf.matrix <- table(heart_data$target, depvar_pred2)[c('0','1'),c('0','1')]
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ": target=")
colnames(conf.matrix) <- paste("Prediction", colnames(conf.matrix), sep = ": target=")

# Print nicely formatted confusion matrix
print("Confusion Matrix")
format(conf.matrix,justify="centre",digit=2)

# ROC curve
library(pROC)

labels <- heart_data$target
predictions <- logit2$fitted.values

roc <- roc(labels ~ predictions)

print("Area Under the Curve (AUC)")
round(auc(roc),4)

print("ROC Curve")
# True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity)
plot(roc, legacy.axes = TRUE)

# Predictions
# What is the probability of a male individual having heart disease who is 30 years old; has a maximum heart rate
# of 145; experiences exercise-induced angina;
# and does not experience chest pain related to typical angina, atypical angina, or non-anginal pain? 
print("age=30, trestbps=100, sex=1, exang=1, cp=0, thalach=145")
newdata3 <- data.frame(age=30, trestbps=100, sex='1', exang='1', cp='0', thalach=145)
pred3 <- predict(logit2, newdata3, type='response')
round(pred3, 4)
# The probability of a male individual having heart disease who is 30 years old, has a maximum heart rate of 145,
# and does not experience exercise-induced angina but experiences typical angina.
print("age=30, trestbps=100, sex=1, exang=0, cp=1, thalach=145")
newdata4 <- data.frame(age=30, trestbps=100, sex='1', exang='0', cp='1', thalach=145)
pred4 <- predict(logit2, newdata4, type='response')
round(pred4, 4)

set.seed(511038)

# Partition the data set into training and testing data
samp.size = floor(0.80*nrow(heart_data))

# Training set
print("Number of rows for the training set")
train_ind = sample(seq_len(nrow(heart_data)), size = samp.size)
train.data = heart_data[train_ind,]
nrow(train.data)

# Testing set 
print("Number of rows for the testing set")
test.data = heart_data[-train_ind,]
nrow(test.data)

# Graph the training and testing error against the number of trees using a classification random forest model for
# the presence of heart disease (target) using variables age (age), sex (sex), chest pain type (cp), resting blood pressure (trestbps), cholesterol
# measurement (chol), resting electrocardiographic measurement (restecg), exercise-induced angina (exang), 
# slope of peak exercise (slope), and number of major vessels (ca). Use a maximum of 200 trees. Use set.seed(511038).
set.seed(511038)
library(randomForest)

# checking
#=====================================================================
train = c()
test = c()
trees = c()

for(i in seq(from=1, to=200, by=1)) {
    #print(i)
    
    trees <- c(trees, i)
    
    model_rf1 <- randomForest(target ~ age+sex+cp+trestbps+chol+restecg+exang+slope+ca, data=train.data, ntree = i)
    
    train.data.predict <- predict(model_rf1, train.data, type = "class")
    conf.matrix1 <- table(train.data$target, train.data.predict)
    train_error = 1-(sum(diag(conf.matrix1)))/sum(conf.matrix1)
    train <- c(train, train_error)
    
    test.data.predict <- predict(model_rf1, test.data, type = "class")
    conf.matrix2 <- table(test.data$target, test.data.predict)
    test_error = 1-(sum(diag(conf.matrix2)))/sum(conf.matrix2)
    test <- c(test, test_error)
}
 
#matplot (trees, cbind (train, test), ylim=c(0,0.5) , type = c("l", "l"), lwd=2, col=c("red","blue"), ylab="Error", xlab="number of trees")
#legend('topright',legend = c('training set','testing set'), col = c("red","blue"), lwd = 2 )

plot(trees, train,type = "l",ylim=c(0,1.0),col = "red", xlab = "Number of Trees", ylab = "Classification Error")
lines(test, type = "l", col = "blue")
legend('topright',legend = c('training set','testing set'), col = c("red","blue"), lwd = 2 )

# 20 tree model with confusion matrices for training and testing sets
model_rf2 <- randomForest(target ~ age+sex+cp+trestbps+chol+restecg+exang+slope+ca, data=train.data, ntree = 20)

# Confusion matrix
print("======================================================================================================================")
print('Confusion Matrix: TRAINING set based on random forest model built using 20 trees')
train.data.predict <- predict(model_rf2, train.data, type = "class")

# Construct the confusion matrix
conf.matrix <- table(train.data$target, train.data.predict)[,c('1','0')]
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ": ")
colnames(conf.matrix) <- paste("Prediction", colnames(conf.matrix), sep = ": ")

# Print a nicely formatted confusion matrix
format(conf.matrix,justify="centre",digit=2)


print("======================================================================================================================")
print('Confusion Matrix: TESTING set based on random forest model built using 20 trees')
test.data.predict <- predict(model_rf2, test.data, type = "class")

# Construct the confusion matrix
conf.matrix <- table(test.data$target, test.data.predict)[,c('1','0')]
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ": ")
colnames(conf.matrix) <- paste("Prediction", colnames(conf.matrix), sep = ": ")

# Print a nicely formatted confusion matrix
format(conf.matrix,justify="centre",digit=2)


# Random Forest Regression Model!
# Graph the mean squared error against the number of trees for a random forest regression model for maximum heart rate
#achieved using age (age), sex (sex), chest pain type (cp), resting blood pressure (trestbps), 
#cholesterol measurement (chol), resting electrocardiographic measurement (restecg), exercise-induced angina (exang),
#slope of peak exercise (slope), and number of major vessels (ca). Use a maximum of 80 trees. 
set.seed(511038)
library(randomForest)

# Root mean squared error
RMSE = function(pred, obs) {
    return(sqrt( sum( (pred - obs)^2 )/length(pred) ) )
}


# checking
#=====================================================================
train = c()
test = c()
trees = c()

for(i in seq(from=1, to=80, by=1)) {
    trees <- c(trees, i)
    model_rf3 <- randomForest(thalach ~ age+sex+cp+trestbps+chol+restecg+exang+slope+ca, data=train.data, ntree = i)
    
    pred <- predict(model_rf3, newdata=train.data, type='response')
    rmse_train <-  RMSE(pred, train.data$thalach)
    train <- c(train, rmse_train)
    
    pred <- predict(model_rf3, newdata=test.data, type='response')
     rmse_test <-  RMSE(pred, test.data$thalach)
    test <- c(test, rmse_test)
}
 
plot(trees, train,type = "l",ylim=c(00,100),col = "red", xlab = "Number of Trees", ylab = "Root Mean Squared Error")
lines(test, type = "l", col = "blue")
legend('topright',legend = c('training set','testing set'), col = c("red","blue"), lwd = 2 )

# RF Regression model with 10 trees!
rf_ten <- randomForest(thalach ~ age+sex+cp+trestbps+chol+restecg+exang+slope+ca, data=train.data, ntree = 10)


# Root mean squared error
RMSE = function(pred, obs) {
    return(sqrt( sum( (pred - obs)^2 )/length(pred) ) )
}

print("======================================================================================================================")
print('Root Mean Squared Error: TRAINING set based on random forest model built using 20 trees')
pred <- predict(rf_ten, newdata=train.data, type='response')
RMSE(pred, train.data$thalach)


print("======================================================================================================================")
print('Root Mean Squared Error: TESTING set based on random forest model built using 20 trees')
pred <- predict(rf_ten, newdata=test.data, type='response')
RMSE(pred, test.data$thalach)