# Libraries
library(ggplot2) # For graphing
library(tidyverse) # For data manipulation
library(fastDummies) # For creating dummy variables quickly
library(tree) # for regression trees
library(xgboost) # For gradient boosting
library(caret) # For knn model
library(randomForest) # For random forest



firm <- read.csv("./Employee Attrition.csv") # Dataset
firm <- na.omit(firm) # remove rows with NA values
median_value <- median(firm$satisfaction_level)
firm$firm_binary <- ifelse(firm$satisfaction_level > median_value, 1, 0) # Binary Value for KNN


## Setting seed and creating 90/10 set
set.seed(67)
shuffled_firm <- firm[sample(nrow(firm)),] # Shuffle to minimize random effects
train <- createDataPartition(shuffled_firm$firm_binary, p = 0.9, list = FALSE, times  = 1)
firm_train <- shuffled_firm[train,] # Training Set
firm_test <- shuffled_firm[-train,] # Test Set


## Creating dummies for training dataset + changing variables

firm_train$avg_weekly_hours <- firm_train$average_montly_hours/4

# Salary Dummy variables
firm_train$salary_low <- ifelse(firm_train$salary == "low", 1,0)
firm_train$salary_medium <- ifelse(firm_train$salary == "medium", 1,0)
firm_train$salary_high <- ifelse(firm_train$salary == "high", 1,0)

# department dummy variables
firm_train <- dummy_cols(firm_train,
                   select_columns = "dept"
                   )

firm_train$firm_binary <- factor(firm_train$firm_binary, levels = c(0, 1))

firm_train <- firm_train[,c(-1:-2, -5, -9:-10)] # Get rid of employee id, salary, satisfaction level (numeric) avg monthly hours, department, and salary


## creating dummy variables for test set

firm_test$avg_weekly_hours <- firm_test$average_montly_hours/4

# Salary Dummy variables
firm_test$salary_low <- ifelse(firm_test$salary == "low", 1,0)
firm_test$salary_medium <- ifelse(firm_test$salary == "medium", 1,0)
firm_test$salary_high <- ifelse(firm_test$salary == "high", 1,0)

# department dummy variables
firm_test <- dummy_cols(firm_test,
                         select_columns = "dept")

firm_test$firm_binary <- factor(firm_test$firm_binary, levels = c(0, 1))

firm_test <- firm_test[,c(-1:-2, -5, -9:-10)]



## K nearest neighbors

# Building the model
knn_model <- caret::train(x = firm_train[, c(-6)],
                          y = as.factor(firm_train[, c(6)]),
                          method = "knn", # Choosing knn model
                          trControl = trainControl(
                            method ="repeatedcv", # k-fold cross validation
                            number = 10, # number of folds (k in cross validation)
                            repeats = 5 ), # number of times to repeat k-fold cv
                          
                          preProcess = c("center", "scale"),
                          tuneGrid = expand.grid(k=3:15),
                          metric = "Accuracy"
                          )

plot(knn_model$results$k, knn_model$results$Accuracy, type = "o", col = "blue", xlab = "K value", ylab = "Accuracy",
     main = "K vs Accuracy, 10 fold CV, CARET")

best_k <- knn_model$bestTune[1,1]

best_model <- knn3(
                   firm_train$firm_binary ~ .,
                   data = firm_train,
                   k = best_k
                  )

predictions <- predict(best_model, firm_test, type = "class")
cm <- confusionMatrix(predictions, firm_test$firm_binary)
cm # 67% prediction rate



## Regression Tree

tree_firm <- tree(firm_binary ~ ., firm_train)
summary(tree_firm)

plot(tree_firm)
text(tree_firm, pretty = 0)

tree_pred1 <- predict(tree_firm, firm_test, type = "class")
table(tree_pred1, firm_test$firm_binary) # 68.9% classification rate

# Tree pruning
cv_firm <- cv.tree(tree_firm, FUN = prune.misclass)
plot(cv_firm$size, cv_firm$dev, type = "b")

prune_firm <- prune.misclass(tree_firm, best = 4)
plot(prune_firm)
text(prune_firm, pretty = 0)

tree_pred2 <- predict(prune_firm, firm_test, type = "class")
table(tree_pred2, firm_test$firm_binary) # 68.9% classification rate

## Random Forest/Bagging

bag_firm <- randomForest(firm_binary~.,
                         data = firm_train,
                         mtry = 19,
                         importance = TRUE,
                         proximity = TRUE)

bag_firm

bagged_predictions <- predict(bag_firm, newdata = firm_test)

# Confusion Matrix
conf_matrix <- table(Predicted = bagged_predictions, Actual = firm_test$firm_binary)
conf_matrix_df <- as.data.frame(conf_matrix)
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Count")

ggplot(data = conf_matrix_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "white", size = 5) +
  scale_fill_gradient(low = "black", high = "blue") +
  theme_minimal() +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) ## 77% prediction rate

## Gradient Boosting (XGBoosting)

# setting up parameters

params <- list(set.seed = 67,
               eval_metric = "auc",
               objective = "binary:logistic")

# Running XGBoost

xg_model <- xgboost(data = as.matrix(firm_train[, c(-6)]),
                    label = firm_train[, c(6)],
                    params = params,
                    nrounds = 20,
                    verbose = 1)

#' Blue dot = observation
#' above zero = positive contribution, increases likelihood of 1 result
#' below zero = negative contribution, decreased likelihood that result is 1 (aka 0)



xg_pred <- predict(xg_model, firm_test)
err <- mean(as.numeric(xg_pred > 0.5) != firm_test[, c(6)])
print(paste("test-error=", err)) # 73% Prediction rate
pred_acc <- 1 - err
print(pred_acc)

xgb.plot.shap(data = as.matrix(firm_train[, c(-6)]), # Most important variables that predict unhappiness
              model = xg_model,
              top_n = 5)

xgb.plot.shap(data = as.matrix(firm_test[, c(-6)]), # Most important variables that predict unhappiness
              model = xg_model,
              top_n = 5)
