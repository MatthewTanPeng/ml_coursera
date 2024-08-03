# Load necessary libraries
library(randomForest)
library(caret)
library(dplyr)

# Load the dataset
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
train_data <- read.csv(train_url)

# Explore the dataset structure and summary
str(train_data)
summary(train_data)

# Step 1: Data Cleaning
# Remove near-zero variance predictors
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
train_data <- train_data[, !nzv$nzv]

# Remove columns with too many NA values
train_data <- train_data[, colSums(is.na(train_data)) < nrow(train_data) * 0.95]

# Remove irrelevant columns (like timestamp, user_name, etc.)
train_data <- train_data %>% select(-c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))

# Convert 'classe' to a factor
train_data$classe <- as.factor(train_data$classe)

# Step 2: Data Preprocessing
# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(train_data$classe, p = 0.7, list = FALSE)
training_data <- train_data[train_index, ]
testing_data <- train_data[-train_index, ]

# Preprocess the data: Center and Scale
preproc <- preProcess(training_data[, -which(names(training_data) == "classe")], method = c("center", "scale"))
training_data_preprocessed <- predict(preproc, newdata = training_data)
testing_data_preprocessed <- predict(preproc, newdata = testing_data)

# Step 3: Model Building with Random Forest
# Build the random forest model
rf_model <- randomForest(classe ~ ., data = training_data_preprocessed, importance = TRUE, ntree = 100)

# Evaluate variable importance
importance(rf_model)
varImpPlot(rf_model)

# Step 4: Cross-validation
# Set up cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the model using cross-validation
set.seed(123)
cv_model <- train(classe ~ ., data = training_data_preprocessed, method = "rf", trControl = train_control, tuneLength = 3)

# Print the cross-validation results
print(cv_model)

# Step 5: Evaluate Model Performance
# Predict on the testing set
predictions <- predict(rf_model, newdata = testing_data_preprocessed)

# Confusion matrix to evaluate accuracy
confusion_matrix <- confusionMatrix(predictions, testing_data_preprocessed$classe)
print(confusion_matrix)

# Calculate expected out-of-sample error
oob_error <- rf_model$err.rate[ntree, "OOB"]
cat("Out-of-sample error rate:", oob_error)

# Calculate expected out-of-sample error
oob_error <- rf_model$err.rate[rf_model$ntree, "OOB"]
cat("Out-of-sample error rate:", oob_error)

# Load necessary libraries
library(randomForest)
library(caret)
library(dplyr)

# Load the dataset
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
train_data <- read.csv(train_url)

# Explore the dataset structure and summary
str(train_data)
summary(train_data)

# Step 1: Data Cleaning
# Remove near-zero variance predictors
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
train_data <- train_data[, !nzv$nzv]

# Remove columns with too many NA values
train_data <- train_data[, colSums(is.na(train_data)) < nrow(train_data) * 0.95]

# Remove irrelevant columns (like timestamp, user_name, etc.)
train_data <- train_data %>% select(-c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))

# Convert 'classe' to a factor
train_data$classe <- as.factor(train_data$classe)

# Step 2: Data Preprocessing
# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(train_data$classe, p = 0.7, list = FALSE)
training_data <- train_data[train_index, ]
testing_data <- train_data[-train_index, ]

# Preprocess the data: Center and Scale
preproc <- preProcess(training_data[, -which(names(training_data) == "classe")], method = c("center", "scale"))
training_data_preprocessed <- predict(preproc, newdata = training_data)
testing_data_preprocessed <- predict(preproc, newdata = testing_data)

# Step 3: Model Building with Random Forest
# Build the random forest model
rf_model <- randomForest(classe ~ ., data = training_data_preprocessed, importance = TRUE, ntree = 100)

# Evaluate variable importance
importance(rf_model)
varImpPlot(rf_model)

# Step 4: Cross-validation
# Set up cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the model using cross-validation
set.seed(123)
cv_model <- train(classe ~ ., data = training_data_preprocessed, method = "rf", trControl = train_control, tuneLength = 3)

# Print the cross-validation results
print(cv_model)

# Step 5: Evaluate Model Performance
# Predict on the testing set
predictions <- predict(rf_model, newdata = testing_data_preprocessed)

# Confusion matrix to evaluate accuracy
confusion_matrix <- confusionMatrix(predictions, testing_data_preprocessed$classe)
print(confusion_matrix)

# Calculate expected out-of-sample error
oob_error <- rf_model$err.rate[ntree, "OOB"]
cat("Out-of-sample error rate:", oob_error)

# Calculate expected out-of-sample error
oob_error <- rf_model$err.rate[rf_model$ntree, "OOB"]
cat("Out-of-sample error rate:", oob_error)


# Step 6: Predict on New Test Cases
# Load the test dataset
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
test_data <- read.csv(test_url)

# Ensure test data columns match training data columns
common_columns <- intersect(names(training_data_preprocessed), names(test_data))
test_data <- test_data[, common_columns]

# Add any missing columns in test_data that are present in training_data_preprocessed
missing_columns <- setdiff(names(training_data_preprocessed), names(test_data))
for (col in missing_columns) {
  test_data[[col]] <- NA
}

# Reorder test_data columns to match the training_data_preprocessed
test_data <- test_data[, names(training_data_preprocessed)]

# Preprocess the test data using the same preprocessing model
test_data_preprocessed <- predict(preproc, newdata = test_data)

# Make predictions on test data using the cross-validated model
test_predictions <- predict(cv_model, newdata = test_data_preprocessed)
print(test_predictions)
