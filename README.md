# CU-PROJECT


````markdown
---
title: "Classifying Countries by Income Inequality Level"
author: "Ro S"
date: "`r Sys.Date()`"
output: html_document
---

## Introduction

Income inequality is a major concern for policymakers, economists, and development practitioners.  
In this project, we classify countries into different income inequality levels using **machine learning models in R**.  

We will:  
1. Prepare and clean the dataset  
2. Train classification models  
3. Compare their performance  
4. Visualize the results  

The dataset comes from the **World Bank**, containing country-level indicators.

---

## Load Packages

```{r message=FALSE, warning=FALSE}
# Install packages if not already installed
pkgs <- c("tidyverse", "caret", "rpart", "randomForest", 
          "nnet", "e1071", "ggplot2")
new <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if(length(new)) install.packages(new)

# Load packages
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(nnet)
library(e1071)
library(ggplot2)
````

---

## Step 1: Load and Inspect Data

```{r}
# Read dataset
data <- read.csv("income_inequality_dataset.csv")

# Preview the data
head(data)

# Check basic structure
str(data)
```

**Explanation:**
We load the dataset from a CSV file, then check its first few rows and structure to understand:

* How many columns are there?
* What types of variables (numeric, categorical, etc.)?
* Any obvious missing values?

---

## Step 2: Data Cleaning

```{r}
# Remove rows with missing values
data <- na.omit(data)

# Convert 'IncomeInequalityLevel' into a factor (for classification)
data$IncomeInequalityLevel <- as.factor(data$IncomeInequalityLevel)

# Double-check
summary(data$IncomeInequalityLevel)
```

**Explanation:**
Machine learning models in R work best with **clean data**.
We:

* Remove incomplete rows (NA values)
* Ensure the **target variable** (`IncomeInequalityLevel`) is a factor, so R knows it’s categorical

---

## Step 3: Split into Training & Testing Sets

```{r}
set.seed(123) # reproducibility
trainIndex <- createDataPartition(data$IncomeInequalityLevel, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
```

**Explanation:**
We split the dataset into:

* **Training set (80%)** — for the model to learn patterns
* **Testing set (20%)** — for checking how well the model works on new data

---

## Step 4: Train Models

### Decision Tree

```{r}
model_dt <- rpart(IncomeInequalityLevel ~ ., data = trainData, method = "class")
pred_dt <- predict(model_dt, testData, type = "class")
confusionMatrix(pred_dt, testData$IncomeInequalityLevel)
```

### Random Forest

```{r}
model_rf <- randomForest(IncomeInequalityLevel ~ ., data = trainData, ntree = 100)
pred_rf <- predict(model_rf, testData)
confusionMatrix(pred_rf, testData$IncomeInequalityLevel)
```

### Multinomial Logistic Regression

```{r}
model_mlr <- multinom(IncomeInequalityLevel ~ ., data = trainData)
pred_mlr <- predict(model_mlr, testData)
confusionMatrix(pred_mlr, testData$IncomeInequalityLevel)
```

**Explanation:**
We use three different models:

* **Decision Tree** — simple, interpretable structure
* **Random Forest** — ensemble of many decision trees for higher accuracy
* **Multinomial Logistic Regression** — statistical model for multiple categories

Each model is tested on the **testing set** and evaluated with a confusion matrix.

---

## Step 5: Visualization

### Scatter Plot of Countries by Inequality Level

```{r}
ggplot(data, aes(x = GDP.per.capita, y = Gini.Index, color = IncomeInequalityLevel)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Countries by GDP per Capita and Gini Index",
       x = "GDP per Capita",
       y = "Gini Index") +
  theme_minimal()
```

**Explanation:**
This scatter plot helps visually understand the relationship between **economic wealth (GDP per capita)** and **income inequality (Gini Index)**, with colors representing different inequality levels.

---

## Step 6: Conclusion

We built a pipeline that:

* Cleans and prepares data
* Trains multiple classification models
* Evaluates accuracy
* Visualizes country positions

**Next steps** could include:

* Adding more economic and social indicators
* Hyperparameter tuning for better performance
* Testing with other classification algorithms

---

```

---

If you want, I can now **add a short diagram explaining the pipeline visually** so it’s even easier for someone on GitHub to follow. That would make the Markdown more beginner-friendly while keeping your exact workflow.  

Do you want me to add that diagram?
```
