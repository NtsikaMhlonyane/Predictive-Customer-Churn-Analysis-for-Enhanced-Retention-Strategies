# Predictive-Customer-Churn-Analysis-for-Enhanced-Retention-Strategies

## Overview

This project aims to predict customer churn using various models and analyze their performance.
## Objective:
The primary objective of this data science project is to develop a predictive model that accurately identifies potential customer churn. By leveraging various customer-centric data sources, the project aims to enhance our understanding of the factors contributing to churn within our user base. The ultimate goal is to empower the business with actionable insights, enabling the implementation of targeted retention strategies to reduce churn rates and increase overall customer satisfaction.
## Data

The dataset used in this project includes information about customer transactions, vintage, dependents, occupation, city, credit-debit ratio, time-to-churn, and churn status.

## Exploratory Data Analysis (EDA)

After conducting exploratory data analysis, we found that the dataset contains both numerical and categorical features. Missing values were present in the 'city' column.

## Models

### Logistic Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Data preprocessing
# ...

# Train-test split
# ...

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, predictions))
Cox Proportional Hazard Model
python
Copy code
from lifelines import CoxPHFitter

# Data preprocessing
# ...

# Initialize the model
cph = CoxPHFitter()

# Fit the model
cph.fit(df, duration_col='time_to_churn', event_col='churn')
Models Tried and Failed



### The classification report provides a detailed summary of the model's performance on each class. In your case, since you have a binary classification problem (churn or not churn), there is only one class (0 or 1).

## Here's the summary of the results:

## Precision (for class 0): 
Precision is the ratio of correctly predicted instances of class 0 to the total predicted instances of class 0. In your case, it's 1.00, indicating that all instances predicted as class 0 were correct.

## Recall (for class 0): 
Recall, or sensitivity, is the ratio of correctly predicted instances of class 0 to the total actual instances of class 0. Again, it's 1.00, suggesting that the model captured all instances of class 0.

## F1-score (for class 0): 
The F1-score is the harmonic mean of precision and recall. It's also 1.00 for class 0, indicating a perfect balance between precision and recall.

## Support (for class 0): 
Support is the number of actual occurrences of class 0 in the test set. In your case, it's 1.

## Accuracy:
Accuracy is the ratio of correctly predicted instances (both class 0 and class 1) to the total instances in the test set. It's 1.00, meaning that all predictions were correct.

In summary, your model achieved perfect performance on the test set for class 0, resulting in an accuracy of 1.00. However, keep in mind that these results might be indicative of overfitting, especially if your dataset is small. It's always essential to assess the model on new, unseen data to evaluate its generalization performance.






