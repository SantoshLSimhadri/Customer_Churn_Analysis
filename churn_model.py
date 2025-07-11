# Churn Prediction Model using Logistic Regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
df = pd.read_csv('../data/customer_data.csv')

# Basic EDA
print("Data Shape:", df.shape)
print("Null Values:
", df.isnull().sum())
print("Churn Rate: {:.2f}%".format(df['churn'].mean() * 100))

# Feature Engineering
df['tenure_bins'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 36], labels=['<6', '6-12', '12-24', '24-36'])

# Prepare input/output
X = df[['tenure', 'monthly_charges']]
y = df['churn']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:
", confusion_matrix(y_test, y_pred))
print("Classification Report:
", classification_report(y_test, y_pred))
print("Model Accuracy: {:.2f}%".format(model.score(X_test, y_test) * 100))
