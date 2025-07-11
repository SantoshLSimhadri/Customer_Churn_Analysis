# Python Example: Basic churn model
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/customer_data.csv')
X = df[['tenure', 'monthly_charges']]
y = df['churn']
model = LogisticRegression().fit(X, y)
print(f'Model accuracy: {model.score(X, y):.2f}')
