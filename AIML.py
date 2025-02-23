#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv(r"./diabetes.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Visualizations
sns.histplot(df['Glucose'], kde=True)
plt.show()

sns.histplot(df['Pregnancies'], kde=True)
plt.show()

sns.pairplot(df)
plt.show()

# Fill missing values
df.fillna(df.mean(), inplace=True)
df['Glucose'].fillna(df['Glucose'].mode()[0], inplace=True)

# Define features and target
X = df[["Glucose", "BMI", "Age", "BloodPressure", "DiabetesPedigreeFunction"]]
y = df["Outcome"]

# Scale features used for training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression (not used in API, kept for reference)
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Logistic Regression Performance:")
print(classification_report(y_test, lr.predict(X_test)))
print("Accuracy:", accuracy_score(y_test, lr.predict(X_test)))

print("\nRandom Forest Performance:")
print(classification_report(y_test, rf.predict(X_test)))
print("Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# Hyperparameter tuning for Random Forest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Use the best random forest estimator
best_rf = grid_search.best_estimator_
print("\nImproved Random Forest Accuracy:", accuracy_score(y_test, best_rf.predict(X_test)))

# Save the best model and the scaler
joblib.dump(best_rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
