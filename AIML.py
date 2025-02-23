#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

df = pd.read_csv(r"C:\Users\Ananya R\Downloads\diabetes.csv")
print(df.head())  # First few rows
print(df.info())  # Data types & missing values
print(df.describe())  # Summary statistics


# In[3]:


print(df.isnull().sum())


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['Glucose'], kde=True)
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['Pregnancies'], kde=True)
plt.show()


# In[6]:


sns.pairplot(df)
plt.show()


# In[6]:


from sklearn.preprocessing import StandardScaler
df.fillna(df.mean(), inplace=True)  
df['Glucose'].fillna(df['Glucose'].mode()[0], inplace=True)
scaler = StandardScaler()
df[['Pregnancies', 'Glucose', 'BMI', 'Age']] = scaler.fit_transform(df[['Pregnancies', 'Glucose', 'BMI', 'Age']])


# In[7]:


X = df[["Glucose", "BMI", "Age", "BloodPressure", "DiabetesPedigreeFunction"]]
y = df["Outcome"]  # Target variable


# In[8]:


X


# In[9]:


y


# In[10]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale the selected features


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[12]:


X_train


# In[13]:


X_test


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Model 1: Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[15]:


y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)


# In[16]:


print(y_lr_train_pred)


# In[17]:


print(y_lr_test_pred)


# In[18]:


# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# In[19]:


y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)


# In[20]:


print(y_rf_train_pred)


# In[21]:


print(y_rf_test_pred)


# In[22]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Evaluation
print("Logistic Regression Performance:")
print(classification_report(y_test, y_lr_test_pred))
print("Accuracy:", accuracy_score(y_test, y_lr_test_pred))

print("\nRandom Forest Performance:")
print(classification_report(y_test, y_lr_test_pred))
print("Accuracy:", accuracy_score(y_test, y_lr_test_pred))


# In[23]:


import matplotlib.pyplot as plt
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)
plt.plot()


# In[24]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Train the best model
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print("\nImproved Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))


# In[26]:


import joblib

# Save the trained model
joblib.dump(best_rf, "diabetes_model.pkl")
print("Model saved as diabetes_model.pkl")


# In[28]:


get_ipython().system('jupyter nbconvert --to AIML.ipynb')


# In[ ]:




