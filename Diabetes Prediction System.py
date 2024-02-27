#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score


# In[27]:


data=pd.read_csv(r"C:\MBA in Business Analytics\Data Analyst BA\diabetes.csv")


# In[28]:


data


# In[29]:


data.head()


# In[30]:


data.tail()


# In[31]:


data.shape


# In[32]:


data.columns


# In[33]:


sns.heatmap(data.isnull())


# In[34]:


correlation = data.corr()
print(correlation)


# In[35]:


data.columns


# # Train test split

# In[36]:


X=data.drop("Outcome",axis=1)
Y=data['Outcome']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# # Training the model

# In[37]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[38]:


prediction=model.predict(X_test)


# In[39]:


print(prediction)


# In[40]:


accuracy = accuracy_score(prediction,Y_test)


# In[41]:


print(accuracy)


# In[43]:


data.isnull()


# In[45]:


data.isnull().count()


# In[46]:


not_null = df.dropna()


# In[47]:


not_null


# In[48]:


not_null


# In[49]:


not_null.isnull().count()


# In[ ]:





# In[50]:


not_null.isnull().sum()


# In[52]:


# Diabetes Prevalence
diabetes_prevalence = (data['Outcome'].sum() / len(data)) * 100

# Patient Demographics
average_age = data['Age'].mean()
age_distribution = pd.cut(data['Age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])

# Health Metrics
average_glucose = data['Glucose'].mean()
blood_pressure_distribution = data['BloodPressure']
bmi_distribution = data['BMI']

# Pregnancy-related Metrics
average_pregnancies = data['Pregnancies'].mean()
pregnancy_distribution = data['Pregnancies']

# Insulin Levels
average_insulin = data['Insulin'].mean()
insulin_distribution = data['Insulin']

# Body Composition
bmi_distribution = data['BMI']

# DiabetesPedigreeFunction
average_diabetes_pedigree_function = data['DiabetesPedigreeFunction'].mean()

# Age-related Metrics
average_age_diabetic = data[data['Outcome'] == 1]['Age'].mean()
average_age_non_diabetic = data[data['Outcome'] == 0]['Age'].mean()
age_distribution_diabetic = pd.cut(data[data['Outcome'] == 1]['Age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])
age_distribution_non_diabetic = pd.cut(data[data['Outcome'] == 0]['Age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])

# Correlation Analysis
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Display the computed KPIs
print(f"Diabetes Prevalence: {diabetes_prevalence:.2f}%")
print(f"Average Age: {average_age:.2f}")
print(f"Age Distribution:\n{age_distribution.value_counts()}")
print(f"Average Glucose Level: {average_glucose:.2f}")
print(f"Average Pregnancies: {average_pregnancies:.2f}")
print(f"Average Insulin Level: {average_insulin:.2f}")
print(f"Average Diabetes Pedigree Function: {average_diabetes_pedigree_function:.2f}")
print(f"Average Age (Diabetic): {average_age_diabetic:.2f}")
print(f"Average Age (Non-Diabetic): {average_age_non_diabetic:.2f}")
print(f"Correlation Matrix:\n{correlation_matrix['Outcome']}")

# Additional visualizations can be added based on specific KPIs


# In[ ]:





# In[ ]:




