#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[7]:


data = pd.read_csv(r'C:\Users\ashic\Desktop\Leads.csv')


# In[8]:


data.head()


# In[10]:


data.shape


# In[15]:


data.info()


# # DATA CLEANING

# In[22]:


#Identify columns with missing values:

data.isnull().sum()


# # Handling Missing Values:

# In[26]:


data = data.dropna(thresh=len(data) * 0.7, axis=1)  # Drop columns with more than 70% missing values
data = data.dropna()  # Drop rows with any missing values


# In[28]:


data['TotalVisits'].fillna(data['TotalVisits'].median(), inplace=True)  # Replace missing values with the median
data['Lead Source'].fillna('Unknown', inplace=True)  # Replace missing values with a specific category


# In[29]:


data['Country'].fillna('Not Specified', inplace=True)  # Replace missing values with a new category
data['Last Activity'].fillna(method='ffill', inplace=True)  # Forward fill missing values


# # Handling Categorical Variables:

# In[30]:


data['Lead Origin'] = data['Lead Origin'].astype('category')
data['Last Notable Activity'] = data['Last Notable Activity'].astype('category')


# In[31]:


data.replace('Select', np.nan, inplace=True)


# In[33]:


data = data.drop(['Prospect ID', 'Lead Number'], axis=1)


# In[35]:


data.isnull().sum()


# In[36]:


data['Specialization'].fillna('Not Specified', inplace=True)  # Replace missing values in 'Specialization'
data['How did you hear about X Education'].fillna('Not Specified', inplace=True)  # Replace missing values in 'How did you hear about X Education'
data['Lead Profile'].fillna('Not Specified', inplace=True)  # Replace missing values in 'Lead Profile'
data['City'].fillna('Not Specified', inplace=True)  # Replace missing values in 'City'


# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Select the relevant features and target variable
X = data[['Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call', 'TotalVisits', 'Total Time Spent on Website',
          'Page Views Per Visit', 'Last Activity', 'Country', 'Specialization', 'How did you hear about X Education',
          'What is your current occupation', 'What matters most to you in choosing a course', 'Search', 'Magazine',
          'Newspaper Article', 'X Education Forums', 'Newspaper', 'Digital Advertisement', 'Through Recommendations',
          'Receive More Updates About Our Courses', 'Update me on Supply Chain Content', 'Get updates on DM Content',
          'Lead Profile', 'City', 'I agree to pay the amount through cheque',
          'A free copy of Mastering The Interview', 'Last Notable Activity']]
y = data['Converted']

# Perform one-hot encoding on categorical columns
X_encoded = pd.get_dummies(X, drop_first=True)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Step 2: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Model Training
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Step 4: Make Predictions
y_pred = logreg.predict(X_test_scaled)

# Step 5: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[ ]:




