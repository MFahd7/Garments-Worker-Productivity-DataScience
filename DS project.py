#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dset = pd.read_csv("garments_worker_productivity.csv")


# In[2]:


dset.head()


# In[3]:


dset[0:250:25]


# In[4]:


dset.info()


# In[5]:


dset.isna().any()


# In[6]:


dset.isnull().sum()


# In[7]:


dset['wip'].fillna(0,inplace=True)
dset[0:11]


# In[8]:


dset['quarter'].unique()


# In[9]:


Q_dic = {'Quarter1':1, 'Quarter2':2, 'Quarter3':3, 'Quarter4':4, 'Quarter5':5}

# to assign the above change to the dataset in the quarter column
dset['quarter'] = dset['quarter'].map(Q_dic)


# In[10]:


dset[0:11]


# In[11]:


dset['department'].unique()


# In[12]:


dep_dic = {'sweing': 1, 'finishing': 2}
dset['department'] = dset['department'].map(dep_dic)


# In[13]:


dset.head()


# In[14]:


dset['day'].unique()


# In[15]:


day_dic = {'Saturday':1, 'Sunday':2, 'Monday':3, 'Tuesday':4, 'Wednesday':5, 'Thursday':6}

dset['day'] = dset['day'].map(day_dic)


# In[16]:


dset.dtypes


# In[17]:


dset = dset.drop('date', axis=1)
dset.head()


# In[18]:


# Data Visualisation
import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


x,y = plt.subplots(figsize=(12,9))
sns.heatmap(dset.corr(), cmap='YlGnBu', square=True, linewidth=.5, annot=True)

plt.show()


# In[20]:


dset['department'].value_counts().plot.pie()


# In[21]:


dset['actual_productivity'].value_counts()


# In[22]:


dset.hist(bins=50, figsize=(20,15))
plt.show()


# In[23]:


#feature selection
corr_matrix = dset.corr()
corr_matrix['no_of_workers'].sort_values(ascending = False)


# In[24]:


dset = dset.drop('department', axis=1)
dset.head()


# In[25]:


dset.shape


# In[26]:


# To solve outliers
q1 = dset.quantile(0.25)
q3 = dset.quantile(0.75)
IQR =q3 -q1
print(IQR)


# In[27]:


dset = dset[~((dset < (q1 - 1.5 * IQR)) | (dset > (q3 + 1.5*IQR))).any(axis=1)]
print(dset.shape)


# In[28]:


dset.columns


# In[29]:


#Min Max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(dset)
dset = pd.DataFrame(scaler.transform(dset), columns = ['quarter', 'department', 'day', 'team', 'smv', 'wip', 'over_time',
       'incentive', 'idle_time', 'idle_men', 'no_of_style_change',
       'no_of_workers', 'actual_productivity'])


# In[30]:


from sklearn.model_selection import train_test_split

X = dset.iloc[:,0:12].values
y = dset.iloc[:,12].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2 ,random_state=49)


# In[31]:


dset_labels = dset['actual_productivity']
dset = dset.drop('actual_productivity', axis=1)


# In[32]:


train_set_size = int(len(dset)*0.8)
train_set = dset[:train_set_size][:] # 80%
test_set = dset[train_set_size:][:] # 20%

train_lab = dset_labels[:train_set_size] # 80% labels y 
test_lab = dset_labels[train_set_size:] # 20% labels y
print(len(train_set), "train +", len(test_set), 'test')


# In[33]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_set, train_lab)


# In[34]:


from sklearn.metrics import mean_squared_error
import numpy as np
dset_predictions =lin_reg.predict(test_set)
lin_mse = mean =mean_squared_error(test_lab,dset_predictions)
lin_rmse = np.sqrt(lin_mse)


# In[35]:


lin_rmse


# In[36]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# In[37]:


dt = DecisionTreeRegressor(min_samples_leaf=0.1, random_state=49)
dt.fit(train_set, train_lab)


# In[38]:


#Predict test-set labels
y_pred = dt.predict(test_set)
mse_dt = mean_squared_error(test_lab, y_pred)
print("MSE: ",mse_dt)

#Compute test-set RMSE
rmse_dt = mse_dt ** (1/2)
print("RMSE: ",rmse_dt)


# In[39]:


from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


# In[41]:


#Predicting the test set result

y_pred= regressor.predict(X_test)
score_svc = mean_squared_error(y_pred,y_test)*100

print("The mse achieved using SVR is: "+str(score_svc)+" %")


# In[ ]:




