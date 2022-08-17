#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


# In[4]:


df=pd.read_csv("/Users/sayanbanerjee/Downloads/Data analysis projects/Machine learning projects/Column transformer/covid_toy.csv")


# In[5]:


df.head()


# In[13]:


df['cough'].unique()


# In[6]:


df.isnull().sum()


# In[48]:


x_train_1=df.drop(['has_covid'], axis=1)
y_train_1=df['has_covid']


# In[49]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x_train_1, y_train_1, test_size=0.2)


# In[50]:


x_train_1.head()


# In[42]:


df.city.unique()


# # 1. Normal method

# In[51]:


# Adding simple imputer to fever col to fill the missing value
si=SimpleImputer()
x_train_fever=si.fit_transform(x_train[['fever']])
x_test_fever=si.fit_transform(x_test[['fever']])


# In[52]:


x_train_fever.shape


# In[53]:


# ordinal encoding -> cough
oe=OrdinalEncoder(categories=[['Mild','Strong']])
x_train_cough=oe.fit_transform(x_train[['cough']])
x_test_cough=oe.fit_transform(x_test[['cough']])


# In[54]:


# One hot encoding
ohe=OneHotEncoder(drop='first', sparse=False)
x_train_gender_city=ohe.fit_transform(x_train[['gender','city']])
x_test_gender_city=ohe.fit_transform(x_test[['gender','city']])


# In[55]:


x_train_gender_city


# In[56]:


# Extracting age
x_train_age=x_train.drop(['gender','fever','cough','city'], axis=1)
x_test_age=x_test.drop(['gender','fever','cough','city'], axis=1)


# In[34]:


x_train_age.shape


# In[35]:


x_train_age.head()


# In[36]:


x_train_transformed=np.concatenate((x_train_age ,x_train_fever, x_train_gender_city, x_train_cough), axis=1)
x_test_transformed=np.concatenate((x_test_age, x_test_fever, x_test_gender_city, x_test_cough), axis=1)


# In[46]:


x_train_trans= pd.DataFrame(x_train_transformed, columns=['Age','Fever','Gender','Delhi', 'Mumbai',' Bangalore','Strong'])


# In[47]:


x_train.head()


# # Using column transformer

# In[45]:


from sklearn.compose import ColumnTransformer


# In[59]:


transformer=ColumnTransformer(transformers=[
    ('tran1', SimpleImputer(), ['fever']),
    ('tran2', OrdinalEncoder(categories=[['Mild','Strong']]), ['cough'] ),
    ('tran3', OneHotEncoder(sparse=False, drop='first'), ['gender','city'])
], remainder='passthrough')


# In[62]:


x_train_updated=transformer.fit_transform(x_train)


# In[61]:


transformer.transform(x_test).shape


# In[63]:


x_train_updated


# In[ ]:




