#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


import io
get_ipython().run_line_magic('cd', "'/Users/rajeshprabhakarkaila/Desktop/Imarticus/PGA 30/MPGAPP'")


# In[18]:


mpg_df=pd.read_csv("/Auto MPG Reg.csv")


# In[19]:


mpg_df.horsepower=mpg_df.horsepower.fillna(mpg_df.horsepower.median())


# In[20]:


y=mpg_df.mpg
X=mpg_df[['cylinders','displacement','horsepower','weight','acceleration','modelyear','origin']]


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


reg=LinearRegression()


# In[23]:


reg.fit(X,y)


# In[24]:


reg.score(X,y)


# In[25]:


import joblib


# In[26]:


joblib.dump(reg,"reg_model.sav")


# In[ ]:




