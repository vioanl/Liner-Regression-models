#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

MSE=mean_squared_error


# In[2]:


from sklearn.datasets import load_diabetes
diabetes=load_diabetes()
diabetes= pd.DataFrame(data= np.c_[diabetes['data'], diabetes['target']],
                     columns= diabetes['feature_names'] + ['target'])


# In[3]:


#χωρίζουμε σε σετ εκαπίδευσης και δοκιμής, τα π΄ρωτα 300 δειγματα για εκπαίδευση

x_diabetes = diabetes.iloc[:, 0:10]
y_diabetes = diabetes.iloc[:, 10]

x_train = x_diabetes.loc[0:299,:]
x_test = x_diabetes.loc[300:442,:]

y_train = y_diabetes.loc[0:299]
y_test = y_diabetes.loc[300:442]



# In[4]:
#1. linear model στο σετ εκπαίδευσης
lr=LinearRegression()
scores = cross_val_score(lr, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[5]:


#ridge μοντέλο στο σετ εκπαίδευσης
ridge=Ridge()
scores = cross_val_score(ridge, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[7]:
# lasso μοντέλο 
lasso=Lasso()
scores = cross_val_score(lasso, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
scores.mean()

# In[9]:


#svm(rbf) model 
svm_rbf=svm.SVR(kernel='rbf')

C_range=[]
gamma_range=[]

for i in range(-5, 16):
    C_range.append(2**i)
    gamma_range.append(2**i)

param_grid= dict(C=C_range, gamma=gamma_range)
grid_rbf=GridSearchCV(svm_rbf, param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True, verbose=1)
grid_rbf.fit(x_train, y_train)
print(grid_rbf.best_score_)
print(grid_rbf.best_params_)


# In[10]:


#svm(poly) μοντέλο 
C_range=[0.1, 1, 10, 100, 1000]
gamma_range=[0.1, 0.01, 0.001, 0.0001, 0.00001]
svm_poly=svm.SVR(kernel='poly')

param_grid= dict(C=C_range, gamma=gamma_range)
grid_poly=GridSearchCV(svm_poly, param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True, verbose=1)
grid_poly.fit(x_train, y_train)

print(grid_poly.best_score_)
print(grid_poly.best_params_)
scores = cross_val_score(svm_poly, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[11]:


#random forest
regressor=RandomForestRegressor(n_jobs=-1)

regressor.fit(x_train, y_train)
print(regressor.feature_importances_)
param_grid= {'max_depth': range(3,7),'n_estimators': (10, 20, 50, 100, 1000)}
grid_regressor=GridSearchCV(regressor, 
                            param_grid, 
                            cv=10, 
                            scoring="neg_mean_squared_error", 
                            n_jobs=-1, 
                            return_train_score=True, 
                            verbose=1)
grid_regressor.fit(x_train, y_train)
print(grid_regressor.best_score_)
print(grid_regressor.best_params_)


# In[12]:


#2.2   εξετάζουμε την επίδοση του καλύτερου μοντέλου στο σύνολο αξιολόγησης
y_pred=grid_rbf.predict(x_test)
MSE(y_test,y_pred)


# 

# In[13]:


#2.3 διαδικασία για δημιουργία μοντέλου σε άγνωστα δείγματα
# Linear Regression
scores = cross_val_score(lr, x_diabetes, y_diabetes, cv=10, scoring='neg_mean_squared_error')
scores.mean()

# In[19]:


#πινακας συσχετισης
import pandas as pd
import numpy as np

rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = df.corr()
corr.style.background_gradient()


# In[14]:


# Ridge Linear Regression
scores = cross_val_score(ridge, x_diabetes, y_diabetes, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[15]:


# Lasso Linear Regression
scores = cross_val_score(lasso, x_diabetes, y_diabetes, cv=10, scoring='neg_mean_squared_error')
scores.mean()


# In[16]:


#rbf
C_range=[]
gamma_range=[]

for i in range(-5, 16):
    C_range.append(2**i)
    gamma_range.append(2**i)
    
param_grid= dict(C=C_range, gamma=gamma_range)
grid_rbf=GridSearchCV(svm_rbf, param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True, verbose=1)

grid_rbf.fit(x_diabetes, y_diabetes)

print(grid_rbf.best_score_)
print(grid_rbf.best_params_)


# In[17]:


#poly
C_range=[0.1, 1, 10, 100, 1000]
gamma_range=[0.1, 0.01, 0.001, 0.0001, 0.00001]

param_grid= dict(C=C_range, gamma=gamma_range)
grid_poly=GridSearchCV(svm_poly, param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True, verbose=1)

grid_poly.fit(x_diabetes, y_diabetes)

print(grid_poly.best_score_)
print(grid_poly.best_params_)


# In[18]:

#forest
param_grid= {'max_depth': range(3,7),'n_estimators': (10, 20, 50, 100, 1000)}

grid_regressor=GridSearchCV(regressor, 
                            param_grid, 
                            cv=10, 
                            scoring="neg_mean_squared_error", 
                            n_jobs=-1, 
                            return_train_score=True, 
                            verbose=1)
grid_regressor.fit(x_diabetes, y_diabetes)
print(grid_regressor.best_score_)
print(grid_regressor.best_params_)


# In[ ]:




