#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:55:06 2022

@author: akash
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
df = pd.read_csv(r"/media/akash/5E820BDA820BB595/CDAC ACTS/Machine Learning/Cases/Concrete Strength/Concrete_Data.csv")

#################################################mean###################################################


X = df.iloc[:,0:8]
y = df.iloc[:,8]

##########Grid Search#######
params= dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2]  )
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
clf=Ridge()
cv_scale=GridSearchCV(estimator=clf, param_grid=params,scoring='r2',cv=kfold)
cv_scale.fit(X,y)
pd_cv = pd.DataFrame(cv_scale.cv_results_)
print(cv_scale.best_params_)
print(cv_scale.best_score_)