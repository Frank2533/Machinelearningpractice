#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:12:33 2022

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
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"/media/akash/5E820BDA820BB595/CDAC ACTS/Machine Learning/Cases/Chemical Process Data/ChemicalProcess.csv")
df
#################################################mean###################################################
impt= SimpleImputer(strategy='mean')
imputed_data=impt.fit_transform(df)

X = imputed_data[:,1:]
y = imputed_data[:,0]
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)

lin_reg=LinearRegression()

results = cross_val_score(lin_reg, X, y, scoring = "r2_score",cv=kfold)
results.mean(results)
print(lin_reg.coef_)
print(lin_reg.intercept_)
y_pred= lin_reg.predict(X_test)
print(r2_score(y_test, y_pred))
##########Grid Search#######
params={'n_neighbors':np.arange(1,21)}
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
knn= KNeighborsRegressor()
cv_scale=GridSearchCV(estimator=knn, param_grid=params,scoring='r2',cv=kfold)
cv_scale.fit(scaled_X,y)
pd_cv = pd.DataFrame(cv_scale.cv_results_)
print(cv_scale.best_params_)
print(cv_scale.best_score_)



#################################################median###################################################



df = pd.read_csv(r"/media/akash/5E820BDA820BB595/CDAC ACTS/Machine Learning/Cases/Chemical Process Data/ChemicalProcess.csv")
df
# by median
impt= SimpleImputer(strategy='median')
imputed_data=impt.fit_transform(df)

X = imputed_data[:,1:]
y = imputed_data[:,0]

scaler=StandardScaler()
scaled_X=scaler.fit_transform(X)
##########Grid Search#######
params={'n_neighbors':np.arange(1,21)}
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
knn= KNeighborsRegressor()
cv_scale=GridSearchCV(estimator=knn, param_grid=params,scoring='r2',cv=kfold)
cv_scale.fit(scaled_X,y)
pd_cv = pd.DataFrame(cv_scale.cv_results_)
print(cv_scale.best_params_)
print(cv_scale.best_score_)