#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:32:22 2022

@author: akash
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV


df = pd.read_csv(r"/media/akash/5E820BDA820BB595/CDAC ACTS/Machine Learning/Cases/Medical Cost Personal/insurance.csv")
dum_df = pd.get_dummies(df,drop_first=True)
X = dum_df.drop(['charges'],axis=1)
y = dum_df['charges']

############## LINEAR REGRESSION ######################
lin_reg = LinearRegression()
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
results = cross_val_score(lin_reg, X,y,cv=kfold)
print(results.mean())



##################### RIDGE RANDOMIZED REGRESSION ##############
parameters =  dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2])
rdf = Ridge()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
cv_rd = GridSearchCV(estimator=rdf, param_grid=parameters,cv=kfold,scoring='r2')
cv_rd.fit(X,y)
pd_cv = pd.DataFrame(cv_rd.cv_results_)
print(cv_rd.best_params_)
print(cv_rd.best_score_)

################ LASSO RANDOMIZED REGRESSION #############
parameters =  dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2])
lsf = Lasso()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
cv_lsf = GridSearchCV(estimator=lsf, param_grid=parameters,cv=kfold,scoring='r2')
cv_lsf.fit(X,y)
pd_cv = pd.DataFrame(cv_lsf.cv_results_)
print(cv_lsf.best_params_)
print(cv_lsf.best_score_)

##################### ELASTIC NET RANDMIZED SEARCH ###########
parameters =  dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2],
                   l1_ratio=[0.001,0.5,0.7,1])

clf = ElasticNet()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
cv_en = GridSearchCV(estimator=clf, param_grid=parameters,cv=kfold,scoring='r2')

cv_en.fit(X,y)
pd_cv = pd.DataFrame(cv_en.cv_results_)
print(cv_en.best_params_)
print(cv_en.best_score_)

################## POLYNIMIAL REGRESSION ################
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
results = cross_val_score(poly_reg,X_poly,y,cv=kfold)
print(results.mean())

##################### RIDGE RANDOMIZED REGRESSION ##############
parameters =  dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2])
rdf = Ridge()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
cv_rd = GridSearchCV(estimator=rdf, param_grid=parameters,cv=kfold,scoring='r2')
cv_rd.fit(X_poly,y)
pd_cv = pd.DataFrame(cv_rd.cv_results_)
print(cv_rd.best_params_)
print(cv_rd.best_score_)

################ LASSO RANDOMIZED REGRESSION #############
parameters =  dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2])
lsf = Lasso()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
cv_lsf = GridSearchCV(estimator=lsf, param_grid=parameters,cv=kfold,scoring='r2')
cv_lsf.fit(X_poly,y)
pd_cv = pd.DataFrame(cv_lsf.cv_results_)
print(cv_lsf.best_params_)
print(cv_lsf.best_score_)

##################### ELASTIC NET RANDMIZED SEARCH ###########
parameters =  dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2],
                   l1_ratio=[0.001,0.5,0.7,1])

clf = ElasticNet()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
cv_en = GridSearchCV(estimator=clf, param_grid=parameters,cv=kfold,scoring='r2')

cv_en.fit(X_poly,y)
pd_cv = pd.DataFrame(cv_en.cv_results_)
print(cv_en.best_params_)
print(cv_en.best_score_)

################## POLYNIMIAL REGRESSION ################
poly = PolynomialFeatures(degree=3)
X1_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
results = cross_val_score(poly_reg,X_poly,y,cv=kfold)
print(results.mean())

##################### RIDGE RANDOMIZED REGRESSION ##############
parameters =  dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2])
rdf = Ridge()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
cv_rd = GridSearchCV(estimator=rdf, param_grid=parameters,cv=kfold,scoring='r2')
cv_rd.fit(X1_poly,y)
pd_cv = pd.DataFrame(cv_rd.cv_results_)
print(cv_rd.best_params_)
print(cv_rd.best_score_)

################ LASSO RANDOMIZED REGRESSION #############
parameters =  dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2])
lsf = Lasso()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
cv_lsf = GridSearchCV(estimator=lsf, param_grid=parameters,cv=kfold,scoring='r2')
cv_lsf.fit(X1_poly,y)
pd_cv = pd.DataFrame(cv_lsf.cv_results_)
print(cv_lsf.best_params_)
print(cv_lsf.best_score_)

##################### ELASTIC NET RANDMIZED SEARCH ###########
parameters =  dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2],
                   l1_ratio=[0.001,0.5,0.7,1])

clf = ElasticNet()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
cv_en = GridSearchCV(estimator=clf, param_grid=parameters,cv=kfold,scoring='r2')

cv_en.fit(X1_poly,y)
pd_cv = pd.DataFrame(cv_en.cv_results_)
print(cv_en.best_params_)
print(cv_en.best_score_)

#########################Test data prediction using train lasso################################


df_tst = pd.read_csv(r"/media/akash/5E820BDA820BB595/CDAC ACTS/Machine Learning/Cases/Medical Cost Personal/tst_insure.csv")
dum_df_tst = pd.get_dummies(df_tst,drop_first=True)
X_tst_poly=poly.transform(dum_df_tst)

best_model=cv_en.best_estimator_
y_pred1=best_model.predict(X_tst_poly)
