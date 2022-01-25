#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 12:43:32 2022

@author: akash
"""

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("/media/akash/5E820BDA820BB595/CDAC ACTS/Machine Learning/Cases/Bankruptcy/Bankruptcy.csv")
dum_df = pd.get_dummies(df, drop_first=True)
param = dum_df.iloc[:,2:]
result = dum_df.iloc[:,1]

np_param = param.values
np_result = result.values

params = {'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25]}
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
knn=KNeighborsClassifier()
cv = GridSearchCV(estimator=knn, param_grid=params, scoring = 'roc_auc', cv=kfold)
cv.fit(np_param, np_result)
pd.cv = pd.DataFrame(cv.cv_results_)

print(cv.best_params_)
print(cv.best_score_)
 