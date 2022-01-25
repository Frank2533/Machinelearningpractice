#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:23:32 2022

@author: akash
"""

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("/media/akash/5E820BDA820BB595/CDAC ACTS/Machine Learning/Cases/Glass Identification/Glass.csv")
dum_df = pd.get_dummies(df, drop_first=True)
param = dum_df.iloc[:,:-1]
result = dum_df.iloc[:,-1]

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results = cross_val_score(GaussianNB(), param, result, scoring = "roc_auc_ovr",cv=kfold)
print(results.mean())
results1 = cross_val_score(GaussianNB(), param, result, scoring = "roc_auc_ovo",cv=kfold)
print(results1.mean())
