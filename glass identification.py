#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:37:25 2022

@author: akash
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("/media/akash/5E820BDA820BB595/CDAC ACTS/Machine Learning/Cases/Glass Identification/Glass.csv")
dum_df = pd.get_dummies(df, drop_first=True)
param = dum_df.iloc[:,:-1]
result = dum_df.iloc[:,-1]

param_train, param_test, result_train, result_test = train_test_split(param,result,test_size=0.3, 
                                                                      random_state=2022, 
                                                                      stratify=result)

mul=GaussianNB()
mul.fit(param_train,result_train)

result_prob = mul.predict_proba(param_test)
result_pred = mul.predict(param_test)
result_pred_proba = mul.predict_proba(param_test)
print(confusion_matrix(result_test, result_pred))
print(classification_report(result_test, result_pred))
print(accuracy_score(result_test, result_pred))
print(roc_auc_score(result_test,result_pred_proba, multi_class='ovr'))
