# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:45:46 2023

@author: Reece
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

from model_preprocessing import hcv_df_binary_health

hcv_df = hcv_df_binary_health

# Extracting the X and y columns and converting to np arrays
data_df = hcv_df[['Health_Status', 'ALB', 'ALP', 'ALT', 'AST', 'BIL',
                  'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]

X = data_df[['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 
             'PROT']]
X = X.to_numpy()

y = data_df["Health_Status"]
y = y.to_numpy()

# -----------------------------------------------------------------------------
"""Support Vector Machine (SVM)"""

# Creating the pipeline object for data normalization and model fitting
svm_pipe = make_pipeline(StandardScaler(), svm.SVC())

# Splitting data into training and testing sets
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y, random_state=0)

# Fitting the pipeline
svm_pipe.fit(X_train_svm, y_train_svm)

# Establishing the SVM accuracy score
svm_accuracy_score = accuracy_score(svm_pipe.predict(X_test_svm), y_test_svm)

# -----------------------------------------------------------------------------

"""At this point I realize how our outlier removal step eliminated a signficantly
larger proportion of hepatitis patients than donor patients. This leads me to
believe that these 'outliers' weren't actually outliers; the value disparities
was most likely due to hcv and its progression."""