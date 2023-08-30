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
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from model_preprocessing import hcv_df_binary_health, hcv_df_binary_health_outliers

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

""" 
1) Support Vector Machine (SVM)
"""

# Creating the pipeline object for data normalization and model fitting
svm_pipe = make_pipeline(StandardScaler(), svm.SVC())

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Fitting the pipeline
svm_pipe.fit(X_train, y_train)

# Establishing the SVM accuracy score
svm_accuracy_score = accuracy_score(svm_pipe.predict(X_test), y_test)

# Because our dataset is imbalanced, we'll also get the f1 score
svm_f1_score = f1_score(y_test, svm_pipe.predict(X_test), pos_label="Hepatitis")

# -----------------------------------------------------------------------------

"""
At this point I realize how our outlier removal step eliminated a signficantly
larger proportion of hepatitis patients than donor patients. This leads me to
believe that these 'outliers' weren't actually outliers; the value disparities
was most likely due to hcv and its progression.
"""

# Trying with the 'outlier' data points

hcv_df_outliers = hcv_df_binary_health_outliers

# Extracting the X and y columns and converting to np arrays
data_df_outliers = hcv_df_outliers[['Health_Status', 'ALB', 'ALP', 'ALT', 'AST', 
                                    'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]

X_out = data_df_outliers[['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 
                          'CREA', 'GGT', 'PROT']]
X_out = X_out.to_numpy()

y_out = data_df_outliers["Health_Status"]
y_out = y_out.to_numpy()

# -----------------------------------------------------------------------------

""" 
2) Support Vector Machine (SVM) with 'Outliers'
"""

# Creating the pipeline object for data normalization and model fitting
svm_pipe_out = make_pipeline(StandardScaler(), svm.SVC())

# Splitting data into training and testing sets
X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X_out, y_out, random_state=0)

# Fitting the pipeline
svm_pipe_out.fit(X_train_out, y_train_out)

# Establishing the SVM accuracy score
svm_accuracy_score_out = accuracy_score(svm_pipe_out.predict(X_test_out), y_test_out)

# Because our dataset is imbalanced, we'll also get the f1 score
svm_f1_score_out = f1_score(y_test_out, svm_pipe_out.predict(X_test_out), pos_label="Hepatitis")

# -----------------------------------------------------------------------------

""" 
(AMMENDED) - See f1 score section
We notice that the accuracy score for the df with 'outliers' still remains
slightly higher:
    
Without outliers: 0.9814814814814815
With outliers: 0.9797297297297297
"""

"""
Another thing we failed to consider prior to model creation was how our
data is imbalanced. Our df without outliers has 422 donor data points and 10
hepatitis data points. Our df with outliers has 533 donor data points and 56
hepatitis data points.
"""

# -----------------------------------------------------------------------------

""" 
3) Support Vector Machine (SVM) without 'Outliers' and addressed class imbalance.
The weight of 1 and 42.2 for donor and hepatitis data points is derived from
the ratio of donor:hepatitis patients, or 10:422.
"""

# Creating the pipeline object for data normalization and model fitting
# The 'class_weight' parameter for SVC is utilized to address the imbalanced class
svm_pipe_balanced = make_pipeline(StandardScaler(), svm.SVC(class_weight={"Donor": 1,
                                                                          "Hepatitis": 42.2}))

# Splitting data into training and testing sets
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X, y, random_state=0)

# Fitting the pipeline
svm_pipe_balanced.fit(X_train_bal, y_train_bal)

# Establishing the SVM accuracy score
svm_accuracy_score_bal = accuracy_score(svm_pipe_balanced.predict(X_test_bal), y_test_bal)

# Because our dataset is imbalanced, we'll also get the f1 score
svm_f1_score_bal = f1_score(y_test_bal, svm_pipe_balanced.predict(X_test_bal), pos_label="Hepatitis")

""" 
(AMMENDED) - See f1 score section
The accuracy score for the SVM model with outliers and the addressed class
imbalance is 0.9907407407407407, and improvement from the original, unbalanced
model.
"""

"""
It was at this point that I decided to review which evaluation of model
performance was most accurate for our dataset. While an accuracy score provides
an intuitive score for how well our model predicted values, using f1 scores 
seems to be the better option for assessing how well our model performs. This 
is due to two reasons: f1 scores are more apporopriate for imbalanced datasets,
and we care more about how well our model does in terms of true and false
positives for hepatitis. While predicting donor patients is somewhat useful,
the aim of this study is to accurately identify hepatitis patients.

f1 scores:
    1) SVM without outliers, imbalanced: 0.5
    2) SVM with outliers, imbalanced: 0.888888888888888
    3) SVM without outliers, balanced: 0.8
"""

"""
As we can see, the f1 score for our model that the outliers included, as well
as no class weights (imbalanced) performed the best. This brings back the question
of if the outliers are actually outliers/are they important for identifying
health status. To answer this questions, we will once again train another variant
of the SVM model, this time with outliers and class weights included.
"""

# -----------------------------------------------------------------------------

""" 
4) Support Vector Machine (SVM) with 'outliers' included and class weights 
added due to the imbalanced ratio of donor:hepatitis data points.
"""

# Creating the pipeline object for data normalization and model fitting
# The 'class_weight' parameter for SVC is utilized to address the imbalanced class
svm_pipe_balanced_out = make_pipeline(StandardScaler(), svm.SVC(class_weight={"Donor": 1,
                                                                          "Hepatitis": 42.2}))

# Splitting data into training and testing sets
X_train_bal_out, X_test_bal_out, y_train_bal_out, y_test_bal_out = train_test_split(X_out, y_out, random_state=0)

# Fitting the pipeline
svm_pipe_balanced_out.fit(X_train_bal_out, y_train_bal_out)

# Establishing the SVM accuracy score
svm_accuracy_score_bal_out = accuracy_score(svm_pipe_balanced_out.predict(X_test_bal_out), y_test_bal_out)

# Because our dataset is imbalanced, we'll also get the f1 score
svm_f1_score_bal_out = f1_score(y_test_bal_out, svm_pipe_balanced_out.predict(X_test_bal_out), pos_label="Hepatitis")

"""
Updated f1 scores:
    1) SVM without outliers, imbalanced: 0.5
    2) SVM with outliers, imbalanced: 0.888888888888888
    3) SVM without outliers, balanced: 0.8
    4) SVM with outliers, balanced: 0.9285714285714286

No we can see that the last model performs best in terms of f1 scores.
"""

# -----------------------------------------------------------------------------

"""
Hyper-parameter Optimization using GridSearchCV
"""

# Creating our scoring object that uses f1_score and Hepatitis as the positive label
f1_scorer = make_scorer(f1_score, pos_label="Hepatitis")

# Extracting the estimator from our chosen pipeline
svm_optimal = svm_pipe_balanced_out[1]

# Parameter grid that GridSearchCV will go through to optimize
parameters = {"C": [0.1, 1, 10, 100], "kernel": ["poly", "rbf", "sigmoid"],
              "gamma": [0.01, 0.001, 0.0001]}

svm_grid = GridSearchCV(svm_optimal, parameters, refit=True, scoring=f1_scorer)
svm_grid.fit(X_train_bal_out, y_train_bal_out)

# The .best_estimator_ attribute returns the optimal hyper-parameters
optimal_params = svm_grid.best_estimator_

# Generating the f1 score from our model with optimized parameters
svm_optimal_f1 = f1_score(y_test_bal_out, svm_grid.predict(X_test_bal_out), 
                          pos_label="Hepatitis")

"""
GridSearchCV returns the following hyper-parameters as being optimal:
C=10, class_weight={'Donor': 1, 'Hepatitis': 42.2}, gamma=0.0001, kernel='poly')

Using the optimized params, we get the following f1 score:
    f1 score = 0.9600000000000001
"""

