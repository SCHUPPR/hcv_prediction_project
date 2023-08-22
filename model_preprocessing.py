# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:08:06 2023

@author: Reece
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
sns.set_theme()

"""Preprocessing of Data for Model Creation"""

# Loading cleaned dataset
hcv_df = pd.read_csv("hcv_data_clean.csv")

# Lab Result Columns:
lab_tests = ['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# Examining the distributions of the lab test results
# for test in lab_tests:
#     fig, ax = plt.subplots()
#     hcv_df[test].plot.hist(title=test)

"""The lab result distributions indicate that there are a quite a few outliers
in the dataset. Many of the plots appear to have a lot of empty space on either
the right or left hand side, this is due to an outlier. No we'll attempt to 
isolate and remove the outliers."""

# To identify outliers we will use the IQR method

def get_outlier_fences(df, columns):
    """Generate the lower and upper fences for outlier detection using the 
    IQR method.
    Input ---
        df: pandas df, the dataframe in which to replace outliers
        columns: list, list of columns in which to identify and replace outliers
    Output ---
        col_fences: list, list of tuples, the respective index matching the
            lab test index"""
    col_fences = []
    for col in columns:
        first_q = np.quantile(df[col], 0.25)
        third_q = np.quantile(df[col], 0.75)
        iqr = third_q - first_q
        lower_fence = first_q - (1.5 * iqr)
        upper_fence = third_q + (1.5 * iqr)
        col_fences.append((lower_fence, upper_fence))
    return col_fences
 
   
column_fences = get_outlier_fences(hcv_df, lab_tests)

# Replacing outliers with NaN values
outlier_alb = lambda x: np.nan if (x < column_fences[0][0]) or (x > column_fences[0][1]) else x
hcv_df["ALB"] = hcv_df["ALB"].apply(outlier_alb)
outlier_alp = lambda x: np.nan if (x < column_fences[1][0]) or (x > column_fences[1][1]) else x
hcv_df["ALP"] = hcv_df["ALP"].apply(outlier_alp)
outlier_alt = lambda x: np.nan if (x < column_fences[2][0]) or (x > column_fences[2][1]) else x
hcv_df["ALT"] = hcv_df["ALT"].apply(outlier_alt)
outlier_ast = lambda x: np.nan if (x < column_fences[3][0]) or (x > column_fences[3][1]) else x
hcv_df["AST"] = hcv_df["AST"].apply(outlier_ast)
outlier_bil = lambda x: np.nan if (x < column_fences[4][0]) or (x > column_fences[4][1]) else x
hcv_df["BIL"] = hcv_df["BIL"].apply(outlier_bil)
outlier_che = lambda x: np.nan if (x < column_fences[5][0]) or (x > column_fences[5][1]) else x
hcv_df["CHE"] = hcv_df["CHE"].apply(outlier_che)
outlier_chol = lambda x: np.nan if (x < column_fences[6][0]) or (x > column_fences[6][1]) else x
hcv_df["CHOL"] = hcv_df["CHOL"].apply(outlier_chol)
outlier_crea = lambda x: np.nan if (x < column_fences[7][0]) or (x > column_fences[7][1]) else x
hcv_df["CREA"] = hcv_df["CREA"].apply(outlier_crea)
outlier_ggt = lambda x: np.nan if (x < column_fences[8][0]) or (x > column_fences[8][1]) else x
hcv_df["GGT"] = hcv_df["GGT"].apply(outlier_ggt)
outlier_prot = lambda x: np.nan if (x < column_fences[9][0]) or (x > column_fences[9][1]) else x
hcv_df["PROT"] = hcv_df["PROT"].apply(outlier_prot)

# Counting the number of NaN's to validate lambda functions
# print(hcv_df.isna().sum())

hcv_no_outliers = hcv_df.dropna()
# print(hcv_no_outliers.shape)

"""The new shape of the dataframe without outliers is (432,14) which is still
relatively large, large enough to justify removing outliers."""

# Examining the lab test distributions again:
# for test in lab_tests:
#     fig, ax = plt.subplots()
#     hcv_no_outliers[test].plot.hist(title=test)

"""The distributions no longer have large empty spaces, again validating
that the outliers were removed from the dataset."""

