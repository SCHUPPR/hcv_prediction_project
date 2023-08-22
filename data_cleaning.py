# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:15:55 2023

@author: Reece
"""

import pandas as pd

pd.set_option('display.max_columns', None)

# Data Cleaning

from load_data import raw_data

# Establishing df columns
raw_columns = raw_data.columns
# 'Unnamed: 0', 'Category', 'Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST',
# 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'

# Establishing dimensions
raw_dim = raw_data.shape
# (615, 14)

"""Handling Missing Data"""

# Establishing number of NaN's in each column
raw_nans = raw_data.isna().sum()
# ALP has 18 NaN's and CHOL has 10, the two largest amount of NaN's in a column
# Based on the number of rows, 615, removing NaN rows should not hurt data
# analysis

# Dropping all rows with NaN values
raw_data = raw_data.dropna()

# Establishing the new df dimensions
raw_dim = raw_data.shape
# (589, 14)

# Establishing that all NaN rows were removed
raw_nans = raw_data.isna().sum()
# All columns sums are 0, all NaN's have been removed

# Establishing if there are duplicate entries
raw_duplicates = raw_data[raw_data.duplicated()]
# No duplicate rows


"""Fixing structural characteristics such as column names, dtypes, etc."""

# Renaming columns
raw_data = raw_data.rename(columns={"Unnamed: 0": "Patient_ID", "Category": "Health_Status"})

# Health_Status unique values
health_status_uniques = raw_data["Health_Status"].unique()
# '0=Blood Donor' '0s=suspect Blood Donor' '1=Hepatitis' '2=Fibrosis' '3=Cirrhosis'

# Per the dataset description, the health status/diagnosis can be grouped into
# hepatitis and healthy/donor patients. We'll map values to better describe
# the patient.
raw_data["Health_Status"] = raw_data["Health_Status"].map({"0=Blood Donor": "Donor",
                                                           "0s=suspect Blood Donor": "Donor",
                                                           "1=Hepatitis": "Hepatitis",
                                                           "2=Fibrosis": "Fibrosis",
                                                           "3=Cirrhosis": "Cirrhosis"})
# There's no provided definition for how a 'donor' patient differs from a 
# 'suspect donor' patient, so these values are combined

clean_data = raw_data

clean_data.to_csv("hcv_data_clean.csv", index=False)