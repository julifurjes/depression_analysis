from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTENC
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreparation():

    def __init__(self) -> None:
        pass
        
    def standardize_variables(self, X):
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Exclude columns starting with 'Score' and the 'Time' column
        numeric_cols = numeric_cols[~numeric_cols.str.startswith('Score')]
        numeric_cols = numeric_cols.drop('Time', errors='ignore')

        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        return X

    def variable_screening(self, fixed_effects, df):
        X = df[fixed_effects]
        y = df['DepressivePathGroup']

        # Drop non-numeric columns (e.g., Timestamp columns)
        X = X.select_dtypes(include=[np.number])

        # Standardize variables
        X = self.standardize_variables(X)

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Use mutual information for feature selection
        selector = SelectKBest(mutual_info_classif, k=20)
        selector.fit(X, y)
        selected_vars = list(X.columns[selector.get_support()])

        # List of variables to ensure are included
        required_vars = ['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']

        # Manually add required variables to selected_vars if they are in X.columns but not in selected_vars
        for var in required_vars:
            if var in X.columns and var not in selected_vars:
                selected_vars.append(var)

        X_selected = X[selected_vars]
        
        return selected_vars, X_selected, y