
import argparse
import json
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================================
# STEP 1: HANDLE MISSING VALUES & DUPLICATES
# ============================================================================

def replace_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace common missing value representations with NaN.
    
    Example:
        >>> df = pd.DataFrame({'A': ['NA', '1.5', 'N/A']})
        >>> df = replace_missing_values(df)
        >>> df['A'].isna().sum()
        2
    
    This function is PROVIDED as an example.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Replace common missing value strings with np.nan
    missing_values = ["NA", "N/A", "na", "n/a", "NaN", "nan", ""]
    df = df.replace(missing_values, np.nan)
    
    return df


def remove_duplicates(df: pd.DataFrame, id_cols: List[str]) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows, keeping id_cols unchanged.
    
    Returns:
        (cleaned_df, num_removed)
    
    Example:
        >>> df = pd.DataFrame({'id': [1,1,2], 'value': [10,10,20]})
        >>> df_clean, n = remove_duplicates(df, id_cols=['id'])
        >>> len(df_clean)
        2
    
    This function is PROVIDED as an example.
    """
    # Count how many duplicates we have
    num_duplicates = df.duplicated().sum()
    
    # Remove duplicate rows (keeps first occurrence)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    return df_clean


# ============================================================================
# STEP 2: IDENTIFY FEATURE TYPES
# ============================================================================

def detect_feature_types(df: pd.DataFrame, target: str, id_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Identify which columns are categorical vs numeric features.
    
    Example:
        >>> df = pd.DataFrame({'age': [25, 30], 'city': ['NYC', 'LA'], 'target': [0, 1]})
        >>> cat, num = detect_feature_types(df, target='target', id_cols=[])
        >>> cat
        ['city']
        >>> num
        ['age']
    """
    # TODO: Implement feature type detection
    # 1. Get all columns except target and id_cols:
    #    feature_cols = [c for c in df.columns if c not in id_cols and c != target]
    # 2. Identify categorical columns (dtype == 'object'):
    #    cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
    # 3. Identify numeric columns (dtype in [int, float]):
    #    num_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
    # 4. Return (cat_cols, num_cols)

    feature_cols = [c for c in df.columns if c not in id_cols and c != target]
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
    num_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
    return cat_cols


# ============================================================================
# STEP 3: ENCODE CATEGORICAL COLUMNS
# ============================================================================

def encode_categorical(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical columns.
    
    Returns:
        (df_encoded, encoded_column_names)
    
    Example:
        >>> df = pd.DataFrame({'color': ['red', 'blue', 'red']})
        >>> df_enc, cols = encode_categorical(df, ['color'])
        >>> df_enc.columns.tolist()
        ['color_blue', 'color_red']
    
    IMPORTANT: This function is called separately on train and test data in run_preprocessing().
    You must ensure they produce the SAME columns. If test has a category not in train, 
    either exclude it or handle missing columns when encoding test.
    """
    # TODO: Implement one-hot encoding
    # 1. Create a copy of the dataframe
    # 2. For each column in cat_cols:
    #    a. Use pd.get_dummies(df[col], prefix=col, dtype=int) to one-hot encode
    #    b. Drop the original column: df.drop(col, axis=1, inplace=True)
    #    c. Add the new encoded columns: df = pd.concat([df, encoded], axis=1)
    # 3. Keep track of all new column names created
    # 4. Return (df_with_encoded_cols, list_of_encoded_column_names)
    #
    # HINT: When called in run_preprocessing(), you encode TRAIN first to get column names,
    # then when encoding TEST, you should only create those same columns (don't add new ones).
    # You can use pd.get_dummies(..., columns=...) or post-process to match columns.
    df_with_encoded_cols = df.copy()
    list_of_encoded_column_names = []
    for col in cat_cols:
        encoded = pd.get_dummies(df_with_encoded_cols[col], prefix=col, dtype=int)
        list_of_encoded_column_names.extend(encoded.columns.tolist())
        df_with_encoded_cols.drop(col, axis=1, inplace=True)
        df_with_encoded_cols = pd.concat([df_with_encoded_cols, encoded], axis=1)
    return df_with_encoded_cols


# ============================================================================
# STEP 4: SCALE NUMERIC COLUMNS
# ============================================================================

def scale_numeric(df: pd.DataFrame, num_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Standardize numeric columns (mean=0, std=1).
    
    Returns:
        (df_scaled, means_dict, stds_dict)
    
    Example:
        >>> df = pd.DataFrame({'age': [20, 30, 40]})
        >>> df_scaled, means, stds = scale_numeric(df, ['age'])
        >>> abs(df_scaled['age'].mean()) < 0.001
        True
    """
    # TODO: Implement numeric scaling
    # 1. Create a copy of the dataframe
    # 2. For each column in num_cols:
    #    a. Handle missing values first: col.fillna(col.median())
    #    b. Calculate mean and std: mean = col.mean(), std = col.std()
    #    c. Standardize: (col - mean) / std
    # 3. Return (scaled_df, means_dict, stds_dict)
    
    means_dict = {}
    stds_dict = {}
    scaled_df = df.copy()

    for col in num_cols:
        scaled_df[col] = df[col].fillna(df[col].median())
        means_dict[col] = scaled_df[col].mean()
        stds_dict[col] = scaled_df[col].std()

        scaled_df[col] = (scaled_df[col] - scaled_df[col].mean()) / scaled_df[col].std()
    
    return scaled_df