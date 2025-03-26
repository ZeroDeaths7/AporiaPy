
#handle_missing_values(data)

#remove_duplicates()

#correct_data_types()

#balance_classes(data, method='SMOTE')  # SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler (look into these)

import numpy as np
import pandas as pd
from collections import Counter

def handle_missing_values(data, strategy="mean"):
    """
    Handles missing values in the dataset.
    - strategy: "mean", "median", or "mode".
    """
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:  # Numerical columns
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
            else:  # Categorical columns
                df[col].fillna(df[col].mode()[0], inplace=True)

    return df.to_numpy()



def remove_duplicates(data):
    """
    Removes duplicate rows from the dataset.
    """
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    return df.to_numpy()



def correct_data_types(data):
    """
    Ensures consistent data types across the dataset.
    - Converts numerical-like strings to numbers.
    - Converts boolean-like values to True/False.
    - Converts datetime-like strings to proper datetime format.
    """
    df = pd.DataFrame(data)

    for col in df.columns:
        # Try converting to numeric type
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass  # Ignore if conversion fails

        # Convert categorical-like data
        unique_values = df[col].dropna().unique()
        if set(unique_values) <= {"True", "False", "Yes", "No", 0, 1}:  # Boolean-like
            df[col] = df[col].map({"True": True, "False": False, "Yes": True, "No": False, 0: False, 1: True})
        
        # Convert datetime-like columns
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass  # Ignore if not a datetime column

    return df.to_numpy()




def balance_classes(data, method="undersampling"):
    """
    Balances class distribution using undersampling or oversampling.
    - method: "undersampling" (default), "oversampling"
    """
    df = pd.DataFrame(data)
    label_col = df.columns[-1]  # Assuming last column is the label (run some testing on this)
    class_counts = df[label_col].value_counts()

    min_class = class_counts.idxmin()
    min_count = class_counts.min()
    max_class = class_counts.idxmax()
    max_count = class_counts.max()

    if method == "undersampling":
        df_balanced = df.groupby(label_col).apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    elif method == "oversampling":
        df_balanced = df.copy()
        for cls in class_counts.index:
            if class_counts[cls] < max_count:
                additional_samples = df[df[label_col] == cls].sample(max_count - class_counts[cls], replace=True)
                df_balanced = pd.concat([df_balanced, additional_samples])

    return df_balanced.to_numpy()
