

import os
import numpy as np
import pandas as pd



def detect_file_type(fp):
    """
    Detects the file type based on the extension.

    Args:
        fp (str): File path.

    Returns:
        str: File type ('csv', 'json', 'image', 'audio', 'unknown').
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Error: File '{fp}' not found.")

    ext = os.path.splitext(fp)[1].lower()
    if ext == ".csv":
        return "csv"
    elif ext == ".json":
        return "json"
    elif ext in [".png", ".jpg", ".jpeg"]:
        return "image"
    elif ext in [".wav", ".mp3", ".flac"]:
        return "audio"
    else:
        return "unknown"




def preprocess_dataframe(df):
    """
    Preprocesses a DataFrame by converting categorical columns to numerical format.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame with numeric data.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.factorize(df[col])[0]  # Converts categories to numerical values
    return df





def load_dataset(fp):
    """
    Loads a CSV file and converts it into a 2D NumPy array.

    Args:
        fp (str): Path to the CSV file.

    Returns:
        np.ndarray: 2D array representation of the dataset.
    """
    if detect_file_type(fp) != "csv":
        raise ValueError(f"Error: Unsupported file type. Expected a CSV file, got '{fp}'.")

    try:
        df = pd.read_csv(fp)
        df = preprocess_dataframe(df)  # Convert categorical columns if needed
        data_array = df.to_numpy(dtype=np.float32)  # Ensure numeric conversion
        return data_array
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file '{fp}': {e}")




def load_json_dataset(fp):
    """
    Loads a JSON file and converts it into a 2D NumPy array.

    Args:
        fp (str): Path to the JSON file.

    Returns:
        np.ndarray: 2D array representation of the dataset.
    """
    if detect_file_type(fp) != "json":
        raise ValueError(f"Error: Unsupported file type. Expected a JSON file, got '{fp}'.")

    try:
        df = pd.read_json(fp)
        df = preprocess_dataframe(df)  # Convert categorical columns if needed
        data_array = df.to_numpy(dtype=np.float32)  # Ensure numeric conversion
        return data_array
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON file '{fp}': {e}")
