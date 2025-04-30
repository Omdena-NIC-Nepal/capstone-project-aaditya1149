import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import streamlit as st

@st.cache_data

def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the preprocessed climate dataset from a CSV file.

    Args:
        file_path (Optional[str]): Path to the CSV file. If None, loads from the default path.

    Returns:
        pd.DataFrame: Loaded dataset with the 'year' column as integer type.
    """
    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', 'data', 'climate', 'climate_data.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    data = pd.read_csv(file_path)
    data['year'] = data['year'].astype(int)
    return data


def preprocess_data(
    data: pd.DataFrame,
    target_col: str = 'avg_max_temp',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Prepare the dataset for modeling by splitting and scaling.

    Args:
        data (pd.DataFrame): The full dataset.
        target_col (str): The name of the target column to predict.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple: Scaled training and testing data, and the fitted scaler.
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Identify numeric columns and scale them
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train, X_test, y_train, y_test, scaler


def save_model(model, model_name: str) -> None:
    """
    Save a trained machine learning model to disk.

    Args:
        model: Trained model to save.
        model_name (str): The name for saving the model (without extension).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)


def load_model(model_name: str):
    """
    Load a saved machine learning model from disk.

    Args:
        model_name (str): The name of the model file to load (without extension).

    Returns:
        The loaded model object.

    Raises:
        FileNotFoundError: If the model file is not found.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', f"{model_name}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found at: {model_path}")

    return joblib.load(model_path)
