from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

# Fields
RANDOM_STATE = 42
TEST_SIZE = 0.30

# ========================================
# --- 1. Data Loading and Preparation  ---
# ========================================

def load_dataset(csv_path: Path, target_col="diagnosis") -> tuple[pd.DataFrame, pd.Series]:
    # Load the breast cancer data set
    df = pd.read_csv(csv_path)

    # Drop id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Separate target and features
    y = df[target_col].map({"B": 0, "M": 1})
    X = df.drop(columns = [target_col])

    # Replace 0s with NaN
    X = X.replace(0, np.nan)

    return X, y

def split_data(X: pd.DataFrame, y: pd.Series) -> SplitData:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Testing
X, y = load_dataset(Path("breast-cancer.csv"))
print("Missing values per column:")
print(X.isna().sum())

split = split_data(X, y)
print(f"Train size: {split.X_train.shape}, Test size: {split.X_test.shape}")