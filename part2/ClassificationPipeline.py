from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
# --- 0. Data Loading and Preparation  ---
# ========================================

def load_dataset(csv_path: Path, target_col="diagnosis") -> tuple[pd.DataFrame, pd.Series]:
    # Load the breast cancer data set
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print("Dataset loaded")

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
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# ========================================
# --- 1. Missing Data Imputation  ---
# ========================================

def impute_data(split: SplitData, strategy: str) -> SplitData:
    # Impute missing values in the training and testing sets
    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(split.X_train),
        columns=split.X_train.columns,
        index=split.X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(split.X_test),
        columns=split.X_test.columns,
        index=split.X_test.index
    )
    return SplitData(X_train=X_train_imputed, X_test=X_test_imputed, y_train=split.y_train, y_test=split.y_test)

def evaluate_imputation_strategies(split: SplitData, max_depth: int = 5) -> dict[str, float]:
    results = {}
    for strategy in ["mean", "median"]:
        imputed_split = impute_data(split, strategy)
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
        clf.fit(imputed_split.X_train, imputed_split.y_train)
        predictions = clf.predict(imputed_split.X_test)

        accuracy = accuracy_score(imputed_split.y_test, predictions)
        results[strategy] = accuracy
        print(f"{strategy.capitalize()} imputation -> Decision Tree (max_depth={max_depth}) "
              f"accuracy: {accuracy * 100:.2f}%")

    return results

X, y = load_dataset(Path("breast-cancer.csv"))
split = split_data(X, y)

imputation_results = evaluate_imputation_strategies(split, max_depth=5)

# Carry forward one strategy as "the complete dataset" for the rest of Part 2
full_split = impute_data(split, strategy="median")