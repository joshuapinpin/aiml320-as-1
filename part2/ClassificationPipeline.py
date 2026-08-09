from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

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

# ========================================
# --- 2. Normalisation Comparison  ---
# ========================================

def scale_data(split: SplitData, scaler) -> SplitData:
    """
    Fits a scaler on X_train only, then applies it to both X_train and X_test.
    scaler: an unfitted sklearn scaler instance, e.g. StandardScaler() or MinMaxScaler()
    """
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(split.X_train),
        columns=split.X_train.columns,
        index=split.X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(split.X_test),
        columns=split.X_test.columns,
        index=split.X_test.index
    )

    return SplitData(X_train=X_train_scaled, X_test=X_test_scaled, y_train=split.y_train, y_test=split.y_test)

def evaluate_normalisation_impact(
        split: SplitData,
        knn_neighbours: int = 9,
        dt_max_depth: int = 5
) -> dict[str, dict[str,float]]:
    """
    Compares KNN and Decision Tree accuracy with no scaling, standardization,
    and min-max scaling. One fixed hyperparameter setting per model.
    Returns {condition: {"KNN": accuracy, "DecisionTree": accuracy}}
    """

    conditions = {
        "No scaling": split,
        "Standardised": scale_data(split, StandardScaler()),
        "Min-Max scaled": scale_data(split, MinMaxScaler())
    }

    results = {}
    for condition_name, cond_split in conditions.items():
        knn = KNeighborsClassifier(n_neighbors=knn_neighbours)
        knn.fit(cond_split.X_train, cond_split.y_train)
        knn_accuracy = accuracy_score(cond_split.y_test, knn.predict(cond_split.X_test))

        dt = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=RANDOM_STATE)
        dt.fit(cond_split.X_train, cond_split.y_train)
        dt_accuracy = accuracy_score(cond_split.y_test, dt.predict(cond_split.X_test))

        results[condition_name] = {"KNN": knn_accuracy, "DecisionTree": dt_accuracy}
        print(f"{condition_name:15s} -> KNN (k={knn_neighbours}): {knn_accuracy * 100:.2f}%  |  "
              f"Decision Tree (max_depth={dt_max_depth}): {dt_accuracy * 100:.2f}%")
    return results

# ========================================
# --- Saving Results to CSV  ---
# ========================================

def save_imputation_results_csv(
        results: dict[str, float],
        path: Path=Path("imputation_results.csv")
):
    """
    results: {"mean": acc, "median": acc}
    """
    df = pd.DataFrame([
        {"Strategy": strategy, "Accuracy (%)": acc * 100}
        for strategy, acc in results.items()
    ])
    df.to_csv(path, index=False)
    print(f"Saved imputation results to {path}")

def save_normalisation_results_csv(
        results: dict[str, dict[str, float]],
        path: Path=Path("normalisation_results.csv")
):
    """
    results: {condition: {"KNN": acc, "DecisionTree": acc}}
    """
    df = pd.DataFrame([
        {
            "Condition": condition,
            "KNN Accuracy (%)": model_accs["KNN"] * 100,
            "Decision Tree Accuracy (%)": model_accs["DecisionTree"] * 100,
        }
        for condition, model_accs in results.items()
    ])
    df.to_csv(path, index=False)
    print(f"Saved normalization results to {path}")


# ================================================================================================================
# ================================================================================================================

def load_and_split():
    X, y = load_dataset(Path("breast-cancer.csv"))
    print("Missing values per column:")
    print(X.isna().sum())

    split = split_data(X, y)
    print(f"Train size: {split.X_train.shape}, Test size: {split.X_test.shape}")

def imputation_and_normalisation():
    X, y = load_dataset(Path("breast-cancer.csv"))
    split = split_data(X, y)

    imputation_results = evaluate_imputation_strategies(split, max_depth=5)

    # Carry forward one strategy as "the complete dataset" for the rest of Part 2
    full_split = impute_data(split, strategy="median")

    # This is to double-check that the scaling worked as standardised values are genuinely transformed
    std_split = scale_data(full_split, StandardScaler())
    print(full_split.X_train.iloc[0, :5].values)  # raw
    print(std_split.X_train.iloc[0, :5].values)  # should look nothing like the raw values

    # This is to check that the predictions are not identical between the raw and standardised data
    knn_raw = KNeighborsClassifier(n_neighbors=9).fit(full_split.X_train, full_split.y_train)
    knn_std = KNeighborsClassifier(n_neighbors=9).fit(std_split.X_train, std_split.y_train)
    preds_raw = knn_raw.predict(full_split.X_test)
    preds_std = knn_std.predict(std_split.X_test)
    print((preds_raw == preds_std).all())  # True = literally identical predictions
    print((preds_raw != preds_std).sum(), "differing predictions out of", len(preds_raw))

    normalisation_results = evaluate_normalisation_impact(full_split, knn_neighbours=9, dt_max_depth=5)

    save_imputation_results_csv(imputation_results)
    save_normalisation_results_csv(normalisation_results)

def main():
    print("\n========== Loading and Splitting ==========")
    # load_and_split()
    print("\n========== Imputation and Normalisation ==========")
    imputation_and_normalisation()

if __name__ == "__main__":
    main()