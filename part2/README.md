# Part 2 — Exploring Classification Algorithms

## Requirements

- Python 3.10+
- Packages: `pandas`, `numpy`, `scikit-learn`

Install dependencies if needed:

```bash
pip install pandas numpy scikit-learn
```

## Files

- `ClassificationPipeline.py` — main pipeline (data loading, imputation, normalisation, classifier exploration, feature selection, PCA)
- `breast-cancer.csv` — dataset (must be in the same directory as the script, or update the path in the script)

## How to Run

The script is run directly from the command line:

```bash
python ClassificationPipeline.py
```

This executes `main()`, which runs all pipeline stages in sequence:

1. **Load & split** — loads `breast-cancer.csv`, drops the `id` column, encodes `diagnosis` (M=1, B=0), replaces `0` values with `NaN` (missing data indicator), and performs a 70:30 stratified train/test split (`random_state=42`).
2. **Imputation & normalisation** — compares mean vs. median imputation using a Decision Tree (`max_depth=5`), then compares no scaling, `StandardScaler`, and `MinMaxScaler` for KNN (`k=9`) and Decision Tree (`max_depth=5`).
3. **Classifier exploration** — sweeps KNN, Decision Tree, AdaBoost, and Random Forest across the hyperparameter grids specified in the assignment handout, reporting Accuracy, Precision, Recall, and F1 (positive class = malignant, `pos_label=1`).
4. **Feature selection** — computes Pearson correlation of each feature with the target, selects features with `abs(r) > 0.6`, and compares Decision Tree accuracy on the full vs. selected feature set.
5. **PCA** — applies PCA (after standardisation) and compares Decision Tree accuracy with and without PCA.

Each stage prints its results to the console and also saves them to CSV files in the working directory:

- `imputation_results.csv`
- `normalisation_results.csv`
- `classifier_results.csv`
- `correlation_results.csv`
- `feature_selection_results.csv`
- `pca_results.csv`

## Notes

- All preprocessing (imputation, scaling, PCA) is fit on `X_train` only and applied to `X_test` without refitting, to avoid data leakage.
- Feature selection uses `abs(pearson_r) > 0.6` rather than a signed threshold; this is documented and justified in the accompanying report.
- No AIML420-specific components (10-fold cross-validation, PCA-vs-feature-selection comparison) are included, as these are out of scope for this submission.