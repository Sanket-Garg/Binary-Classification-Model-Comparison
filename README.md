# Binary Classification Model Comparison

This notebook walks through a full supervised learning pipeline on a binary
classification dataset — from exploratory data analysis to training and
comparing three models: **Decision Tree**, **Random Forest**, and **Support
Vector Machine (SVM)**.

## Contents

- `binary-classification-model-comparison.ipynb` – the main analysis and modeling notebook
- `dataset_assignment1.csv` – input dataset (9 numeric features + binary `class` target)

## Project Structure / Workflow

1. **Data Loading & Inspection**
   - Load the CSV, check shape, dtypes, and missing values
   - Review value counts per column to check for class imbalance

2. **Exploratory Data Analysis (EDA)**
   - Univariate analysis: distribution plots for each feature
   - Boxplots to detect outliers (notably in `feature4`, `feature5`, `feature8`, `feature9`)
   - Bivariate analysis: feature distributions split by `class`
   - Multivariate analysis: pairplots and a correlation heatmap
     (e.g. `feature2` and `feature3` show high correlation, ~0.91)

3. **Preprocessing**
   - Train/test split (80/20, `random_state=42`)
   - Feature scaling with `StandardScaler`

4. **Model Training & Hyperparameter Tuning**
   For each of the three models, the notebook:
   - Trains a baseline (default hyperparameters) model
   - Runs `GridSearchCV` (5-fold CV, scored on F1) to tune hyperparameters
   - Retrains using the best parameters found

   | Model | Key hyperparameters tuned |
   |---|---|
   | Decision Tree | `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
   | Random Forest | `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
   | SVM | `C`, `gamma`, `kernel` |

5. **Evaluation & Comparison**
   - Confusion matrices for each tuned model
   - Accuracy, Precision, Recall, F1 Score
   - ROC curves per model, plus a combined ROC comparison plot
   - Final side-by-side comparison table across all three models

## Requirements

```
python >= 3.8
pandas
numpy
seaborn
matplotlib
scikit-learn
```

Install with:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## How to Run

1. Clone this repository
2. Make sure `dataset_assignment1.csv` is in the same directory as the notebook
3. Launch Jupyter and run all cells:

```bash
jupyter notebook binary-classification-model-comparison.ipynb
```

## Results Summary

The notebook produces, for each model:
- A confusion matrix
- Accuracy / Precision / Recall / F1 scores
- An ROC curve with AUC

These are compiled at the end into a combined comparison table and a combined
ROC plot so the three models can be evaluated side by side.

## Notes

- Random seed fixed at `0` / `random_state=42` for reproducibility.
- No missing values or major cleaning was required in the source dataset.
- Class imbalance was observed in the target (`class`) and in `feature9`,
  which is worth keeping in mind when interpreting accuracy vs. F1/recall.

## License

Add a license of your choice (e.g. MIT) if you intend to share this publicly.
