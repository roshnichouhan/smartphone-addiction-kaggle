import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier

from feature_engineering import create_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "smartphone_addiction_model.cbm"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("SMARTPHONE ADDICTION PREDICTION")
print("=" * 60)

print("\nLoading training data...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{DATA_PATH}"
    )

train = pd.read_csv(DATA_PATH)

print("Original shape:", train.shape)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

print("\nApplying feature engineering...")

train = create_features(train)

print("Shape after feature engineering:", train.shape)


# ============================================================
# 3. TARGET
# ============================================================

target = "addicted_label"

if target not in train.columns:
    raise ValueError(
        f"Target column '{target}' not found in dataset."
    )


# ============================================================
# 4. X AND y
# ============================================================

drop_columns = [target]

if "id" in train.columns:
    drop_columns.append("id")

X = train.drop(columns=drop_columns).copy()
y = train[target].copy()


print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 5. DETECT CATEGORICAL COLUMNS
# ============================================================
#
# IMPORTANT:
#
# DO NOT use:
#
# X.select_dtypes(
#     include=["object", "str", "category"]
# )
#
# because "str" can trigger:
#
# TypeError:
# numpy string dtypes are not allowed
#
# Instead, check every column safely.
# ============================================================

categorical_cols = []

for column in X.columns:

    dtype = X[column].dtype

    if (
        dtype == "object"
        or pd.api.types.is_categorical_dtype(dtype)
    ):
        categorical_cols.append(column)


# ============================================================
# 6. HANDLE STRING COLUMNS SAFELY
# ============================================================

for column in categorical_cols:

    # Convert to Python object/string values.
    X[column] = X[column].astype("object")

    # Handle missing values.
    X[column] = X[column].fillna("Missing")

    # Convert all categorical values to Python strings.
    X[column] = X[column].astype(str)


# ============================================================
# 7. IDENTIFY NUMERICAL COLUMNS
# ============================================================

numerical_cols = [
    column
    for column in X.columns
    if column not in categorical_cols
]


# ============================================================
# 8. HANDLE NUMERICAL VALUES
# ============================================================

for column in numerical_cols:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

    # Replace missing numerical values
    # with the median.

    median_value = X[column].median()

    if pd.isna(median_value):
        median_value = 0

    X[column] = X[column].fillna(median_value)


# ============================================================
# 9. DISPLAY FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

print("\nCategorical columns:")

if categorical_cols:
    for column in categorical_cols:
        print(" -", column)
else:
    print("None")


print("\nNumerical columns:")

if numerical_cols:
    for column in numerical_cols:
        print(" -", column)
else:
    print("None")


print("\nTotal features:", len(X.columns))
print("Categorical features:", len(categorical_cols))
print("Numerical features:", len(numerical_cols))


# ============================================================
# 10. FINAL DTYPE CHECK
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(X.dtypes)


# ============================================================
# 11. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAIN / VALIDATION SPLIT")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)


# ============================================================
# 12. CATBOOST MODEL
# ============================================================

model = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=100,
    allow_writing_files=False
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

model.fit(
    X_train,
    y_train,

    # Pass categorical column names
    cat_features=categorical_cols,

    # Validation dataset
    eval_set=(X_valid, y_valid),

    # Stop if validation AUC does not improve
    early_stopping_rounds=50
)

print("\nTraining completed successfully!")


# ============================================================
# 14. VALIDATION PROBABILITY
# ============================================================

y_valid_pred = model.predict_proba(
    X_valid
)[:, 1]


# ============================================================
# 15. ROC-AUC
# ============================================================

auc = roc_auc_score(
    y_valid,
    y_valid_pred
)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nValidation ROC-AUC: {auc:.6f}")


# ============================================================
# 16. SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

model.save_model(
    str(MODEL_PATH)
)

print("\nModel saved successfully!")
print("Model path:")
print(MODEL_PATH)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

feature_importance = model.get_feature_importance()

feature_names = X_train.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance
})


importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df.to_string(index=False)
)


# ============================================================
# 18. SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = MODEL_DIR / "feature_importance.csv"

importance_df.to_csv(
    importance_path,
    index=False
)

print("\nFeature importance saved:")
print(importance_path)


# ============================================================
# 19. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(f"\nROC-AUC: {auc:.6f}")

print("\nSaved files:")

print("Model:")
print(MODEL_PATH)

print("\nFeature importance:")
print(importance_path)

print("\nYou can now run the Streamlit app.")