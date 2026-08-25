import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from feature_engineering import create_features


# ==========================================
# 1. LOAD DATA
# ==========================================

train = pd.read_csv("../data/train.csv")

# Apply feature engineering
train = create_features(train)

target = "addicted_label"


# ==========================================
# 2. X AND y
# ==========================================

X = train.drop(columns=["id", target])
y = train[target]


# ==========================================
# 3. CATEGORICAL COLUMNS
# ==========================================

categorical_cols = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("Categorical columns:")
print(categorical_cols)


# ==========================================
# 4. HANDLE CATEGORICAL MISSING VALUES
# ==========================================

X[categorical_cols] = X[categorical_cols].fillna("Missing")


# ==========================================
# 5. TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nX_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)


# ==========================================
# 6. MODEL
# ==========================================

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


# ==========================================
# 7. TRAIN
# ==========================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
    cat_features=categorical_cols,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=50
)

print("\nTraining completed!")


# ==========================================
# 8. VALIDATION PROBABILITY
# ==========================================

y_valid_pred = model.predict_proba(X_valid)[:, 1]


# ==========================================
# 9. ROC-AUC
# ==========================================

auc = roc_auc_score(
    y_valid,
    y_valid_pred
)

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")
print(f"Validation ROC-AUC: {auc:.6f}")

# ==========================================
# SAVE TRAINED MODEL
# ==========================================

model.save_model(
    "../model/smartphone_addiction_model.cbm"
)

print("\nModel saved successfully!")
print("Model path: ../model/smartphone_addiction_model.cbm")


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

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

print("\n===================================")
print("FEATURE IMPORTANCE")
print("===================================")

print(importance_df.to_string(index=False))

import matplotlib.pyplot as plt

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

print(f"Validation ROC-AUC: {auc:.6f}")