import pandas as pd

from catboost import CatBoostClassifier

from feature_engineering import create_features


# ==========================================
# 1. LOAD TEST DATA
# ==========================================

print("Loading test data...")

test = pd.read_csv("../data/test.csv")

print("Test shape:", test.shape)


# ==========================================
# 2. SAVE TEST IDs
# ==========================================

test_ids = test["id"]


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

print("\nCreating features...")

test = create_features(test)

print("Features created!")


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

print("\nLoading trained model...")

model = CatBoostClassifier()

model.load_model(
    "../model/smartphone_addiction_model.cbm"
)

print("Model loaded successfully!")


# ==========================================
# 5. REMOVE ID
# ==========================================

X_test = test.drop(columns=["id"])


# ==========================================
# 6. HANDLE CATEGORICAL MISSING VALUES
# ==========================================

categorical_cols = X_test.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

X_test[categorical_cols] = X_test[
    categorical_cols
].fillna("Missing")


# ==========================================
# 7. PREDICT PROBABILITY
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 8. CREATE SUBMISSION DATAFRAME
# ==========================================

submission = pd.DataFrame({
    "id": test_ids,
    "addicted_label": predictions
})


# ==========================================
# 9. SAVE SUBMISSION
# ==========================================

submission.to_csv(
    "../submission.csv",
    index=False
)


# ==========================================
# 10. CHECK SUBMISSION
# ==========================================

print("\n===================================")
print("SUBMISSION CREATED")
print("===================================")

print("Submission shape:", submission.shape)

print("\nFirst 5 predictions:")
print(submission.head())

print("\nPrediction statistics:")
print(submission["addicted_label"].describe())

print("\nSaved to:")
print("../submission.csv")