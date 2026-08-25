import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. LOAD DATA
# =========================================================

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
sample = pd.read_csv("../data/sample_submission.csv")

# =========================================================
# 2. CHECK SHAPE
# =========================================================

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("Sample Submission Shape:", sample.shape)


# =========================================================
# 3. FIRST 5 ROWS
# =========================================================

print("\n===== TRAIN HEAD =====")
print(train.head())

print("\n===== TEST HEAD =====")
print(test.head())


# =========================================================
# 4. DATA INFORMATION
# =========================================================

print("\n===== TRAIN INFO =====")
train.info()

print("\n===== TEST INFO =====")
test.info()


# =========================================================
# 5. STATISTICAL SUMMARY
# =========================================================

print("\n===== DESCRIPTIVE STATISTICS =====")
print(train.describe(include="all"))


# =========================================================
# 6. MISSING VALUES
# =========================================================

print("\n===== MISSING VALUES =====")
print(train.isnull().sum())


# =========================================================
# 7. UNIQUE VALUES
# =========================================================

print("\n===== UNIQUE VALUES =====")
print(train.nunique())


# =========================================================
# 8. TARGET DISTRIBUTION
# =========================================================

print("\n===== TARGET COUNTS =====")
print(train["addicted_label"].value_counts())

print("\n===== TARGET PERCENTAGE =====")
print(train["addicted_label"].value_counts(normalize=True))


# =========================================================
# 9. TARGET VISUALIZATION
# =========================================================

train["addicted_label"].value_counts().plot(kind="bar")

plt.title("Target Distribution")
plt.xlabel("Addicted Label")
plt.ylabel("Count")
plt.show()


# =========================================================
# 10. SEPARATE FEATURES AND TARGET
# =========================================================

target = "addicted_label"

X = train.drop(columns=[target])
y = train[target]


# =========================================================
# 11. FIND CATEGORICAL AND NUMERICAL COLUMNS
# =========================================================

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()


print("\n===== CATEGORICAL COLUMNS =====")
print(categorical_cols)

print("\n===== NUMERICAL COLUMNS =====")
print(numerical_cols)


print("\n===== CATEGORICAL VALUE COUNTS =====")

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(train[col].value_counts(dropna=False))


print("\n===== NUMERICAL SUMMARY =====")
print(train[numerical_cols].describe().T)


print("\n===== CORRELATION WITH TARGET =====")

correlation = train[numerical_cols].corrwith(
    train["addicted_label"]
).sort_values(ascending=False)

print(correlation)


print("\n===== TARGET RATE BY CATEGORY =====")

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(train.groupby(col, dropna=False)["addicted_label"].mean())