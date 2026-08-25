import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Load data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")


# Target
target = "addicted_label"

X = train.drop(columns=["id", target])
y = train[target]

X_test = test.drop(columns=["id"])


# Columns
numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["str", "object", "category"]
).columns.tolist()


# Numerical preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])


print("Numerical columns:")
print(numerical_cols)

print("\nCategorical columns:")
print(categorical_cols)