import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# =========================
# VERİYİ ÇEK
# =========================
data = fetch_ucirepo(id=144)
X = data.data.features
y = data.data.targets["class"]

# 1 = good -> 0
# 2 = bad  -> 1
y = y.map({1: 0, 2: 1})

# =========================
# FEATURE AYRIMI
# =========================
numerical_features = [
    "Attribute2", "Attribute5", "Attribute8",
    "Attribute11", "Attribute13", "Attribute16", "Attribute18"
]

categorical_features = [
    col for col in X.columns if col not in numerical_features
]

# =========================
# PREPROCESSOR
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

# =========================
# MODEL
# =========================
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# =========================
# TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

# =========================
# DEĞERLENDİRME
# =========================
y_pred = pipeline.predict(X_test)

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# =========================
# PERMUTATION IMPORTANCE
# =========================
result = permutation_importance(
    pipeline,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="f1"
)

feature_names = X_test.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values(by="importance_mean", ascending=False)

print("\n=== PERMUTATION FEATURE IMPORTANCE ===")
print(importance_df.head(15))

importance_df.to_csv("permutation_feature_importance.csv", index=False)
